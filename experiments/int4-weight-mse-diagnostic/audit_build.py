#!/usr/bin/env python3
"""Read-only input/ELF/image binding for the fixed-MSE host diagnostic."""
from pathlib import Path
import hashlib
import importlib.util
import json

TREE = Path(__file__).resolve().parents[2]
ROOT = TREE.parent
RUN = ROOT / 'runs/a004-int4-weight-mse-device-build'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def index(artifact):
    assert all(not x['key'] for x in artifact['records'])
    result = {x['path']: x for x in artifact['records']}
    assert len(result) == len(artifact['records'])
    return result


def main():
    run = read(RUN / 'run.json')
    assert run['exit_code'] == 0 and run['build_inputs_unchanged']
    assert run['fresh_package']['contents_changed'] is False
    inputs = read(RUN / 'source-hashes.json')
    assert len(inputs) == 427
    for path, expected in inputs.items():
        assert sha(TREE / path) == expected, path
    previous = read(ROOT / 'runs/a012-int4-grouped-threegraph-device-build/source-hashes.json')
    assert len(previous) == 424
    for path, expected in previous.items():
        assert inputs[path] == expected, path
    extra = {'src/bin/support/int4_weight_mse.rs',
             'src/bin/support/int4_weight_mse_fixture.rs', 'src/bin/test_int4_weight_mse.rs'}
    assert set(inputs) - set(previous) == extra
    plan = read(TREE / 'experiments/int4-weight-mse-diagnostic/plan.json')
    for path, expected in plan['dependencies'].items():
        assert inputs[path] == expected, path
    artifact = run['built_artifacts']['target/release/test_int4_weight_mse']
    captured = Path(artifact['path'])
    assert sha(captured) == artifact['sha256'] and captured.stat().st_size == artifact['size']
    parser = ROOT / 'b16-f4-mse-lm-fp8/experiments/sdk09-cache-audit/audit_registry.py'
    spec = importlib.util.spec_from_file_location('registry', parser)
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    current = registry.read_artifact(captured)
    prior_path = ROOT / 'runs/a002-int4-grouped-complete-ffn-device-build/built/target/release/test_int4_grouped_complete_ffn'
    prior = registry.read_artifact(prior_path)
    assert prior['elf_sha256'] == '3444ef561e2d39d6a15ec6340548d07e65b873d2884b62ac04f9ea2ccd66cdd4'
    new, old = index(current), index(prior)
    assert len(new) == len(old) == 19 and set(new) == set(old)
    outarg = next(x for x in run['command'] if x.startswith('FURIOSA_OPT_OUT_DIR='))
    out = ROOT / outarg.split('=', 1)[1].removeprefix('/round2/') / 'furiosa-opt-gemma4'
    comparisons = []
    for name, item in sorted(new.items()):
        assert (item['sha256'], item['bytes']) == (old[name]['sha256'], old[name]['bytes']), name
        fresh = out / (name + '.bin')
        assert sha(fresh) == item['sha256'] and fresh.stat().st_size == item['bytes'], name
        comparisons.append({'path': name, 'sha256': item['sha256'], 'bytes': item['bytes'],
                            'reference_actual': 323213, 'prior_equal': True, 'fresh_out_equal': True})
    q020_path = ROOT / 'runs/a003-q020-mse-parallel-server-build/built/target/release/server'
    q020 = registry.read_artifact(q020_path)
    control = 'furiosa_opt_gemma4::ops_group_ffn_projection_storegroup::sliding_projection_owned'
    q = [x for x in q020['records'] if x['path'] == control and not x['key']]
    assert len(q) == 1
    assert (new[control]['sha256'], new[control]['bytes']) == (q[0]['sha256'], q[0]['bytes'])
    return {'run': run['name'], 'source_commit': run['commit'], 'functional_source': '4d3dd463',
            'inputs': len(inputs), 'prior_424_inputs_unchanged': True, 'new_inputs': sorted(extra),
            'dependencies_unchanged': True, 'artifact': artifact, 'registry': current,
            'comparisons': comparisons, 'all19_prior_and_fresh_equal': True,
            'q020_control_exact': True, 'new_device_functions': 0,
            'prior_actual_323213': {'path': str(prior_path), 'sha256': prior['elf_sha256']},
            'q020': {'path': str(q020_path), 'sha256': q020['elf_sha256']},
            'parser_sha256': sha(parser),
            'scope': 'Seven-call numerical diagnostic build identity only; no actual MSE output, model/task quality or speed result.'}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
