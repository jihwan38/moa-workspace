#!/usr/bin/env python3
"""Read-only input/ELF/image binding for the complete fixed-H64 host numerical diagnostic."""
from pathlib import Path
import hashlib
import importlib.util
import json

TREE = Path(__file__).resolve().parents[2]
ROOT = TREE.parent
RUN = ROOT / 'runs/a003-int4-h64-complete-device-build'


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
    assert len(inputs) == 432
    for path, expected in inputs.items():
        assert sha(TREE / path) == expected, path
    previous = read(ROOT / 'runs/a005-int4-h64-device-build/source-hashes.json')
    assert len(previous) == 429
    for path, expected in previous.items():
        assert inputs[path] == expected, path
    extra = {'src/bin/support/int4_hadamard64_transform.rs',
             'src/bin/support/int4_hadamard64_complete_fixture.rs', 'src/bin/test_int4_hadamard64_complete.rs'}
    assert set(inputs) - set(previous) == extra
    plan = read(TREE / 'experiments/int4-hadamard64-complete-diagnostic/plan.json')
    for path, expected in plan['dependencies'].items():
        assert inputs[path] == expected, path
    artifact = run['built_artifacts']['target/release/test_int4_hadamard64_complete']
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
    mse_path = ROOT / 'runs/a004-int4-weight-mse-device-build/built/target/release/test_int4_weight_mse'
    mse = registry.read_artifact(mse_path)
    assert mse['elf_sha256'] == '64dd3b267d6bfe13003172d57a26414d1daa76a7f0b26ed13a9cb3824e7caefc'
    new, old, mse_images = index(current), index(prior), index(mse)
    assert len(new) == len(old) == len(mse_images) == 19 and set(new) == set(old) == set(mse_images)
    outarg = next(x for x in run['command'] if x.startswith('FURIOSA_OPT_OUT_DIR='))
    out = ROOT / outarg.split('=', 1)[1].removeprefix('/round2/') / 'furiosa-opt-gemma4'
    comparisons = []
    for name, item in sorted(new.items()):
        assert (item['sha256'], item['bytes']) == (old[name]['sha256'], old[name]['bytes']), name
        assert (item['sha256'], item['bytes']) == (mse_images[name]['sha256'], mse_images[name]['bytes']), name
        fresh = out / (name + '.bin')
        assert sha(fresh) == item['sha256'] and fresh.stat().st_size == item['bytes'], name
        comparisons.append({'path': name, 'sha256': item['sha256'], 'bytes': item['bytes'],
                            'reference_actuals': [323213, 323322], 'both_prior_equal': True, 'fresh_out_equal': True})
    q020_path = ROOT / 'runs/a003-q020-mse-parallel-server-build/built/target/release/server'
    q020 = registry.read_artifact(q020_path)
    control = 'furiosa_opt_gemma4::ops_group_ffn_projection_storegroup::sliding_projection_owned'
    q = [x for x in q020['records'] if x['path'] == control and not x['key']]
    assert len(q) == 1
    assert (new[control]['sha256'], new[control]['bytes']) == (q[0]['sha256'], q[0]['bytes'])
    return {'run': run['name'], 'source_commit': run['commit'], 'functional_source': '70812693',
            'inputs': len(inputs), 'prior_429_inputs_unchanged': True, 'new_inputs': sorted(extra),
            'dependencies_unchanged': True, 'artifact': artifact, 'registry': current,
            'comparisons': comparisons, 'all19_prior_and_fresh_equal': True,
            'q020_control_exact': True, 'new_device_functions': 0,
            'prior_actual_323213': {'path': str(prior_path), 'sha256': prior['elf_sha256']},
            'prior_actual_323322': {'path': str(mse_path), 'sha256': mse['elf_sha256']},
            'q020': {'path': str(q020_path), 'sha256': q020['elf_sha256']},
            'parser_sha256': sha(parser),
            'scope': 'Complete fixed-H64 numerical diagnostic with CPU activation roundtrips. No new device function. Identity only; no actual numerical, task-quality or speed result. Replay checks all13 field values, but residual/rotated_prefix/rotated_hidden are reallocated and do not prove same-allocation reuse.'}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
