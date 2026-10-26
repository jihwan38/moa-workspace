#!/usr/bin/env python3
"""Read-only captured-input and reused-device-image audit. No compiler/NPU calls."""
from pathlib import Path
import argparse, hashlib, importlib.util, json


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/home/jihwan38/moa-worktrees/round2'))
    ap.add_argument('--run', default='a002-int4-grouped-complete-ffn-device-build')
    args = ap.parse_args()
    root = args.root.resolve()
    tree = root / 'int4-grouped-complete-ffn'
    parser = root / 'b16-f4-mse-lm-fp8/experiments/sdk09-cache-audit/audit_registry.py'
    spec = importlib.util.spec_from_file_location('registry', parser)
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    runroot = root / 'runs' / args.run
    run = json.loads((runroot / 'run.json').read_text())
    assert run['exit_code'] == 0 and run['build_inputs_unchanged']
    assert run['fresh_package']['contents_changed'] is False
    inputs = json.loads((runroot / 'source-hashes.json').read_text())
    for p, h in inputs.items():
        assert sha(tree / p) == h, p
    old_inputs = json.loads((root / 'runs/a004-int4-grouped-activation-projection-device-build/source-hashes.json').read_text())
    assert len(old_inputs) == 418
    for p, h in old_inputs.items():
        if p != 'src/lib.rs':
            assert inputs[p] == h, p
    extra = {'src/ops_int4_grouped_activation_store.rs', 'src/bin/test_int4_grouped_complete_ffn.rs',
             'src/bin/support/int4_grouped_complete_fixture.rs'}
    assert set(inputs) - set(old_inputs) == extra
    quant_inputs = json.loads((root / 'runs/a007-grouped-activation-store-device/source-hashes.json').read_text())
    assert inputs['src/ops_int4_grouped_activation_store.rs'] == quant_inputs['src/ops_int4_grouped_activation_store.rs']
    artifact = run['built_artifacts']['target/release/test_int4_grouped_complete_ffn']
    current = registry.read_artifact(artifact['path'])
    assert current['elf_sha256'] == artifact['sha256']
    assert Path(artifact['path']).stat().st_size == artifact['size']
    refs = {
        'actual323067_complete': root / 'runs/a005-int4-complete-ffn-device-build/built/target/release/test_int4_complete_ffn',
        'actual323173_quantizer': root / 'runs/a007-grouped-activation-store-device/built/target/release/test_int4_grouped_activation_store',
        'actual323172_projection': root / 'runs/a004-int4-grouped-activation-projection-device-build/built/target/release/test_int4_grouped_activation_projection',
        'q020_production': root / 'runs/a003-q020-mse-parallel-server-build/built/target/release/server',
    }
    ref_artifacts = {k: registry.read_artifact(p) for k, p in refs.items()}
    def index(artifact):
        assert all(not x['key'] for x in artifact['records'])
        rows = {x['path']: x for x in artifact['records']}
        assert len(rows) == len(artifact['records'])
        return rows
    ci = index(current)
    ri = {k: index(v) for k, v in ref_artifacts.items() if k != 'q020_production'}
    # q020 contains generic entries too; select this exact nongeneric control.
    qi = {x['path']: x for x in ref_artifacts['q020_production']['records'] if not x['key']}
    old = ri['actual323067_complete']
    removed_prefixes = ('furiosa_opt_gemma4::ops_int4_group960_', 'furiosa_opt_gemma4::ops_int4_rational_fused_pack::')
    reused = {k: v for k, v in old.items() if not k.startswith(removed_prefixes)}
    expected = set(reused) | set(ri['actual323173_quantizer']) | set(ri['actual323172_projection'])
    assert len(reused) == 13 and len(ci) == 19 and set(ci) == expected
    out_arg = next(x for x in run['command'] if x.startswith('FURIOSA_OPT_OUT_DIR='))
    out = root / out_arg.split('=', 1)[1].removeprefix('/round2/') / 'furiosa-opt-gemma4'
    comparisons = []
    for name in sorted(ci):
        group = ('actual323067_complete' if name in reused else
                 'actual323173_quantizer' if name in ri['actual323173_quantizer'] else 'actual323172_projection')
        prior = ri[group][name]
        assert (ci[name]['bytes'], ci[name]['sha256']) == (prior['bytes'], prior['sha256']), name
        fresh_bin = out / (name + '.bin')
        assert fresh_bin.stat().st_size == ci[name]['bytes'] and sha(fresh_bin) == ci[name]['sha256'], name
        comparisons.append({'path': name, 'prior_actual': group, 'sha256': ci[name]['sha256'],
                            'bytes': ci[name]['bytes'], 'prior_image_equal': True, 'fresh_out_bin_equal': True})
    control = 'furiosa_opt_gemma4::ops_group_ffn_projection_storegroup::sliding_projection_owned'
    assert (ci[control]['bytes'], ci[control]['sha256']) == (qi[control]['bytes'], qi[control]['sha256'])
    return {'run': args.run, 'source_commit': run['commit'], 'build_result': run['result'],
            'input_hashes': len(inputs), 'projection_donor_inputs_preserved_except_additive_lib': len(old_inputs) - 1,
            'new_inputs': sorted(extra), 'quantizer_source_matches_actual_component': True,
            'all_inputs_match_capture': True, 'parser_sha256': sha(parser), 'captured_artifact': artifact,
            'registry': current, 'comparisons': comparisons,
            'reference_artifacts': {k: {'path': str(refs[k]), 'elf_sha256': v['elf_sha256']} for k, v in ref_artifacts.items()},
            'q020_whole_graph_control_exact': True, 'all19_prior_and_fresh_images_exact': True,
            'scope': 'Build/input/image binding only. Complete grouped correctness, 10% speed gate and model quality are not established.'}

if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
