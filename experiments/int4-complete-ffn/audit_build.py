#!/usr/bin/env python3
"""Read-only captured-ELF/source parity. No compiler, SDK or NPU execution."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument('--run', default='a005-int4-complete-ffn-device-build')
    args = ap.parse_args()
    root = args.root
    tree = root / 'int4-complete-ffn-probe'
    reader = root / 'b16-f4-mse-lm-fp8/experiments/sdk09-cache-audit/audit_registry.py'
    spec = importlib.util.spec_from_file_location('registry_reader', reader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runpath = root / 'runs' / args.run
    run = json.loads((runpath / 'run.json').read_text())
    assert run['exit_code'] == 0 and run['build_inputs_unchanged']
    sources = json.loads((runpath / 'source-hashes.json').read_text())
    for name, expected in sources.items():
        assert digest(tree / name) == expected, name
    art = run['built_artifacts']['target/release/test_int4_complete_ffn']
    current = module.read_artifact(art['path'])
    assert current['elf_sha256'] == art['sha256']
    indexed = {r['path']: r for r in current['records'] if not r['key']}
    comparisons = []
    old_artifacts = {}

    def compare_entries(label, path, names, required=True):
        old = module.read_artifact(root / path)
        old_artifacts[label] = {k: v for k, v in old.items() if k != 'records'}
        old_index = {r['path']: r for r in old['records'] if not r['key']}
        for name in names:
            full = 'furiosa_opt_gemma4::' + name
            assert full in indexed, ('missing current', full)
            if full not in old_index:
                comparisons.append({'label': label, 'path': full, 'status': 'prior_image_unavailable'})
                assert not required, ('missing prior', full)
                continue
            a, b = indexed[full], old_index[full]
            equal = (a['bytes'], a['sha256']) == (b['bytes'], b['sha256'])
            comparisons.append({'label': label, 'path': full, 'status': 'equal' if equal else 'MISMATCH', 'current': a, 'prior': b})
            assert equal, full

    compare_entries('actual323014', 'runs/a002-int4-group960-bound-device-build/built/target/release/test_int4_group960_bound',
                    [f'ops_int4_group960_{m}::project_{p}' for m in ['projection', 'reference'] for p in ['upgate', 'down']])
    compare_entries('actual323029_323030', 'runs/a006-rational-fused-pack-device-build/built/target/release/test_int4_rational_fused_pack',
                    ['ops_int4_rational_fused_pack::quantize_pack_input_aligned', 'ops_int4_rational_fused_pack::quantize_pack_hidden'])
    compare_entries('q020_frozen_server', 'runs/a003-q020-mse-parallel-server-build/built/target/release/server',
                    ['ops_group_ffn_projection_storegroup::sliding_projection_owned',
                     'ops_norm_blocked_residual::post_feedforward', 'ops_norm_wide::post_attention_p128',
                     'ops_norm_wide::rmsnorm_p128', 'ops_norm_wide::post_feedforward_p128',
                     'ops_prepack_alloc::prepack_up', 'ops_prepack_alloc::reblock_up',
                     'ops_prepack_alloc::prepack_down', 'ops_prepack_alloc::reblock_down',
                     'ops_ffn_projection_join::join'], required=False)
    for name, directory in [('prefix_sliding_bf16', 'a002'), ('geglu_bf16', 'a004')]:
        full = 'furiosa_opt_gemma4::ops_int4_ffn_bridges::' + name
        path = root / f'int4-complete-ffn-bridges/experiments/int4-complete-ffn-bridges/kernel-out-{directory}/furiosa-opt-gemma4/{full}.bin'
        before = path.stat()
        prior_hash = digest(path)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        a = indexed[full]
        equal = a['sha256'] == prior_hash and a['bytes'] == before.st_size
        comparisons.append({'label': 'bridge_resource_only', 'path': full, 'status': 'equal' if equal else 'MISMATCH',
                            'current': a, 'prior_path': str(path), 'prior_sha256': prior_hash})
        assert equal, full
    print(json.dumps({'run': args.run, 'build_commit': run['commit'], 'source_inputs_unchanged': len(sources),
                      'parser_sha256': digest(reader), 'current': current, 'prior_artifacts': old_artifacts,
                      'comparisons': comparisons, 'no_actual_complete_chain_claim': True}, indent=2))


if __name__ == '__main__':
    main()
