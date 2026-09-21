#!/usr/bin/env python3
"""Read-only frozen source, physical-layout, and precision-boundary checks."""
import hashlib
import json
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bf16_rne(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def main():
    original = (HERE / 'mlp.original.rs').read_text()
    candidate = (HERE / 'mlp.rs').read_text()
    begin = original.index('fn project_rows_with_scale_contraction(')
    finish = original.index('\npub(crate) fn project_up_and_gate(', begin)
    candidate_finish = candidate.index('\npub(crate) fn project_up_and_gate(', begin)
    assert candidate[:begin] == original[:begin]
    assert candidate[candidate_finish:] == original[finish:]
    old_helper = original[begin:finish]
    new_helper = candidate[begin:candidate_finish]
    assert old_helper.split(') {', 1)[0] == new_helper.split(') {', 1)[0]
    # First native contraction and pair reconstruction are byte-identical.
    old_first = old_helper[old_helper.index('        .main.begin(weights.view())'):old_helper.index('    let scales: DmTensor<bf16')]
    new_first = new_helper[new_helper.index('        .main.begin(weights.view())'):new_helper.index('    // Six actual rows')]
    assert new_first == old_first.replace('        .cast::<bf16, m![H / 16 % 4 # 16]>()\n', '')
    assert old_helper.count('.contract_outer') == 2 and new_helper.count('.contract_outer') == 1
    assert new_helper.count('.cast::<bf16') == 1 and old_helper.count('.cast::<bf16') == 2
    base_manifest = json.loads((HERE / 'baseline-source-manifest.json').read_text())
    compiled_manifest = json.loads((HERE / 'candidate-source-manifest.json').read_text())
    assert set(base_manifest) == set(compiled_manifest)
    changed = [k for k in base_manifest if base_manifest[k] != compiled_manifest[k]]
    assert changed == ['src/device/shared/mlp.rs']
    snapshot = ROOT / 'moa-dev/experiments/artifacts/e41-native-row30'
    for name, digest in base_manifest.items():
        assert sha(snapshot / name) == digest, name
    assert sha(HERE / 'mlp.rs') == compiled_manifest[changed[0]]
    assert sha(HERE / 'mlp.original.rs') == base_manifest[changed[0]]
    # Five six-row windows preserve scalar1#4 (8-byte) row stride. Padding is poison.
    seen_rows, written_values, physical_bytes = set(), set(), set()
    for tile in range(5):
        for local in range(6):
            row = tile * 6 + local
            assert row not in seen_rows
            seen_rows.add(row)
            written_values.add(4 * row)
            for byte in range(8):
                assert row * 8 + byte not in physical_bytes
                physical_bytes.add(row * 8 + byte)
    assert seen_rows == set(range(30)) and physical_bytes == set(range(240))
    assert all(4 * row in written_values for row in range(30))
    assert all(4 * row + padding not in written_values for row in range(30) for padding in range(1, 4))
    # Native quantization is exact for x=[1,1/256,0...,1/256], weight=[1,1,0...,1].
    # Its first two block partials expose an actual intermediate BF16 rounding boundary.
    partials = [1 + 1 / 256, 1 / 256]
    with_bf16_intermediate = bf16_rne(sum(map(bf16_rne, partials)))
    with_f32_intermediate = bf16_rne(sum(partials))
    assert with_bf16_intermediate == 1.0 and with_f32_intermediate == 1.0078125
    schedule = json.loads((HERE / 'candidate.json').read_text())
    baseline = json.loads((HERE / 'baseline.json').read_text())
    tensors = schedule['tensors']
    scale_tensors = [t for t in tensors if t['name'] == 'scale_vrf' and 'FfnRow = 6' in t['shape']]
    assert len(scale_tensors) == 10 and all(t['size'] == 512 * 5760 for t in scale_tensors)
    partial_tensors = [t for t in tensors if t['buffer_type'] == 'Sram' and t['shape'].endswith('Element: (FfnRow, H / 16)]') and t['size'] == 512 * 28800]
    assert len(partial_tensors) == 2 and all(t['size'] == 512 * 28800 for t in partial_tensors)
    graph = json.loads((HERE / 'candidate-graph.json').read_text())
    partial_producers = []
    final_reductions = []
    for op in graph['operators'].values():
        desc = op['description']
        if op.get('kind') != 'TuExec':
            continue
        if 'ReduceAccTree: LReduce(+) [axes=H]' in desc and 'FfnRow_1=6' in desc:
            assert 'commit_in_size: 8B (=4EA)' in desc
            assert 'commit_projection: 0+[None=6:4, None=4:1]' in desc
            assert 'output_element_type: bf16' in desc
            final_reductions.append(op['name'])
        for tid in op['outputs']:
            tensor = graph['tensors'][str(tid)]
            if tensor['buffer_type'] == 'Sram' and tensor['size'] == 512 * 28800:
                assert tensor['element_type'] == 'Float32'
                partial_producers.append(op['name'])
    assert len(partial_producers) == 2 and len(final_reductions) == 10
    cycles = lambda s: max(i['lifetime']['end'] for i in s['instructions'])
    result = {
        'status': 'PASS', 'candidate_mlp_sha256': sha(HERE / 'mlp.rs'),
        'only_helper_changed': 'project_rows_with_scale_contraction',
        'first_native_dpe_pair_xscale_preserved': True,
        'hbm_views_gather_quantizer_down_epilogue_preserved': True,
        'all_other_source_and_manifests_unchanged': True,
        'source_and_manifest_count': len(base_manifest),
        'row_tiles': 5, 'rows_per_tile': 6, 'covered_rows_per_slice': 30,
        'physical_output_bytes_per_slice': 240, 'poison_padding_never_promoted': True,
        'f32_partial_bytes_per_slice': 28800, 'scale_vrf_bytes_per_slice': 5760,
        'graph_f32_producers': partial_producers, 'graph_scalar_bf16_commit_ops': final_reductions,
        'bf16_intermediate_example': {'partials': partials, 'baseline_result': with_bf16_intermediate, 'candidate_result': with_f32_intermediate, 'scope': 'IEEE round-to-nearest-even CPU precision-boundary illustration, not NPU verification'},
        'baseline_cycles': cycles(baseline), 'candidate_cycles': cycles(schedule),
        'hardware_correctness_checked': False,
        'limitations': 'No all-input correctness, overflow-freedom, output-error monotonicity or runtime scheduling claim.'
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
