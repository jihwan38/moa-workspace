#!/usr/bin/env python3
"""Verify frozen O16->O32 source scope and factored physical coordinates."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sections(text):
    a = text.index('pub(crate) fn project_output(')
    b = text.index('pub(crate) type OutputEpilogueSlices', a)
    c = text.index('// O shard-specific quantization:', b)
    d = text.index('// Independently derived runtime power-of-two scaling.', c)
    return text[:a], text[a:b], text[b:c], text[c:d], text[d:]


def main():
    b = sections((HERE / 'projection.original.rs').read_text())
    c = sections((HERE / 'projection.rs').read_text())
    assert b[0] == c[0] and b[2] == c[2] and b[4] == c[4]
    assert b[1].split('{', 1)[0] == c[1].split('{', 1)[0]
    expected = b[1]
    for old, new in [('type OutputRows = m![H / 120 % 16, 1 # 16];', 'type OutputRows = m![H / 240 % 8, 1 # 32];'), ('Qs % 256', 'Qs % 128'), ('Qs / 32 % 8', 'Qs / 32 % 4'), ('Qs / 64 % 4', 'Qs / 64 % 2'), ('H % 120', 'H % 240'), ('H / 4 % 30', 'H / 4 % 60'), ('all 120 local', 'all 240 local'), ('H=1920*cluster+120*group+local', 'H=1920*cluster+240*group+local')]:
        expected = expected.replace(old, new)
    assert c[1] == expected
    expected = b[3]
    for old, new in [('256', '128'), ('H / 120 % 16', 'H / 240 % 8'), ('Qs / 16 % 16', 'Qs / 16 % 8'), ('Qs / 8 % 32', 'Qs / 8 % 16'), ('Qs / 4 % 64', 'Qs / 4 % 32'), ('Qs / 32 % 8', 'Qs / 32 % 4')]:
        expected = expected.replace(old, new)
    assert c[3] == expected
    baseline = json.loads((HERE / 'baseline-source-manifest.json').read_text())
    compiled = json.loads((HERE / 'candidate-source-manifest.json').read_text())
    assert set(baseline) == set(compiled)
    assert [n for n in baseline if baseline[n] != compiled[n]] == ['src/device/sliding/projection.rs']
    for name, digest in baseline.items():
        assert sha(ROOT / 'moa-dev/experiments/artifacts/e47-output-sixteen-columns' / name) == digest, name
    assert compiled['src/device/sliding/projection.rs'] == sha(HERE / 'projection.rs')
    rows, tiles, columns = set(), set(), set()
    for h in range(3840):
        cluster, group, local = h // 1920, h // 240 % 8, h % 240
        assert h == 1920 * cluster + 240 * group + local
        rows.add((cluster, group, local))
        for column in range(32):
            coordinate = (cluster, group * 32 + column, local)
            assert coordinate not in tiles
            tiles.add(coordinate)
        assert (local // 4) * 4 + local % 4 == local
        assert (h // 480) * 480 + h % 480 == h
    for column in range(32):
        for local in range(128):
            q = 128 * column + local
            assert q // 128 == column and q % 128 == local and q not in columns
            columns.add(q)
    assert len(rows) == 3840 and len(tiles) == 122880 and columns == set(range(4096))
    graph = json.loads((HERE / 'candidate-graph.json').read_text())
    reductions = [v['description'] for v in graph['operators'].values()
                  if 'GReduceAddF [ratio=32]' in v['description']]
    assert len(reductions) == 1
    reduction = reductions[0]
    assert 'output_element_type: bf16' in reduction
    assert 'commit_in_size: 8B (=4EA)' in reduction
    assert 'commit_projection: 0+[None=60:4, None=4:1]' in reduction
    schedule = json.loads((HERE / 'candidate.json').read_text())
    assert max(i['lifetime']['end'] for i in schedule['instructions']) == 37344
    result = {
        'status': 'PASS', 'candidate_projection_sha256': sha(HERE / 'projection.rs'),
        'baseline_files_match_e47_snapshot': True, 'checked_source_and_manifest_files': len(baseline),
        'only_O_core_and_private_quantizer_layout_changed': True,
        'qkv_public_ops_h480_epilogue_unchanged': True,
        'row_coordinates_checked': len(rows), 'physical_row_column_tiles_checked': len(tiles),
        'column_coordinates_checked': len(columns), 'cartesian_weight_elements': 3840 * 4096,
        'clusters': 2, 'row_groups_per_cluster': 8, 'columns_per_row_group': 32,
        'local_rows': 240, 'local_columns': 128, 'all512_slices_used': True,
        'weight_fp8_bytes_per_slice': 30720, 'bf16_input_bytes_per_slice': 256,
        'paired_fp8_and_trf_payload_bytes_per_slice': 256, 'hi_f32_vrf_bytes_per_slice': 512,
        'f32_partial_with_lane_padding_bytes_per_slice': 7680,
        'reduced_bf16_physical_local_bytes_per_slice': 480,
        'bf16_commit_packets_per_slice': 60, 'bf16_values_per_commit': 4,
        'compiled_reduce_ratio': 32, 'compiled_commit_stride_verified': True,
        'compiled_candidate_cycles': 37344,
        'precision_change': 'Per-column scale groups256->128; F32 column reduction16->32. Same BF16 boundaries, no bitwise/all-input guarantee.',
        'hardware_correctness_checked': False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
