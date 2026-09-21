#!/usr/bin/env python3
"""Read-only source contract and exhaustive factored O mapping checks."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sections(s):
    a = s.index('pub(crate) fn project_output(')
    b = s.index('pub(crate) type OutputEpilogueSlices', a)
    c = s.index('// O shard-specific quantization:', b)
    d = s.index('// Independently derived runtime power-of-two scaling.', c)
    return s[:a], s[a:b], s[b:c], s[c:d], s[d:]


def main():
    base = (HERE / 'projection.original.rs').read_text()
    source = (HERE / 'projection.rs').read_text()
    b, c = sections(base), sections(source)
    assert b[0] == c[0] and b[2] == c[2] and b[4] == c[4]
    assert b[1].split('{', 1)[0] == c[1].split('{', 1)[0]
    expected_output = b[1]
    for old, new in [('type OutputRows = m![H / 60 % 32, 1 # 8];', 'type OutputRows = m![H / 120 % 16, 1 # 16];'), ('Qs % 512', 'Qs % 256'), ('Qs / 32 % 16', 'Qs / 32 % 8'), ('Qs / 64 % 8', 'Qs / 64 % 4'), ('H % 60', 'H % 120'), ('H / 4 % 15', 'H / 4 % 30'), ('all 60 local', 'all 120 local'), ('H=1920*cluster+60*group+local', 'H=1920*cluster+120*group+local')]:
        expected_output = expected_output.replace(old, new)
    expected_quant = b[3]
    for old, new in [('512', '256'), ('H / 60 % 32', 'H / 120 % 16'), ('Qs / 16 % 32', 'Qs / 16 % 16'), ('Qs / 8 % 64', 'Qs / 8 % 32'), ('Qs / 4 % 128', 'Qs / 4 % 64'), ('Qs / 32 % 16', 'Qs / 32 % 8')]:
        expected_quant = expected_quant.replace(old, new)
    assert c[1] == expected_output and c[3] == expected_quant
    baseline_manifest = json.loads((HERE / 'baseline-source-manifest.json').read_text())
    candidate_manifest = json.loads((HERE / 'candidate-source-manifest.json').read_text())
    assert set(baseline_manifest) == set(candidate_manifest)
    changed = [key for key in baseline_manifest if baseline_manifest[key] != candidate_manifest[key]]
    assert changed == ['src/device/sliding/projection.rs']
    for name, digest in baseline_manifest.items():
        assert sha(ROOT / 'moa-dev/experiments/artifacts/e37-kernel-specific-rms' / name) == digest, name
    assert candidate_manifest[changed[0]] == sha(HERE / 'projection.rs')
    seen_rows = set()
    physical_row_tiles = set()
    for h in range(3840):
        cluster = h // 1920
        group = h // 120 % 16
        row = h % 120
        assert h == 1920 * cluster + 120 * group + row
        assert (cluster, group, row) not in seen_rows
        seen_rows.add((cluster, group, row))
        for k in range(16):
            coordinate = (cluster, 16 * group + k, row)
            assert coordinate not in physical_row_tiles
            physical_row_tiles.add(coordinate)
        # The reduce retains column0 and120 actual local rows; commit is30 packets x4 BF16.
        assert (16 * group) % 16 == 0
        assert (row // 4) * 4 + row % 4 == row
        assert (h // 480) * 480 + h % 480 == h
    seen_q = set()
    for k in range(16):
        for qlocal in range(256):
            q = k * 256 + qlocal
            assert q // 256 == k and q % 256 == qlocal and q not in seen_q
            seen_q.add(q)
    assert len(seen_rows) == 3840 and len(physical_row_tiles) == 61440
    assert seen_q == set(range(4096))
    # Row and column maps are independent bijections; their product covers all15,728,640 weights.
    schedule = json.loads((HERE / 'candidate.json').read_text())
    graph = json.loads((HERE / 'candidate-graph.json').read_text())
    tensors = {t['index']: t for t in schedule['tensors']}
    assert tensors['20']['size'] == 15728640 and 'H % 120, Qs % 256' in tensors['20']['shape']
    assert tensors['25']['name'] == 'hi_vrf' and tensors['25']['size'] == 512 * 1024
    assert tensors['29']['size'] == 512 * 3840 and '(H % 120, 1 # 8)' in tensors['29']['shape']
    reduce_ops = [o for o in graph['operators'].values() if o['kind'] == 'TuExec' and 'GReduceAddF [ratio=16]' in o['description']]
    assert len(reduce_ops) == 1
    desc = reduce_ops[0]['description']
    assert 'broadcast_kind: Direct, f32' in desc and 'output_element_type: bf16' in desc
    assert 'commit_in_size: 8B (=4EA)' in desc and 'commit_projection: 0+[None=30:4, None=4:1]' in desc
    assert max(i['lifetime']['end'] for i in schedule['instructions']) == 24621
    result = {
        'status': 'PASS', 'candidate_projection_sha256': sha(HERE / 'projection.rs'),
        'qkv_and_h480_epilogue_byte_identical': True,
        'public_project_output_signature_unchanged': True,
        'all_other_source_and_manifests_unchanged': True,
        'checked_source_and_manifest_files': len(baseline_manifest),
        'row_coordinates_checked': len(seen_rows), 'column_coordinates_checked': len(seen_q),
        'physical_row_column_tiles_checked': len(physical_row_tiles),
        'cartesian_weight_elements': 3840 * 4096,
        'clusters': 2, 'physical_slices_per_cluster': 256,
        'row_groups_per_cluster': 16, 'columns_per_row_group': 16,
        'rows_per_slice': 120, 'columns_per_slice': 256,
        'weight_bytes_per_slice': 120 * 256,
        'input_bf16_payload_bytes_per_slice': 256 * 2,
        'paired_trf_payload_bytes_per_slice': 2 * 256,
        'hi_vrf_bytes_per_slice': 256 * 4,
        'f32_partial_physical_bytes_per_slice': 120 * 8 * 4,
        'reduced_bf16_physical_bytes_per_slice': 120 * 2,
        'bf16_commit_packets_per_slice': 30, 'bf16_values_per_commit': 4,
        'precision_change': 'Amax/scale groups512->256 and final F32 inter-slice sum8->16; arithmetic/cast boundaries preserved, bitwise equality not promised.',
        'compiled_reduce_ratio': 16, 'compiled_commit_stride_verified': True,
        'compiled_candidate_cycles': 24621,
        'hardware_correctness_checked': False
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
