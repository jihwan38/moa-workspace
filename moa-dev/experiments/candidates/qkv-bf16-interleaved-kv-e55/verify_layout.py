#!/usr/bin/env python3
"""Independent source/coordinate audit of the fused-KV design, not NPU validation."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace
    base = root / 'moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/crate'
    candidate = args.candidate
    original = (base / 'src/device/sliding/projection.rs').read_text()
    source = (candidate / 'src/device/sliding/projection.rs').read_text()
    assert sha(base / 'src/device/sliding/projection.rs') == '380e57d6502e26086bd6d933ce762f4ad5de18587d52a8ef4aac7fa87e3b9c9f'
    assert source.split('type KvCluster', 1)[0] == original.split('type KvCluster', 1)[0]
    assert source.split('pub(crate) fn project_output(', 1)[1] == original.split('pub(crate) fn project_output(', 1)[1]
    changes = []
    inputs = sorted(p for p in (base / 'src').rglob('*') if p.is_file())
    assert len(inputs) == 43
    for before in inputs:
        relative = before.relative_to(base)
        if before.read_bytes() != (candidate / relative).read_bytes():
            changes.append(relative.as_posix())
    assert changes == ['src/device/sliding/projection.rs'], changes
    for name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'):
        assert (base / name).read_bytes() == (candidate / name).read_bytes(), name

    raw_seen, canonical_seen, destination_seen = set(), set(), set()
    vrf_offsets = set()
    for matrix in range(2):
        for p in range(2048):
            cluster, remainder = divmod(p, 1024)
            group, remainder = divmod(remainder, 32)
            row, low_slice = divmod(remainder, 8)
            raw_slice = 8 * group + low_slice
            raw = matrix, cluster, raw_slice, row
            assert raw not in raw_seen
            raw_seen.add(raw)
            assert p == 1024 * cluster + 32 * group + 8 * row + low_slice
            # Both HBM matrices and their channel scales use the same row p.
            weight_begin, weight_end = p * 3840, (p + 1) * 3840
            assert weight_end - weight_begin == 3840
            # 16-wide BF16 fetch groups pair into 32-wide DPE packets. Only H
            # is reduced; the independent matrix selector and row survive.
            assert 240 * 16 == 120 * 32 == 3840
            canonical_slice, canonical_time = 8 * group + row, low_slice
            assert (canonical_slice // 8 * 8 + canonical_time, canonical_slice % 8) == (raw_slice, row)
            scale_vrf_byte = matrix * 256 + row * 32
            assert 0 <= scale_vrf_byte <= 352 and row < 4
            vrf_offsets.add(scale_vrf_byte)
            # A BF16 [matrix=2,row=4] commit is 16B/slice. Splitting on the
            # outer matrix preserves the selected base; unpad changes no byte.
            result_byte = matrix * 8 + row * 2
            tile_base = matrix * 8
            assert tile_base % 8 == 0
            assert tile_base + 2 * row == result_byte < 16
            # The existing Switch then uses the exact old per-matrix layout.
            canonical = matrix, cluster, canonical_slice, canonical_time
            assert canonical not in canonical_seen and canonical_slice % 8 < 4
            canonical_seen.add(canonical)
            head, d = divmod(p, 256)
            assert (cluster, canonical_slice, canonical_time) == (
                head // 4, (head % 4) * 64 + (d // 32) * 8 + (d // 8) % 4, d % 8)
            destination = matrix, head, d
            assert destination not in destination_seen
            destination_seen.add(destination)
    assert len(raw_seen) == len(canonical_seen) == len(destination_seen) == 4096
    assert vrf_offsets == {0, 32, 64, 96, 256, 288, 320, 352}
    print(json.dumps({
        'scope': 'Independent source scope and intended coordinates only; does not execute SDK lowering or prove hardware correctness.',
        'candidate_projection_sha256': sha(candidate / 'src/device/sliding/projection.rs'),
        'changed_source_files': changes, 'source_files': 43,
        'other_42_source_files_and_cargo_inputs_unchanged': True,
        'query_prefix_and_output_suffix_unchanged': True,
        'matrix_output_coordinates_checked': len(raw_seen),
        'covered_weight_elements_in_full_contiguous_h_spans': 4096 * 3840,
        'matrix_selector_kept_in_independent_coordinate_model': True,
        'vrf_valid_scalar_offsets': sorted(vrf_offsets),
        'vrf_physical_bytes_per_slice': 512,
        'contraction_bf16_bytes_per_slice': 16,
        'tile_bases_bytes': [0, 8],
        'padding_promoted_to_valid_in_coordinate_model': False,
        'npu_validation': False,
    }, indent=2))


if __name__ == '__main__':
    main()
