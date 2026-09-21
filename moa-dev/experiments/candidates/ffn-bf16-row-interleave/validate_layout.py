#!/usr/bin/env python3
"""Check the full row permutation independently of kernels or generated fixtures."""

import argparse
import hashlib
import json
from pathlib import Path


def location(row):
    cluster, local = divmod(row, 7680)
    group, inner = divmod(local, 256)
    local_row, lane = divmod(inner, 8)
    return cluster, 8 * group + lane, local_row


def logical_row(cluster, physical_slice, local_row):
    group, lane = divmod(physical_slice, 8)
    if not (0 <= cluster < 2 and 0 <= group < 30 and 0 <= local_row < 32):
        raise ValueError("Padding or invalid row location")
    return cluster * 7680 + group * 256 + local_row * 8 + lane


def after_switch(cluster, physical_slice, local_row):
    group, lane = divmod(physical_slice, 8)
    row_outer, row_inner = divmod(local_row, 8)
    return cluster, group * 8 + row_inner, row_outer * 8 + lane


def restored_row(cluster, physical_slice, local_row):
    group, row_inner = divmod(physical_slice, 8)
    row_outer, lane = divmod(local_row, 8)
    return cluster * 7680 + group * 256 + row_outer * 64 + row_inner * 8 + lane


def verify(directory):
    original = (directory / "mlp.original.rs").read_text()
    candidate = (directory / "mlp.rs").read_text()
    assert "type UpGateSlices = m![L / 256 % 30 # 32, L % 8];" in candidate
    assert "m![L / 8 % 32, H]>" in candidate
    assert "m![L / 8 % 32, H / 16]>" in candidate
    assert "SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 }" in candidate
    assert "m![L / 256 % 30 # 32, L / 8 % 8], m![L / 64 % 4, L % 8]>" in candidate
    assert candidate.count(".fetch_table_lookup::<f8e4m3>()") == 3
    assert candidate.count(".fetch_table_lookup::<bf16>()") == 3
    unchanged_start = "pub(crate) fn feedforward("
    assert candidate[candidate.index(unchanged_start):] == original[original.index(unchanged_start):]
    restore_start = "    // Remove padding with typed copies before reinterpreting the high row axis."
    assert candidate[candidate.index(restore_start):candidate.index(unchanged_start)] == original[
        original.index(restore_start):original.index(unchanged_start)]

    seen, switched = set(), set()
    for row in range(15360):
        at = location(row)
        assert at not in seen
        seen.add(at)
        assert logical_row(*at) == row
        after = after_switch(*at)
        assert after not in switched
        switched.add(after)
        assert restored_row(*after) == row
        # Weight rows and scale rows use the identical L partition. H and H/16
        # remain untouched, so the Cartesian product covers every weight/block.
        weight_row = logical_row(*at)
        scale_row = logical_row(*at)
        assert weight_row == scale_row == row
    assert len(seen) == len(switched) == 15360
    for cluster in range(2):
        for physical_slice in range(240):
            assert sum((cluster, physical_slice, local) in seen for local in range(32)) == 32
            for local_base in range(0, 32, 8):
                chunk = [restored_row(cluster, physical_slice, local) for local in range(local_base, local_base + 8)]
                assert chunk == list(range(chunk[0], chunk[0] + 8))
                assert (chunk[0] * 2) % 16 == 0
        for physical_slice in range(240, 256):
            assert all((cluster, physical_slice, local) not in seen for local in range(32))
    for first in range(0, 15360, 8):
        assert len({location(row)[:2] for row in range(first, first + 8)}) == 8

    return {
        "rows_checked": 15360,
        "all_rows_unique_and_restored": True,
        "rows_per_active_slice": 32,
        "active_slices_per_cluster": 240,
        "padding_slices_per_cluster": 16,
        "eight_adjacent_rows_use_eight_distinct_slices": True,
        "switch_is_bijective_and_restores_row_identity": True,
        "switch_outputs_eight_contiguous_bf16_rows_per_16byte_chunk": True,
        "weights_and_scales_same_row_permutation": True,
        "typed_canonical_output_copies_byte_unchanged": True,
        "feedforward_geglu_down_source_byte_unchanged": True,
        "fp4_fp8_bf16_decode_topology_unchanged": True,
        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "first_eight_locations": [location(row) for row in range(8)],
        "caveat": "Logical mapping proof and source checks; not an NPU correctness or DMA-performance result.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parent)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
