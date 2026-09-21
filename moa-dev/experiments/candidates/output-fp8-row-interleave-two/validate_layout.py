#!/usr/bin/env python3
"""Enumerate all O row/column partitions and the proposed Switch restoration."""

import argparse
import hashlib
import json
from pathlib import Path


def encode(row, column):
    cluster, within = divmod(row, 1920)
    group, inner = divmod(within, 120)
    local_row, row_lane = divmod(inner, 2)
    return cluster, (group * 2 + row_lane) * 8 + column, local_row


def decode(cluster, physical_slice, local_row):
    row_group, column = divmod(physical_slice, 8)
    group, row_lane = divmod(row_group, 2)
    return cluster * 1920 + group * 120 + local_row * 2 + row_lane, column


def switch(cluster, physical_slice, local_row):
    row_group, column = divmod(physical_slice, 8)
    group, row_lane = divmod(row_group, 2)
    outer, inner = divmod(local_row, 30)
    return cluster, (group * 2 + outer) * 8 + column, inner * 2 + row_lane


def verify(directory):
    original = (directory / "projection.original.rs").read_text()
    candidate = (directory / "projection.rs").read_text()
    start, end = "pub(crate) fn project_output(", "\ntype HiddenRows"
    assert candidate[:candidate.index(start)] == original[:original.index(start)]
    old_alias = "type OutputColumns = m![H / 60 % 32, Qs / 512];"
    new_alias = "type OutputColumns = m![H / 120 % 16, H % 2, Qs / 512];"
    assert new_alias in candidate
    assert candidate[candidate.index(end):].replace(new_alias, old_alias) == original[original.index(end):]
    assert "SwitchConfig::InterTranspose { slice1: 2, slice0: 8, time0: 30 }" in candidate
    assert "m![H / 2 % 60, Qs % 512]>" in candidate
    assert ".vector_inter_slice_reduce::<OutputRows, m![H / 2 % 60]>" in candidate

    seen, restored = set(), set()
    for row in range(3840):
        for column in range(8):
            at = encode(row, column)
            assert at not in seen
            seen.add(at)
            assert decode(*at) == (row, column)
            after = switch(*at)
            assert after not in restored
            restored.add(after)
            cluster, physical_slice, local_row = after
            canonical_row_group, canonical_column = divmod(physical_slice, 8)
            assert canonical_column == column
            assert cluster * 1920 + canonical_row_group * 60 + local_row == row
        # The eight Qs/512 shards remain the same contiguous 8-slice group.
        slots = [encode(row, column)[1] for column in range(8)]
        assert slots == list(range(slots[0], slots[0] + 8))
    assert len(seen) == len(restored) == 3840 * 8
    for cluster in range(2):
        for physical_slice in range(256):
            assert all((cluster, physical_slice, local) in seen for local in range(60))
            assert all((cluster, physical_slice, local) in restored for local in range(60))
    for first in range(0, 3840, 2):
        assert len({encode(row, 0)[:2] for row in range(first, first + 2)}) == 2
    return {
        "rows_checked": 3840,
        "column_shards_per_row": 8,
        "row_column_coordinates_checked": 30720,
        "every_physical_slice_has_60_rows": True,
        "two_adjacent_hbm_rows_use_two_row_groups": True,
        "column_shards_and_reduce_order_unchanged": True,
        "switch_restores_canonical_row_group_and_local_row": True,
        "qkv_source_byte_unchanged": True,
        "quantizer_and_channel_scale_epilogue_source_unchanged_except_type_alias": True,
        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "examples": [{"row": r, "before": encode(r, 0), "after": switch(*encode(r, 0))}
                     for r in (0, 1, 3, 4, 59, 60, 239, 240, 1919, 1920, 3839)],
        "caveat": "Independent logical permutation proof; NPU correctness and performance remain unverified.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parent)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))
