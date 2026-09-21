#!/usr/bin/env python3
"""Check logical scale coordinates; this does not execute SDK DMA or an NPU."""
import json


def main():
    count = 3840 * 960
    seen = bytearray(count)
    for h in range(3840):
        cluster = h // 1920
        group = (h % 1920) // 60
        row = h % 60
        for block in range(960):
            column, local_block = divmod(block, 120)
            # New HBM -> DM stage holds the full row on column-zero slice.
            intermediate_slice = 8 * group
            intermediate_element = row * 960 + block
            restored_h = 1920 * cluster + 60 * (intermediate_slice // 8) + intermediate_element // 960
            restored_block = intermediate_element % 960
            assert (restored_h, restored_block) == (h, block)
            assert restored_h * 960 + restored_block == h * 960 + block
            # Typed DM -> DM scatter recovers the original DownRowsByColumns.
            target_slice = intermediate_slice + restored_block // 120
            target_element = row * 120 + restored_block % 120
            assert (target_slice, target_element) == (8 * group + column, row * 120 + local_block)
            target_flat = (cluster * 256 + target_slice) * 7200 + target_element
            assert not seen[target_flat], (h, block, target_flat)
            seen[target_flat] = 1
    assert all(seen)
    print(json.dumps({
        "status": "PASS",
        "coordinates": count,
        "unique_destination_coordinates": sum(seen),
        "intermediate_valid_slices": 64,
        "intermediate_elements_per_valid_slice": 57600,
        "source_and_destination_logical_bytes": count,
        "intermediate_physical_extent_bytes": 512 * 57600,
        "scope": "Integer coordinate bijection only; no SDK execution or hardware correctness claim.",
    }, indent=2))


if __name__ == "__main__":
    main()
