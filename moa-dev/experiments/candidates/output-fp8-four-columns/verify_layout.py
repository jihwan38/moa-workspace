#!/usr/bin/env python3
"""Independent coordinate-factor proof, not a device or quantization test."""
import json


def main():
    row_coordinates = set()
    output_coordinates = set()
    for h in range(3840):
        cluster = h // 1920
        group = (h % 1920) // 60
        row = h % 60
        assert 1920 * cluster + 60 * group + row == h
        row_coordinates.add((cluster, group, row))
        output_coordinates.add((h // 480, h % 480))
    assert len(row_coordinates) == len(output_coordinates) == 3840

    column_coordinates = set()
    for q in range(4096):
        column, local = divmod(q, 1024)
        assert 0 <= column < 4
        assert column * 1024 + local == q
        column_coordinates.add((column, local))
    assert len(column_coordinates) == 4096

    valid_slices = {(c, 8 * g + col) for c in range(2) for g in range(32) for col in range(4)}
    assert len(valid_slices) == 256
    assert all(s % 8 < 4 for _, s in valid_slices)
    # Row and column factors are independent. Their Cartesian product covers
    # every HBM weight exactly once: h*4096+q -> (c,8*g+col,row*1024+local).
    assert 256 * 60 * 1024 == 3840 * 4096
    print(json.dumps({
        "status": "PASS",
        "row_axis_coordinates_checked": 3840,
        "column_axis_coordinates_checked": 4096,
        "weight_coordinates_proved_by_cartesian_product": 3840 * 4096,
        "valid_physical_slices_across_two_clusters": len(valid_slices),
        "physical_slices_across_two_clusters": 512,
        "valid_column_indices": [0, 1, 2, 3],
        "epilogue_output_coordinates_checked": len(output_coordinates),
        "scope": "Integer layout factors only; no SDK execution, FP8 numerical comparison, or NPU test.",
    }, indent=2))


if __name__ == "__main__":
    main()
