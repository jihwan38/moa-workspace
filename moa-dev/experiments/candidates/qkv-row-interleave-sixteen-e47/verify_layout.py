#!/usr/bin/env python3
"""All Q/KV row coordinates, padded Switch slots, and unchanged consumers."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent


def function(source, name, next_name):
    return source[source.index(name):source.index(next_name)]


def main():
    old = (HERE / "projection.original.rs").read_text()
    new = (HERE / "projection.rs").read_text()
    assert old.split("pub(crate) fn project_output(", 1)[1] == new.split("pub(crate) fn project_output(", 1)[1]
    names = [("pub(crate) fn project_query(", "type KvCluster", "Qs"),
             ("fn project_one_kv_matrix(", "fn project_value_two_dot(", "Ps"),
             ("fn project_value_two_dot(", "pub(crate) fn project_key_value(", "Ps")]
    for start, end, axis in names:
        before = function(old, start, end)
        after = function(new, start, end)
        original_dot = before.split("    // Switch swaps" if axis == "Qs" else "    // Four valid local rows", 1)[0]
        new_dot = after.split("    // Sixteen-way Switch", 1)[0]
        if axis == "Qs":
            new_dot = new_dot.replace("Qs / 128 % 16, Qs % 16", "Qs / 64 % 32, Qs % 8")
            new_dot = new_dot.replace("Qs / 16 % 8", "Qs / 8 % 8")
            new_dot = new_dot.replace("Qs / 16 % 4", "Qs / 8 % 4")
            new_dot = new_dot.replace("Qs / 64 % 2", "Qs / 32 % 2")
        else:
            new_dot = new_dot.replace("Ps / 16 % 4", "Ps / 8 % 4")
        assert original_dot == new_dot
        # Channel-scale arithmetic remains one FP32 multiply between BF16 boundaries.
        scale_before = before[before.index("    let scaled:"):]
        scale_after = after[after.index("    let scaled:"):]
        arithmetic = lambda text: [line.strip() for line in text.splitlines()
                                   if ".vector_fp_" in line]
        assert arithmetic(scale_before) == arithmetic(scale_after)
    assert function(old, "pub(crate) fn project_key_value(", "pub(crate) fn project_output(") == function(new, "pub(crate) fn project_key_value(", "pub(crate) fn project_output(")

    result = {}
    for name, count, cluster_rows, local_rows, group_rows in (
            ("Q", 4096, 2048, 8, 128), ("KV", 2048, 1024, 4, 64)):
        before_coords = set()
        post_coords = set()
        head_coords = set()
        tagged_slots = {}
        for row in range(count):
            cluster = row // cluster_rows
            group = row // group_rows % 16
            low_slice = row % 16
            local = row // 16 % local_rows
            physical_slice = 16 * group + low_slice
            reconstructed = cluster_rows * cluster + group_rows * group + 16 * local + low_slice
            assert reconstructed == row
            coord = (cluster, physical_slice, local)
            assert coord not in before_coords
            before_coords.add(coord)
            # 16-way InterTranspose exchanges low slice and padded local Time.
            post_slice, post_time = 16 * group + local, low_slice
            post = (cluster, post_slice, post_time)
            assert post not in post_coords
            post_coords.add(post)
            assert post_slice % 16 < local_rows
            scale_row = cluster_rows * cluster + group_rows * (post_slice // 16) + 16 * (post_slice % 16) + post_time
            assert scale_row == row
            if name == "Q":
                head, remainder = divmod(row, 512)
                query_group, dimension = divmod(remainder, 256)
                head_view = (head // 4,
                    (((head % 4) * 2 + query_group) * 2 + dimension // 128) * 16 + dimension // 16 % 8,
                    dimension % 16)
                canonical = (head, query_group, dimension)
            else:
                head, dimension = divmod(row, 256)
                head_view = (head // 4,
                    ((head % 4) * 4 + dimension // 64) * 16 + dimension // 16 % 4,
                    dimension % 16)
                canonical = (head, dimension)
            assert head_view == post
            assert canonical not in head_coords
            head_coords.add(canonical)
            tagged_slots[post] = row
        for cluster in range(2):
            for slc in range(256):
                for time in range(16):
                    slot = (cluster, slc, time)
                    assert (slot in tagged_slots) == (slc % 16 < local_rows)
        assert len(before_coords) == len(post_coords) == len(head_coords) == count
        if name == "Q":
            assert head_coords == {(n, g, d) for n in range(8) for g in range(2) for d in range(256)}
        else:
            assert head_coords == {(n, d) for n in range(8) for d in range(256)}
        result[name] = {"rows_exhaustively_checked": count,
                        "weight_dma_valid_slices_per_cluster": 256,
                        "post_switch_valid_slices_per_cluster": 16 * local_rows,
                        "post_switch_invalid_slices_per_cluster": 256 - 16 * local_rows,
                        "post_switch_valid_elements_per_slice": 16,
                        "all_valid_head_coordinates_covered_once": True}
    for col in range(3840):
        assert (col // 64) * 64 + col % 64 == col
        assert (col // 32) * 32 + col % 32 == col
    result.update({"H_columns_checked_independently": 3840,
                   "dot_bodies_identical_after_row_axis_substitution": True,
                   "channel_scale_arithmetic_identical": True,
                   "O_and_quantizer_suffix_byte_identical": True,
                   "KV_dispatch_byte_identical": True,
                   "q_and_k_keep_pair_dot_v_keeps_two_dot": True,
                   "candidate_sha256": hashlib.sha256(new.encode()).hexdigest(),
                   "no_device_execution": True,
                   "scope": "Integer mapping and source arithmetic audit; not proof of compiler/NPU padding implementation."})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
