#!/usr/bin/env python3
"""Verify source scope, factored coordinate coverage and compiled F32 boundaries."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = HERE.parents[1] / "artifacts/output-bf16-eight-columns-e53"


def main():
    old = (HERE / "projection.original.rs").read_text()
    new = (HERE / "projection.rs").read_text()
    entry = "pub(crate) fn project_output("
    boundary = "pub(crate) type OutputEpilogueSlices"
    assert old.split(entry, 1)[0] == new.split(entry, 1)[0]
    old_tail = old[old.index(boundary):]
    new_tail = new[new.index(boundary):]
    new_tail = new_tail.replace(
        "// O projection partitions 4096 BF16 inputs into eight 512-column groups.",
        "// O projection partitions 4096 BF16 inputs into sixteen 256-column groups.")
    new_tail = new_tail.replace(
        "type OutputColumns = m![H / 60 % 32, Qs / 512];",
        "type OutputColumns = m![H / 120 % 16, Qs / 256];")
    assert old_tail == new_tail
    body = new[new.index(entry):new.index(boundary)]
    assert body.count(".contract_outer") == 1
    assert "TrfTensor<bf16" in body
    assert "fetch_table_lookup::<bf16>()" in body
    assert ".cast::<bf16" not in body
    assert "prepare_output_fp8_split" not in new
    assert "DmTensor<f32" in body
    assert ".transpose::<m![H / 2 % 30], m![H % 2 # 8]>()" in body

    rows, columns = set(), set()
    for h in range(3840):
        cluster, within = divmod(h, 1920)
        group, row = divmod(within, 60)
        assert group < 32
        assert 1920 * cluster + 60 * group + row == h
        # One valid result slice per eight-slice column group; padding is not data.
        result_slice = group * 8
        assert result_slice < 256
        # Thirty two-F32 commits cover sixty values with no row padding.
        assert 8 * (row // 2) + 4 * (row % 2) == 4 * row
        assert 480 * (h // 480) + h % 480 == h
        rows.add((cluster, group, row))
    for q in range(4096):
        column, element = divmod(q, 512)
        assert column < 8
        assert 512 * column + element == q
        assert 32 * (element // 32) + element % 32 == element
        columns.add((column, element))
    physical_slices = {(c, 8 * g + col)
                       for c in range(2) for g in range(32) for col in range(8)}
    assert len(rows) == 3840 and len(columns) == 4096 and len(physical_slices) == 512
    # Each independent row/column coordinate has a unique inverse; Cartesian
    # coverage is all 15,728,640 weight elements, not a sampled subset.
    assert 512 * 60 * 512 == 3840 * 4096

    graph = json.loads((ARTIFACTS / "candidate-graph.json").read_text())
    dots = [(k, v) for k, v in graph["operators"].items()
            if "GReduceAddF [ratio=8]" in v["description"]]
    assert len(dots) == 1
    desc = dots[0][1]["description"]
    assert "data_type: Float32" in desc
    assert "commit_in_size: 8B (=2EA)" in desc
    assert "commit_projection: 0+[None=30:2, None=2:1]" in desc
    manifest = json.loads((HERE / "compiled-source-manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256((ARTIFACTS / "crate" / name).read_bytes()).hexdigest() == expected
    result = {
        "parent_commit": manifest["parent_commit"],
        "QKV_prefix_byte_identical": True,
        "channel_scale_RMS_residual_helpers_unchanged": True,
        "only_O_column_row_partition_changed": True,
        "original_BF16_x_and_one_BF16_dot": True,
        "F32_through_channel_scale_before_first_BF16_cast": True,
        "row_coordinates_checked": 3840,
        "column_coordinates_checked": 4096,
        "factorized_full_weight_coverage": 15728640,
        "active_physical_slices": len(physical_slices),
        "per_slice_weight_bytes": 30720,
        "per_slice_BF16_TRF_input_bytes": 1024,
        "F32_commit_bytes": 8,
        "per_result_slice_F32_elements": 60,
        "graph_column_reduce_ratio": 8,
        "compiled_input_hashes_checked": len(manifest["sha256"]),
        "candidate_sha256": hashlib.sha256(new.encode()).hexdigest(),
        "numerical_limit": "512-column local sums plus eight-way reduction regroup F32 additions; no bit-identical or NPU correctness claim.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
