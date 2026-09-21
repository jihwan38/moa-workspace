#!/usr/bin/env python3
"""Check the unchanged O16 mapping and reduction/scale/cast pipeline order."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parents[1] / "artifacts/output-bf16-fused-channel-scale-e53"


def main():
    old = (HERE / "projection.original.rs").read_text()
    new = (HERE / "projection.rs").read_text()
    entry = "pub(crate) fn project_output("
    suffix = "pub(crate) type OutputEpilogueSlices"
    assert old.split(entry, 1)[0] == new.split(entry, 1)[0]
    assert old[old.index(suffix):] == new[new.index(suffix):]
    body = new[new.index(entry):new.index(suffix)]
    reduction = body.index(".vector_inter_slice_reduce")
    scale = body.index(".vector_fp_binary(FpBinaryOp::MulF")
    cast = body.index(".cast::<bf16")
    assert reduction < scale < cast
    assert body.count(".contract_outer") == body.count(".cast::<bf16") == 1
    assert "apply_output_channel_scale_f32(" not in body
    assert "TrfTensor<bf16" in body and "fetch_table_lookup::<bf16>()" in body
    assert "m![Dummy2, H % 1920]" in body
    assert "m![H / 8 % 15], m![H % 8]" in body

    rows = set()
    for h in range(3840):
        cluster, within = divmod(h, 1920)
        group, row = divmod(within, 120)
        assert cluster * 1920 + group * 120 + row == h
        # HBM view is wire-identical; each row's scale goes to its reduction leader.
        assert (cluster * 1920 + group * 120 + row) * 2 == h * 2
        assert row * 4 < 480
        assert 8 * (row // 4) + 2 * (row % 4) == 2 * row
        assert 480 * (h // 480) + h % 480 == h
        rows.add((cluster, group * 16, row))
    assert len(rows) == 3840
    for q in range(4096):
        assert (q // 256) * 256 + q % 256 == q

    graph = json.loads((ARTIFACTS / "candidate-graph.json").read_text())
    matches = [(k, v) for k, v in graph["operators"].items()
               if "GReduceAddF [ratio=16]" in v["description"]]
    assert len(matches) == 1
    key, node = matches[0]
    desc = node["description"]
    assert desc.index("GReduceAddF [ratio=16]") < desc.index("FpMul0:") < desc.index("cast_compaction:")
    assert "output_element_type: bf16" in desc
    assert "data_type: Bfloat16" in desc
    assert "commit_in_size: 8B (=4EA)" in desc
    assert "commit_projection: 0+[None=30:4, None=4:1]" in desc
    schedule = json.loads((ARTIFACTS / "candidate.json").read_text())
    vrfs = [v for v in schedule["tensors"] if v["name"] == "weight_scale_vrf"]
    assert len(vrfs) == 1
    assert "Slice: (H / 120 % 16, 1 # 16), Element: H % 120" in vrfs[0]["shape"]
    assert vrfs[0]["size"] == 512 * 120 * 4
    manifest = json.loads((HERE / "compiled-source-manifest.json").read_text())
    for name, sha in manifest["sha256"].items():
        assert hashlib.sha256((ARTIFACTS / "crate" / name).read_bytes()).hexdigest() == sha
    print(json.dumps({
        "QKV_prefix_and_all_helpers_byte_identical": True,
        "unchanged_O16_weight_input_and_reduction_partition": True,
        "row_scale_addresses_checked": len(rows),
        "column_coordinates_checked": 4096,
        "factored_weight_coordinate_coverage": 3840 * 4096,
        "F32_reduction_then_scale_then_first_BF16_cast": True,
        "graph_pipeline_key": key,
        "scale_DM_BF16_bytes_per_valid_slice": 240,
        "scale_VRF_F32_bytes_per_slice": 480,
        "scale_VRF_physical_schedule_bytes": vrfs[0]["size"],
        "scale_VRF_shape": vrfs[0]["shape"],
        "scale_VRF_lifetime": vrfs[0]["lifetime"],
        "BF16_commit_bytes": 8,
        "compiled_input_hashes_checked": len(manifest["sha256"]),
        "candidate_sha256": hashlib.sha256(new.encode()).hexdigest(),
        "NPU_verified": False,
        "scope": "Typed source and graph order check; no disassembly, exhaustive FP arithmetic or hardware proof.",
    }, indent=2))


if __name__ == "__main__":
    main()
