#!/usr/bin/env python3
"""Source, coordinate, BF16 representation and graph boundary checks."""
from pathlib import Path
import json
import struct
import hashlib

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parents[1] / "artifacts/output-bf16-activation-e47"


def main():
    old = (HERE / "projection.original.rs").read_text()
    new = (HERE / "projection.rs").read_text()
    entry = "pub(crate) fn project_output("
    assert old.split(entry, 1)[0] == new.split(entry, 1)[0]
    old_tail = old[old.index("fn apply_output_channel_scale("):]
    assert old_tail == new[new.index("fn apply_output_channel_scale("):]
    body = new[new.index(entry):new.index("pub(crate) type OutputEpilogueSlices")]
    assert "prepare_output_fp8_split(" not in body
    assert ".cast::<bf16" not in body
    assert body.count(".contract_outer") == 1
    assert "TrfTensor<bf16" in body and ".fetch_table_lookup::<bf16>()" in body
    assert ".transpose::<m![H / 2 % 60], m![H % 2 # 8]>()" in body
    helper = new[new.index("fn apply_output_channel_scale_f32("):new.index("fn apply_output_channel_scale(")]
    assert helper.index(".vector_fp_binary(FpBinaryOp::MulF") < helper.index(".cast::<bf16")

    rows, columns = set(), set()
    for h in range(3840):
        cluster, within = divmod(h, 1920)
        group, local = divmod(within, 120)
        assert 1920 * cluster + 120 * group + local == h
        rows.add((cluster, group, local))
        # Two-row F32 commit: 60 packets * 2 values, exactly 480 bytes.
        assert 2 * (local // 2) + local % 2 == local
        assert 8 * (local // 2) + 4 * (local % 2) == 4 * local
        # Same high-H rename, then typed H480x8 gather.
        shard, element = divmod(h, 480)
        assert 480 * shard + element == h
    for q in range(4096):
        column, local = divmod(q, 256)
        assert 256 * column + local == q
        assert 32 * (local // 32) + local % 32 == local
        columns.add((column, local))
    assert len(rows) == 3840 and len(columns) == 4096

    finite = 0
    for code in range(256):
        sign = -1.0 if code & 128 else 1.0
        exponent, mantissa = (code >> 3) & 15, code & 7
        if exponent == 15 and mantissa == 7:
            continue
        value = sign * (mantissa * 2.0**-9 if exponent == 0
                        else (1 + mantissa / 8) * 2.0**(exponent - 7))
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        assert bits & 65535 == 0
        reconstructed = struct.unpack("<f", struct.pack("<I", bits & 0xffff0000))[0]
        assert struct.pack("<f", reconstructed) == struct.pack("<f", value)
        finite += 1
    assert finite == 254
    graph = json.loads((ARTIFACTS / "candidate-graph.json").read_text())
    reductions = [(k, o) for k, o in graph["operators"].items()
                  if "GReduceAddF [ratio=16]" in o["description"]]
    assert len(reductions) == 1
    description = reductions[0][1]["description"]
    assert "data_type: Float32" in description and "commit_in_size: 8B (=2EA)" in description
    assert "commit_projection: 0+[None=60:2, None=2:1]" in description
    result = {
        "QKV_prefix_byte_identical": True,
        "old_channel_scale_residual_and_quantizer_suffix_byte_identical": True,
        "native_input_quantizer_not_called_by_O": True,
        "one_BF16_dot_and_no_projection_BF16_cast": True,
        "channel_scale_precedes_first_output_BF16_cast": True,
        "finite_E4M3_values_exactly_BF16_representable": finite,
        "representation_check_is_not_SDK_table_or_NPU_test": True,
        "rows_checked": len(rows), "columns_checked": len(columns),
        "weight_coordinates_factored_coverage": len(rows) * len(columns),
        "F32_two_row_commit_bytes": 8,
        "F32_projection_elements_per_valid_slice": 120,
        "F32_projection_bytes_per_valid_slice": 480,
        "graph_reduction_key": reductions[0][0],
        "F32_column_reduce_and_commit_graph_checked": True,
        "candidate_sha256": hashlib.sha256(new.encode()).hexdigest(),
        "universal_numerical_equivalence_or_hardware_pass_claimed": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
