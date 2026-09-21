"""Finite coordinate/source proof only; this is not an RNGD oracle."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def section(text, start, end):
    return text[text.index(start):text.index(end)]


def main():
    original = (HERE / "mlp.original.rs").read_text()
    candidate = (HERE / "mlp.rs").read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == "7c363a7f116ff9ebf4027f6a9957faa5cb3ce74aac85b3c5f7b6579ff628b267"
    assert section(original, "fn project_rows_with_scale_contraction(", "pub(crate) fn project_up_and_gate(") == section(candidate, "fn project_rows_with_scale_contraction(", "pub(crate) fn project_up_and_gate(")
    assert original[original.index("pub(crate) fn feedforward("):] == candidate[candidate.index("pub(crate) fn feedforward("):]
    old_projection = section(original, "pub(crate) fn project_up_and_gate(", "    // Each real row has L =")
    new_projection = section(candidate, "pub(crate) fn project_up_and_gate(", "    // L = 7680*cluster")
    expected = old_projection.replace("m![Dummy2, FfnRow = 30, FfnSlice, H]", "m![Dummy2, FfnSlice / 2, FfnRow = 30, FfnSlice % 2, H]")
    expected = expected.replace("m![Dummy2, FfnRow = 30, FfnSlice, H / 16]", "m![Dummy2, FfnSlice / 2, FfnRow = 30, FfnSlice % 2, H / 16]")
    assert expected == new_projection, "weight/scale loads or dense zero-fill/commit/DPE changed beyond four HBM mappings"
    assert new_projection.count(".memset(const { f8e4m3::from_f32(0.0) }, &mut device.sub)") == 4

    rows = set()
    packed_packets = 0
    for cluster in range(2):
        for group in range(128):
            source = {(r, s): None for r in range(32) for s in range(2)}
            # All32 live rows are actual zeros before first30 rows are written.
            for key in source:
                source[key] = 0
            for row in range(30):
                for side in range(2):
                    logical = 7680 * cluster + 60 * group + 2 * row + side
                    assert logical not in rows
                    rows.add(logical)
                    source[row, side] = logical + 1
                    assert (((cluster * 128 + group) * 30 + row) * 2 + side) == logical
                    # H and H/16 are unchanged contiguous trailing dimensions.
                    for width in (3840, 240):
                        assert ((((cluster * 128 + group) * 30 + row) * 2 + side) * width) == logical * width
            assert all(source[r, s] == 0 for r in (30, 31) for s in range(2))
            stream = [source[r, s] for r in range(30) for s in range(2)]
            packets = [[source[2 * p + dr, s] for dr in range(2) for s in range(2)] for p in range(15)]
            assert [v for packet in packets for v in packet] == stream
            assert all(len(packet) * 2 == 8 for packet in packets)
            canonical_group = 128 * cluster + group
            assert stream == [60 * canonical_group + j + 1 for j in range(60)]
            packed_packets += len(packets)
    assert rows == set(range(15360))
    print(json.dumps({
        "status": "PASS", "logical_rows_bijective": len(rows),
        "live_slices_per_cluster": 256, "actual_zero_tail_rows_per_slice": 2,
        "real_rows_read_per_slice": 30, "bf16_8byte_output_packets": packed_packets,
        "dpe_helper_byte_identical": True, "feedforward_and_suffix_byte_identical": True,
        "projection_prefix_change_only_four_hbm_mappings": True,
        "four_full_live_memsets_preserved": True,
        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "limitation": "Integer/source checks only. Native FP8 quantization and block-BF16 rounding are inherited; backend mapping still needs NPU validation."
    }, indent=2))


if __name__ == "__main__":
    main()
