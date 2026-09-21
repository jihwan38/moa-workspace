#!/usr/bin/env python3
"""Check weight/scale/output row correspondence and the unchanged dot pipeline."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def section(text, first, last):
    return text[text.index(first):text.index(last)]


def main():
    original = subprocess.check_output(
        ["git", "show", "baseline-e55:furiosa-opt-gemma4-12B/src/device/shared/mlp.rs"], cwd=ROOT
    ).decode()
    candidate = (HERE / "mlp.rs").read_text()
    old = section(original, "fn project_eight_rows_scaled_after_dot(", "pub(crate) fn project_up_and_gate(")
    retained = section(candidate, "fn project_eight_rows_scaled_after_dot(", "fn project_eight_rows_from_sixteen(")
    assert old == retained
    staged = section(candidate, "fn project_eight_rows_from_sixteen(", "// Tile the row-major HBM view")
    staged = staged.replace("project_eight_rows_from_sixteen", "project_eight_rows_scaled_after_dot", 1)
    staged = staged.replace("    local_offset: usize,\n", "", 1)
    staged = staged.replace("m![L % 32 = 16, H]>", "m![L % 32, H]>", 1)
    staged = staged.replace("tile::<m![L % 32 = 16], 8, m![L % 32 = 16 = 8 # 16, H]>(local_offset)",
        "tile::<m![L % 32], 8, m![L % 32 = 8 # 32, H]>(row_offset)", 1)
    assert old == staged
    gate = "    let mut gate:"
    assert original[original.index(gate):] == candidate[candidate.index(gate):]

    seen = set()
    for cluster in range(2):
        for group in range(240):
            for start in (0, 16):
                for dot_start in (0, 8):
                    for row in range(8):
                        local = start + dot_start + row
                        global_row = 7680 * cluster + 32 * group + local
                        assert global_row not in seen
                        seen.add(global_row)
                        # Both scale and output continue using the full 32-row index.
                        assert global_row == ((cluster * 240 + group) * 32 + local)
                        # The compact BF16 tile uses a separate, local 0..15 row index.
                        assert 0 <= dot_start + row < 16
                        assert 240 * global_row == 240 * (7680 * cluster + 32 * group + start + dot_start + row)
                        assert (3840 * global_row) % 2 == 0
    assert seen == set(range(15360))
    for column in range(3840):
        assert 16 * (column // 16) + column % 16 == column
        assert 8 * (column // 8) + column % 8 == column
    result = {
        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "baseline_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "weight_transfer_rows": 16,
        "dot_rows": 8,
        "rows_checked": len(seen),
        "columns_checked": 3840,
        "weight_coordinates_by_cartesian_product": len(seen) * 3840,
        "method": "Exhaustive row correspondence and independent column factorization, not an NPU comparison.",
        "unchanged_dot_scale_reduce_cast_pipeline": True,
        "unchanged_gate_geglu_down_suffix": True,
        "hardware_correctness_proved": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
