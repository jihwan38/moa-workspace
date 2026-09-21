"""Check this frozen scratch candidate; does not execute or emulate an NPU."""
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent / "artifacts/e31-switch-gather-rms/src"


def functions(text):
    result = {}
    for match in re.finditer(r"(?m)^(?:pub(?:\(crate\))? )?fn (\w+)[\s\S]*?\{", text):
        start, depth = match.end() - 1, 1
        end = start + 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        result[match[1]] = (text[match.start():start], text[start:end])
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"mlp": "device/shared/mlp.rs", "residual": "device/shared/residual.rs", "ops": "ops.rs"}
    originals = {}
    candidates = {}
    for name, rel in paths.items():
        original = (HERE / f"{name}.original.rs").read_text()
        assert original == (BASE / rel).read_text(), f"baseline snapshot drift: {name}"
        originals[name] = functions(original)
        candidates[name] = functions((HERE / f"{name}.rs").read_text())
        for fn, (header, _) in originals[name].items():
            assert candidates[name][fn][0] == header, f"signature changed: {name}::{fn}"

    assert len(originals["ops"]) == 10
    unchanged_ops = sorted(set(originals["ops"]) - {"decoder_feedforward"})
    for fn in unchanged_ops:
        assert originals["ops"][fn] == candidates["ops"][fn], f"unrelated ops body changed: {fn}"
    for fn in ("project_eight_rows_scaled_after_dot", "project_up_and_gate", "geglu"):
        assert originals["mlp"][fn] == candidates["mlp"][fn], f"FFN core changed: {fn}"
    # Everything through the last down BF16 commit is byte-identical.
    old_down = originals["mlp"]["project_down"][1].split("    // Compact each cluster before joining its high row axis back into H.")[0]
    new_down = candidates["mlp"]["project_down_sharded"][1].split("    // Only rename the existing high-row cluster axis.")[0]
    assert old_down == new_down, "down contraction/reduction changed"
    old_ff = originals["mlp"]["feedforward"][1].split("    let down = project_down(")[0]
    new_ff = candidates["mlp"]["feedforward_sharded"][1].split("    let down = project_down_sharded(")[0]
    assert old_ff == new_ff, "activation/TRF/GeGLU flow changed"
    for fn, entry in originals["residual"].items():
        assert entry == candidates["residual"][fn], f"existing residual helper changed: {fn}"

    # Label every value by its canonical H row, then apply the exact source
    # and destination layout identities, including the physical slice stride.
    seen, source_coords = {}, set()
    for cluster in range(2):
        for group in range(32):
            for local_row in range(60):
                h = 1920 * cluster + 60 * group + local_row
                src = (cluster, 8 * group, local_row)
                dst = (0, 4 * cluster + group // 8, 60 * (group % 8) + local_row)
                assert dst[1:] == (h // 480, h % 480)
                assert src not in source_coords and dst not in seen
                source_coords.add(src)
                seen[dst] = h
    restored = [seen[(0, s, r)] for s in range(8) for r in range(480)]
    assert restored == list(range(3840))

    print(json.dumps({
        "status": "PASS", "base_snapshot": str(BASE),
        "public_ops_signatures_unchanged": 10,
        "other_ops_bodies_byte_identical": unchanged_ops,
        "core_functions_byte_identical": ["project_eight_rows_scaled_after_dot", "project_up_and_gate", "geglu"],
        "down_body_through_final_bf16_commit_byte_identical": True,
        "existing_residual_helpers_byte_identical": sorted(originals["residual"]),
        "canonical_down_rows_bijective": len(restored),
        "rmsnorm_frozen_unchanged_sha256": sha(BASE / "device/shared/rmsnorm.rs"),
        "candidate_sha256": {name: sha(HERE / f"{name}.rs") for name in paths},
        "limitations": "Source and coordinate checks, not backend padding/allocator or NPU numerical validation. RMS reduction order changes."
    }, indent=2))


if __name__ == "__main__":
    main()
