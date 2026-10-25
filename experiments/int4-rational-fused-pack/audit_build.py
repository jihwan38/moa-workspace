#!/usr/bin/env python3
"""Verify captured source/ELF, fresh registry and strong control image lineage."""
from pathlib import Path
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
TREE, ROOT = HERE.parents[1], HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = ROOT / "b16-f4-mse-lm-fp8/experiments/sdk09-cache-audit/audit_registry.py"
    spec = importlib.util.spec_from_file_location("registry", parser)
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    run_dir = ROOT / "runs/a006-rational-fused-pack-device-build"
    run = json.loads((run_dir / "run.json").read_text())
    assert run["exit_code"] == 0 and run["result"] == "command_succeeded"
    assert run["build_inputs_unchanged"] and run["fresh_package"]["contents_changed"] is False
    hashes = json.loads((run_dir / "source-hashes.json").read_text())
    for relative, expected in hashes.items():
        assert sha(TREE / relative) == expected, relative
    assert {str(p.relative_to(TREE)) for p in (TREE / "src").rglob("*") if p.is_file()} == {p for p in hashes if p.startswith("src/")}
    artifact = run["built_artifacts"]["target/release/test_int4_rational_fused_pack"]
    elf = Path(artifact["path"])
    assert sha(elf) == artifact["sha256"] == "be18ad3ee9fcc13c04e19df4776e99ac95599ff2a12167538aba7444fe430c68"
    assert elf.stat().st_size == artifact["size"] == 3357912
    current = registry.read_artifact(elf)
    prefix = "furiosa_opt_gemma4::"
    expected = {
        prefix + "ops_int4_rational_fused_pack::quantize_pack_input_aligned",
        prefix + "ops_int4_rational_fused_pack::quantize_pack_hidden",
        prefix + "ops_int4_rational_fused_pack::pack_hidden_control",
        prefix + "ops_int4_pack_input_aligned::pack_input",
        prefix + "ops_int4_rational_lut_hidden::quantize_row",
        prefix + "ops_int4_rational_lut_p512_compact::quantize_row",
    }
    assert current["record_count"] == 6
    assert {r["path"] for r in current["records"]} == expected
    assert all(r["key"] == [] for r in current["records"])
    fresh = HERE / "kernel-out-a006-device/furiosa-opt-gemma4"
    assert {p.stem for p in fresh.glob("*.bin")} == expected
    for row in current["records"]:
        assert sha(fresh / (row["path"] + ".bin")) == row["sha256"]
    by_path = {r["path"]: r for r in current["records"]}
    resources = {}
    for function, directory in [
        ("quantize_pack_input_aligned", "kernel-out-a004-aligned"),
        ("quantize_pack_hidden", "kernel-out-a003-hidden"),
    ]:
        name = prefix + "ops_int4_rational_fused_pack::" + function
        path = HERE / directory / "furiosa-opt-gemma4" / (name + ".bin")
        assert sha(path) == by_path[name]["sha256"]
        resources[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "byte_equal": True}
    controls = [
        ("ops_int4_rational_lut_p512_compact::quantize_row", "ops_int4_rational_lut_p512_compact::quantize_row",
         "runs/a013-rational-lut-device-build/built/target/release/test_int4_rational_lut", "43664ed77db36c0493c92874b6d07ed88e9d44423a1d13c4f7a3820245229b7c", 322892),
        ("ops_int4_rational_lut_hidden::quantize_row", "ops_int4_rational_lut_hidden::quantize_row",
         "runs/a005-rational-hidden-device-build/built/target/release/test_int4_rational_lut_hidden", "a5a1b579e32f474ae5f505fecf7627ba8014bcd7cea7b2472267c2aaeddad5de", 322970),
        ("ops_int4_pack_input_aligned::pack_input", "ops_int4_pack_input_aligned::pack_input",
         "runs/a004-int4-pack-input-aligned-build/built/target/release/test_int4_pack_input_aligned", "9900b5af49813e0e3a8ad167ccea3f8d9f3ba540ce78153999a378ff4ef23715", 323012),
        ("ops_int4_rational_fused_pack::pack_hidden_control", "ops_int4_pack_staged_store::pack_hidden",
         "runs/a005-int4-pack-staged-device-build/built/target/release/test_int4_pack_staged_store", "eb0c888759665175abb1e11bd1a1b24660038b30ec6d778cb3a1bc24df2ac0c0", 322987),
    ]
    control_records = []
    for new_name, old_name, relative, old_sha, job in controls:
        old = registry.read_artifact(ROOT / relative)
        assert old["elf_sha256"] == old_sha
        row = next(r for r in old["records"] if r["path"] == prefix + old_name and not r["key"])
        actual = by_path[prefix + new_name]
        assert (row["bytes"], row["sha256"]) == (actual["bytes"], actual["sha256"])
        control_records.append({"current_path": prefix + new_name, "original_path": row["path"],
                                "image_sha256": row["sha256"], "byte_equal": True,
                                "validated_artifact": relative, "validated_artifact_sha256": old_sha,
                                "previous_actual_job": job})
    result = {
        "result": "BUILD_CAPTURE_AND_SIX_IMAGE_PARITY_PASS_NO_FUSED_DEVICE_EXECUTION",
        "source_commit": run["commit"], "artifact": artifact,
        "source_inputs": {"count": len(hashes), "all_unchanged": True, "manifest_sha256": sha(run_dir / "source-hashes.json")},
        "run": str(run_dir.relative_to(ROOT)),
        "raw_hashes": {name: sha(run_dir / name) for name in ["run.json", "source-hashes.json", "source.diff", "output.log"]},
        "registry_parser": {"path": str(parser.relative_to(ROOT)), "sha256": sha(parser)},
        "embedded_registry": current, "all_fresh_images_equal": True,
        "selected_resource_parity": resources, "strong_control_parity": control_records,
        "old_unaligned_fused_input_embedded": False,
        "source_reviews": {"root": "607c5c93 PASS", "round2_path": "607c5c93 PASS; unchanged at33d7d3cf"},
        "root_only_dispatch": [{"args": [arg], "timeout_seconds": 70} for arg in ["input", "hidden"]],
        "scope": "Full quantize+pack correctness and paired cost remain pending. No I4 consumer/GEMM/FFN/model claim.",
    }
    (HERE / "device-build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": artifact, "source_inputs": len(hashes), "images": current["record_count"], "all_parity": True}, indent=2))


if __name__ == "__main__":
    main()
