#!/usr/bin/env python3
"""Verify M01 frozen source scope, coordinate mapping, and build provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def check():
    plan = json.loads((HERE / "plan-dma-v2.json").read_text())
    commit = plan["source_base_commit"]
    origin = subprocess.check_output(["git", "show", commit + ":furiosa-opt-gemma4-12B/src/device/sliding/projection.rs"], cwd=ROOT, text=True)
    a = origin.index("pub(crate) fn project_output(")
    b = origin.index("\npub(crate) type OutputEpilogueSlices", a)
    variants = {}
    for variant in ("dma-v1", "switch-v1", "dma-v2", "switch-v2"):
        code = (HERE / variant / "projection.rs").read_text()
        x = code.index("pub(crate) fn project_output(")
        y = code.index("\npub(crate) type OutputEpilogueSlices", x)
        assert code[:x] == origin[:a] and code[y:] == origin[b:]
        body = code[x:y]
        assert body.count("weight.to_dm(&mut device.tdma)") == 1
        assert body.count("fetch_table_lookup::<bf16>()") == 1
        assert ".cast" not in body
        assert "IngestSlices = m![Qs / 256, H / 120 % 16]" in body
        assert ("partial.to_dm" in body) == variant.startswith("dma")
        variants[variant] = {"projection": digest(code.encode()), "only_project_output_changed": True,
                             "weight_dm_repack": False, "partial_bf16_cast": False}

    # Independent row/K identity tags expose mix-ups across cluster, row, column
    # and the consumer's eight-F32 packet. This does not execute Rust or emulate DPE.
    cases = 0
    for c in range(2):
        for r in range(16):
            for packet in range(15):
                for lane in range(8):
                    h = 1920*c + 120*r + 8*packet + lane
                    source = {16*k+r: 10000*h+k for k in range(16)}
                    remapped = {16*(s % 16)+s//16: value for s,value in source.items()}
                    assert sum(remapped[16*r+k] for k in range(16)) == 16*10000*h+120
                    cases += 1
    assert cases == 3840

    builds = {}
    for path in sorted(HERE.glob("build-*-repro.json")):
        record = json.loads(path.read_text())
        for target, frozen in record["frozen_files"].items():
            expected = record["source_inputs"].get(target, record["logs"].get(target))
            assert digest((ROOT / frozen).read_bytes()) == expected, frozen
        recipe = record["source_plan"]["recipe"]
        assert digest((ROOT / recipe["path"]).read_bytes()) == {k: recipe[k] for k in ("bytes", "sha256")}
        for target, override in record["source_plan"]["source_overrides"].items():
            assert digest((ROOT / override["path"]).read_bytes()) == record["source_inputs"][target]
        if record.get("host_build_exit_code") == 0:
            artifacts = ROOT / "moa-dev/experiments/artifacts/m01-partial-remap" / record["label"]
            package = artifacts / "package"
            assert {p.name: digest(p.read_bytes()) for p in package.iterdir()} == record["package_files"]
            host = (package / "test_runtime").read_bytes()
            for kernel, image in record["images"].items():
                data = (artifacts / "kernels" / f"furiosa_opt_gemma4::ops::{kernel}.bin").read_bytes()
                assert digest(data) == {k: image[k] for k in ("bytes", "sha256")}
                assert host.count(data) == image["host_occurrences"] == 1
                if kernel == record["target_kernel"]:
                    assert digest(data) == record["exact_image"]
                else:
                    assert image["sha256"] == record["source_plan"]["expected_unchanged_images"][kernel]
            first = json.loads((HERE / f"build-{record['label'].removesuffix('-package')}-repro.json").read_text())
            assert first["exact_image"] == record["exact_image"]
            assert first["schedule"] == record["schedule"]
        builds[path.name] = {"frozen_files_checked": len(record["frozen_files"]),
                             "exact_compile_exit_code": record.get("exact_compile_exit_code"),
                             "host_build_exit_code": record.get("host_build_exit_code"),
                             "failed_exit_code": record.get("failed_exit_code")}
    arena = None
    result_path = HERE / "m01-remap-screen-results.json"
    if result_path.exists():
        sys.path.insert(0, str(ROOT / "moa-dev"))
        from measure import parse_log
        result = json.loads(result_path.read_text())
        raw = (HERE / "evidence" / f"arena-{result['job']['id']}.log").read_bytes()
        assert digest(raw)["sha256"] == result["log_sha256"]
        manifest = json.loads((HERE / "m01-remap-screen-build.json").read_text())
        assert {k:v for k,v in result["package_files"].items() if k != "job.yaml"} == manifest["package_files"]
        sections = re.findall(r"^STRUCTURAL_BEGIN (\w+)\n(.*?)^STRUCTURAL_END \1$", raw.decode(), re.M | re.S)
        assert [n for n,_ in sections] == [s["name"] for s in manifest["stages"]]
        for stage,text in sections:
            assert parse_log(text) == result["stages"][stage]["kernels"]
            assert result["stages"][stage]["valid"]
        assert result["valid"] and result["job"]["exit_code"] == 0
        arena = {"job": result["job"]["id"], "complete_passing_stages": len(sections),
                 "total_comparisons": sum(s["comparisons_passed"] for s in result["stages"].values()),
                 "log_and_downloaded_package_manifest_checked": True}
    return {"scope": "Source boundaries, coordinate model, immutable build/package provenance; NPU basic results verified from the preserved log when present.",
            "variants": variants, "row_packet_transpose_reduce_cases": cases, "builds": builds,
            "arena": arena,
            "logical_coverage_report": "docs/research/rngd-dataflow-accounting-20260925.json"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = check()
    output = HERE / "verification.json"
    if args.check:
        assert json.loads(output.read_text()) == result
    else:
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(result['variants'])} source snapshots, {result['row_packet_transpose_reduce_cases']} coordinates, {len(result['builds'])} builds.")
