#!/usr/bin/env python3
"""Verify M04 source, frozen builds, schedule accounting and Arena provenance."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from analyze import analyze, coordinates, digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FAMILY = "m04-tiled-weight-supply"
LABEL = "m04-row-tiles-screen"


def check():
    recipe = json.loads((HERE / "row-tiles-v1/recipe.json").read_text())
    origin = subprocess.check_output(["git", "show", recipe["base_commit"]+
                                      ":furiosa-opt-gemma4-12B/src/device/shared/mlp.rs"], cwd=ROOT, text=True)
    candidate = (HERE / "row-tiles-v1/mlp.rs").read_text()
    def section(source, first, last):
        a = source.index(first)
        return source[a:source.index(last, a)]
    # GeGLU's global scales and BF16 boundaries remain byte-identical.
    assert section(origin, "pub(crate) fn geglu(", "pub(crate) type DownEpilogueSlices") == section(
        candidate, "pub(crate) fn geglu(", "pub(crate) type DownEpilogueSlices")
    assert section(origin, "fn cast_gathered_rows(", "pub(crate) fn feedforward(") == section(
        candidate, "fn cast_gathered_rows(", "pub(crate) fn feedforward(")
    # Arithmetic of quantization is unchanged; only cluster ownership changed.
    old_quant = origin[origin.index("    // Amax on the canonical input"):]
    new_quant = candidate[candidate.index("    // Amax on the canonical input"):]
    assert old_quant.replace("Chip, m![2],", "Chip, Cluster,") == new_quant
    builds = {}
    for path in sorted(HERE.glob("build-*-repro.json")):
        r = json.loads(path.read_text())
        assert r["exact_compile_exit_code"] == 0
        plan = r["source_plan"]
        expected = dict(plan["source_origin_inputs"])
        for target, override in plan["source_overrides"].items():
            actual = digest((ROOT / override["path"]).read_bytes())
            assert actual == {k: override[k] for k in ("bytes", "sha256")}
            expected[target] = actual
        assert expected == r["source_inputs"]
        assert set(plan["source_overrides"]) == {"src/device/shared/mlp.rs"}
        for target, frozen in r["frozen_files"].items():
            assert digest((ROOT / frozen).read_bytes()) == r["source_inputs"].get(target, r["logs"].get(target))
        assert digest((ROOT / plan["recipe"]["path"]).read_bytes()) == {
            k: plan["recipe"][k] for k in ("bytes", "sha256")}
        artifacts = ROOT / "moa-dev/experiments/artifacts" / FAMILY / r["label"]
        assert digest((artifacts / "schedule.json").read_bytes()) == r["schedule"]
        assert digest((artifacts / "exact-kernel.bin").read_bytes()) == r["exact_image"]
        if r.get("host_build_exit_code") == 0:
            package = artifacts / "package"
            assert {p.name: digest(p.read_bytes()) for p in package.iterdir()} == r["package_files"]
            host = (package / "test_runtime").read_bytes()
            for kernel, image in r["images"].items():
                data = (artifacts / "kernels" / f"furiosa_opt_gemma4::ops::{kernel}.bin").read_bytes()
                assert digest(data) == {k: image[k] for k in ("bytes", "sha256")}
                assert host.count(data) == image["host_occurrences"] == 1
                if kernel == r["target_kernel"]:
                    assert digest(data) == r["exact_image"]
                else:
                    assert image["sha256"] == plan["expected_unchanged_images"][kernel]
            first = json.loads((HERE / f"build-{r['label'].removesuffix('-package')}-repro.json").read_text())
            assert first["exact_image"] == r["exact_image"] and first["schedule"] == r["schedule"]
        builds[path.name] = {"source_inputs": len(expected), "frozen_files": len(r["frozen_files"]),
                             "compile": r["exact_compile_exit_code"], "host": r.get("host_build_exit_code")}
    static = json.loads((HERE / "static-analysis.json").read_text())
    assert analyze(HERE / "evidence/m04-row-tiles-v1/schedule.json", True) == static["candidate"]
    assert analyze(HERE.parent / "e104-weight-path/evidence/profile-v1/schedule.json", False) == static["control"]
    assert coordinates() == static["coordinates"]
    sys.path.insert(0, str(ROOT / "moa-dev"))
    from measure import parse_log
    result = json.loads((HERE / f"{LABEL}-results.json").read_text())
    raw = (HERE / f"evidence/arena-{result['job']['id']}.log").read_bytes()
    assert digest(raw)["sha256"] == result["log_sha256"]
    manifest = json.loads((HERE / f"{LABEL}-build.json").read_text())
    assert {k: v for k,v in result["package_files"].items() if k != "job.yaml"} == manifest["package_files"]
    sections = re.findall(r"^STRUCTURAL_BEGIN (\w+)\n(.*?)^STRUCTURAL_END \1$", raw.decode(), re.M | re.S)
    assert [n for n,_ in sections] == [s["name"] for s in manifest["stages"]]
    assert raw.count(b"STRUCTURAL_COMPLETE") == 1
    for stage, log in sections:
        assert f"STRUCTURAL_EXIT {stage} 0" in log
        assert parse_log(log) == result["stages"][stage]["kernels"]
    assert result["valid"] and result["job"]["status"] == "SUCCEEDED"
    return {"scope": "Source scope, exact row coverage and immutable compilation/Arena artifacts; basic NPU correctness, not expanded accuracy or independent repeats.",
            "coordinates": static["coordinates"], "builds": builds,
            "unchanged_geglu_and_quantization_arithmetic": True,
            "unchanged_host_inputs_and_qkv_o_images": True,
            "arena": {"job": result["job"]["id"], "valid": True,
                      "comparisons_passed": sum(s["comparisons_passed"] for s in result["stages"].values()),
                      "candidate_ffn_inputs": 3, "candidate_ffn_output_elements": 11520},
            "schedule_recomputed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = check()
    path = HERE / "verification.json"
    if args.check:
        assert json.loads(path.read_text()) == data
    else:
        path.write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps(data, indent=2))
