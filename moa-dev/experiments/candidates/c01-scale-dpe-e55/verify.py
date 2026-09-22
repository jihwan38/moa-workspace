#!/usr/bin/env python3
"""Audit C01's frozen source -> images -> uploaded packages -> measurements."""
import argparse
import json
import statistics
import sys

from build import ARTIFACTS, HERE, IMAGES, ROOT, digest
sys.path.insert(0, str(ROOT / "moa-dev"))
from measure import parse_log
from extra_seeds import parse_log as parse_extra_log


def read(path):
    return json.loads(path.read_text())


def union(intervals):
    total = end = 0
    for begin, stop in sorted(intervals):
        if stop > end:
            total += stop - max(end, begin)
            end = stop
    return total


def audit():
    control = read(ROOT / "docs/experiments/e55-c01-control-repeats.json")
    control_files = control["runs"][0]["package_files"]
    assert control["valid"] and len(control["runs"]) == 3
    assert all(r["valid"] and r["package_files"] == control_files for r in control["runs"])
    assert control_files["test_runtime"]["sha256"] == "b22444012a01de7f700bb4ecb8316a87093d5c5bde7cdd6ec96c8bfb8afa682f"
    recipe = read(ROOT / "baseline/e55.json")
    baseline_inputs = read(ROOT / recipe["compile_inputs"])
    base_cycles = statistics.median(r["kernels"]["decoder_feedforward"]["median_cycles"] for r in control["runs"])
    variants = [
        ("E65", "upgate-streamed", "build-upgate-streamed.json", "upgate-streamed-package", "upgate-streamed-kernels", "e65-c01-upgate-scale-dpe-repeats.json", "inconclusive"),
        ("E66", "down", "build-down-repro.json", "down/package", "down/kernels", "e66-c01-down-scale-dpe.json", "regressed_single_screen"),
        ("E67", "combined", "build-combined-repro.json", "combined/package", "combined/kernels", "e67-c01-combined-scale-dpe.json", "regressed_single_screen"),
    ]
    results = []
    reports = [control]
    for number, variant, build_file, package, kernels, report_file, decision in variants:
        build = read(HERE / build_file)
        source = dict(baseline_inputs)
        source["src/device/shared/mlp.rs"] = digest((HERE / f"mlp.{variant}.rs").read_bytes())
        assert build["source_inputs"] == source
        assert len(source) == 46
        report = read(ROOT / "docs/experiments" / report_file)
        reports.append(report)
        assert report["valid"] and all(r["valid"] for r in report["runs"])
        for run in report["runs"]:
            assert {k: v for k, v in run["package_files"].items() if k != "job.yaml"} == build["package_files"]
            for name in ("fixtures.safetensors", "remote_entrypoint.sh"):
                assert run["package_files"][name] == control_files[name]
        host = (ARTIFACTS / package / "test_runtime").read_bytes()
        for name, expected in build["package_files"].items():
            assert digest((ARTIFACTS / package / name).read_bytes()) == expected
        for name, baseline_hash in IMAGES.items():
            image = (ARTIFACTS / kernels / f"furiosa_opt_gemma4::ops::{name}.bin").read_bytes()
            assert host.count(image) == 1
            assert digest(image)["sha256"] == build["images"][name]["sha256"]
            if name != "decoder_feedforward":
                assert digest(image)["sha256"] == baseline_hash
        values = [r["kernels"]["decoder_feedforward"]["median_cycles"] for r in report["runs"]]
        cycles = statistics.median(values)
        assert cycles == report["aggregate"]["decoder_feedforward"]["median_cycles"]
        results.append({"id": number, "variant": variant, "jobs": [r["job"]["id"] for r in report["runs"]],
                        "ffn_job_medians": values, "ffn_cycles": cycles,
                        "change_percent": 100 * (cycles / base_cycles - 1), "decision": decision,
                        "source_inputs_verified": 46, "qkv_output_images_unchanged": True,
                        "package_hashes_match_upload": True, "adopted": False})
    extra = read(ROOT / "docs/experiments/e65-c01-upgate-extra.json")
    extra_build = read(HERE / "build-upgate-extra.json")
    extra_log = (ROOT / extra["log"]).read_bytes()
    assert digest(extra_log)["sha256"] == extra["log_sha256"]
    assert parse_extra_log(extra_log.decode()) == extra["results"]
    assert extra["results"]["output_comparisons_passed"] == 30
    assert {k: v for k, v in extra["package_files"].items() if k != "job.yaml"} == extra_build["package_files"]
    extra_host = (ARTIFACTS / "upgate-extra-package/test_runtime").read_bytes()
    assert digest(extra_host) == extra_build["package_files"]["test_runtime"]
    for name in IMAGES:
        assert extra_host.count((ARTIFACTS / "upgate-streamed-kernels" / f"furiosa_opt_gemma4::ops::{name}.bin").read_bytes()) == 1
    diagnostics = read(HERE / "diagnostics.json")
    fixed = diagnostics["runs"][-1]
    assert fixed["job"]["id"] == 72783 and fixed["job"]["exit_code"] == 0
    assert len(fixed["stages"]) == 12 and all(s["mismatches"] == 0 for s in fixed["stages"])
    probe = read(HERE / "build-probe-clear8.json")
    for path, name in (("src/lib.rs", "probe_lib.rs"), ("src/scale_probe.rs", "scale_probe.rs"), ("src/bin/probe_scale.rs", "probe_scale.rs")):
        assert digest((HERE / name).read_bytes()) == probe["source_inputs"][path]
    assert {k: v for k, v in fixed["package_files"].items() if k != "job.yaml"} == probe["package_files"]
    comparisons = 0
    for report in reports:
        for run in report["runs"]:
            log = (ROOT / run["log"]).read_bytes()
            assert digest(log)["sha256"] == run["log_sha256"]
            assert parse_log(log.decode()) == run["kernels"]
            comparisons += sum(k["output_comparisons_passed"] for k in run["kernels"].values())
    schedules = {}
    for label, filename in (("E55", "e55-control-schedule.json"), ("materialized", "upgate-materialized-schedule.json"),
                            ("E65", "upgate-streamed-schedule.json"), ("E66", "down/schedule.json"), ("E67", "combined/schedule.json")):
        data = (ARTIFACTS / filename).read_bytes()
        instructions = json.loads(data)["instructions"]
        by_engine = {engine: union([(i["lifetime"]["begin"], i["lifetime"]["end"]) for i in instructions if i["tpe"] == engine])
                     for engine in sorted({i["tpe"] for i in instructions})}
        schedules[label] = dict(digest(data), instructions=len(instructions),
            makespan=max(i["lifetime"]["end"] for i in instructions), engine_interval_unions=by_engine)
    return {"base": "baseline-e55", "control_jobs": [r["job"]["id"] for r in control["runs"]],
            "control_ffn_job_medians": [r["kernels"]["decoder_feedforward"]["median_cycles"] for r in control["runs"]],
            "control_ffn_cycles": base_cycles, "variants": results, "static_schedules": schedules,
            "official_harness_comparisons_passed": comparisons,
            "extra_input_comparisons_passed": extra["results"]["output_comparisons_passed"],
            "probe_full120_checks": sum(s["checked"] for s in fixed["stages"]),
            "adopted_candidate": None, "model_e2e_tested": False, "official_submission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = audit()
    path = HERE / "summary.json"
    if args.check:
        assert read(path) == summary
    else:
        path.write_text(json.dumps(summary, indent=2) + "\n")
    print("Verified 3 candidates, 46 source inputs each, unchanged QKV/O, uploaded packages, 150 output comparisons and D01.")
