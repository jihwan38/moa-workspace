#!/usr/bin/env python3
"""Verify C02/C03 sources, compiler outputs, submitted bytes and recorded results."""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys

from build import ARTIFACTS, HERE, IMAGES, ROOT, digest
from edges import edge_harness
sys.path.insert(0, str(ROOT / "moa-dev"))
from measure import parse_log
from extra_seeds import parse_log as parse_extra


def read(path):
    return json.loads(path.read_text())


def signatures(text):
    return re.findall(r"#\[device\(chip = 1\)\]\s*(pub fn .*?\)\s*)\{", text, re.S)


def audit_builds():
    baseline = read(ROOT / "baseline/e55.json")
    inputs = read(ROOT / baseline["compile_inputs"])
    assert len(inputs) == 46
    builds = {}
    for path in sorted(HERE.glob("build-*-repro.json")):
        build = read(path)
        label, variant = build["label"], build["variant"]
        out = ARTIFACTS / label
        expected = dict(inputs)
        for name in build["changed_sources"]:
            source = (HERE / "attempts/c02-combined-dma.rs" if label == "c02-combined-first"
                      else HERE / variant / Path(name).name)
            expected[name] = digest(source.read_bytes())
        if build.get("edge_inputs"):
            original = (ARTIFACTS / "c03-cluster-heads-first/crate/src/bin/test_kernels.rs").read_bytes()
            expected["src/bin/test_kernels.rs"] = digest(edge_harness(original))
        assert build["source_inputs"] == expected, label
        assert build["source_inputs_verified_in_container"]
        assert build["exact_compile_exit_code"] == build["host_build_exit_code"] == 0
        for name, value in expected.items():
            assert digest((out / "crate" / name).read_bytes()) == value, (label, name)
        public_count = 0
        for filename in ("ops.rs", "ops_audio.rs", "ops_vision.rs"):
            original_ops = (ROOT / "furiosa-opt-gemma4-12B/src" / filename).read_text()
            assert signatures((out / "crate/src" / filename).read_text()) == signatures(original_ops)
            public_count += len(signatures(original_ops))
        assert public_count == 15
        host = (out / "package/test_runtime").read_bytes()
        for name, value in build["package_files"].items():
            assert digest((out / "package" / name).read_bytes()) == value
        for name, baseline_hash in IMAGES.items():
            image = (out / "kernels" / f"furiosa_opt_gemma4::ops::{name}.bin").read_bytes()
            assert dict(digest(image), host_occurrences=host.count(image)) == build["images"][name]
            assert host.count(image) == 1
            if name == build["target_kernel"]:
                assert image == (out / "exact-kernel.bin").read_bytes()
            else:
                assert digest(image)["sha256"] == baseline_hash
        for name, value in build["logs"].items():
            assert digest((out / name).read_bytes()) == value
        schedule = read(out / "schedule.json")
        assert max(i["lifetime"]["end"] for i in schedule["instructions"]) == build["static_cycles"]
        builds[label] = build
    probe = read(HERE / "build-packet-probe.json")
    for source, file in {"src/lib.rs": "probe_lib.rs", "src/packet_probe.rs": "packet_probe.rs", "src/bin/probe_packet.rs": "probe_packet.rs"}.items():
        assert digest((HERE / file).read_bytes()) == probe["source_inputs"][source]
    out = ARTIFACTS / "packet-probe-sequencer"
    host = (out / "probe_packet").read_bytes()
    assert digest(host) == probe["host"]
    image = next((out / "crate/kernels").rglob("*.bin")).read_bytes()
    assert dict(digest(image), host_occurrences=host.count(image)) == probe["image"]
    assert host.count(image) == 1
    return builds


def repeated(path, build=None):
    report = read(path)
    assert report["valid"] and len(report["runs"]) == 3
    files = report["runs"][0]["package_files"]
    for run in report["runs"]:
        assert run["valid"] and run["package_files"] == files
        log = (ROOT / run["log"]).read_bytes()
        assert digest(log)["sha256"] == run["log_sha256"]
        assert parse_log(log.decode()) == run["kernels"]
        if build:
            assert {k: v for k, v in files.items() if k != "job.yaml"} == build["package_files"]
    for kernel, aggregate in report["aggregate"].items():
        assert statistics.median(r["kernels"][kernel]["median_cycles"] for r in report["runs"]) == aggregate["median_cycles"]
    return report


def audit_results(builds, check):
    c03 = repeated(ROOT / "docs/experiments/e68-c03-cluster-heads.json", builds["c03-cluster-heads-first"])
    control = repeated(ROOT / "docs/experiments/e69-c02-c03-control.json")
    assert control["runs"][0]["package_files"]["test_runtime"]["sha256"] == "b22444012a01de7f700bb4ecb8316a87093d5c5bde7cdd6ec96c8bfb8afa682f"
    batch = read(HERE / "batch-results.json")
    payload = read(HERE / "batch-build.json")["package_files"]
    assert {k: v for k, v in batch["package_files"].items() if k != "job.yaml"} == payload
    for name, value in payload.items():
        assert digest((ARTIFACTS / "batch-package" / name).read_bytes()) == value
    data = (ROOT / batch["log"]).read_bytes()
    assert digest(data)["sha256"] == batch["log_sha256"]
    sections = dict(re.findall(r"^C02_C03_BEGIN (\w+)\n(.*?)^C02_C03_END \1$", data.decode(), re.M | re.S))
    assert batch["valid"], "Inspect failed stages before creating performance reports"
    assert len(sections) == 7 and len(batch["stages"]) == 7
    for name in ("baseline_before", "baseline_after", "upgate", "down", "combined"):
        assert parse_log(sections[name]) == batch["stages"][name]["kernels"]
    extra = parse_extra(sections["edges"])
    assert all(batch["stages"]["edges"][k] == v for k, v in extra.items())
    assert batch["stages"]["edges"]["untouched_cache_elements_checked"] == 25141248
    for stage, label in {"upgate": "c02-upgate-sequencer", "down": "c02-down-sequencer", "combined": "c02-combined-sequencer", "edges": "c03-edges"}.items():
        assert payload[stage + "_runtime"] == builds[label]["package_files"]["test_runtime"]
        assert payload["edges.safetensors" if stage == "edges" else "fixtures.safetensors"] == builds[label]["package_files"]["fixtures.safetensors"]
    assert payload["probe_packet"] == read(HERE / "build-packet-probe.json")["host"]
    assert payload["baseline_runtime"] == control["runs"][0]["package_files"]["test_runtime"]
    for name in ("fixtures.safetensors", "remote_entrypoint.sh"):
        assert payload[name] == control["runs"][0]["package_files"][name] == c03["runs"][0]["package_files"][name]
    baseline_values = [batch["stages"][label]["kernels"]["decoder_feedforward"]["median_cycles"] for label in ("baseline_before", "baseline_after")]
    local_control = statistics.median(baseline_values)
    results = []
    for number, stage in ((70, "upgate"), (71, "down"), (72, "combined")):
        kernels = batch["stages"][stage]["kernels"]
        run = {k: batch[k] for k in ("job", "log", "log_sha256", "package_files")}
        run.update(valid=True, kernels=kernels, batch_stage=stage, executed_host=stage + "_runtime")
        report = {"schema_version": 1, "valid": True, "label": f"e{number}-c02-{stage}", "runs": [run],
                  "method": batch["method"], "aggregate": {k: {"median_cycles": v["median_cycles"], "job_medians": [v["median_cycles"]], "min_cycles": v["median_cycles"], "max_cycles": v["median_cycles"], "relative_range": 0} for k, v in kernels.items()},
                  "control_processes_same_job": baseline_values, "independent_arena_jobs": 1, "batch_evidence": str((HERE / "batch-results.json").relative_to(ROOT))}
        path = ROOT / f"docs/experiments/e{number}-c02-{stage}.json"
        if check:
            assert read(path) == report
        else:
            path.write_text(json.dumps(report, indent=2) + "\n")
        value = kernels["decoder_feedforward"]["median_cycles"]
        results.append({"id": f"E{number}", "stage": stage, "ffn_cycles": value,
                        "delta_vs_batch_control_percent": 100 * (value / local_control - 1),
                        "delta_vs_three_job_control_percent": 100 * (value / control["aggregate"]["decoder_feedforward"]["median_cycles"] - 1),
                        "above_both_batch_controls": value > max(baseline_values), "independent_jobs": 1, "adopted": False})
    q = c03["aggregate"]["sliding_project_qkv"]
    reference = control["aggregate"]["sliding_project_qkv"]
    return {"base": "baseline-e55", "builds_verified": len(builds), "compile_inputs_per_build": 46,
            "public_device_signatures_preserved": 15, "unrelated_device_images_unchanged": True,
            "c03": {"qkv_cycles": q["median_cycles"], "candidate_job_medians": q["job_medians"], "control_job_medians": reference["job_medians"],
                    "change_percent": 100 * (q["median_cycles"] / reference["median_cycles"] - 1),
                    "decision": "regressed" if q["min_cycles"] > reference["max_cycles"] else "inconclusive",
                    "basic_output_comparisons": 45, "edge_output_comparisons": 30, "cache_sentinels_checked": 25141248},
            "c02": results, "c02_probe": batch["stages"]["probe"], "batch_job": batch["job"]["id"],
            "batch_control_ffn_cycles": baseline_values, "batch_control_midpoint": local_control,
            "basic_output_comparisons_including_controls": 165,
            "adopted_candidate": None, "model_e2e_tested": False, "candidate_official_submission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--builds-only", action="store_true")
    args = parser.parse_args()
    builds = audit_builds()
    if not args.builds_only:
        summary = audit_results(builds, args.check)
        path = HERE / "summary.json"
        if args.check:
            assert read(path) == summary
        else:
            path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Verified {len(builds)} frozen builds, 46 inputs each, 15 public signatures and exact embedded images.")
