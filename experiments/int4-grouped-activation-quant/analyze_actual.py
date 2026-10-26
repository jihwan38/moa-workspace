#!/usr/bin/env python3
"""Replay job323173 evidence; read files/Git only, never submit or execute ELF.

Usage: PYTHONDONTWRITEBYTECODE=1 python3 experiments/int4-grouped-activation-quant/analyze_actual.py
Output is deterministic JSON. Failed terminal executions retain partial records
and cannot produce ACTUAL_CORRECTNESS_PASS, even if stdout contains PASS text.
"""
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import registry_audit
from analyze_metadata_store import git_sources

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RAW = HERE / "actual-323173"
WIDTHS = (3840, 15360)
BUILD_COMMIT = "ff51c81255ea272086daccc2318320dc8b568df2"
SUBMIT_COMMIT = "0dbea7d94e22b25127d79f2aa8fc7057d259dacb"
ELF_SHA = "b71c05511a3217c8d6a3758360996203232df88d6bd5dfb6ae3d429edd37bfb3"
SCOPE = "complete_group_quant_offset_write_read_pack_scale_no_GEMM"
TIME = re.compile(r"GROUPED_A4_PACK_TIME width=(\d+) case=(\d+) repeat=(\d+) samples=(\d+) us=([0-9.]+) scope=(\S+)")
EXACT = re.compile(r"GROUPED_A4_PACK_EXACT width=(\d+) case=(\d+) codes=(\d+) packed_bytes=(\d+) scales=(\d+) offsets=(\d+) postcheck=true")
PROFILE = re.compile(r"PROFILE_GROUPED_A4_PACK width=(\d+)")
PASS = re.compile(r"PASS_GROUPED_A4_PACK codes=(\d+) packed_bytes=(\d+) group_scales=(\d+) offsets=(\d+) no_GEMM_complete_FFN_model_quality_claim")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def provenance():
    manifest = read_json(RAW / "manifest.json")
    assert manifest["job_id"] == 323173
    for name, item in manifest["files"].items():
        p = RAW / name
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"], name
    run = read_json(RAW / "run.json")
    build = read_json(HERE / "device-build/run.json")
    submit_hashes = read_json(RAW / "source-hashes.json")
    build_hashes = read_json(HERE / "device-build/source-hashes.json")
    assert run["commit"] == SUBMIT_COMMIT and run["status"] == ""
    assert build["commit"] == BUILD_COMMIT and build["build_inputs_unchanged"]
    assert run["result"] == build["result"] == "command_succeeded"
    assert run["exit_code"] == build["exit_code"] == 0
    assert (RAW / "source.diff").read_bytes() == (HERE / "device-build/source.diff").read_bytes() == b""
    assert submit_hashes == build_hashes and len(submit_hashes) == 418
    git_sources(BUILD_COMMIT, build_hashes)
    git_sources(SUBMIT_COMMIT, submit_hashes)
    artifact = run["submitted_artifact"]
    built = build["built_artifacts"]["target/release/test_int4_grouped_activation_store"]
    assert artifact["sha256"] == built["sha256"] == ELF_SHA
    paths = [Path(artifact["path"]), Path(artifact["source_path"]), Path(built["path"])]
    for p in paths:
        assert sha(p) == ELF_SHA and p.stat().st_size == 2973256, p
    registry = registry_audit.read_artifact(paths[0])
    assert registry["record_count"] == 2
    symbols = set()
    for row in registry["records"]:
        mode = row["path"].rsplit("::quantize_pack_", 1)[-1]
        assert mode in ("input", "hidden") and mode not in symbols
        symbols.add(mode)
        assert row["path"] == f"furiosa_opt_gemma4::ops_int4_grouped_activation_store::quantize_pack_{mode}"
        assert row["key"] == [] and row["sha256"] == sha(HERE / f"metadata-resource-images/{mode}.bin")
    cmd = run["command"]
    assert cmd[cmd.index("--timeout") + 1] == "70"
    assert "submitted job 323173" in (RAW / "output.log").read_text()
    return {
        "build_commit": BUILD_COMMIT, "submit_commit": SUBMIT_COMMIT,
        "functional_source_commit": "f6ad9c83", "source_inputs_exact": 418,
        "original_dependencies": {k: submit_hashes[k] for k in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")},
        "submitted_artifact": artifact, "captured_artifact": built,
        "two_embedded_images_equal_frozen_resources": registry,
        "raw_manifest": manifest,
        "binding_files": {str(p.relative_to(HERE)): sha(p) for p in [RAW / "manifest.json", HERE / "device-build.json", HERE / "standalone-plan.json"]},
    }


def parse_log(text):
    times, exact, profiles, passes, order, other = {}, {}, [], [], [], []
    for line in text.splitlines():
        if m := TIME.fullmatch(line):
            width, case, repeat, samples = map(int, m.groups()[:4])
            us = float(m[5]); key = (width, case, repeat)
            assert key not in times and width in WIDTHS and case in range(5) and repeat in (0, 1)
            assert samples == 10 and m[6] == SCOPE and math.isfinite(us) and us > 0
            times[key] = us; order.append(("time", width, case, repeat))
        elif m := EXACT.fullmatch(line):
            width, case, codes, packed, scales, offsets = map(int, m.groups())
            key = (width, case)
            assert key not in exact and width in WIDTHS and case in range(5)
            assert (codes, packed, scales, offsets) == (512 * width, 256 * width, 512 * (width // 960), 512 * (width // 960))
            exact[key] = dict(codes=codes, packed_bytes=packed, group_scales=scales, offsets=offsets)
            order.append(("exact", width, case))
        elif m := PROFILE.fullmatch(line):
            width = int(m[1]); assert width in WIDTHS and width not in profiles
            profiles.append(width); order.append(("profile", width))
        elif m := PASS.fullmatch(line):
            passes.append(tuple(map(int, m.groups()))); order.append(("pass",))
        else:
            assert not line.startswith(("GROUPED_A4_PACK_", "PASS_GROUPED_A4_PACK", "PROFILE_GROUPED_A4_PACK")), line
            if line.strip(): other.append(line)
    return times, exact, profiles, passes, order, other


def main():
    proof = provenance()
    status = read_json(RAW / "arena-status.json")
    assert status["id"] == 323173
    log = (RAW / "arena.log").read_text()
    times, exact, profiles, passes, order, other = parse_log(log)
    terminal_ok = status["status"] == "SUCCEEDED" and status["exit_code"] == 0
    out = {"provenance": proof, "terminal_status": status, "decision": "TERMINAL_NOT_SUCCESS; no correctness PASS",
           "partial_marker_counts": {"times": len(times), "exact": len(exact), "profiles": len(profiles), "pass": len(passes)}}
    if not terminal_ok:
        out["partial_stdout"] = log
        print(json.dumps(out, indent=2, sort_keys=True)); return
    expected_order = []
    for width in WIDTHS:
        for case in range(5):
            expected_order.extend(("time", width, case, repeat) for repeat in (0, 1))
            if case == 0: expected_order.append(("profile", width))
            expected_order.append(("exact", width, case))
    expected_order.append(("pass",))
    assert order == expected_order
    assert passes == [(49152000, 24576000, 51200, 51200)]
    counts = {name: sum(row[name] for row in exact.values()) for name in ("codes", "packed_bytes", "group_scales", "offsets")}
    assert tuple(counts.values()) == passes[0]
    case_times = []
    for width in WIDTHS:
        for case in range(5):
            values = [times[width, case, r] for r in (0, 1)]
            case_times.append({"width": width, "case": case, "two_ten_call_means_us": values,
                               "min_us": min(values), "max_us": max(values), "mean_us": statistics.mean(values)})
    assert read_json(RAW / "arena-records.json") == []
    spans = [x for x in other if "NPU_SPAN" in x]
    assert not spans
    out.update({
        "decision": "ACTUAL_CORRECTNESS_PASS; standalone quantization/packing cost only",
        "unique_declared_checks": counts,
        "source_bound_additional_checks": {
            "output_bytes_compared_twice": 2 * (counts["packed_bytes"] + 4 * counts["group_scales"] + 4 * counts["offsets"]),
            "explicit_permutation_nibbles": 2 * 512 * sum(WIDTHS),
            "explicit_permutation_scales_and_offsets_each": 2 * 512 * sum(w // 960 for w in WIDTHS),
            "immutable_read_bytes": 5 * 512 * sum(WIDTHS) * 2 + 10 * 4096 + 10 * 32 + 4096,
            "completed_kernel_calls": 2 * (5 * (1 + 2 * 10) + 1),
            "scope": "These checks are assertions in the frozen probe reached before each EXACT marker, not separately printed markers. Initial and post-timing outputs, exact quotient oracle, all input/LUT/shifts, output reuse, and explicit permutations are checked.",
        },
        "fixtures": {"per_width": 5, "base_fixtures": 3, "true_permutations": {"3": 0, "4": 1},
                     "permutation": "row=(73p+17)%512; group=(3g+1)%G; channel=(73k+127)%960; all are bijections",
                     "case2": "Exact half thresholds A=14*2^shift. Its normalized mantissa is224, hence every relative LUT offset is(224-128)*32=3072. Slower wall time is observed; gather contention is not established."},
        "timings": case_times,
        "timing_scope": {"unit": "microseconds", "each_observation": "mean of10 completed calls; two repeats; individual-call latency distribution unavailable",
                         "includes": SCOPE, "excludes": "CPU oracle, upload, readbacks; no GEMM, full FFN or serving",
                         "control": "No same-job production/control arm. No speedup claim."},
        "profile": {"markers": profiles, "npu_span_records": 0, "arena_records": [],
                    "limitation": "Two profiled calls were requested, but no actual device spans were emitted. No DMA or critical-path attribution from this run."},
        "preserved_resource_history": {"original_grouped_H_L_cycles": [60292, 74636], "metadata_repair_H_L_cycles": [29069, 67271],
                                       "scope": "Compiler resources only; no proportional wall-time inference and no overwrite of previous evidence."},
        "limits": ["Synthetic finite BF16 fixtures only; no actual model activation distribution.",
                   "No GEMM/full-FFN/model-quality/serving activation or current-production performance conclusion.",
                   "Case2 timing is retained; complete FFN may have a different LUT-offset distribution and latency."],
    })
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
