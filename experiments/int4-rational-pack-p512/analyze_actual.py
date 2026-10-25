#!/usr/bin/env python3
"""Replay frozen files into a local JSON summary; no device/tool invocation."""
import argparse
import ast
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
EXPECTED_ELF = "cd870cffa309a89bee202ef24e9821a96ef79fdeb97f7f386e8eb3a8369bf1b2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    assert len(values) == 4 and all(math.isfinite(x) and x > 0 for x in values)
    return {"samples_us": values, "minimum_us": min(values), "maximum_us": max(values),
            "mean_us": statistics.mean(values), "median_us": statistics.median(values),
            "sample_count": len(values), "completed_calls_per_sample": 16}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round2-root", type=Path, default=TREE.parent)
    args = parser.parse_args()
    raw = HERE / "actual-322961"
    log = (raw / "arena.log").read_text()
    status = json.loads((raw / "arena-status.json").read_text())
    run = json.loads((raw / "run.json").read_text())
    build = json.loads((HERE / "actual-build.json").read_text())
    source = json.loads((raw / "source-hashes.json").read_text())
    assert status["id"] == 322961 and status["status"] == "SUCCEEDED" and status["exit_code"] == 0
    assert run["exit_code"] == 0 and run["result"] == "command_succeeded"
    assert len(source) == 401 and all(sha(TREE / p) == h for p, h in source.items())
    assert run["submitted_artifact"]["sha256"] == EXPECTED_ELF == build["artifact"]["sha256"]
    artifact_paths = {
        "submitted": args.round2_root / "runs/rational-pack-p512-20261026/submitted/test_int4_pack_p512",
        "captured": args.round2_root / "runs/a007-rational-pack-p512-actual-build/built/target/release/test_int4_pack_p512",
    }
    assert all(sha(p) == EXPECTED_ELF for p in artifact_paths.values())
    assert build["captured_registry_records"] == 4 and build["all_four_embedded_records_match_fresh_bins"]
    registry = json.loads((HERE / "actual-build-registry.json").read_text())
    assert registry["elf_sha256"] == EXPECTED_ELF and len(registry["records"]) == 4
    for entry in registry["records"]:
        bins = list((HERE / "kernel-out-a007").glob("**/" + entry["path"] + ".bin"))
        assert len(bins) == 1 and sha(bins[0]) == entry["sha256"]

    cases = []
    for m in re.finditer(r"^PACK_P512_CASE width=(\d+) case=(\d+) codes=(\d+) packed_bytes=(\d+) "
                         r"whole_owner_exact=true typed_raw_canonical_exact=true$", log, re.M):
        width, case, codes, packed = map(int, m.groups())
        assert width in (3840, 15360) and case in range(5)
        assert codes == 512 * width and packed == codes // 2
        cases.append({"width": width, "case": case, "codes": codes, "packed_bytes": packed})
    assert len(cases) == 10 and {(x["width"], x["case"]) for x in cases} == {
        (w, c) for w in (3840, 15360) for c in range(5)}
    finals = re.findall(r"^PASS_PACK_P512 codes=(\d+) packed_bytes=(\d+) owner_rejection_checks=(\d+) "
                        r"scope=integer_packing_and_raw_blocked_layout_only no_quantizer_gemm_model_gain_claim=true$", log, re.M)
    assert finals == [("49152000", "24576000", "6")]
    assert sum(x["codes"] for x in cases) == 49_152_000
    assert sum(x["packed_bytes"] for x in cases) == 24_576_000

    timings = []
    for m in re.finditer(r"^PACK_P512_TIMING width=(\d+) case=(\d+) repetitions=16 "
                         r"pack_us=(\[[^\]]+\]) pack_plus_raw_copy_us=(\[[^\]]+\]) "
                         r"complete_awaited_calls=true excludes_quantizer_gemm_setup_readback=true$", log, re.M):
        width, case = map(int, m.groups()[:2])
        pack, copy = map(ast.literal_eval, m.groups()[2:])
        assert len(pack) == len(copy) == 2
        timings.append({"width": width, "case": case, "pack_us": pack, "pack_plus_raw_copy_us": copy})
    assert len(timings) == 4 and {(x["width"], x["case"]) for x in timings} == {
        (w, c) for w in (3840, 15360) for c in (0, 4)}
    timing_summary = {}
    for width in (3840, 15360):
        timing_summary[str(width)] = {
            key: summary([v for x in timings if x["width"] == width for v in x[key]])
            for key in ("pack_us", "pack_plus_raw_copy_us")}
    span_count = sum(line.startswith("NPU_SPAN ") for line in log.splitlines())
    profile_count = sum(line.startswith("PACK_P512_PROFILE ") for line in log.splitlines())
    records = json.loads((raw / "arena-records.json").read_text())
    assert span_count == profile_count == 0 and records == []
    resource = json.loads((HERE / "resource-result.json").read_text())
    assert resource["input"]["resource_gate"] == "HOLD" and resource["input"]["complete_timeline_cycles"] == 52085

    report = {
        "job": status["id"], "status": status["status"], "duration_seconds": status["duration_sec"],
        "source_commit": build["source_commit"], "run_evidence_commit": run["commit"],
        "elf_sha256": EXPECTED_ELF, "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
        "source_config_hash_count_verified": len(source), "fresh_embedded_kernel_count_verified": 4,
        "correctness": {"result": "PASS", "cases": cases, "codes": 49_152_000,
                        "packed_bytes": 24_576_000, "rejected_bindings": 6,
                        "scope": "Independent byte/nibble/row/shard layout, raw owner versus typed I4 reader, full outputs and post-reuse immutability; no quantizer or GEMM"},
        "timing_records": timings, "complete_call_timing": timing_summary,
        "timing_scope": {"included": "Completed awaited pack call including input/shift loads, bit packing/compaction and real blocked output store; second mode additionally checked raw-I4 layout-copy call",
                         "excluded": "Fixture generation, allocation/upload, CPU oracles/readback, activation quantization, GEMM, GeGLU, complete FFN/model",
                         "samples": "Two cases per shape; warm ABBA blocks; four samples per mode/shape,16 completed calls per sample",
                         "no_baseline_speedup": "No production or full-FFN speedup is inferred from these standalone costs"},
        "profile": {"NPU_SPAN_count": span_count, "profile_marker_count": profile_count,
                    "arena_records": records, "status": "NOT_COLLECTED",
                    "limit": "No actual per-operation DMA/TU attribution. Earlier operation spans remain static resource estimates only. No rerun performed."},
        "prior_resource_screen": {"input": "HOLD52,085>50,000 unchanged", "hidden": "PASS180,627<=200,000", "no_DramStack": True},
        "raw_file_sha256": {p.name: sha(p) for p in sorted(raw.iterdir()) if p.is_file()},
        "build_record_sha256": sha(HERE / "actual-build.json"),
        "registry_record_sha256": sha(HERE / "actual-build-registry.json"),
        "not_claimed": ["No complete quantize/pack/GEMM/FFN timing", "No model quality or serving gain", "No promotion or relaxation of failed H static screen"],
    }
    (HERE / "actual-result-322961.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"job": status["id"], "correctness": "PASS", "timing_us": timing_summary,
                      "profile": "NOT_COLLECTED", "H_resource": "HOLD retained"}, indent=2))


if __name__ == "__main__":
    main()
