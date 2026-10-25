#!/usr/bin/env python3
"""Replay strict actual gates, all timing blocks and frozen image/source lineage."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import re
import statistics

HERE = Path(__file__).resolve().parent
TREE, ROOT = HERE.parents[1], HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(line):
    return {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)(?:\s|$)", line)}


def analyze(mode, width, job, build):
    directory = HERE / "actual" / mode
    manifest = json.loads((directory / "manifest.json").read_text())
    for filename, record in manifest.items():
        path = directory / filename
        assert sha(path) == record["sha256"] and path.stat().st_size == record["bytes"]
        assert sha(ROOT / record["source"]) == record["sha256"]
    status = json.loads((directory / "arena-status.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    assert status["id"] == job and status["status"] == "SUCCEEDED" and status["exit_code"] == 0
    assert 0 < status["duration_sec"] <= 70
    assert run["result"] == "command_succeeded" and run["exit_code"] == 0
    argv = run["command"]
    assert argv[argv.index("--arg") + 1] == mode and argv.count("--arg") == 1
    assert argv[argv.index("--timeout") + 1] == "70"
    submitted = run["submitted_artifact"]
    assert sha(Path(submitted["path"])) == submitted["sha256"] == build["artifact"]["sha256"]
    assert Path(submitted["path"]).stat().st_size == submitted["size"] == build["artifact"]["size"]
    hashes = json.loads((directory / "source-hashes.json").read_text())
    frozen_hashes = json.loads((ROOT / build["run"] / "source-hashes.json").read_text())
    assert hashes == frozen_hashes and len(hashes) == 406
    for relative, expected in hashes.items():
        assert sha(TREE / relative) == expected
    lines = (directory / "arena.log").read_text().splitlines()
    times, exact, gates, profiles, final, performance = {}, {}, {}, [], [], []
    for line in lines:
        if line.startswith("RATIONAL_FUSED_ABBA "):
            m = re.fullmatch(r"RATIONAL_FUSED_ABBA width=(\d+) case=(\d+) samples=20 candidate_us=(\[[^]]+\]) current_separate_us=(\[[^]]+\]) scope=complete_quantize_pack_no_consumer_GEMM_FFN_model", line)
            assert m and int(m[1]) == width
            case = int(m[2]); assert case not in times
            c, o = json.loads(m[3]), json.loads(m[4])
            assert len(c) == len(o) == 2 and all(math.isfinite(x) and x > 0 for x in c + o)
            times[case] = {"candidate_block_means_us": c, "current_separate_block_means_us": o,
                           "samples_per_block": 20, "order": ["candidate", "control", "control", "candidate"],
                           "mean_reduction_fraction": 1 - statistics.mean(c) / statistics.mean(o),
                           "worst_candidate_vs_best_control_reduction_fraction": 1 - max(c) / min(o)}
        elif line.startswith("RATIONAL_FUSED_EXACT "):
            row = fields(line); assert row["width"] == width and row["case"] not in exact
            exact[row["case"]] = row
        elif line.startswith("RATIONAL_FUSED_SPEED_GATE "):
            m = re.fullmatch(r"RATIONAL_FUSED_SPEED_GATE width=(\d+) case=(\d+) candidate_worst_us=(\S+) control_best_us=(\S+) required_gain=0.20 pass=true", line)
            assert m and int(m[1]) == width and int(m[2]) not in gates
            gates[int(m[2])] = (float(m[3]), float(m[4]))
        elif line.startswith("PROFILE_RATIONAL_FUSED "):
            profiles.append(line)
        elif line.startswith("PASS_RATIONAL_FUSED_CORRECTNESS "):
            assert f"mode={mode} " in line and line.endswith("no_consumer_GEMM_FFN_model_claim")
            final.append(fields(line))
        elif line == "RATIONAL_FUSED_PERFORMANCE pass=true correctness_complete=true":
            performance.append(line)
        else:
            raise AssertionError(f"unrecognized log line: {line}")
    assert set(times) == set(exact) == set(gates) == set(range(4))
    assert len(final) == len(performance) == 1
    assert profiles == [f"PROFILE_RATIONAL_FUSED width={width} arm=candidate", f"PROFILE_RATIONAL_FUSED width={width} arm=current_separate"]
    n = 512 * width
    for case in range(4):
        assert exact[case] == {"width": width, "case": case, "codes": n,
                               "candidate_packed_bytes": n // 2, "control_packed_bytes": n // 2,
                               "each_scales": 512, "post_reuse_output_bytes": 5 * n + 4096}
        c, o = times[case]["candidate_block_means_us"], times[case]["current_separate_block_means_us"]
        assert gates[case] == (max(c), min(o)) and max(c) <= 0.8 * min(o)
    expected = {"candidate_codes": 4*n, "control_canonical_codes": 4*n,
                "candidate_packed_bytes": 2*n, "control_packed_bytes": 2*n, "each_scales": 2048,
                "permutation_codes": 2*n, "permutation_scales": 1024,
                "post_reuse_output_bytes": 4*(5*n+4096), "immutable_bytes": 4*(2*n+8192+32), "kernel_calls": 495}
    assert final[0] == expected
    c = [v for row in times.values() for v in row["candidate_block_means_us"]]
    o = [v for row in times.values() for v in row["current_separate_block_means_us"]]
    trace = json.loads((directory / "arena-records.json").read_text())
    assert trace == []
    return {"job": job, "status": status, "source_commit_at_submission": run["commit"],
            "source_input_count": len(hashes), "source_and_artifact_equal_build": True,
            "submitted_artifact": submitted, "raw_manifest": manifest,
            "counts": expected, "fixtures": [times[i] | {"case": i, "exact_counts": exact[i], "speed_gate_pass": True} for i in range(4)],
            "timing": {"candidate_mean_us": statistics.mean(c), "candidate_block_mean_range_us": [min(c), max(c)],
                       "current_separate_mean_us": statistics.mean(o), "current_separate_block_mean_range_us": [min(o), max(o)],
                       "mean_reduction_fraction": 1-statistics.mean(c)/statistics.mean(o),
                       "timed_quantize_pack_invocations_each_arm": 160,
                       "scope": "Whole awaited quantize+pack; candidate1 kernel versus control2 kernels. Excludes setup/CPUoracle/readback equally."},
            "actual_per_operation_trace_available": False,
            "trace_limit": "Profile markers are present but arena-records is empty; no actual per-operation attribution is established."}


def main():
    old_build_hash = sha(HERE / "device-build.json")
    spec = importlib.util.spec_from_file_location("build_audit", HERE / "audit_build.py")
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    with contextlib.redirect_stdout(io.StringIO()): audit.main()
    assert sha(HERE / "device-build.json") == old_build_hash
    build = json.loads((HERE / "device-build.json").read_text())
    results = {"input": analyze("input",3840,323029,build), "hidden": analyze("hidden",15360,323030,build)}
    totals = {k: sum(r["counts"][k] for r in results.values()) for k in results["input"]["counts"]}
    assert totals["candidate_codes"] == totals["control_canonical_codes"] == 39_321_600
    assert totals["candidate_packed_bytes"] == totals["control_packed_bytes"] == 19_660_800
    assert totals["each_scales"] == 4096
    assert totals["post_reuse_output_bytes"] == 196_640_768 and totals["immutable_bytes"] == 78_708_992
    result = {
        "result": "ACTUAL_STRICT_QUANTIZE_PLUS_PACK_AND_ALL8_SPEED_GATES_PASS",
        "build_audit_sha256": sha(HERE / "audit_build.py"), "device_build_evidence_sha256": old_build_hash,
        "compiled_source": build["source_commit"], "frozen_artifact": build["artifact"],
        "six_embedded_images_and_actual_control_resource_parity_rechecked": True,
        "aggregate_counts": totals, "runs": results,
        "standalone_sum_caveat": {
            "previous_approximate_sum_us": {"input":64.0,"hidden":157.0},
            "same_job_two_call_control_mean_us": {k:v["timing"]["current_separate_mean_us"] for k,v in results.items()},
            "interpretation": "The measured paired two-call controls are slower than sums of prior standalone timings. The cause is not established. Preserve both observations; do not substitute standalone sums for the measured control or attribute the discrepancy to an unmeasured mechanism."
        },
        "scope": "PASS ends at complete exact-rational BF16 quantization plus blocked packed-I32 bytes/scales. GEMM consumer, complete FFN, model quality, model performance and serving gains remain unverified. Counts include separate repeated fixtures/arms, not unique model data."
    }
    (HERE / "actual-result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v["timing"] for k,v in results.items()},indent=2))


if __name__ == "__main__": main()
