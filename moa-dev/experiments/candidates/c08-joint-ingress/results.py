#!/usr/bin/env python3
"""Record the one authorized corrected-joint screen, with two distinct controls."""
import argparse
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def report(batch, stage):
    data = batch["stages"][stage]
    assert batch["valid"] and data["valid"] and data["comparisons_passed"] == 15
    run = {key: batch[key] for key in ("job", "log", "log_sha256", "package_files")}
    run.update(data, batch_stage=stage, executed_host=stage + "_test_runtime",
               benchmark_files={name: stage + "_" + name for name in ("test_runtime", "fixtures.safetensors", "remote_entrypoint.sh")})
    return {"schema_version": 1, "valid": True, "runs": [run], "independent_arena_jobs": 1,
            "method": "One upload-verified Arena allocation; E104 before, corrected joint QKV, E104 after; basic three inputs, not independent repetitions or an adopted speedup.",
            "aggregate": {kernel: {"median_cycles": value["median_cycles"], "job_medians": [value["median_cycles"]]}
                          for kernel, value in data["kernels"].items()}}


def outputs():
    batch = json.loads((HERE / "e124-c08-corrected-v2-results.json").read_text())
    before_path = "docs/experiments/e124-c08-e104-control-before.json"
    after_path = "docs/experiments/e124-c08-e104-control-after.json"
    before, candidate, after = (report(batch, stage) for stage in ("control_before", "candidate_0", "control_after"))
    medians = [value["aggregate"]["sliding_project_qkv"]["median_cycles"] for value in (before, after)]
    cycles = candidate["aggregate"]["sliding_project_qkv"]["median_cycles"]
    change = 100 * (cycles / statistics.median(medians) - 1)
    candidate.update(candidate_build=str((HERE / "build-corrected-v2-repro.json").relative_to(ROOT)),
                     reference=before_path, control_after_report=after_path,
                     control_after_qkv_cycles=medians[1], change_percent_vs_bracketing_median=change,
                     adopted=False, accuracy_scope="Basic original three inputs only; candidate QKV nine output comparisons passed")
    decision = {"id": "E124", "status": "rejected_measured_regression", "adopted": False,
                "source_origin": "Frozen C08 v2 joint QKV with only corrected compact K/V ingress and exact packet copies",
                "comparison": "Frozen E104, independently selected from the source recipe; O/FFN images unchanged from E104",
                "job": batch["job"]["id"], "independent_allocations": 1,
                "correctness_comparisons_passed": 45, "candidate_qkv_output_comparisons_passed": 9,
                "qkv_control_before": medians[0], "qkv_candidate": cycles, "qkv_control_after": medians[1],
                "qkv_change_percent_vs_bracketing_median": change,
                "candidate_cycles_by_run": batch["stages"]["candidate_0"]["kernels"]["sliding_project_qkv"]["cycles_by_run"],
                "compile_attempts": {"corrected-v1": "Rejected 8-byte BF16 Collect packet; frozen before repair",
                                     "corrected-v2": "Collect padded to one 32-byte flit, commit trims to four BF16 elements; full compile and NPU basic accuracy PASS"},
                "descriptor_gate": "All six HBM input sources have exact unique bounds/coverage and intended destination row ownership; four raw copies and three output gathers verified",
                "structural_costs": "One shared lookup preparation/TRF, but K/V ingress adds compact DM read/write copies and the single joint dot waits for all weights/scales. These costs and lost early-head overlap are structural observations, not a measured cycle breakdown.",
                "interpretation": "E105's invalid ingress was repaired and basic QKV accuracy recovered. This corrected prototype is slower than contemporary controls; neither the original failure nor this one screen proves all joint QKV architectures impossible.",
                "stop": "No repeats, extra-input screen, tile tuning, adoption or official submission; concrete transfer repair and one authorized full-kernel screen are complete"}
    return {ROOT / before_path: before, ROOT / after_path: after,
            ROOT / "docs/experiments/e124-c08-corrected-joint.json": candidate,
            HERE / "decision.json": decision}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, value in outputs().items():
        if args.check:
            assert json.loads(path.read_text()) == value, path
        else:
            path.write_text(json.dumps(value, indent=2) + "\n")
    print("E124: 45 comparisons PASS; QKV 103390 -> 129258 -> 104408; +24.407%")
