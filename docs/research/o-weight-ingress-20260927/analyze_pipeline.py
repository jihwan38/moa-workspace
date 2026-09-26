#!/usr/bin/env python3
"""Conditional two-stage arithmetic from preserved traces; no runtime prediction."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROFILE = ROOT / "moa-dev/experiments/candidates/e104-qkv-output-profile/profile-analysis.json"
STATIC = ROOT / "moa-dev/experiments/candidates/c09-output-pipeline/static-screen.json"


def digest(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def calculate():
    profile = json.loads(PROFILE.read_text())["kernels"]["sliding_attention_output"]
    static = json.loads(STATIC.read_text())
    rows = []
    for sample in profile["run_cluster_observations"]:
        spans = sample["selected_relative_intervals"]
        weight = spans["weight"]
        dot = spans["dot_and_16slice_reduce"]
        w = weight["end"] - weight["begin"]
        d = dot["end"] - dot["begin"]
        ready = max(spans["input_trf"]["end"], spans["table_prepare"]["end"])
        serial_end = max(weight["end"], ready) + d
        first_end = max(weight["begin"] + w / 2, ready) + d / 2
        second_end = max(weight["end"], first_end) + d / 2
        rows.append({"job": sample["job"], "run": sample["run"],
                     "cluster": sample["cluster"], "weight_interval": w,
                     "dot_reduce_interval": d,
                     "weight_end_to_dot_begin": dot["begin"] - weight["end"],
                     "conditional_two_stage_saving": serial_end - second_end})
    median = lambda key: statistics.median(row[key] for row in rows)
    control = static["builds"]["c09-e104-control"]
    candidate = static["builds"]["c09-two-stage-v1"]
    return {
        "scope": "Read-only arithmetic on existing evidence. No new compile, Arena run, or speedup claim.",
        "inputs": [digest(PROFILE), digest(STATIC)],
        "two_stage_assumptions": [
            "Exactly two equal row stages; each costs half the observed whole weight interval and dot/reduction interval.",
            "Same total ingress throughput despite concurrent Main consumption; no extra startup, copies, table preparation, or register/resource waits.",
            "Input and a reusable lookup table are ready at their original observed times. Public SDK table reuse is NOT established.",
            "The final gather/scale/RMS/residual and wrapper are excluded, not presumed faster.",
            "This is a conditional sensitivity model, not a hardware lower bound or a prediction for C09."
        ],
        "observations": rows,
        "medians": {key: median(key) for key in (
            "weight_interval", "dot_reduce_interval", "weight_end_to_dot_begin",
            "conditional_two_stage_saving")},
        "c09_static": {
            "control_makespan": control["makespan_cycles"],
            "candidate_makespan": candidate["makespan_cycles"],
            "overlap": static["candidate_first_dot_second_weight_dma_overlap_cycles"],
            "control_weight_interval": control["weight_load_duration_sum"],
            "control_dot_reduce_interval": control["lookup_dot_duration_sum"],
            "scope": "Compiler estimates; C09 has no NPU result. Ratios against trace durations are not a calibration model."
        },
        "logical_accounting": {
            "weight_hbm_bytes": 3840 * 4096,
            "weight_dm_bytes_per_slice": 120 * 256,
            "unique_input_hbm_bytes": 4096 * 2,
            "input_dm_replica_bytes": 512 * 256 * 2,
            "complete_f32_output_bytes": 3840 * 4,
            "minimum_pair_for_8byte_f32_commit": 2,
            "candidate_extra_weight_materialization_bytes": 0,
            "scope": "Payload/representation accounting, not measured bus traffic. Candidate zero extra weight materialization is a design requirement, not an emitted-graph result."
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = HERE / "pipeline-model.json"
    result = calculate()
    if args.check:
        if json.loads(output.read_text()) != result:
            raise SystemExit("pipeline-model.json differs from preserved evidence")
        print("PASS: existing trace arithmetic and source digests")
    else:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
