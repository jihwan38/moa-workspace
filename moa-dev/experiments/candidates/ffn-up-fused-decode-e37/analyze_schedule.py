#!/usr/bin/env python3
"""Summarize preserved schedules only. No compilation, device use, or source edits."""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PATHS = {
    "e37_reference": ROOT / "moa-dev/experiments/candidates/ffn-parallel-epilogue/candidate.json",
    "e48": ROOT / "moa-dev/experiments/artifacts/ffn-up-fused-decode-e37/schedule.json",
}

def summarize(path):
    j = json.loads(path.read_text())
    tensors = {int(t["index"]): t for t in j["tensors"]}
    sums, counts = collections.Counter(), collections.Counter()
    decodes, dmas, down_dots, up_gate_dots = [], [], [], []
    for op in j["instructions"]:
        duration = op["lifetime"]["end"] - op["lifetime"]["begin"]
        sums[op["tpe"]] += duration
        counts[op["tpe"]] += 1
        short = {k: op[k] for k in ["index", "tpe", "contexts", "lifetime", "input_tensors", "output_tensors"]}
        short["duration"] = duration
        description = op["description"]
        if "fetch_table_lookup" in description:
            short["description"] = description
            short["schedule_tensors"] = [tensors[t] for t in op["input_tensors"] + op["output_tensors"] if t in tensors]
            decodes.append(short)
        if op["tpe"] == "DmaLoad":
            short["description"] = description
            dmas.append(short)
        if op["tpe"] == "Main" and "down_weight_bf16.view().tile" in description:
            down_dots.append(short)
        if op["tpe"] == "Main" and ".begin(weights.view().tile" in description:
            up_gate_dots.append(short)
    makespan = max(op["lifetime"]["end"] for op in j["instructions"])
    last_down_dot = max(op["lifetime"]["end"] for op in down_dots)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "makespan": makespan,
        "instruction_duration_sum_by_type": dict(sums),
        "instruction_count_by_type": dict(counts),
        "decode_main_duration_sum": sum(op["duration"] for op in decodes),
        "decode_ops": decodes,
        "dma_load_ops": dmas,
        "up_gate_dot_ops_in_scheduled_order": up_gate_dots,
        "down_dot_ops": down_dots,
        "last_down_dot_end": last_down_dot,
        "tail_cycles": makespan - last_down_dot,
        "explicit_sync_ops": [op for op in j["instructions"] if str(op["index"]).startswith("ExplicitSync")],
    }

if __name__ == "__main__":
    records = {name: summarize(path) for name, path in PATHS.items()}
    base, candidate = records.values()
    records["difference"] = {
        "makespan": candidate["makespan"] - base["makespan"],
        "makespan_fraction": candidate["makespan"] / base["makespan"] - 1,
        "instruction_duration_sum_by_type": {
            name: candidate["instruction_duration_sum_by_type"].get(name, 0) - value
            for name, value in base["instruction_duration_sum_by_type"].items()
        },
    }
    records["limitations"] = [
        "Duration sums overlap; Core sums include explicit waiting, not only CPU work.",
        "Schedule tensor lifetime is SSA lifetime; tile views can retain backing allocation after the parent tensor's displayed end.",
        "No E48 lowered graph was saved. Table internals and register-level cast packing are not directly exposed by this schedule.",
    ]
    (HERE / "schedule-summary.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({k: {n: v[n] for n in ["makespan", "decode_main_duration_sum", "tail_cycles"]} for k, v in records.items() if k in PATHS}, indent=2))
