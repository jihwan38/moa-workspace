#!/usr/bin/env python3
"""Trace full M01 schedules against the image-verified E104 control."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def union(intervals):
    end = total = 0
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def analyze(candidate, label):
    record_path = ROOT / "moa-dev/experiments/candidates" / candidate / f"build-{label}-repro.json"
    record = json.loads(record_path.read_text())
    schedule_path = HERE / "schedules" / f"{label}.json"
    raw = schedule_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record["schedule"]["sha256"]
    schedule = json.loads(raw)
    contexts = {}
    nodes = []
    for item in schedule["instructions"]:
        a, b = item["lifetime"]["begin"], item["lifetime"]["end"]
        for context in item["contexts"]:
            contexts.setdefault(context, []).append((a, b))
        if item["tpe"] == "Core":
            continue
        operation = item["description"]
        kind = None
        if "projection.rs:" in operation:
            if "\n\nweight.to_dm(" in operation:
                kind = "weight_ingress"
            elif "\n\nx.to_dm(" in operation:
                kind = "activation_ingress"
            elif "contract_outer" in operation:
                kind = "dot_and_fused_operations"
            elif "\n\npartial.to_dm(" in operation:
                kind = "partial_dma_remap"
            elif "SwitchConfig::Transpose" in operation:
                kind = "partial_switch_and_reduce"
            elif "vector_inter_slice_reduce" in operation:
                kind = "partial_reduce"
            elif "\n\nresult.to_dm(" in operation:
                kind = "epilogue_gather"
        if kind:
            nodes.append({"kind": kind, "index": item["index"], "engine": item["tpe"],
                          "begin": a, "end": b, "duration": b-a,
                          "input_tensors": item["input_tensors"],
                          "output_tensors": item["output_tensors"], "operation": operation})
    makespan = max(i["lifetime"]["end"] for i in schedule["instructions"])
    assert makespan == record["static_cycles"]
    return {"makespan": makespan, "schedule": record["schedule"],
            "source_build_record": str(record_path.relative_to(ROOT)),
            "image": record["exact_image"], "nodes": nodes,
            "context_interval_unions": {k: union(v) for k,v in contexts.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = {
        "E104": analyze("c09-output-pipeline", "c09-e104-control"),
        "dma-v2": analyze("m01-partial-remap", "m01-dma-v2"),
        "switch-v2": analyze("m01-partial-remap", "m01-switch-v2"),
    }
    for key in ("dma-v2", "switch-v2"):
        data[key]["change_percent_vs_e104"] = 100 * (data[key]["makespan"] / data["E104"]["makespan"] - 1)
    output = {"scope": "Compiler schedule and explicit producer/consumer intervals; not NPU stall times or measured bus traffic.",
              "builds": data,
              "finding": "Weight ingress and dot do not shorten in these schedules. The new remap/reduction waits for all dot partials, and the epilogue waits for that consumer. Compact partials avoid E64's full-weight repack but add work absent from E104.",
              "limits": ["Context intervals overlap and are not additive runtime contributions.",
                         "Dot timing includes fused operations; no isolated decode or arithmetic latency is measured.",
                         "Whole-kernel NPU measurements cannot identify hardware stall fractions."]}
    path = HERE / "static-cost.json"
    if args.check:
        assert json.loads(path.read_text()) == output
    else:
        path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: {"makespan": value["makespan"],
                           "nodes": [{k: n[k] for k in ("kind", "begin", "end", "duration")} for n in value["nodes"]]}
                      for key,value in data.items()}, indent=2))


if __name__ == "__main__":
    main()
