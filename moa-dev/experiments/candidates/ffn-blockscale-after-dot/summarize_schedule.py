#!/usr/bin/env python3
"""Compare exact-kernel resource schedules; durations are static estimates, not NPU cycles."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def union(intervals):
    total, end = 0, -1
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total


def summarize(directory, label):
    path = directory / f"{label}.json"
    data = json.loads(path.read_text())
    summary = json.loads((directory / f"{label}-summary/summary.json").read_text())
    graph_path = directory / f"{label}-graph.json"
    graph = json.loads(graph_path.read_text()) if graph_path.exists() else None
    contexts, groups = defaultdict(list), defaultdict(lambda: {"count": 0, "cycles": 0})
    generated, large_dma = [], []
    for instruction in data["instructions"]:
        begin, end = instruction["lifetime"]["begin"], instruction["lifetime"]["end"]
        source = instruction["description"].split("\n")[0] or "(compiler-generated)"
        for context in instruction["contexts"]:
            contexts[context].append((begin, end))
            group = groups[source + " " + context]
            group["count"] += 1
            group["cycles"] += end - begin
        item = {"id": instruction["index"], "contexts": instruction["contexts"],
                "begin": begin, "end": end, "duration": end - begin, "source": source}
        if source == "(compiler-generated)" and any(c in instruction["contexts"] for c in ("DmaEngine", "SubContext")):
            generated.append(item)
        if "DmaEngine" in instruction["contexts"] and end - begin > 5000:
            large_dma.append({**item, "operation": instruction["description"].split("\n\n")[-1]})
    table_tensors = []
    if graph:
        table_tensors = [{"id": key, "bytes": value["size"], "address": value["optionals"].get("address")}
                         for key, value in graph["tensors"].items()
                         if value["buffer_type"] == "Dram" and "table_lookup" in str(value["shape"])]
    return {"schedule_path": str(path), "schedule_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "makespan": max(item["lifetime"]["end"] for item in data["instructions"]),
            "instruction_count": len(data["instructions"]),
            "context_union_cycles": {name: union(times) for name, times in contexts.items()},
            "compiler_resource_stats": summary["rlir_stats"],
            "source_context_sums": dict(sorted(groups.items(), key=lambda item: item[1]["cycles"], reverse=True)),
            "large_dma_timeline": sorted(large_dma, key=lambda item: item["begin"]),
            "compiler_generated_dma_sub": generated,
            "lookup_dram_constants": table_tensors,
            "peak_memory_usage_bytes": summary["peak_memory_usage_bytes"],
            "caveats": ["PeCore interval includes DramReuse lifetime markers; it is not CPU ALU busy time.",
                        "Makespan is the final scheduled completion path including resource constraints, not a sum of engine work.",
                        "Source groups cannot isolate Mul ALU cycles from the combined pipeline; no 16x hardware speedup is inferred."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {label: summarize(args.directory, label) for label in args.labels}
    with args.output.open("w") as output:
        json.dump(results, output, indent=2)
        output.write("\n")
    print(json.dumps({name: {key: result[key] for key in ("makespan", "context_union_cycles", "peak_memory_usage_bytes")}
                      for name, result in results.items()}, indent=2))
