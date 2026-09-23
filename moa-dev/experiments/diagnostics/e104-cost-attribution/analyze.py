#!/usr/bin/env python3
"""Audit a frozen E104 schedule; resource intervals are not hardware stall times."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
C07 = ROOT / "moa-dev/experiments/candidates/c07-native-scale-dpe"
LABEL = "c07-e104-reviewed-v1"


def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def union_duration(intervals):
    total = 0
    previous_end = 0
    for begin, end in sorted(intervals):
        assert 0 <= begin <= end
        total += max(0, end - max(begin, previous_end))
        previous_end = max(previous_end, end)
    return total


def analyze():
    baseline = json.loads((ROOT / "baseline/e104.json").read_text())
    build_path = C07 / f"build-{LABEL}-repro.json"
    build = json.loads(build_path.read_text())
    prior = json.loads((C07 / "static-summary.json").read_text())["builds"][LABEL]
    schedule_bytes = (HERE / "e104-ffn-schedule.json").read_bytes()
    assert digest(schedule_bytes) == prior["schedule"]
    source = subprocess.check_output([
        "git", "show", baseline["git"]["commit"]
        + ":furiosa-opt-gemma4-12B/src/device/shared/mlp.rs",
    ], cwd=ROOT)
    assert digest(source) == build["source_inputs"]["src/device/shared/mlp.rs"]
    functions = [
        (line, match.group(1))
        for line, text in enumerate(source.decode().splitlines(), 1)
        if (match := re.match(r"(?:pub(?:\(crate\))? )?fn (\w+)\(", text))
    ]
    schedule = json.loads(schedule_bytes)
    makespan = max(i["lifetime"]["end"] for i in schedule["instructions"])
    assert makespan == build["static_cycles"] == prior["makespan_cycles"]
    intervals = {}
    long_nodes = []
    for instruction in schedule["instructions"]:
        begin, end = (instruction["lifetime"][key] for key in ("begin", "end"))
        if begin == end:
            continue
        for context in instruction["contexts"]:
            intervals.setdefault(context, []).append((begin, end))
        if end - begin < 2000 or instruction["tpe"] == "Core":
            continue
        description = instruction["description"]
        match = re.search(r"--> (src/[^:\n]+):(\d+):(\d+)", description)
        function = "outside_mlp_or_unattributed"
        if match and match.group(1) == "src/device/shared/mlp.rs":
            names = [name for line, name in functions if line <= int(match.group(2))]
            if names:
                function = names[-1]
        long_nodes.append({
            "index": instruction["index"],
            "engine": instruction["tpe"],
            "contexts": instruction["contexts"],
            "begin": begin, "end": end, "duration": end - begin,
            "source_location": match.group(0).removeprefix("--> ") if match else None,
            "source_function": function,
            "input_tensors": instruction["input_tensors"],
            "output_tensors": instruction["output_tensors"],
            "operation": description.split("\n\n", 1)[-1],
        })
    contexts = {name: union_duration(value) for name, value in sorted(intervals.items())}
    assert all(0 <= value <= makespan for value in contexts.values())
    main = intervals["MainContext"]
    dma = intervals["DmaEngine"]
    sub = intervals["SubContext"]
    return {
        "scope": "Existing compiler schedule, not a new hardware measurement or an optimization candidate.",
        "source_commit": baseline["git"]["commit"],
        "mlp_source": digest(source),
        "source_build_record": str(build_path.relative_to(ROOT)),
        "schedule": digest(schedule_bytes),
        "makespan_cycles": makespan,
        "instruction_count": len(schedule["instructions"]),
        "context_interval_unions": contexts,
        "main_dma_overlap": union_duration(main) + union_duration(dma) - union_duration(main + dma),
        "no_main_dma_sub_interval": makespan - union_duration(main + dma + sub),
        "long_nodes": sorted(long_nodes, key=lambda item: (item["begin"], item["index"])),
        "hardware_control_reference": "docs/experiments/c13-v7-screen-e104-before.json",
        "limitations": [
            "Context intervals overlap; their durations are not additive latency contributions.",
            "Long nodes are a source-linked shortlist, not a demonstrated critical path.",
            "Tensor views, aliases and resource dependencies require tracing before assigning wait causes.",
            "No hardware traffic, stall fractions or isolated component timings were measured here.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze()
    output = HERE / "static-audit.json"
    if args.check:
        assert json.loads(output.read_text()) == result
    else:
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified E104: {result['instruction_count']} instructions, "
          f"{result['makespan_cycles']} static cycles, {len(result['long_nodes'])} long nodes")
