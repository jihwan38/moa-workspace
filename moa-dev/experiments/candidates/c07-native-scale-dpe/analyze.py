#!/usr/bin/env python3
"""Attribute compiler resource durations to source functions, without summing overlaps into latency."""
import argparse
import json
import re
from collections import defaultdict
from build import HERE, common


def analyze(check=False):
    summary = {"scope": "Compiler schedules, not hardware cycles. Resource-duration sums can overlap and do not establish critical paths or additive speedups.", "builds": {}}
    records = [(p, common.ARTIFACTS / p.stem.removeprefix("build-").removesuffix("-repro"))
               for p in sorted(HERE.glob("build-*-repro.json"))]
    records.append((HERE.parent / "c04-down-transfer/build-c04-eight-ring32-v1-repro.json",
                    common.ROOT / "moa-dev/experiments/artifacts/c04-down-transfer/c04-eight-ring32-v1"))
    for path, out in records:
        record = json.loads(path.read_text())
        if "static_cycles" not in record:
            continue
        source = (out / "crate/src/device/shared/mlp.rs").read_bytes()
        functions = [(i, m.group(1)) for i, line in enumerate(source.decode().splitlines(), 1)
                     if (m := re.match(r"(?:pub(?:\(crate\))? )?fn (\w+)\(", line))]
        schedule_bytes = (out / "schedule.json").read_bytes()
        schedule = json.loads(schedule_bytes)
        duration = defaultdict(lambda: defaultdict(int))
        for instruction in schedule["instructions"]:
            match = re.search(r"src/device/shared/mlp\.rs:(\d+):", instruction["description"])
            function = "outside_mlp_or_unattributed"
            if match:
                line = int(match[1])
                eligible = [name for start, name in functions if start <= line]
                if eligible:
                    function = eligible[-1]
            duration[function][instruction["tpe"]] += instruction["lifetime"]["end"] - instruction["lifetime"]["begin"]
        makespan = max(i["lifetime"]["end"] for i in schedule["instructions"])
        assert makespan == record["static_cycles"]
        summary["builds"][record["label"]] = {
            "variant": record["variant"], "source": common.digest(source),
            "schedule": common.digest(schedule_bytes), "makespan_cycles": makespan,
            "resource_duration_sums_by_source_function": {k: dict(v) for k, v in sorted(duration.items())}}
    path = HERE / "static-summary.json"
    if check:
        assert json.loads(path.read_text()) == summary
    else:
        path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Attributed {len(summary['builds'])} compiler schedules")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    analyze(p.parse_args().check)
