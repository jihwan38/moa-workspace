#!/usr/bin/env python3
"""Compare exact O schedules and prove overlap without calling it NPU speedup."""
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / "moa-dev/experiments/artifacts/c09-output-pipeline"
spec = importlib.util.spec_from_file_location("profile", HERE.parent / "ffn-up-tiled-fused-decode-e47/analyze_schedule.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)

records = {}
for label in ["c09-e104-control", "c09-two-stage-v1"]:
    data, record = profile.profile(ARTIFACTS / label / "schedule.json")
    record["projection_nodes"] = [
        {k: i[k] for k in ("index", "tpe", "lifetime", "description", "contexts")}
        for i in data["instructions"]
        if (i["tpe"] == "Main" and "fetch_table_lookup" in i["description"])
        or (i["tpe"] == "DmaLoad" and re.search(r"\bweight\.(?:tile|to_dm)", i["description"]))
    ]
    record["dma_load_count"] = sum(i["tpe"] == "DmaLoad" for i in data["instructions"])
    for kind, field in [("DmaLoad", "weight_load_duration_sum"), ("Main", "lookup_dot_duration_sum")]:
        record[field] = sum(i["lifetime"]["end"] - i["lifetime"]["begin"] for i in record["projection_nodes"] if i["tpe"] == kind)
    records[label] = record
control = json.loads((HERE / "build-c09-e104-control-repro.json").read_text())
assert control["exact_image"]["sha256"] == "359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b"
nodes = records["c09-two-stage-v1"]["projection_nodes"]
dots = sorted((i for i in nodes if i["tpe"] == "Main"), key=lambda i: i["lifetime"]["begin"])
loads = sorted((i for i in nodes if i["tpe"] == "DmaLoad"), key=lambda i: i["lifetime"]["begin"])
assert len(dots) == len(loads) == 2
a, b = dots[0]["lifetime"], loads[1]["lifetime"]
overlap = max(0, min(a["end"], b["end"]) - max(a["begin"], b["begin"]))
result = {
    "scope": "Compiler schedule only, not NPU performance. Resource durations overlap and are not additive latency.",
    "control_image_matches_e104": True,
    "makespan_change_percent": (records["c09-two-stage-v1"]["makespan_cycles"] / records["c09-e104-control"]["makespan_cycles"] - 1) * 100,
    "candidate_first_dot_second_weight_dma_overlap_cycles": overlap,
    "builds": records,
}
(HERE / "static-screen.json").write_text(json.dumps(result, indent=2) + "\n")
for label, record in records.items():
    print(label, {k: record[k] for k in ("makespan_cycles", "instruction_count", "dma_load_count", "weight_load_duration_sum", "lookup_dot_duration_sum")})
