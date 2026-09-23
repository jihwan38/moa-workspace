#!/usr/bin/env python3
"""Trace native DPE output storage and compare whole FFN compiler schedules."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / "moa-dev/experiments/artifacts/c11-ffn-partial-vrf"
spec = importlib.util.spec_from_file_location("profile", HERE.parent / "ffn-up-tiled-fused-decode-e47/analyze_schedule.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
records = {}
for label in ["c11-e104-control", "c11-two-row-v1"]:
    data, record = profile.profile(ARTIFACTS / label / "schedule.json")
    tensors = {int(t["index"]): t for t in data["tensors"]}
    dots = [i for i in data["instructions"] if i["tpe"] == "Main" and "begin(weights.view()" in i["description"] and "contract_outer" in i["description"]]
    reductions = [i for i in data["instructions"] if i["tpe"] == "Main" and "FfnRow" in i["description"] and "vector_intra_slice_reduce" in i["description"]]
    record["dot_count"] = len(dots)
    record["dot_duration_sum"] = sum(i["lifetime"]["end"] - i["lifetime"]["begin"] for i in dots)
    record["reduction_count"] = len(reductions)
    record["reduction_duration_sum"] = sum(i["lifetime"]["end"] - i["lifetime"]["begin"] for i in reductions)
    record["dot_nodes"] = [{k: i[k] for k in ("index", "contexts", "lifetime", "output_tensors")} for i in dots]
    record["dot_outputs"] = [tensors[index] for i in dots for index in i["output_tensors"]]
    records[label] = record
assert records["c11-e104-control"]["dot_count"] == 2
assert records["c11-two-row-v1"]["dot_count"] == 30
assert all(t["buffer_type"] == "Sram" for t in records["c11-e104-control"]["dot_outputs"])
assert all(t["buffer_type"] == "Vrf" and t["size"] // 512 == 3840 for t in records["c11-two-row-v1"]["dot_outputs"])
control = json.loads((HERE / "build-c11-e104-control-repro.json").read_text())
assert control["exact_image"]["sha256"] == "6178c158c093eb40febd091e271c1cca35241c6c361ff6d3a0f6eefc9f3e1742"
before, after = (records[k]["makespan_cycles"] for k in ("c11-e104-control", "c11-two-row-v1"))
result = {"scope": "Compiler schedule only, not NPU cycles; resource sums overlap.",
          "control_image_matches_e104": True,
          "makespan_change_percent": (after / before - 1) * 100,
          "computed_partial_vrf_supported": True,
          "partial_vrf_bytes_per_slice": 3840,
          "builds": records}
(HERE / "static-screen.json").write_text(json.dumps(result, indent=2) + "\n")
print(before, after, result["makespan_change_percent"])
