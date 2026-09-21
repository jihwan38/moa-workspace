#!/usr/bin/env python3
"""Compare complete static schedules, including DMA/compute overlap."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / "moa-dev/experiments/artifacts"
ANALYZER = HERE.parent / "ffn-up-tiled-fused-decode-e47/analyze_schedule.py"
spec = importlib.util.spec_from_file_location("ffn_schedule_profile", ANALYZER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    paths = {
        "e55": HERE.parent / "ffn-parallel-epilogue/candidate.json",
        "up8": ARTIFACTS / "ffn-up-tiled-fused-decode-e47/candidate.json",
        "up16": ARTIFACTS / "ffn-up-sixteen-row-e55/candidate.json",
    }
    records = {}
    for label, path in paths.items():
        data, record = module.profile(path)
        record["path"] = str(path.relative_to(ROOT))
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        record["main_operations"] = [
            {k: op[k] for k in ("index", "lifetime", "description", "input_tensors", "output_tensors")}
            for op in data["instructions"] if op["tpe"] == "Main"
        ]
        records[label] = record
    base, candidate = records["e55"], records["up16"]
    records["comparison"] = {
        "makespan_delta": candidate["makespan_cycles"] - base["makespan_cycles"],
        "makespan_change_fraction": candidate["makespan_cycles"] / base["makespan_cycles"] - 1,
        "analyzer_sha256": hashlib.sha256(ANALYZER.read_bytes()).hexdigest(),
        "scope": "Compiler schedule only. Overlapping engine times cannot be added; NPU speed and correctness are unmeasured.",
    }
    (HERE / "schedule-summary.json").write_text(json.dumps(records, indent=2) + "\n")
    for label in paths:
        record = records[label]
        print(label, record["makespan_cycles"], record["context_interval_union_cycles"], "main_dma_overlap", record["main_dma_overlap_cycles"])
    print(json.dumps(records["comparison"], indent=2))


if __name__ == "__main__":
    main()
