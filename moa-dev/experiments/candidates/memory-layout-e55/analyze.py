#!/usr/bin/env python3
"""Extract whole schedules, DMA descriptors, and actual Arena package identity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from prepare import ARTIFACTS, HERE, ROOT, VARIANTS, kernel_name

ANALYZER = HERE.parent / "ffn-up-tiled-fused-decode-e47/analyze_schedule.py"
spec = importlib.util.spec_from_file_location("schedule_profile", ANALYZER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule(path):
    data, record = module.profile(path)
    record["path"] = str(path.relative_to(ROOT))
    record["sha256"] = sha(path)
    record["large_dma_operations"] = [
        {"cycles": op["lifetime"]["end"] - op["lifetime"]["begin"],
         "schedule_index": op["index"], "description": op["description"]}
        for op in data["instructions"]
        if "DmaEngine" in op["contexts"] and op["lifetime"]["end"] - op["lifetime"]["begin"] > 4000
    ]
    record["main_cycles_with_lookup"] = sum(
        op["lifetime"]["end"] - op["lifetime"]["begin"]
        for op in data["instructions"]
        if op["tpe"] == "Main" and "fetch_table_lookup" in op["description"]
    )
    record["large_sram_tensors"] = [t for t in data["tensors"]
                                     if t["buffer_type"] == "Sram" and t["size"] >= 15_000_000]
    graph_path = path.with_name("candidate-graph.json")
    graph = json.loads(graph_path.read_text())
    record["graph_sha256"] = sha(graph_path)
    record["hbm_large_transfers"] = {}
    # Graph and schedule tensor/operator IDs belong to different IR stages.
    # Identify graph transfers by their external IO address, not schedule IDs.
    for key, op in graph["operators"].items():
        if op["kind"] != "Dma.DtoS":
            continue
        for inp in op["inputs"]:
            tensor = graph["tensors"][str(inp)]
            addr = tensor["optionals"].get("address", "")
            if re.fullmatch(r"IO\(\d+\)\+0", addr) and tensor["size"] >= 3_000_000:
                record["hbm_large_transfers"][addr] = {
                    "graph_operator": key, "dtype": tensor["element_type"],
                    "logical_bytes": tensor["size"],
                    "source_descriptor_count": len(re.findall(r"^src\d+:", op["description"], re.M)),
                    "descriptor": op["description"],
                }
    return record


def main():
    records = {"e55": schedule(HERE.parent / "ffn-parallel-epilogue/candidate.json")}
    records["e55_qkv"] = schedule(ROOT / "moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/candidate.json")
    records["e55_output"] = schedule(ROOT / "moa-dev/experiments/artifacts/output-bf16-activation-e47/candidate.json")
    for variant in VARIANTS:
        evidence = HERE / variant / "build-evidence.json"
        if not evidence.exists():
            continue
        build = json.loads(evidence.read_text())
        if build["exact_compile_exit_code"] == 0:
            records[variant] = schedule(ARTIFACTS / variant / "candidate.json")
            key = "e55_qkv" if variant.startswith("qkv-") else "e55_output" if variant.startswith("output-") else "e55"
            records[variant]["base_schedule"] = key
            records[variant]["makespan_change_fraction"] = records[variant]["makespan_cycles"] / records[key]["makespan_cycles"] - 1
    (HERE / "schedule-summary.json").write_text(json.dumps(records, indent=2) + "\n")
    for label, rec in records.items():
        print(label, rec["makespan_cycles"], rec["context_interval_union_cycles"], rec["sram_physical_lifetime_union"]["peak_bytes"])

    cases = {
        "up-gate-packed": "e57-up-gate-packed-screen.json",
        "down-packed-split": "e58-down-packed-screen.json",
        "down-scale-split": "e59-down-scale-repeats.json",
        "down-both-split": "e60-down-both-screen.json",
        "up-gate-bf16-skew32": "e61-up-gate-bf16-skew-screen.json",
        "down-bf16-skew32": "e62-down-bf16-skew-screen.json",
        "qkv-sixteen-rows": "e63-qkv-sixteen-row-repeats.json",
        "output-column-major-staged": "e64-output-column-major-screen.json",
    }
    hardware = {}
    control_path = ROOT / "docs/experiments/e55-memory-layout-control-repeats.json"
    control = json.loads(control_path.read_text())
    assert control["valid"] and len(control["runs"]) == 3
    job_ids = {run["job"]["id"] for run in control["runs"]}
    for run in control["runs"]:
        assert run["package_files"]["test_runtime"]["sha256"] == "b22444012a01de7f700bb4ecb8316a87093d5c5bde7cdd6ec96c8bfb8afa682f"
    for variant, name in cases.items():
        path = ROOT / "docs/experiments" / name
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        build = json.loads((HERE / variant / "build-evidence.json").read_text())
        for run in report["runs"]:
            for file, identity in build["package_files"].items():
                assert run["package_files"][file] == identity, (variant, file)
        hardware[variant] = {
            "report": str(path.relative_to(ROOT)), "report_sha256": sha(path),
            "actual_uploaded_package_matches_build": True,
            "all_runs_valid": report["valid"],
            "jobs": [run["job"]["id"] for run in report["runs"]],
            "aggregate": report["aggregate"],
            "adopted": False,
            "target_kernel": kernel_name(variant),
            "target_cycles_change_from_control": report["aggregate"][kernel_name(variant)]["median_cycles"] / control["aggregate"][kernel_name(variant)]["median_cycles"] - 1,
        }
        job_ids.update(run["job"]["id"] for run in report["runs"])
    result = {"control_report": str(control_path.relative_to(ROOT)), "control_sha256": sha(control_path),
              "unique_jobs": sorted(job_ids), "variants": hardware,
              "decision": "Retain E55. No tested memory layout demonstrates a repeatable performance improvement.",
              "scope": "Arena development measurements on the unchanged official three-input harness; not official scores or full-model throughput."}
    (HERE / "hardware-summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
