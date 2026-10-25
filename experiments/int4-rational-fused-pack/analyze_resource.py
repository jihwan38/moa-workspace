#!/usr/bin/env python3
"""Replay exact resource gates and allocation/provenance facts; no runtime claims."""
from pathlib import Path
import hashlib
import json
import subprocess

EXP = Path(__file__).resolve().parent
TREE = EXP.parents[1]
ROUND2 = TREE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_size(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def analyze(width, name, run_name, output_dir):
    schedule_path = EXP / f"{name}-schedule.json"
    schedule = json.loads(schedule_path.read_text())
    desc, timeline = schedule["description"], schedule["timeline"]
    run_dir = ROUND2 / "runs" / run_name
    run = json.loads((run_dir / "run.json").read_text())
    hashes = json.loads((run_dir / "source-hashes.json").read_text())
    assert run["result"] == "command_succeeded" and run["exit_code"] == 0
    assert run["fresh_package"]["contents_changed"] is False
    for relative, expected in hashes.items():
        if sha(TREE / relative) != expected:
            frozen = subprocess.check_output(["git", "-C", str(TREE), "show", f"{run['commit']}:{relative}"])
            assert hashlib.sha256(frozen).hexdigest() == expected, relative
    actual_sources = set(subprocess.check_output(["git", "-C", str(TREE), "ls-tree", "-r", "--name-only", run["commit"], "src"]).decode().splitlines())
    assert actual_sources == {p for p in hashes if p.startswith("src/")}
    records = [dict(tensor=tensor, **record)
               for tensor, records in timeline["tensors"].items() for record in records]
    classes = sorted({r["allocation"]["buffer_type"] for r in records})
    assert not any("Stack" in c for c in classes)
    assert {c for c in classes if c.startswith("Dram")} == {f"DramIO({i})" for i in range(5)}
    expected_io = [512 * width * 2, 4096, 32, 512 * width // 2, 2048]
    for slot, expected in enumerate(expected_io):
        assert {r["allocation"]["size"] for r in records
                if r["allocation"]["buffer_type"] == f"DramIO({slot})"} == {expected}
    operators = []
    for op_id, op in desc["operators"].items():
        if "dma" not in op["kind"]:
            continue
        for record in timeline["operators"].get(op_id, []):
            lifetime = record["lifetime"]
            operators.append({"id": op_id, "kind": op["kind"], **lifetime,
                              "cycles": lifetime["end"] - lifetime["begin"],
                              "source": op.get("description", "").split("\n\n")[-1]})
    operators.sort(key=lambda op: (op["begin"], op["id"]))
    total = max(r["lifetime"]["end"] for values in timeline["operators"].values() for r in values)
    all_sram = [r for r in records if r["allocation"]["buffer_type"] == "Sram"]
    sram = [r for r in all_sram if r["allocation"]["address"] is not None]
    times = sorted({t for r in sram for t in r["lifetime"].values()})
    peak = {"bytes": 0}
    for begin, end in zip(times, times[1:]):
        live = [r for r in sram if r["lifetime"]["begin"] <= begin < r["lifetime"]["end"]]
        amount = union_size([(r["allocation"]["address"], r["allocation"]["address"] + r["allocation"]["size"]) for r in live])
        if amount > peak["bytes"]:
            peak = {"bytes": amount, "begin": begin, "end": end,
                    "tensors": sorted(r["tensor"] for r in live)}
    main_tensors = []
    for r in sram:
        if r["allocation"]["size"] >= 512 * width // 2:
            main_tensors.append({**r, "properties": desc["tensors"][r["tensor"]]["properties"]})
    main_tensors.sort(key=lambda r: (r["lifetime"]["begin"], r["tensor"]))
    images = sorted((EXP / output_dir).rglob("*.bin"))
    assert len(images) == 1
    plan = json.loads((EXP / "plan.json").read_text())
    if name == "aligned-input":
        extension = json.loads((EXP / "aligned-input-plan.json").read_text())
        baseline, limit = extension["gate"]["baseline_cycles"], extension["gate"]["complete_resource_max_cycles"]
    else:
        baseline_key = "input" if name == "input" else "hidden_staged_pack"
        baseline = plan["resource_gate"]["baseline_separate_cycles"][baseline_key]
        limit = plan["resource_gate"]["maximum_cycles"][name]
    return {
        "result": "RESOURCE_PASS_ONLY" if total <= limit else "RESOURCE_HOLD",
        "rows": 512, "width": width, "estimated_total_cycles": total,
        "maximum_cycles": limit, "current_separate_baseline_cycles": baseline,
        "estimated_reduction_fraction": 1 - total / baseline,
        "no_dram_stack": True, "canonical_i32_hbm_bridge_present": False,
        "explicit_dram_io_sizes": expected_io, "memory_classes": classes,
        "main_tensor_allocations": main_tensors,
        "peak_live_sram_tensor_address_union": peak,
        "sram_records_without_address": [r for r in all_sram if r["allocation"]["address"] is None],
        "memory_scope": "Union of scheduler-reported live SRAM tensor address ranges with concrete addresses, aliases deduplicated; unassigned records separately reported; not total runtime allocator or model peak memory",
        "dma_operations": operators,
        "source_commit": run["commit"], "source_input_count": len(hashes),
        "source_inputs_verified": True,
        "source_verification_scope": "Captured inputs match current bytes or exact captured Git commit; ignored original lock must match current frozen bytes. Source file set matches captured commit.",
        "run": str(run_dir.relative_to(ROUND2)),
        "raw_hashes": {p: sha(run_dir / p) for p in ["run.json", "source-hashes.json", "output.log"]},
        "schedule_sha256": sha(schedule_path),
        "compiled_device_image": {"path": str(images[0].relative_to(TREE)), "size": images[0].stat().st_size, "sha256": sha(images[0])},
    }


if __name__ == "__main__":
    check_dir = ROUND2 / "runs/a001-rational-fused-pack-check"
    check = json.loads((check_dir / "run.json").read_text())
    assert check["result"] == "command_succeeded" and check["exit_code"] == 0
    result = {
        "scope": "Source/typecheck/resource only. No executable or Arena run; no device equivalence, actual speed, FFN, or model claim.",
        "host_check": {"run": str(check_dir.relative_to(ROUND2)), "source_commit": check["commit"], "run_sha256": sha(check_dir / "run.json")},
        "input": analyze(3840, "input", "a002-rational-fused-pack-input-resource", "kernel-out-a002-input"),
        "hidden": analyze(15360, "hidden", "a003-rational-fused-pack-hidden-resource", "kernel-out-a003-hidden"),
        "aligned_input": analyze(3840, "aligned-input", "a004-rational-fused-pack-aligned-resource", "kernel-out-a004-aligned"),
        "source_review": {"reviewer":"root", "commit":"c38b773e", "result":"PASS", "scope":"Quantization prefix/scale-store and pack-body extraction; runtime equivalence remains separate"},
        "remaining_gates": ["Separately authorized executable build", "Strict packed-code/scalebit equivalence and post-reuse/immutability", "Same-job complete quantize+current-pack actual comparison", "Any consumer/GEMM/FFN integration is separate"],
    }
    (EXP / "resource-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {key: result[name][key] for key in ["estimated_total_cycles", "estimated_reduction_fraction", "result"]} for name in ["input", "hidden", "aligned_input"]}, indent=2))
