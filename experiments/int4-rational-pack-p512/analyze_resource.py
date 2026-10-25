#!/usr/bin/env python3
"""Read frozen SDK schedules; no tools, builds, devices, or network calls."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_bytes(ranges):
    end = -1
    total = 0
    for begin, stop in sorted(set(ranges)):
        total += max(0, stop - max(begin, end))
        end = max(end, stop)
    return total


def analyze(label, width, cap, run_name):
    path = HERE / f"{label}-schedule.json"
    schedule = json.loads(path.read_text())
    desc, timeline = schedule["description"], schedule["timeline"]
    end = max(x["lifetime"]["end"] for xs in timeline["operators"].values() for x in xs)
    allocations = []
    for tensor, entries in timeline["tensors"].items():
        for entry in entries:
            allocations.append({"tensor": tensor, **entry})
    stack = [x for x in allocations if "DramStack" in x["allocation"]["buffer_type"]]
    sram = [x for x in allocations if x["allocation"]["buffer_type"] == "Sram"]
    # A consumed same-wire reshape aliases its predecessor's allocation. Count
    # the union of live address intervals, not the sum of tensor aliases.
    times = sorted({x["lifetime"][k] for x in sram for k in ("begin", "end")})
    peak = max(union_bytes([
        (x["allocation"]["address"], x["allocation"]["address"] + x["allocation"]["size"])
        for x in sram if x["lifetime"]["begin"] <= t < x["lifetime"]["end"]
    ]) for t in times)
    ops = []
    for key, op in desc["operators"].items():
        if op["kind"] not in ("dma.dto_s", "dma.sto_d", "rngd_cmd.tu_exec", "rngd_cmd.sto_vrf"):
            continue
        entries = timeline["operators"].get(key, [])
        ops.append({"id": key, "kind": op["kind"], "source": op["description"],
                    "cycles": sum(x["lifetime"]["end"] - x["lifetime"]["begin"] for x in entries),
                    "entries": entries})
    run = json.loads((HERE / "runs" / run_name / "run.json").read_text())
    assert run["exit_code"] == 0 and run["commit"] == "9f9a276293eef64d8de4fb26f8b070c8e5d65a9f", run
    # This assertion is intentionally updated only to the actual frozen commit,
    # never inferred from the currently checked-out branch.
    input_bytes, packed_bytes = 512 * width * 4, 512 * width // 2
    assert any(x["allocation"]["size"] == input_bytes for x in sram)
    assert any(x["allocation"]["size"] == packed_bytes for x in sram)
    return {
        "schedule": path.name, "schedule_sha256": sha(path), "run": run_name,
        "source_commit": run["commit"], "compile_exit": run["exit_code"],
        "complete_timeline_cycles": end, "predeclared_cycle_cap": cap,
        "resource_gate": "PASS" if end <= cap and not stack else "HOLD",
        "ordinary_i32_input_bytes": input_bytes, "blocked_packed_output_bytes": packed_bytes,
        "actual_sram_peak_live_payload_union_bytes": peak,
        "sram_allocations_including_reshape_aliases": sram,
        "dram_stack_allocations": stack, "operations": ops,
    }


if __name__ == "__main__":
    report = {
        "scope": "Static full-P512 resource only, no ELF, actual device execution, quantizer, GEMM or model claim",
        "input": analyze("input", 3840, 50000, "a002-rational-pack-p512-h-resource"),
        "hidden": analyze("hidden", 15360, 200000, "a003-rational-pack-p512-l-resource"),
        "decision": "HOLD overall: input exceeds frozen cap; hidden passes full shape without spill or tiling",
        "failed_first_variant": {
            "source": "7a43bf6b664a8d646ced7d1078660b208f683ee6",
            "run": "a001-rational-pack-p512-h-resource",
            "failure": "K-named reduction drops retained K/8 time, so final transpose mismatches; no resource result",
        },
    }
    (HERE / "resource-result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
