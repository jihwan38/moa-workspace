#!/usr/bin/env python3
"""Collect immutable all-owner QKV model gates; no device execution."""
import hashlib
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
queue = root / "state/r1099-r1109-micro-queue-status.json"
status = json.loads(queue.read_text())
assert not status["failed"] and not status["active"] and not status["pending"]
assert len(status["finished"]) == 11
result = {"source": "fc3f009", "functional_source": "3deaf9e",
          "probe_sha256": "71cf54b68bc6ef12c305f8f7b003a7356ee3f3f6919a4f74650fb7ee6fc6b182",
          "queue": str(queue), "runs": [],
          "scope": "Exact same grouped owner routes, current O and projection-owned FFN; no HTTP claim"}
total = {"logit_values": 0, "owner_cache_bf16_values": 0, "sentinel_bf16_values": 0}
warm_savings = []
for row in status["finished"]:
    assert row["status"] == "SUCCEEDED" and row["pass_marker"]
    raw = (root / "runs" / row["run"] / "arena.log").read_bytes()
    text = raw.decode()
    assert "PASS original vs scoped weight-stationary sliding QKV:" in text
    assert not re.search(r"bitdiff=[1-9]|nonfinite=[1-9]", text)
    logits = sum(map(int, re.findall(r"^QKV_STATIONARY_MODEL_EXACT .*?values=(\d+) ", text, re.M)))
    cache = sum(map(int, re.findall(r"^QKV_STATIONARY_MODEL_CACHE .*?bf16_values=(\d+) ", text, re.M)))
    sentinel = sum(map(int, re.findall(r"^QKV_STATIONARY_MODEL_SENTINEL .*?bf16_values=(\d+) ", text, re.M)))
    timings = []
    cases = set()
    for case, round_, control, candidate in re.findall(
            r"^QKV_STATIONARY_MODEL_PASS case=(\d+) round=(\d+) control_ms=([\d.]+) candidate_ms=([\d.]+)", text, re.M):
        cases.add(int(case)); control, candidate = float(control), float(candidate)
        timings.append({"round": int(round_), "control_ms": control, "candidate_ms": candidate,
                        "savings_ms": control-candidate, "savings_fraction": 1-candidate/control})
        if round_ == "1": warm_savings.append(control-candidate)
    assert len(cases) == 1 and {t["round"] for t in timings} == {0, 1}
    counts = dict(zip(total, (logits, cache, sentinel)))
    for key in total: total[key] += counts[key]
    stages = []
    for label, stage, calls, ms in re.findall(
            r"^QKV_STATIONARY_MODEL_STAGE label=(\S+) stage=(\S+) calls=(\d+) ms=([\d.]+)", text, re.M):
        stages.append({"label": label, "stage": stage, "calls": int(calls), "ms": float(ms)})
    result["runs"].append({"run": row["run"], "job": row["job"], "case": cases.pop(),
                           "duration_sec": row["duration_sec"],
                           "arena_log_sha256": hashlib.sha256(raw).hexdigest(),
                           "exact_counts": counts, "timings": timings, "stages": stages})
result["runs"].sort(key=lambda r: r["case"])
assert [r["case"] for r in result["runs"]] == list(range(11))
assert total == {"logit_values": 68943872, "owner_cache_bf16_values": 9437184000,
                 "sentinel_bf16_values": 4152360960}
result["total_exact_counts"] = total
result["warm_round_savings_ms"] = {"min": min(warm_savings), "median": statistics.median(warm_savings),
                                   "max": max(warm_savings)}
result["timing_limits"] = [
    "Each case has two rounds, current-first then candidate-first reset; not repeated statistical ABBA",
    "Total includes preparation and complete prompt layer/finalization work; excludes logit downloads, interleaved decode and cache comparison",
    "Stage profile includes candidate-only interleave operations in case0 round0; use clean round1 for stage attribution",
    "No production/HTTP throughput inference from this model-only gate",
]
print(json.dumps(result, indent=2))
