#!/usr/bin/env python3
"""Read-only trace reduction; excludes prompts and generated text."""
import collections
import hashlib
import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[3]
RUNS = [
    ("r971-c071-c15-warm-long-submit", 15),
    ("r972-c071-c16-warm-long-submit", 16),
    ("r981-q016-c16-warm-long-submit", 16),
]
report = {}
for name, width in RUNS:
    path = ROOT / "runs" / name
    raw = (path / "arena.log").read_bytes()
    candidate = json.loads((path / "tested-candidate.json").read_text())
    rows, http, cohorts, totals, summaries = {}, [], [], [], {}
    for line in raw.decode().splitlines():
        if line.startswith("SCHED_REQ "):
            fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
            rows[int(fields["id"])] = fields
        elif line.startswith("SCHED_COHORT "):
            cohorts.append(dict(re.findall(r"(\w+)=([^ ]+)", line)))
        elif line.startswith("SCHED_TOTAL "):
            totals.append(line)
        elif line.startswith("STAGE2_PERFORMANCE") and " {" in line:
            kind, payload = line.split(" ", 1)
            data = json.loads(payload)
            phase = "warmup" if "_WARMUP_" in kind else "measured"
            if kind.endswith("_SUMMARY"):
                summaries[phase] = data
            elif kind.endswith("_RESULT"):
                kept = {k: v for k, v in data.items() if k not in ("output", "seconds")}
                kept["phase"] = phase
                kept["first_offset_ms"] = data["dispatch_offset_ms"] + data["ttft_ms"]
                http.append(kept)
    bands = []
    for first in range(1, max(rows) + 1, width):
        band = [rows[i] for i in range(first, min(first + width, max(rows) + 1))]
        bands.append({
            "worker_id_range": [first, first + len(band) - 1],
            "note": "Admission-order band, not an explicitly logged snapshot membership list",
            "input_tokens": sum(int(r["p"]) for r in band),
            "length_counts": dict(sorted(collections.Counter(int(r["p"]) for r in band).items())),
            "positive_hold_count": sum(float(r["hold"]) > 0 for r in band),
            "mean_decode_wait_ms": statistics.mean(float(r["dw"].split("/")[0]) for r in band),
            "first_ms_range": [min(float(r["first"]) for r in band), max(float(r["first"]) for r in band)],
        })
    http_bands = []
    measured = [r for r in http if r["phase"] == "measured"]
    for first in range(1, len(measured) + 1, width):
        band = [r for r in measured if first <= r["request_index"] < first + width]
        def spread(key):
            return max(r[key] for r in band) - min(r[key] for r in band)
        http_bands.append({"request_index_range": [first, first + len(band) - 1],
                           "dispatch_span_ms": spread("dispatch_offset_ms"),
                           "first_span_ms": spread("first_offset_ms"),
                           "completion_span_ms": spread("completion_offset_ms")})
    report[name] = {
        "arena_log_sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": candidate["source_commit"],
        "binary_sha256": candidate["binary_sha256"],
        "scheduler_source_hashes": {k: candidate["source_hashes"][k] for k in (
            "src/host/worker.rs", "src/host/startup_cohort.rs")},
        "summaries": summaries, "cohort_releases": cohorts,
        "admission_bands": bands, "http_dispatch_bands": http_bands,
        "http_records": http, "worker_records": rows,
        "final_scheduler_totals": totals[-1],
        "http_over_4s": {phase: sum(r["ttft_ms"] > 4000 for r in http if r["phase"] == phase)
                         for phase in ("warmup", "measured")},
    }
print(json.dumps(report, indent=2))
