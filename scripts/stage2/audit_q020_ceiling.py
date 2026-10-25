#!/usr/bin/env python3
"""Recompute the current-work performance bounds from the frozen q020 log."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counters(line):
    return {
        key: (int(count), float(mean))
        for key, count, mean in re.findall(r"\b([a-z]+\d*)=(\d+):([\d.]+)/", line)
    }


def difference(first, last, keys):
    count = 0
    total = 0.0
    error = 0.0
    for key in keys:
        n0, mean0 = first.get(key, (0, 0.0))
        n1, mean1 = last.get(key, (0, 0.0))
        assert n1 >= n0, key
        count += n1 - n0
        total += n1 * mean1 - n0 * mean0
        # WorkCost::compact prints cumulative means to two decimal places.
        error += (n0 + n1) * 0.005
    return {"calls": count, "milliseconds": total, "rounding_bound_ms": error}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
args = parser.parse_args()
root = args.root.resolve()
run = "runs/r1301-q020-c16-warm-r96-submit"
log_path = root / run / "arena.log"
lines = log_path.read_text().splitlines()
totals = [(i, line) for i, line in enumerate(lines) if line.startswith("SCHED_TOTAL ")]
requests = [
    (i, int(match[1]), int(match[2]), int(match[3]))
    for i, line in enumerate(lines)
    if (match := re.match(r"SCHED_REQ id=(\d+) p=(\d+) .*? n=(\d+) ", line))
]
summaries = [json.loads(line.split(" ", 1)[1]) for line in lines
             if line.startswith("STAGE2_PERFORMANCE_SUMMARY ")]
assert len(summaries) == 1
summary = summaries[0]
concurrency = summary["concurrency"]
assert concurrency == 16 and summary["total"] == summary["passed"] == 96
assert len({row[1] for row in requests}) == len(requests) == 112
warm = [row for row in requests if row[0] < totals[0][0]]
measured = [row for row in requests if totals[0][0] < row[0] < totals[-1][0]]
assert len(warm) == concurrency and len(measured) == summary["total"]
assert all(row[3] == 64 for row in requests)
first, last = counters(totals[0][1]), counters(totals[-1][1])
decode_keys = {key for key in last if re.fullmatch(r"d\d+", key)}
decode = difference(first, last, decode_keys)
prompt = difference(first, last, ["pc1", "pc4"])
assert difference(first, last, ["pc1"])["calls"] == 0
chunks = difference(first, last, ["prep4"])["calls"]
assert prompt["calls"] == chunks * 48
decode_rows = sum(int(key[1:]) * (last[key][0] - first.get(key, (0, 0))[0])
                  for key in decode_keys)
assert decode_rows == sum(row[3] - 1 for row in measured)
outputs = sum(row[3] for row in measured)
useful_prompt = sum(row[2] for row in measured)
window_ms = summary["metrics"]["measurement_window_seconds"] * 1000
assert abs(outputs * 1000 / window_ms - summary["metrics"]["generated_tokens_per_second"]) < 1e-9
official_path = root / "final/submissions/f5abd3e8/submission.json"
official = json.loads(official_path.read_text())
board_path = root / "state/leaderboard-20261025T213340Z.json"
rows = json.loads(board_path.read_text())["data"]
beef = next(row for row in rows if row["teamName"] == "BEEF")
third = next(row for row in rows if row["rank"] == 3)
assert beef["submissionCode"] == official["submission"] == "f5abd3e8"
assert beef["score"] == official["official_score_generated_tokens_per_second"]
worker_path = root / "final/submissions/f5abd3e8/src/host/worker.rs"
assert sha256(worker_path) == official["source_hashes"]["src/host/worker.rs"]
report = {
    "candidate": official["candidate"],
    "source_commit": official["source_commit"],
    "classification": "current-work counterfactual audit; not a universal hardware bound",
    "inputs_sha256": {str(path.relative_to(root)): sha256(path)
                       for path in (log_path, official_path, board_path, worker_path)},
    "local_measured_window": {
        "concurrency": concurrency, "warmup_requests_excluded": len(warm),
        "measured_requests": len(measured), "output_tokens": outputs,
        "duration_ms": window_ms, "prompt": prompt, "decode": decode,
        "unattributed_ms": window_ms - prompt["milliseconds"] - decode["milliseconds"],
        "decode_rows": decode_rows, "dense_chunks": chunks,
        "useful_prompt_tokens": useful_prompt, "dense_prompt_rows": chunks * 512,
        "padding_fraction": 1 - useful_prompt / (chunks * 512),
    },
    "counterfactual_tokens_per_second": {
        "all_prompt_work_free_unchanged_decode": outputs * 1000 / decode["milliseconds"],
        "all_decode_work_free_unchanged_prompt": outputs * 1000 / prompt["milliseconds"],
        "perfect_overlap_unchanged_phases": outputs * 1000 / max(prompt["milliseconds"], decode["milliseconds"]),
        "all_prompt_free_rounding_optimistic": outputs * 1000 / (decode["milliseconds"] - decode["rounding_bound_ms"]),
        "all_decode_free_rounding_optimistic": outputs * 1000 / (prompt["milliseconds"] - prompt["rounding_bound_ms"]),
    },
    "official_observation": {
        "submission": beef["submissionCode"], "score": beef["score"], "rank": beef["rank"],
        "observed_at_utc": "2026-10-25T21:33:40Z", "final_ranking": False,
        "third_score": third["score"], "required_speedup": third["score"] / beef["score"],
    },
    "interpretation": [
        "pc1/pc4 are inclusive work intervals; prep/layer/final counters are subcomponents and must not be added again.",
        "The first cumulative total follows exactly the 16 completed warmup requests; subtract it from the final total.",
        "Counterfactuals omit all unattributed overhead, preserve the observed work/call schedule and use rounded cumulative means.",
        "Local trace covers 96 measured requests. Official score covers 240 and is recorded separately; no official phase profile is available.",
        "Improving only one phase cannot reach 700 TPS under this fixed-work counterfactual. This does not rule out a different algorithm or schedule.",
        "A higher-concurrency path needs separately measured prompt and decode improvements; extra KV capacity alone does not establish that path.",
    ],
}
print(json.dumps(report, indent=2) + "\n", end="")
