#!/usr/bin/env python3
"""Compare complete matched physical64/32 diagnostic logs, never partial runs.

This establishes SHA256 equality for the explicitly emitted logical outputs and
reserved cache bytes. Each executable separately compares paired owners bytewise;
this script does not call a digest match an in-process full-byte comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

MARKERS = (
    "PHYSICAL_POOL_GROUP_EXACT",
    "PHYSICAL_HISTORY_LOGITS",
    "PHYSICAL_HISTORY_PROMPT",
    "KV_ALL_BYTES_EXACT",
)
FINALS = (
    "PASS physical32 same-c079 paired owners",
    "PASS physical64 same-c079 paired owners",
    "PASS physical_history physical=",
)


def parse(path):
    raw = Path(path).read_bytes()
    lines = raw.decode(errors="replace").splitlines()
    finals = [line for line in lines if any(m in line for m in FINALS)]
    if len(finals) != 1:
        raise ValueError(f"{path}: expected one completed diagnostic PASS, got {len(finals)}")
    events = []
    for line in lines:
        marker = next((m for m in MARKERS if m + " " in line), None)
        if marker is None:
            continue
        body = line.split(marker + " ", 1)[1]
        fields = dict(re.findall(r"(\w+)=([^\s]+)", body))
        if not re.fullmatch(r"[0-9a-f]{64}", fields.get("sha256", "")):
            raise ValueError(f"{path}: missing/invalid digest: {line}")
        for flag in ("bitdiff", "nonfinite"):
            if flag in fields and fields[flag] != "0":
                raise ValueError(f"{path}: failed {flag}: {line}")
        events.append({"kind": marker, **fields})
    if not events:
        raise ValueError(f"{path}: no digest events")
    return {
        "path": str(Path(path).resolve()),
        "log_sha256": hashlib.sha256(raw).hexdigest(),
        "events": events,
        "completed_pass": finals[0],
    }


def compare(control, candidate):
    a, b = parse(control), parse(candidate)
    mismatches = []
    if len(a["events"]) != len(b["events"]):
        mismatches.append({"event_count": [len(a["events"]), len(b["events"])]})
    for i, (x, y) in enumerate(zip(a["events"], b["events"])):
        if x != y:
            mismatches.append({"index": i, "control": x, "candidate": y})
    logits = sum(int(x.get("values", 0)) for x in a["events"]
                 if x["kind"] != "KV_ALL_BYTES_EXACT")
    cache = sum(int(x.get("compared_bytes", 0)) for x in a["events"]
                if x["kind"] == "KV_ALL_BYTES_EXACT")
    return {
        "result": "PASS" if not mismatches else "FAIL",
        "claim": "cross-program SHA256 equality of matched logical logits/cache; paired-owner bytewise gates are separate",
        "control": {k: v for k, v in a.items() if k != "events"},
        "candidate": {k: v for k, v in b.items() if k != "events"},
        "digest_events": [len(a["events"]), len(b["events"])],
        "logical_logit_values": logits,
        "cache_bytes_per_program_pair_comparisons": cache,
        "mismatches": mismatches,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("control")
    ap.add_argument("candidate")
    ap.add_argument("--output")
    args = ap.parse_args()
    result = compare(args.control, args.candidate)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["result"] == "PASS" else 1)
