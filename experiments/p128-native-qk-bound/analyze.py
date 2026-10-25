#!/usr/bin/env python3
"""Reproduce the complete compiler screen; no hardware/numerical claim."""
import collections
import hashlib
import json
from pathlib import Path

tree = Path(__file__).resolve().parents[2]
root = tree.parent
out = {"source": "7ec9a1aea06c03349ec1f0dc7511ecca84bf6ca4", "graphs": {}}
for name, run in [("candidate", "r1207-p128-native-qk-bound-resource"),
                  ("control", "r1208-p128-native-qk-bound-control")]:
    record = root / "runs" / run
    assert json.loads((record / "run.json").read_text())["exit_code"] == 0
    hashes = json.loads((record / "source-hashes.json").read_text())
    for rel, want in hashes.items():
        assert hashlib.sha256((tree / rel).read_bytes()).hexdigest() == want, rel
    path = Path(__file__).with_name(name + "-schedule.json")
    d = json.loads(path.read_text())
    sums = collections.Counter()
    contractions = []
    end = 0
    for key, times in d["timeline"]["operators"].items():
        desc = d["description"]["operators"][key]
        for timing in times:
            life = timing["lifetime"]
            end = max(end, life["end"])
            duration = life["end"] - life["begin"]
            sums[desc["kind"]] += duration
            if desc["kind"] == "rngd_cmd.tu_exec" and "contract_outer" in desc["description"]:
                contractions.append({"duration": duration, "description": desc["description"]})
    stack = [entry["allocation"] for entries in d["timeline"]["tensors"].values()
             for entry in entries if "DramStack" in entry["allocation"]["buffer_type"]]
    assert not stack
    out["graphs"][name] = {"estimated_cycles": end, "source_hashes_verified": len(hashes),
        "schedule_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "overlapping_duration_sums": sums, "contractions": contractions,
        "dram_stack": stack}
out["fractional_reduction"] = 1 - out["graphs"]["candidate"]["estimated_cycles"] / out["graphs"]["control"]["estimated_cycles"]
out["decision"] = "HOLD: complete resource gate failed; unscaled casts are not a numerical candidate"
Path(__file__).with_name("result.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"decision": out["decision"], "fractional_reduction": out["fractional_reduction"],
                  "estimated_cycles": {k: v["estimated_cycles"] for k, v in out["graphs"].items()}}))
