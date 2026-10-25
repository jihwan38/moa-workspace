#!/usr/bin/env python3
"""Reproduce r1204/r1205 evidence; no device execution or source mutation."""
import collections
import hashlib
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parents[2]
RUNS = [
    "r1204-f4-partialscale-layers0-5-submit",
    "r1205-f4-partialscale-layers6-47-submit",
]


def fields(line):
    result = {}
    for word in line.split():
        if "=" not in word:
            continue
        key, value = word.split("=", 1)
        if value in ("true", "false"):
            value = value == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        result[key] = value
    return result


def bounds(values):
    return {"min": min(values), "max": max(values), "mean": statistics.mean(values)}


def schedule_dma(mode):
    path = BASE / "b16-f4-partialscale/experiments/b16-f4-partialscale" / (mode + "-schedule.json")
    obj = json.loads(path.read_text())
    events = []
    for key, op in obj["description"]["operators"].items():
        if not op["kind"].startswith("dma."):
            continue
        for event in obj["timeline"]["operators"].get(key, []):
            events.append({"begin": event["lifetime"]["begin"], "operator": key,
                           "description": op["description"], "kind": op["kind"],
                           "iteration": event["iteration_index"]})
    return sorted(events, key=lambda x: (x["begin"], x["operator"]))


def parse(run):
    directory = BASE / "runs" / run
    raw = (directory / "arena.log").read_bytes()
    (HERE / (run.split("-")[0] + ".log")).write_bytes(raw)
    data = {"run": run, "status": json.loads((directory / "arena-status.json").read_text()),
            "log_sha256": hashlib.sha256(raw).hexdigest(), "weights": [], "stages": [],
            "times": [], "cpu_partials": [], "cpu_final": [], "profiles": []}
    weights = []
    current = None
    for line in raw.decode().splitlines():
        if line.startswith("F4_WEIGHT "):
            weights.append(fields(line))
        elif line.startswith("F4_STAGE "):
            record = fields(line)
            if weights:
                for weight in weights:
                    weight["layer"] = record["layer"]
                data["weights"].extend(weights)
                weights = []
            data["stages"].append(record)
            current = None
        elif line.startswith("F4_TIME "):
            data["times"].append(fields(line))
            current = None
        elif line.startswith("F4_CPU_PARTIAL "):
            record = fields(line)
            record["layer"] = data["stages"][-1]["layer"]
            data["cpu_partials"].append(record)
        elif line.startswith("F4_CPU_FINAL "):
            data["cpu_final"].append(fields(line))
        elif line.startswith("F4_PROFILE "):
            current = {**fields(line), "spans": []}
            data["profiles"].append(current)
        elif line.startswith("NPU_SPAN ") and current is not None:
            current["spans"].append(json.loads(line[9:]))
        elif line.startswith("PASS F4_PARTIALSCALE "):
            data["pass"] = fields(line)
    assert data["status"]["status"] == "SUCCEEDED"
    assert all(s["intended_bitexact"] for s in data["stages"])
    assert all(s["original_byte_diff"] == 0 for s in data["stages"] if s["stage"] in (2, 3, 4))
    for p in data["profiles"]:
        schedule = schedule_dma("candidate" if p["candidate"] else "control")
        threads = []
        for tid in sorted({s["tid"] for s in p["spans"]}):
            spans = [s for s in p["spans"] if s["tid"] == tid]
            counts = collections.Counter(s["name"] for s in spans)
            duration = collections.Counter()
            for s in spans:
                duration[s["name"]] += s["dur"]
            dma = sorted((s for s in spans if s["name"] == "DMA"), key=lambda s: s["begin_cycle"])
            assert len(dma) == len(schedule), (len(dma), len(schedule))
            loads = []
            for index, (actual, static) in enumerate(zip(dma, schedule)):
                desc = static["description"].strip()
                if desc.splitlines()[-1:] in (["up.to_dm(&mut device.tdma)"], ["gate.to_dm(&mut device.tdma)"], ["weight.to_dm(&mut device.tdma)"]):
                    loads.append({"ordinal": index, "duration_cycles": actual["dur"], **static})
            assert len(loads) == 3
            threads.append({"tid": tid, "count": counts, "sum_span_cycles": duration,
                            "weight_dma": loads})
        p["threads"] = threads
        del p["spans"]
    data["timing_by_layer"] = []
    for layer in sorted({t["layer"] for t in data["times"]}):
        old = [t["complete_us"] for t in data["times"] if t["layer"] == layer and not t["candidate"]]
        new = [t["complete_us"] for t in data["times"] if t["layer"] == layer and t["candidate"]]
        data["timing_by_layer"].append({"layer": layer, "control_us": bounds(old), "candidate_us": bounds(new),
            "relative_gain": 1-statistics.mean(new)/statistics.mean(old)})
    return data


def main():
    runs = [parse(r) for r in RUNS]
    weights = [w for r in runs for w in r["weights"]]
    stages = [s for r in runs for s in r["stages"] if not s["perm"]]
    total = {"intended_exact_bytes": sum(r["pass"]["intended_exact_bytes"] for r in runs),
             "immutable_bytes": sum(r["pass"]["immutable_bytes"] for r in runs),
             "cpu_partial_samples": sum(x["samples"] for r in runs for x in r["cpu_partials"]),
             "cpu_final_samples": sum(x["samples"] for r in runs for x in r["cpu_final"]),
             "weight_relative_l2": bounds([w["relative_l2"] for w in weights]),
             "weight_row_dequant_l2": bounds([w["row_dequant_l2"] for w in weights]),
             "encode_and_audit_ms_per_projection": bounds([w["cpu_encode_and_audit_ms"] for w in weights]),
             "stage_error_ranges": {str(stage): {
                 "relative_l2": bounds([s["original_relative_l2"] for s in stages if s["stage"] == stage]),
                 "max_abs": max(s["original_max_abs"] for s in stages if s["stage"] == stage)} for stage in range(8)}}
    old = [t["complete_us"] for r in runs for t in r["times"] if not t["candidate"]]
    new = [t["complete_us"] for r in runs for t in r["times"] if t["candidate"]]
    total.update(control_us=bounds(old), candidate_us=bounds(new), relative_gain=1-statistics.mean(new)/statistics.mean(old))
    notes = ["Actual timing is completed one-call post-attention+FFN+postnorm; reset upload/readbacks and static encoding excluded.",
             "Every reported stage error compares same-input current E4 versus new F4 weights; feedback input is the previous candidate state in both arms.",
             "Strict all-stage equality is against unscaled F4 codes expanded to E4 with identical local partial scales, not against original E4 weights.",
             "CPU checks are sampled/toleranced; all observed sample differences happened to be zero.",
             "DMA/other span sums overlap and include queued waiting; they are not additive device busy time.",
             "Weight DMA identity uses flattened static schedule DMA order including loop iterations, checked against actual DMA count.",
             "encode_and_audit_ms includes full per-value independent F64 error/code audit; excludes prior D2H and subsequent upload. It is not encoder-only/startup time."]
    output = {"runs": runs, "aggregate": total, "notes": notes}
    (HERE / "actual-r1204-r1205.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(total, indent=2))


if __name__ == "__main__":
    main()
