#!/usr/bin/env python3
"""Extend frozen overlap reader for appended membership; never join IDs by order."""
import argparse
import importlib.util
import json
import pathlib

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("base", HERE.parent / "recovery-overlap/analyze.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def analyze(path):
    result = base.analyze(path, parse_recovery=False)
    workers = result["scheduler_trace"]["worker_requests"]
    events, cycles, errors = [], [], []
    current, participated = None, set()
    for num, line in enumerate((path / "arena.log").read_text().splitlines(), 1):
        if not line.startswith("SCHED_RECOVERY "):
            continue
        f = base.fields(line)
        event = dict(line=num, action=f["action"], members=base.members(f["members"], True),
                     passthrough=base.members(f["passthrough"], False),
                     oldest_ms=float(f["oldest_ms"]), left_ms=float(f["left_ms"]))
        ids = [r["id"] for r in event["members"]]
        if len(set(ids)) != len(ids): errors.append("duplicate member IDs")
        if set(ids) & {r["id"] for r in event["passthrough"]}: errors.append("member also passed through")
        if event["action"] == "start":
            if current: errors.append("start before release")
            if participated & set(ids): errors.append("previous participant captured again")
            if len(ids) < 2: errors.append("start with fewer than two members")
            if any(r.get("ready") != 0 for r in event["members"]): errors.append("ready member held on start")
            participated.update(ids)
            current = dict(start=event, latest=event, joins=[], origin_shift_ms=0.0,
                           initial_rows={r["id"]: r for r in event["members"]})
        else:
            if current is None:
                errors.append("event without open snapshot (possible truncated log)")
                events.append(event)
                continue
            prior = current["latest"]
            previous_ids = [r["id"] for r in prior["members"]]
            if event["action"] == "join":
                event["previous"] = [int(x) for x in f["previous"].split(",") if x]
                event["added"] = base.members(f["added"], False)
                for k in ("old_oldest_ms", "old_left_ms", "elapsed_ms"):
                    event[k] = float(f[k])
                added_ids = [r["id"] for r in event["added"]]
                if event["previous"] != previous_ids: errors.append("join previous IDs do not match")
                if ids != previous_ids + added_ids: errors.append("membership not monotone append")
                if participated & set(added_ids): errors.append("late member held more than once")
                if event["left_ms"] > event["old_left_ms"] + .003: errors.append("join extended deadline")
                if event["oldest_ms"] + .003 < event["old_oldest_ms"]: errors.append("oldest origin moved later")
                if any(r.get("ready") != 0 for r in event["members"] if r["id"] in added_ids): errors.append("ready late member")
                if not added_ids: errors.append("empty join event")
                participated.update(added_ids)
                current["origin_shift_ms"] += event["oldest_ms"] - event["old_oldest_ms"]
                current["initial_rows"].update({r["id"]: r for r in event["members"] if r["id"] in added_ids})
                current["joins"].append(event)
            elif ids != previous_ids:
                errors.append("release membership differs from final snapshot")
            before = {r["id"]: r for r in prior["members"]}
            for row in event["members"]:
                if row["id"] in before and not row.get("gone") and not before[row["id"]].get("gone"):
                    if row["remaining"] > before[row["id"]]["remaining"]: errors.append("remaining increased")
            if event["action"] not in ("join", "start"):
                if event["action"] == "ready" and any(not (r.get("gone") or r.get("ready")) for r in event["members"]):
                    errors.append("ready release has unfinished member")
                if event["action"] == "deadline" and event["oldest_ms"] < 3750: errors.append("early deadline")
                observed = [current["start"], *current["joins"]]
                pass_ids = {r["id"] for e in observed for r in e["passthrough"] if r["remaining"] == 0}
                completions = [w for w in workers if int(w["id"]) in pass_ids
                               and current["start"]["line"] < w["line"] < event["line"]
                               and any(e["line"] < w["line"] and int(w["id"]) in
                                       {r["id"] for r in e["passthrough"] if r["remaining"] == 0} for e in observed)]
                cycles.append(dict(start=current["start"], joins=current["joins"], release=event,
                    duration_ms=event["oldest_ms"] - current["start"]["oldest_ms"] - current["origin_shift_ms"],
                    ready_count=sum(bool(r.get("ready") or r.get("gone")) for r in event["members"]),
                    remaining_prompt_tokens=sum(r.get("remaining", 0) for r in event["members"]),
                    initial_rows=list(current["initial_rows"].values()),
                    start_passthrough_decoder_ids=[r["id"] for r in current["start"]["passthrough"] if r["remaining"] == 0],
                    all_observed_passthrough_decoder_ids=sorted(pass_ids),
                    old_decoder_completions_inside=completions))
                current = None
            else:
                current["latest"] = event
        if event["action"] in ("start", "join") and abs(event["oldest_ms"] + event["left_ms"] - 3750) > .003:
            errors.append("active receipt budget differs")
        events.append(event)
    trace = result["scheduler_trace"]
    trace.update(events=events, cycles=cycles, recovery_event_count=len(events), recovery_cycle_count=len(cycles),
                 structural_errors=errors, open_snapshot=current, unique_participant_count=len(participated),
                 join_count=sum(e["action"] == "join" for e in events))
    for stream in result["long_stream_overlap"]:
        stream["matching_recovery_passthrough_cycles"] = [i for i, c in enumerate(cycles)
            if stream["unique_worker_id_by_length_and_token_count"] in c["all_observed_passthrough_decoder_ids"]]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/home/jihwan38/moa-worktrees/round2"))
    a = p.parse_args()
    print(json.dumps({name: analyze(a.root / "runs" / name) for name in a.runs}, indent=2))
