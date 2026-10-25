#!/usr/bin/env python3
"""Matched late-admission cost accounting; all times retain rounding bounds."""
import collections
import hashlib
import json
import pathlib
import re
import subprocess

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[2]
RUNS = json.loads((HERE / "results.json").read_text())


def parse(raw):
    return {k: (int(n), float(mean)) for k, n, mean in re.findall(
        r"\b((?:d|pc|prep|layer|final)\d+)=(\d+):([\d.]+)/", raw)}


def account(run):
    trace = run["scheduler_trace"]
    first, last = trace["total_lines"][0], trace["total_lines"][-1]
    warm = run["record_counts"]["warmup"]["record_count"]
    assert warm == 16
    assert sum(w["line"] < first["line"] for w in trace["worker_requests"]) == warm
    assert run["full_measured_summary_present"] and run["expected_measured"] == 96
    assert len(trace["worker_requests"]) == 112
    assert not trace["structural_errors"] and trace["open_snapshot"] is None
    a, b = parse(first["raw"]), parse(last["raw"])
    out = {"warm_checkpoint_line": first["line"], "groups": {}, "widths": {}}
    for prefix in ("d", "pc", "prep", "layer", "final"):
        aa = [v for k, v in a.items() if re.fullmatch(prefix + r"\d+", k)]
        bb = [v for k, v in b.items() if re.fullmatch(prefix + r"\d+", k)]
        na, nb = sum(n for n, _ in aa), sum(n for n, _ in bb)
        out["groups"][prefix] = dict(count=nb-na,
            cost_ms=sum(n*m for n,m in bb)-sum(n*m for n,m in aa), rounding_bound_ms=.005*(na+nb))
    for k, (n, mean) in b.items():
        if not re.fullmatch(r"d\d+", k): continue
        na, ma = a.get(k, (0, 0))
        out["widths"][int(k[1:])] = dict(count=n-na, cost_ms=n*mean-na*ma, mean_ms=(n*mean-na*ma)/(n-na))
    out["rows"] = sum(w*x["count"] for w,x in out["widths"].items())
    out["mean_width"] = out["rows"]/out["groups"]["d"]["count"]
    assert out["rows"] == 6112
    assert out["groups"]["layer"]["count"] == 48*out["groups"]["prep"]["count"]
    owner = lambda s: {int(w):int(n) for w,n in re.findall(r"\bp(\d+)=(\d+)\b",s)}
    oa,ob=owner(first["raw"]),owner(last["raw"])
    out["prompt_owner_quanta"] = {w:n-oa.get(w,0) for w,n in ob.items()}
    out["prompt_owner_chunks"] = {w:n//48 for w,n in out["prompt_owner_quanta"].items()}
    assert all(n%48 == 0 for n in out["prompt_owner_quanta"].values())
    out["metrics"] = run["published_summaries"]["measured"]["metrics"]
    out["workload"] = sorted([list(k)+[n] for k,n in collections.Counter(
        (r["suite"],r["usage"]["prompt_tokens"],r["usage"]["completion_tokens"])
        for r in run["http_records"]["measured"]).items()])
    workers={int(w["id"]):w for w in trace["worker_requests"]}
    out["unfinished_at_release"] = [dict(cycle=i, reason=c["release"]["action"],
        ready=c["ready_count"], members=len(c["release"]["members"]),
        remaining=c["remaining_prompt_tokens"], rows=[dict(worker_id=m["id"],
            prompt_length=int(workers[m["id"]]["p"]), remaining=m["remaining"],
            consumed=int(workers[m["id"]]["p"])-m["remaining"])
            for m in c["release"]["members"] if m.get("remaining")])
        for i,c in enumerate(trace["cycles"])]
    return out


names=list(RUNS)
out={"runs":{n:account(r) for n,r in RUNS.items()}}
a,b=(out["runs"][n] for n in names)
assert a["workload"] == b["workload"]
out["deltas"]={p:dict(count=b["groups"][p]["count"]-a["groups"][p]["count"],
    cost_ms=b["groups"][p]["cost_ms"]-a["groups"][p]["cost_ms"],
    rounding_bound_ms=b["groups"][p]["rounding_bound_ms"]+a["groups"][p]["rounding_bound_ms"])
    for p in a["groups"]}
out["window_delta_ms"] = 1000*(b["metrics"]["measurement_window_seconds"]-a["metrics"]["measurement_window_seconds"])
out["dispatch_delta_ms"] = sum(out["deltas"][p]["cost_ms"] for p in ("d","pc"))
out["dispatch_rounding_bound_ms"] = sum(out["deltas"][p]["rounding_bound_ms"] for p in ("d","pc"))
out["unassigned_ms"] = out["window_delta_ms"]-out["dispatch_delta_ms"]
assert set(b["widths"]) <= set(a["widths"])
out["decode_width_cost_decomposition"] = {
    "count_and_width_mix_at_control_cost_ms": sum(
        (b["widths"].get(w,{"count":0})["count"]-x["count"])*x["mean_ms"] for w,x in a["widths"].items()),
    "within_width_cost_delta_ms": sum(x["count"]*(x["mean_ms"]-a["widths"][w]["mean_ms"])
                                      for w,x in b["widths"].items()),
    "scope":"Arithmetic decomposition using observed control mean costs, not a replay or causal prediction of another scheduler."
}
out["client_hashes"]={}
for file in ("scripts/stage2/performance.py","scripts/stage2/performance_test.sh",
             "scripts/stage2/entrypoint.sh","scripts/stage2/scoring.py"):
    hashes=[hashlib.sha256((ROOT/"evaluations"/n/file).read_bytes()).hexdigest()
            for n in ("c077-c16-overlap-r96","c078-c16-overlap-r96")]
    assert hashes[0] == hashes[1]
    out["client_hashes"][file]=hashes[0]
out["changed_source_files"]=subprocess.check_output(["git","diff","--name-only",
    RUNS[names[0]]["source_commit"],RUNS[names[1]]["source_commit"],"--","src"]).decode().splitlines()
assert out["changed_source_files"] == ["src/host/recovery_late_admission_tests.rs","src/host/startup_cohort.rs","src/host/worker.rs"]
out["limits"]=["Diagnostic has one128-output request and is not an official/rank-comparable workload.",
    "All cost differences subtract the verified warm checkpoint; mean rounding bounds are not confidence intervals.",
    "pc includes prep/layer/final. Do not add both decompositions.",
    "No SCHED_CHUNK records: exact tail geometry/individual batch composition is unavailable.",
    "HTTP IDs and Worker IDs only map for a unique observed prompt/output-length tuple; never by arrival order."]
print(json.dumps(out,indent=2))
