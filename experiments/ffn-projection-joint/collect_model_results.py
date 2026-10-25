"""Verify frozen model runs against their source commits and aggregate exact gates."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
TREE=HERE.parents[1]
RUNS=TREE.parent/"runs"
ARTIFACT=json.loads((HERE/"model-artifact.json").read_text())
SHA=ARTIFACT["sha256"]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for directory in sorted(RUNS.glob("r*")):
    if not (directory/"arena-status.json").exists(): continue
    if not (directory/"run.json").exists(): continue
    run=json.loads((directory/"run.json").read_text())
    artifact=run.get("submitted_artifact",{})
    if artifact.get("sha256")!=SHA: continue
    status=json.loads((directory/"arena-status.json").read_text())
    assert status["status"]=="SUCCEEDED" and status["exit_code"]==0
    assert sha(Path(artifact["path"]))==SHA
    hashes=json.loads((directory/"source-hashes.json").read_text())
    for name,digest in hashes.items():
        source=subprocess.run(["git","show",f'{run["commit"]}:{name}'],cwd=TREE,capture_output=True)
        # Cargo.lock is generated but ignored in this upstream worktree.
        data=source.stdout if source.returncode==0 else (TREE/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==digest,(directory.name,name)
    lines=(directory/"arena.log").read_text().splitlines()
    assert any(l.startswith("PASS current composed preload vs scoped projection-owned FFN") for l in lines)
    exact=[l for l in lines if l.startswith("PROJECTION_FFN_MODEL_EXACT ")]
    cache=[l for l in lines if l.startswith(("PROJECTION_FFN_MODEL_CACHE ","PROJECTION_FFN_MODEL_SENTINEL "))]
    assert all("bitdiff=0" in l and "nonfinite=0" in l for l in exact)
    assert all("bitdiff=0" in l for l in cache)
    pairs=[]
    for line in lines:
        m=re.match(r"PROJECTION_FFN_MODEL_PASS case=(\d+) round=(\d+) control_ms=([\d.]+) candidate_ms=([\d.]+)",line)
        if m:pairs.append(dict(case=int(m[1]),round=int(m[2]),control_ms=float(m[3]),candidate_ms=float(m[4])))
    assert len(pairs)==2 and [p["round"] for p in pairs]==[0,1]
    assert pairs[0]["case"]==pairs[1]["case"]
    memory=[dict(label=m[1],capacity=int(m[2]),available=int(m[3]),largest=int(m[4])) for l in lines if (m:=re.fullmatch(r"PROJECTION_MODEL_MEMORY label=(\S+) capacity=(\d+) available=(\d+) largest=(\d+)",l))]
    assert [x["label"] for x in memory]==["before_join","after_join","all_model_routes_complete"]
    rows.append(dict(memory=memory,case=pairs[0]["case"],run=directory.name,job=status["id"],duration_seconds=status["duration_sec"],
                     source_commit=run["commit"],source_files_verified=len(hashes),artifact=artifact,
                     log_sha256=sha(directory/"arena.log"),source_manifest_sha256=sha(directory/"source-hashes.json"),
                     exact_comparisons=len(exact),exact_logit_values=sum(int(re.search(r"values=(\d+)",l)[1]) for l in exact),
                     cache_comparisons=len(cache),exact_cache_bytes=2*sum(int(re.search(r"bf16_values=(\d+)",l)[1]) for l in cache),
                     timings=pairs))
rows.sort(key=lambda r:r["case"])
assert [r["case"] for r in rows]==list(range(11)),[r["case"] for r in rows]
result=dict(status="all11 joint stationary-O/projection-FFN model routes passed",source=ARTIFACT["source_commit"],elf_sha256=SHA,
            minimum_reported_available_bytes=min(x["available"] for r in rows for x in r["memory"]),
            exact_logit_values=sum(r["exact_logit_values"] for r in rows),exact_cache_bytes=sum(r["exact_cache_bytes"] for r in rows),
            exact_comparisons=sum(r["exact_comparisons"] for r in rows),runs=rows,
            caveats=["Case0 candidate-only independent B16/R32 interleaves are excluded from total_ms.",
                     "Stage-profile aggregates include interleaved decoder calls, so their sum is not the prompt timing.",
                     "Round0 is control-first; reset round1 is candidate-first. These are paired diagnostics, not ABBA or HTTP throughput.",
                     "Untouched sentinel and full physical cache scans include repeated values; byte totals count comparisons, not unique cache capacity.",
                     "Both arms explicitly enable current exact sliding/full stationary O and retain composed norm stages and global preload; projection-FFN default activation is still off. Startup joins are excluded from per-request timings and measured memory includes actual joined storage."])
(HERE/"model-results.json").write_text(json.dumps(result,indent=2)+"\n")
summary=["# All11 joint model routes pass","",f'All **{result["exact_logit_values"]:,} vocabulary values** and **{result["exact_cache_bytes"]:,} raw KV bytes compared** match exactly. Frozen source `acbc934`; all submitted ELFs and recorded source hashes verified.',"", "| Case | Job | Reset control ms | Reset projection-owned ms |", "| --- | --- | ---: | ---: |"]
for r in rows:
    t=r["timings"][1]
    summary.append(f'| {r["case"]} | {r["job"]} | {t["control_ms"]:.3f} | {t["candidate_ms"]:.3f} |')
summary.extend(["",*result["caveats"],"","Coverage: four independent128, single512, pair256+256, both Tail256 owner shapes, one/two/three-owner Tail384 and all doubled-owner positions in ThreeOwner512. Every route independently refills actual prefixes, executes48 layers, reverses owner reservations on reset, and compares all-vocabulary output, whole allocated raw KV, sentinels and teacher-forced postdecodes. Tail reset padding is NaN. Case0 includes six matched B16/R32 decoder calls at complete layer boundaries with disjoint reservations."])
(HERE/"model-results.md").write_text("\n".join(summary)+"\n")
print(json.dumps({k:result[k] for k in ("status","exact_logit_values","exact_cache_bytes","exact_comparisons")}))
