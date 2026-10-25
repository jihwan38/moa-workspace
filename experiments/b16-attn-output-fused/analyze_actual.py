#!/usr/bin/env python3
"""Summarize actual complete spans; category unions are not additive wall time."""
import ast, collections, json, re, statistics, sys
from pathlib import Path
log=Path(sys.argv[1]);groups={};current=None;timings=[];checks=[]
for line in log.read_text().splitlines():
    if line.startswith("PROFILE_VARIANT "):
        current=line.removeprefix("PROFILE_VARIANT ");groups[current]=[]
    elif line=="PROFILE_END":current=None
    elif line.startswith("NPU_SPAN ") and current:
        groups[current].append(json.loads(line.removeprefix("NPU_SPAN ")))
    elif line.startswith("B16_ATTN_O_TIME "):
        row={key:int(re.search(key+r"=(\d+)",line)[1]) for key in ["live","case","round"]}
        row["candidate"]=re.search(r"candidate=(\w+)",line)[1]=="true"
        row["complete_us"]=float(re.search(r"complete_us=([0-9.]+)",line)[1])
        timings.append(row)
    elif line.startswith(("B16_ATTN_O_EXACT ","PASS ")):checks.append(line)
def union(spans):
    pairs=sorted((s["begin_cycle"],s["end_cycle"]) for s in spans)
    total=0;end=-1
    for a,b in pairs:total+=max(0,b-max(a,end));end=max(end,b)
    return total
profiles={}
for label,events in groups.items():
    profile={}
    for tid in sorted(set(e["tid"] for e in events)):
        subset=[e for e in events if e["tid"]==tid];byname={}
        for name in sorted(set(e["name"] for e in subset)):
            ss=[e for e in subset if e["name"]==name]
            byname[name]={"count":len(ss),"union_cycles":union(ss),"sum_cycles":sum(e["dur"] for e in ss),"durations_cycles":[e["dur"] for e in ss]}
        profile[str(tid)]=byname
    profiles[label]=profile
result={"log":str(log),"passed":any(s.startswith("PASS ") for s in checks),"checks":checks,"timings":timings,"profiles":profiles,
"scope":["Complete timing compares original commit+attention+Hsplit O three launches against one retained-stage graph; no host integration.","ABBA entries average12 completed chains; metadata uploads and readbacks excluded.","Actual category durations overlap; do not add category unions to derive wall time."]}
print(json.dumps(result,indent=2))
