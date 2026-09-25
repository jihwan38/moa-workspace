#!/usr/bin/env python3
"""Extract M02's compiler dependencies and allocations; these are not NPU stalls."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def analyze(family, label):
    path = ROOT / "moa-dev/experiments/candidates" / family / f"build-{label}-repro.json"
    build = json.loads(path.read_text())
    raw = (HERE / "schedules" / f"{label}.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == build["schedule"]["sha256"]
    schedule = json.loads(raw)
    nodes = []
    for item in schedule["instructions"]:
        op = item["description"]
        if item["tpe"] == "Core" or "projection.rs:" not in op:
            continue
        begin, end = (item["lifetime"][key] for key in ("begin", "end"))
        if begin == end:
            continue
        if "\n\nweight.to_dm(" in op:
            kind = "weight_ingress"
        elif "\n\nx.to_dm(" in op:
            kind = "activation_ingress"
        elif "CustomBroadcast" in op:
            kind = "activation_broadcast"
        elif "to_trf()" in op:
            kind = "activation_trf"
        elif "contract_outer" in op:
            kind = "dot_and_fused_operations"
        elif "\n\npartial.to_dm(" in op:
            kind = "partial_gather"
        elif "begin(partial.view())" in op and "transpose" in op:
            kind = "partial_compaction"
        elif "tile::<m![Dummy2]" in op:
            kind = "sum_scale" if "AddF" in op else "second_partial_vrf"
        elif "\n\nresult.to_dm(" in op:
            kind = "control_epilogue_gather"
        elif "MulF" in op:
            kind = "control_scale"
        else:
            continue
        nodes.append({"kind":kind, "index":item["index"], "engine":item["tpe"],
                      "begin":begin, "end":end, "duration":end-begin,
                      "input_tensors":item["input_tensors"], "output_tensors":item["output_tensors"],
                      "operation":op})
    ids = {str(i) for n in nodes for i in n["input_tensors"]+n["output_tensors"]}
    allocations = [t for t in schedule["tensors"] if t["index"] in ids and t["buffer_type"] in ("Sram", "Vrf", "Trf")]
    span = max(n["lifetime"]["end"] for n in schedule["instructions"])
    assert span == build["static_cycles"]
    return {"makespan":span, "source_build_record":str(path.relative_to(ROOT)),
            "schedule":build["schedule"], "image":build["exact_image"], "nodes":nodes,
            "compiler_tensor_allocations":allocations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    builds = {"E104":analyze("c09-output-pipeline", "c09-e104-control"),
              "khalf-v3":analyze("m02-cluster-k", "m02-khalf-v3"),
              "khalf-v4":analyze("m02-cluster-k", "m02-khalf-v4")}
    for key in ("khalf-v3", "khalf-v4"):
        builds[key]["change_percent_vs_e104"] = 100*(builds[key]["makespan"]/builds["E104"]["makespan"]-1)
    result = {"scope":"Compiler schedule and tensor allocations; not measured NPU stalls, traffic, or peak live memory.",
              "builds":builds,
              "findings":[
                  "Both M02 mappings keep a13512-cycle weight load vs13383 in E104; segment length is not a supply speedup here.",
                  "M02 dot stream5063 equals E104 dot+inline16-column-reduction stream5063. Removing that reducer did not shorten the enclosing interval.",
                  "v3 full-H gather4658 -> compaction4105 -> compact cross-cluster gather801 are dependencies after dot completion.",
                  "v4 aligned groups replace those with one3314 gather and1225 compaction, then503 second-partial VRF and401 sum+scale.",
                  "Input broadcast4359 overlaps weight loading; do not add it to the critical path. Its setup and dependencies still precede dot.",
                  "Logical active-slot bytes differ from allocation extents: sparse cluster/slice dimensions reserve padding. Compare compiler_tensor_allocations, not a sum of logical payloads."
              ],
              "limits":["Context intervals overlap; durations are not additive NPU runtime contributions.",
                        "Whole-kernel Arena cycles cannot attribute decode, DMA or stall fractions.",
                        "Two realizations of one K-owned architecture; neither a global tile optimum nor an architecture-wide bound."]}
    path = HERE / "static-cost.json"
    if args.check:
        assert json.loads(path.read_text()) == result
    else:
        path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:{"makespan":v["makespan"], "change_percent":v.get("change_percent_vs_e104"),
                         "nodes":[{key:n[key] for key in ("kind", "begin", "end", "duration")} for n in v["nodes"]]}
                      for k,v in builds.items()}, indent=2))


if __name__ == "__main__":
    main()
