#!/usr/bin/env python3
"""Compare compiler schedules; these are estimates, not NPU attribution."""
import importlib.util
import json
from build import HERE, common

source = HERE.parent / "ffn-up-tiled-fused-decode-e47/analyze_schedule.py"
spec = importlib.util.spec_from_file_location("c04_schedule_profile", source)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def analyze():
    paths = {"e55": common.ROOT / "moa-dev/experiments/artifacts/c01-scale-dpe-e55/e55-control-schedule.json",
             "e74": common.ROOT / "moa-dev/experiments/artifacts/c05-streamed-decode/c05-partials-transpose2/schedule.json"}
    paths.update({p.parent.name: p for p in sorted(common.ARTIFACTS.glob("*/schedule.json"))})
    records = {}
    for label, path in paths.items():
        data, record = profile.profile(path)
        record.update(path=str(path.relative_to(common.ROOT)), sha256=common.digest(path.read_bytes())["sha256"])
        record["down_operations"] = [{**{k: i[k] for k in ("index", "tpe", "lifetime", "description", "input_tensors", "output_tensors")},
                                      "duration_cycles": i["lifetime"]["end"] - i["lifetime"]["begin"]}
                                     for i in data["instructions"] if any(x in i["description"] for x in ("down_weight", "begin(partials")) or (
                                         i["lifetime"]["begin"] > 110000 and ("x.to_dm" in i["description"] or "begin(x.view())" in i["description"]))]
        records[label] = record
    return records


if __name__ == "__main__":
    data = analyze()
    (HERE / "static-summary.json").write_text(json.dumps(data, indent=2) + "\n")
    for name, value in data.items():
        print(name, value["makespan_cycles"], value["context_interval_union_cycles"], value["sram_physical_lifetime_union"]["peak_bytes"])
