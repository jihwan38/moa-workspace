#!/usr/bin/env python3
"""Compare compiler schedules; these are estimates, not NPU attribution."""
import importlib.util
import json
from build import HERE, common

source = HERE.parent / "ffn-up-tiled-fused-decode-e47/analyze_schedule.py"
spec = importlib.util.spec_from_file_location("c05_schedule_profile", source)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def analyze():
    paths = {"e55": common.ROOT / "moa-dev/experiments/artifacts/c01-scale-dpe-e55/e55-control-schedule.json",
             "direct": common.ARTIFACTS / "c05-direct-first/schedule.json",
             "partials": common.ARTIFACTS / "c05-partials-transpose2/schedule.json"}
    records = {}
    for label, path in paths.items():
        data, record = profile.profile(path)
        record.update(path=str(path.relative_to(common.ROOT)), sha256=common.digest(path.read_bytes())["sha256"])
        record["main_operations"] = [{k: i[k] for k in ("index", "lifetime", "description", "input_tensors", "output_tensors")}
                                     for i in data["instructions"] if i["tpe"] == "Main"]
        record["full_bf16_weight_tensors"] = [t for t in data["tensors"] if t["buffer_type"] == "Sram" and (
            (t["shape"].endswith("Element: (L % 32, H)]") and t["size"] == 32 * 3840 * 2 * 512) or
            (t["shape"].endswith("Element: (H % 60, L % 1920)]") and t["size"] == 60 * 1920 * 2 * 512))]
        assert bool(record["full_bf16_weight_tensors"]) == (label == "e55")
        records[label] = record
    return records


if __name__ == "__main__":
    data = analyze()
    (HERE / "static-summary.json").write_text(json.dumps(data, indent=2) + "\n")
    for name, value in data.items():
        print(name, value["makespan_cycles"], value["context_interval_union_cycles"], value["sram_physical_lifetime_union"]["peak_bytes"])
