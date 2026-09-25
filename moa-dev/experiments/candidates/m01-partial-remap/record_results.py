#!/usr/bin/env python3
"""Extract both M01 routes from one immutable, bracketing-control Arena screen."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LABEL = "m01-remap-screen"


def write(path, data):
    with path.open("x") as out:
        json.dump(data, out, indent=2)
        out.write("\n")


def main():
    spec = importlib.util.spec_from_file_location("structural_results", ROOT / "moa-dev/experiments/structural_results.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    batch = json.loads((HERE / f"{LABEL}-results.json").read_text())
    manifest = json.loads((HERE / f"{LABEL}-build.json").read_text())
    before, after = (common.report(batch, key) for key in ("control_before", "control_after"))
    before_path = f"docs/experiments/{LABEL}-e104-before.json"
    after_path = f"docs/experiments/{LABEL}-e104-after.json"
    write(ROOT / before_path, before)
    write(ROOT / after_path, after)
    summaries = []
    for stage, experiment in (("candidate_0", "e112-m01-partial-dma"), ("candidate_1", "e113-m01-partial-switch")):
        result = common.report(batch, stage)
        label = next(s["build"] for s in manifest["stages"] if s["name"] == stage)
        build_path = HERE / f"build-{label}-repro.json"
        build = json.loads(build_path.read_text())
        target = build["target_kernel"]
        result.update(reference=before_path, control_after_report=after_path,
                      candidate_build=str(build_path.relative_to(ROOT)), target_kernel=target,
                      variant=build["variant"], candidate_family="M01")
        if result["valid"]:
            c, b, a = (x["aggregate"][target]["median_cycles"] for x in (result, before, after))
            result["screen_change_percent"] = {"vs_before": 100*(c/b-1), "vs_after": 100*(c/a-1),
                                               "vs_bracketing_median": 100*(c/statistics.median([b,a])-1)}
        write(ROOT / f"docs/experiments/{experiment}.json", result)
        summaries.append({"experiment": experiment, "variant": result["variant"], "valid": result["valid"],
                          "cycles": result.get("aggregate", {}).get(target),
                          "change": result.get("screen_change_percent")})
    raw = (ROOT / batch["log"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == batch["log_sha256"]
    (HERE / "evidence" / f"arena-{batch['job']['id']}.log").write_bytes(raw)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
