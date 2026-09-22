#!/usr/bin/env python3
"""Extract two bounded QKV architecture screens, excluding failed-stage cycles."""
import argparse
import json
import statistics
from build import HERE, common

GROUPS = [(105, "fused", "c08-fused-contiguous-v2"),
          (106, "columns8", "c08-columns8-v1")]


def report(batch, stage):
    assert batch["stages"]["control_before"]["valid"]
    assert batch["stages"]["control_after"]["valid"]
    run = {k: batch[k] for k in ("job", "log", "log_sha256", "package_files")}
    run.update(batch["stages"][stage], batch_stage=stage,
               executed_host=stage + "_test_runtime",
               benchmark_files={n: stage + "_" + n for n in ("test_runtime", "fixtures.safetensors", "remote_entrypoint.sh")})
    value = dict(schema_version=1, valid=run["valid"], runs=[run], independent_arena_jobs=1,
                 method="One upload-verified Arena allocation; unchanged benchmark hosts with E104 before and after. A failed candidate has no performance aggregate.")
    if run["valid"]:
        value["aggregate"] = {k: {"median_cycles": v["median_cycles"], "job_medians": [v["median_cycles"]]}
                              for k, v in run["kernels"].items()}
    return value


def write(path, value, check):
    if check:
        assert json.loads(path.read_text()) == value, path
    else:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def results(check):
    summary = {"scope": "Single-allocation basic screens, not adopted speedups or official scores.", "candidates": {}}
    for number, suffix, build in GROUPS:
        path = HERE / f"c08-screen-{suffix}-results.json"
        if not path.exists():
            continue
        batch = json.loads(path.read_text())
        before, after = (batch["stages"][s]["kernels"]["sliding_project_qkv"]["median_cycles"]
                         for s in ("control_before", "control_after"))
        reference = f"docs/experiments/c08-{suffix}-e104-control.json"
        write(common.ROOT / reference, report(batch, "control_before"), check)
        candidate = report(batch, "candidate_0")
        candidate.update(candidate_build=str((HERE / f"build-{build}-repro.json").relative_to(common.ROOT)), reference=reference,
                         control_after_qkv_cycles=after)
        candidate_path = f"docs/experiments/e{number}-c08-{suffix}.json"
        write(common.ROOT / candidate_path, candidate, check)
        record = dict(valid=candidate["valid"], report=candidate_path, job=batch["job"]["id"],
                      control_before=before, control_after=after)
        if candidate["valid"]:
            cycles = candidate["aggregate"]["sliding_project_qkv"]["median_cycles"]
            record.update(qkv_cycles=cycles, change_percent_vs_before=100 * (cycles / before - 1),
                          change_percent_vs_bracketing_median=100 * (cycles / statistics.median([before, after]) - 1))
        summary["candidates"][f"E{number}"] = record
    write(HERE / "performance-summary.json", summary, check)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    results(parser.parse_args().check)
