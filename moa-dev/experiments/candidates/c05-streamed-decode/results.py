#!/usr/bin/env python3
"""Reconstruct performance reports from three byte-identical screening jobs."""
import argparse
import json
import statistics
from build import HERE, common

ROOT = common.ROOT


def results(check=False):
    first = json.loads((HERE / "c05-screen-first-results.json").read_text())
    reruns = json.loads((HERE / "repeats-submission.json").read_text())["reruns"]
    batches = [first, *[json.loads((HERE / f"c05-screen-first-job-{r['job_id']}-results.json").read_text()) for r in reruns]]
    assert len(batches) == 3 and len({b["job"]["id"] for b in batches}) == 3
    assert all(b["valid"] and b["package_files"] == first["package_files"] for b in batches)
    summaries = {}
    for ident, stage, suffix in ((73, "candidate_0", "direct-repeats"), (74, "candidate_1", "partials-repeats"), (75, "control_before", "control-repeats")):
        runs = []
        for batch in batches:
            run = {k: batch[k] for k in ("job", "log", "log_sha256", "package_files")}
            run.update(valid=True, kernels=batch["stages"][stage]["kernels"], batch_stage=stage, executed_host=stage + "_test_runtime")
            run["benchmark_files"] = {name: stage + "_" + name for name in ("test_runtime", "fixtures.safetensors", "remote_entrypoint.sh")}
            runs.append(run)
        aggregate = {}
        for name in runs[0]["kernels"]:
            values = [r["kernels"][name]["median_cycles"] for r in runs]
            median = statistics.median(values)
            aggregate[name] = {"median_cycles": median, "job_medians": values, "min_cycles": min(values), "max_cycles": max(values), "relative_range": (max(values) - min(values)) / median}
        report = {"schema_version": 1, "valid": True, "label": f"e{ident}-c05-{suffix}", "runs": runs,
                  "method": "Three independent Arena allocations of one immutable batch package. Each reported stage runs the unchanged official basic host/fixture/runner. Stage medians within a job are not treated as independent repetitions.",
                  "aggregate": aggregate, "independent_arena_jobs": 3,
                  "control_after_job_medians": [b["stages"]["control_after"]["kernels"]["decoder_feedforward"]["median_cycles"] for b in batches]}
        path = ROOT / f"docs/experiments/e{ident}-c05-{suffix}.json"
        if check:
            assert json.loads(path.read_text()) == report
        else:
            path.write_text(json.dumps(report, indent=2) + "\n")
        summaries[f"E{ident}"] = aggregate["decoder_feedforward"]
    for ident in ("E73", "E74"):
        summaries[ident]["change_percent"] = 100 * (summaries[ident]["median_cycles"] / summaries["E75"]["median_cycles"] - 1)
    summary = {"jobs": [b["job"]["id"] for b in batches], "performance": summaries,
               "basic_comparisons_in_repeats": 3 * 4 * 15, "initial_direct_screen_comparisons": 15,
               "control_after_ffn": [b["stages"]["control_after"]["kernels"]["decoder_feedforward"]["median_cycles"] for b in batches],
               "independent_jobs": 3, "notes": "Only control_before is used as the catalog reference. The after controls are reported separately, never counted as six independent jobs."}
    path = HERE / "performance-summary.json"
    if check:
        assert json.loads(path.read_text()) == summary
    else:
        path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    results(parser.parse_args().check)
