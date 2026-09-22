#!/usr/bin/env python3
"""Reconstruct C04 reports without pooling different upload packages."""
import argparse
import json
import statistics
from build import HERE, common

GROUPS = {
    "first": [(76, "four-ring16", "candidate_0"), (77, "four-ring32", "candidate_1")],
    "second": [(78, "two-ring16", "candidate_0"), (80, "four-ring16-sub", "candidate_1"), (79, "eight-ring32", "candidate_2")],
    "third": [(81, "four-row30-ring16", "candidate_0"), (82, "four-row30-ring8", "candidate_1")],
    "fourth": [(83, "four-row30-gather-direct", "candidate_0"), (84, "eight-gather-direct", "candidate_1")],
}


def write(path, data, check):
    if check:
        assert json.loads(path.read_text()) == data, path
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def stage_report(batches, stage, label):
    first = batches[0]
    assert all(b["valid"] and b["package_files"] == first["package_files"] for b in batches)
    assert len({b["job"]["id"] for b in batches}) == len(batches)
    runs = []
    for b in batches:
        value = {k: b[k] for k in ("job", "log", "log_sha256", "package_files")}
        value.update(valid=True, kernels=b["stages"][stage]["kernels"], batch_stage=stage,
                     executed_host=stage + "_test_runtime",
                     benchmark_files={n: stage + "_" + n for n in ("test_runtime", "fixtures.safetensors", "remote_entrypoint.sh")})
        runs.append(value)
    aggregate = {}
    for kernel in first["stages"][stage]["kernels"]:
        values = [r["kernels"][kernel]["median_cycles"] for r in runs]
        median = statistics.median(values)
        aggregate[kernel] = dict(median_cycles=median, job_medians=values,
            min_cycles=min(values), max_cycles=max(values), relative_range=(max(values)-min(values))/median)
    return dict(schema_version=1, valid=True, label=label, runs=runs, aggregate=aggregate,
        independent_arena_jobs=len(batches),
        method="Stage of an immutable batch: unchanged basic host, fixture and runner. Job medians are independent only across distinct Arena allocations; stages in one job are not repetitions.",
        control_after_job_medians=[b["stages"]["control_after"]["kernels"]["decoder_feedforward"]["median_cycles"] for b in batches])


def results(check=False):
    summaries = {}
    total = 0
    for group, candidates in GROUPS.items():
        label = "c04-screen-" + group
        batches = [json.loads((HERE / (label + "-results.json")).read_text())]
        if group == "second":
            for repeat in (1, 2):
                job = json.loads((HERE / f"repeat-second-{repeat}-submission.json").read_text())["job_id"]
                batches.append(json.loads((HERE / f"{label}-job-{job}-results.json").read_text()))
        assert len(batches) == (3 if group == "second" else 1)
        total += len(batches) * (len(candidates) + 3) * 15
        control_name = "e85-c04-e74-control" if group == "second" else f"c04-{group}-e74-control"
        control = stage_report(batches, "control_before", control_name)
        reference = "docs/experiments/" + control_name + ".json"
        write(common.ROOT / reference, control, check)
        for ident, variant, stage in candidates:
            name = f"e{ident}-c04-{variant}"
            report = stage_report(batches, stage, name)
            report.update(candidate_build=f"moa-dev/experiments/candidates/c04-down-transfer/build-c04-{variant}-v1-repro.json",
                          reference=reference)
            write(common.ROOT / ("docs/experiments/" + name + ".json"), report, check)
            metric = dict(report["aggregate"]["decoder_feedforward"])
            metric.update(variant=variant, independent_jobs=len(batches),
                          control_median_cycles=control["aggregate"]["decoder_feedforward"]["median_cycles"],
                          reference=reference, jobs=[b["job"]["id"] for b in batches])
            metric["change_percent"] = 100 * (metric["median_cycles"] / metric["control_median_cycles"] - 1)
            summaries[f"E{ident}"] = metric
    summary = dict(performance=summaries, basic_comparisons_in_batches=total,
        note="E78/E79/E80 and their E85 control have three independent repeats. Other candidates are single-job screens, not validated speedups. E55 reference and E74 after-control observations remain in each raw batch report.")
    write(HERE / "performance-summary.json", summary, check)
    for ident, value in sorted(summaries.items()):
        print(ident, value["median_cycles"], round(value["change_percent"], 3), value["independent_jobs"])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    results(parser.parse_args().check)
