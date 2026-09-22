#!/usr/bin/env python3
"""Extract valid stages; never use cycles from a failed correctness stage."""
import argparse
import json
import os
import statistics
from build import HERE, common

# Stable IDs describe built candidates, not every compile refinement.
GROUPS = {
    "first": [(86, "pair-row2", "c07-up-pair-v1")],
    "time": [(87, "time-row2", "c07-up-time-v3")],
    "contiguous": [(88, "time-contiguous", "c07-up-time-contig-v3")],
    "full": [(89, "full-pair-row2", "c07-full-pair-v5"), (90, "full-time-contiguous", "c07-full-time-contig-v2"), (91, "pair-contiguous", "c07-up-pair-contig-v2")],
    "precision": [(94, "up-f32", "c07-up-f32-v1"), (95, "full-f32", "c07-full-f32-v1"), (96, "e5-dpe", "c07-up-e5-v1"), (97, "e5-f32", "c07-up-e5-f32-v1")],
    "late": [(92, "late-up", "c07-up-late64-v1"), (93, "full-time4-late", "c07-full-time4-v1")],
    "two-columns": [(98, "two-column-time", "c07-full-time2-v2"), (99, "two-column-pair", "c07-full-pair2-v2")],
    "compensated": [(100, "compensated-dpe", "c07-up-e5-comp-v2")],
    "e5-paths": [(101, "e5-f32-late", "c07-up-e5-f32-late-v2"), (102, "e5-materialized", "c07-up-e5-materialized-v1")],
    "reload": [(103, "e5-reload", "c07-up-e5-reload-v3"), (104, "e5-reload-materialized", "c07-up-e5-reload-materialized-v3")],
}


def write(path, value, check):
    if check:
        assert json.loads(path.read_text()) == value, path
    else:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def report(batches, stage, label):
    assert len({b["job"]["id"] for b in batches}) == len(batches)
    assert all(b["package_files"] == batches[0]["package_files"] for b in batches)
    assert all(b["stages"]["control_before"]["valid"] and b["stages"]["control_after"]["valid"] for b in batches)
    valid = all(b["stages"][stage]["valid"] for b in batches)
    runs = []
    for b in batches:
        run = {k: b[k] for k in ("job", "log", "log_sha256", "package_files")}
        run.update(b["stages"][stage], batch_stage=stage,
                   executed_host=stage + "_test_runtime",
                   benchmark_files={n: stage + "_" + n for n in ("test_runtime", "fixtures.safetensors", "remote_entrypoint.sh")})
        runs.append(run)
    value = dict(schema_version=1, label=label, valid=valid, runs=runs,
                 independent_arena_jobs=len(batches),
                 method="Unchanged benchmark process in an upload-verified serial batch. Only distinct allocations count as repetitions. A valid stage may coexist with a different failed stage; failed-stage cycles are excluded.")
    if valid:
        aggregate = {}
        for kernel in runs[0]["kernels"]:
            values = [r["kernels"][kernel]["median_cycles"] for r in runs]
            median = statistics.median(values)
            aggregate[kernel] = dict(median_cycles=median, job_medians=values,
                                    min_cycles=min(values), max_cycles=max(values),
                                    relative_range=(max(values)-min(values))/median)
        value["aggregate"] = aggregate
        value["control_after_job_medians"] = [b["stages"]["control_after"]["kernels"]["decoder_feedforward"]["median_cycles"] for b in batches]
    return value


def results(check=False):
    summary = {}
    for group, candidates in GROUPS.items():
        label = "c07-screen-" + group
        original = HERE / (label + "-results.json")
        if not original.exists():
            continue
        paths = [original, *sorted(HERE.glob(label + "-job-*-results.json"))]
        batches = [json.loads(p.read_text()) for p in paths]
        build = json.loads((HERE / (label + "-build.json")).read_text())
        control_name = "c07-" + group + "-e79-control"
        reference = "docs/experiments/" + control_name + ".json"
        control = report(batches, "control_before", control_name)
        write(common.ROOT / reference, control, check)
        for i, (ident, variant, candidate) in enumerate(candidates):
            stage = "candidate_" + str(i)
            assert next(s for s in build["stages"] if s["name"] == stage)["build"] == candidate
            name = f"e{ident}-c07-{variant}"
            data = report(batches, stage, name)
            data.update(candidate_build=str((HERE / f"build-{candidate}-repro.json").relative_to(common.ROOT)), reference=reference)
            path = "docs/experiments/" + name + ".json"
            write(common.ROOT / path, data, check)
            metric = dict(report=path, reference=reference, candidate_build=data["candidate_build"],
                          valid=data["valid"], independent_jobs=len(batches), jobs=[b["job"]["id"] for b in batches])
            if data["valid"]:
                metric.update(data["aggregate"]["decoder_feedforward"])
                metric["control_median_cycles"] = control["aggregate"]["decoder_feedforward"]["median_cycles"]
                metric["change_percent"] = 100 * (metric["median_cycles"] / metric["control_median_cycles"] - 1)
            summary[f"E{ident}"] = metric
    value = dict(performance=summary, note="Basic benchmark stages only. Separate diagnostic failures and mathematical precision limits must be considered before adoption. Single allocations are screens, not established speedups.")
    write(HERE / "performance-summary.json", value, check)
    table = ["# C07 기본 입력 측정", "",
             "동일 package를 서로 다른 Arena allocation에서 실행한 경우만 반복으로 센다. 실패 stage의 cycles는 제외한다. "
             "[최종 판정](decision.json)에서 별도 정확도 진단과 채택 여부를 함께 확인한다.", "",
             "| ID | 구조 | E79 대조 → FFN cycles | 변화 | 독립 jobs |",
             "|---|---|---:|---:|---|"]
    variants = {f"E{ident}": variant for group in GROUPS.values() for ident, variant, _ in group}
    for ident, data in sorted(summary.items(), key=lambda x: int(x[0][1:])):
        cycles = f"{data['control_median_cycles']:,} → {data['median_cycles']:,}" if data["valid"] else "정확도/실행 실패"
        change = f"{data['change_percent']:+.2f}%" if data["valid"] else "—"
        link = os.path.relpath(common.ROOT / data["report"], HERE)
        jobs = ", ".join(map(str, data["jobs"]))
        table.append(f"| [{ident}]({link}) | {variants[ident]} | {cycles} | {change} | {jobs} |")
    table_text = "\n".join(table) + "\n"
    if check:
        assert (HERE / "measurements.md").read_text() == table_text
    else:
        (HERE / "measurements.md").write_text(table_text)
    for ident, data in sorted(summary.items(), key=lambda x: int(x[0][1:])):
        print(ident, data.get("median_cycles", "INVALID"), round(data.get("change_percent", 0), 3), data["independent_jobs"])
    return value


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    results(p.parse_args().check)
