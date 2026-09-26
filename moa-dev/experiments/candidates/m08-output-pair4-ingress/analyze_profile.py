#!/usr/bin/env python3
"""Validate and compare M08 O trace windows, separately from whole-O timings.

No compile, packaging or Arena calls. Missing execution is status=not_run.
All event subtraction uses one launch's cluster-local Task clock. Cross-launch
comparisons subtract durations/relative windows, never absolute device clocks.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
QO = HERE.parent / "e104-qkv-output-profile"
KERNEL = "sliding_attention_output"
spec = importlib.util.spec_from_file_location("m08_e104_o_analysis", QO / "analyze.py")
REFERENCE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REFERENCE)
CONFIG = REFERENCE.CONFIG[KERNEL]
SELECTED = CONFIG["selected"]


def digest(path):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def assert_digest(path, expected):
    actual = digest(path)
    assert all(actual[key] == expected[key] for key in ("bytes", "sha256")), str(path)
    return actual


def load(path):
    return json.loads(path.read_text())


def stats(values):
    assert values
    return {"count": len(values), "median": statistics.median(values),
            "range": [min(values), max(values)]}


def make_mapping(schedule_path, metadata_path):
    schedule, metadata = load(schedule_path), load(metadata_path)
    assert len(metadata["spans"]) == 1
    spans = sorted(metadata["spans"][0], key=lambda row: row["begin_marker"])
    assert len({row["label"] for row in spans}) == len(spans)
    markers = [row for row in spans if row["name"] not in REFERENCE.WRAPPERS]
    operations = [node for node in schedule["instructions"] if REFERENCE.operation_kind(node)]
    assert len(markers) == len(operations) == 22
    assert [REFERENCE.operation_kind(node) for node in operations] == [row["name"] for row in markers]
    nodes = {node["index"]: node for node in schedule["instructions"]}
    assert {node["index"] for node in operations} == set(SELECTED.values())
    for prerequisite in ("16", "13", "21"):
        assert nodes[prerequisite]["output_tensors"][0] in nodes["26"]["input_tensors"]
    assert nodes["26"]["output_tensors"][0] in nodes["47"]["input_tensors"]
    assert nodes["47"]["output_tensors"][0] in nodes["54"]["input_tensors"]
    assert nodes["54"]["description"].endswith("result.to_dm(&mut device.tdma)")
    assert nodes["54"]["output_tensors"][0] in nodes["58"]["input_tensors"]
    assert nodes["58"]["output_tensors"][0] in nodes["62"]["input_tensors"]
    task, = [row for row in spans if row["name"] == "Task"]
    names = {index: name for name, index in SELECTED.items()}
    return {"source_schedule": digest(schedule_path), "source_metadata": digest(metadata_path),
            "compiler_makespan": max(n["lifetime"]["end"] for n in schedule["instructions"]),
            "task_label": task["label"], "spans": spans, "mapped_operation_count": 22,
            "operations": [{"marker": marker, "schedule": node, "semantic_name": names[node["index"]]}
                           for marker, node in zip(markers, operations)]}


def cycles_of(stage):
    # Legacy shape is useful for checking this validator against frozen E104 data.
    cycles = stage["cycles"]
    if isinstance(cycles, list):
        assert len(cycles) == 1
        cycles = cycles[0]
    assert len(cycles["values"]) == 3
    assert cycles["median"] == statistics.median(cycles["values"])
    return cycles


def validate_trace(stage, mapping, job, side):
    assert stage["complete"] and stage["correctness_passes"] == 3
    if "valid" in stage:
        assert stage["valid"] is True
    if "exit_code" in stage:
        assert stage["exit_code"] == 0
    cycles = cycles_of(stage)
    spans = stage["spans"]
    assert {(s["run"], s["cluster"]) for s in spans} == {(r, c) for r in range(3) for c in range(2)}
    assert all(s["kernel"] == KERNEL for s in spans)
    labels = {s["label"]: s for s in mapping["spans"]}
    operations = mapping["operations"]
    op_labels = [entry["marker"]["label"] for entry in operations]
    observations, task_maxima = [], []
    for run in range(3):
        task_cycles = []
        for cluster in range(2):
            samples = [s for s in spans if s["run"] == run and s["cluster"] == cluster]
            by_label = {s["name"]: s for s in samples}
            assert len(samples) == len(by_label), (side, run, cluster, "duplicate marker")
            assert set(by_label).issubset(labels), (side, run, cluster, "unknown marker")
            assert set(by_label) & set(op_labels) == set(op_labels), (side, run, cluster, "missing engine marker")
            ordered = [by_label[label] for label in op_labels]
            assert [s["begin"] for s in ordered] == sorted(s["begin"] for s in ordered), (side, run, cluster, "issue order")
            task = by_label[mapping["task_label"]]
            assert all(task["begin"] <= s["begin"] <= s["end"] <= task["end"] for s in samples)
            rows = {entry["schedule"]["index"]: by_label[entry["marker"]["label"]] for entry in operations}
            dependencies = REFERENCE.dependency_observation(KERNEL, rows, task)
            dependencies["windows"].update({
                "weight_begin_from_task": rows["16"]["begin"] - task["begin"],
                "weight_end_from_task": rows["16"]["end"] - task["begin"],
                "dot_end_from_task": rows["26"]["end"] - task["begin"],
                "gather_begin_from_task": rows["54"]["begin"] - task["begin"],
                "gather_end_from_task": rows["54"]["end"] - task["begin"],
                "channel_scale_begin_from_task": rows["62"]["begin"] - task["begin"],
            })
            relative = [{"schedule_index": entry["schedule"]["index"],
                         "marker_label": entry["marker"]["label"],
                         "kind": entry["marker"]["name"],
                         "begin_from_task": by_label[entry["marker"]["label"]]["begin"] - task["begin"],
                         "end_from_task": by_label[entry["marker"]["label"]]["end"] - task["begin"],
                         "duration": by_label[entry["marker"]["label"]]["end"] - by_label[entry["marker"]["label"]]["begin"]}
                        for entry in operations]
            duration = task["end"] - task["begin"]
            task_cycles.append(duration)
            observations.append({"job": job, "side": side, "run": run, "cluster": cluster,
                                 "task_cycles": duration, "observed_engine_count": 22,
                                 **dependencies, "operations": relative,
                                 "unassigned_wrappers": [{"label": s["name"], "kind": labels[s["name"]]["name"],
                                     "begin_from_task": s["begin"] - task["begin"],
                                     "end_from_task": s["end"] - task["begin"]}
                                    for s in samples if s["name"] not in op_labels]})
        task_maxima.append(max(task_cycles))
    assert task_maxima == cycles["values"], (side, "Task max-cluster durations disagree with kernel cycles")
    if "trace_observations" in stage:
        checked = [{key: row[key] for key in ("run", "cluster", "task_cycles", "operations")} for row in observations]
        assert checked == stage["trace_observations"], "Independent spans analysis differs from screen observations"
        assert stage["task_marker_matches_reported_cycles"] is True
    return {"profile_cycles": cycles, "task_marker_max_clusters_matches_reported_cycles": True,
            "observations": observations,
            "window_summary": {name: {"all_run_clusters": stats([o["windows"][name] for o in observations]),
                                     "by_cluster": {str(c): stats([o["windows"][name] for o in observations if o["cluster"] == c]) for c in range(2)}}
                               for name in observations[0]["windows"]},
            "task_cycle_summary": stats([o["task_cycles"] for o in observations])}


def whole_summary(stages):
    whole = {}
    for side in ("control_before", "candidate_whole", "control_after"):
        stage = stages[side]
        assert stage["valid"] and stage["correctness_passes"] == 15
        assert all(k["correctness"] == "passed" for k in stage["kernels"].values())
        output = stage["kernels"][KERNEL]
        assert len(output["cycles_by_run"]) == 3 and output["output_comparisons_passed"] == 3
        assert statistics.median(output["cycles_by_run"]) == output["median_cycles"]
        whole[side] = output
    bracket = statistics.median([whole[side]["median_cycles"] for side in ("control_before", "control_after")])
    candidate = whole["candidate_whole"]["median_cycles"]
    return {"stages": whole, "control_bracket_median_cycles": bracket,
            "candidate_minus_control_cycles": candidate-bracket,
            "candidate_over_control_ratio": candidate/bracket,
            "candidate_change_percent": (candidate/bracket-1)*100,
            "scope": "Standard benchmark host complete O (PROFILE=info); three inputs in one job, not an official score or repeated validation."}


def compare_profiles(control, candidate):
    pairs = []
    for a, b in zip(control["observations"], candidate["observations"]):
        assert (a["run"], a["cluster"]) == (b["run"], b["cluster"])
        pairs.append({"run": a["run"], "cluster": a["cluster"],
                      "task_cycles_delta": b["task_cycles"]-a["task_cycles"],
                      "window_deltas": {name: b["windows"][name]-a["windows"][name] for name in a["windows"]}})
    return {"scope": "Candidate-minus-control relative-window differences from separate launches at the same run/cluster index; no shared absolute clock is assumed.",
            "run_cluster_differences": pairs,
            "task_cycle_delta_summary": stats([p["task_cycles_delta"] for p in pairs]),
            "window_delta_summary": {name: {"all_run_clusters": stats([p["window_deltas"][name] for p in pairs]),
                                            "by_cluster": {str(c): stats([p["window_deltas"][name] for p in pairs if p["cluster"] == c]) for c in range(2)}}
                                     for name in pairs[0]["window_deltas"]}}


def build_analysis(label, result_paths):
    compiled_path = HERE / f"build-{label}-repro.json"
    compiled = load(compiled_path)
    assert compiled["exact_compile_exit_code"] == 0 and compiled["target_kernel"] == KERNEL
    assert compiled["source_inputs_verified_in_container"]
    candidate_schedule = HERE / "evidence" / label / "schedule.json"
    assert_digest(candidate_schedule, compiled["schedule"])
    control_schedule = QO / "evidence/profile-v1/sliding_attention_output.json"
    control_build_path = QO / "build-profile-v1.json"
    control_build = load(control_build_path)
    assert_digest(control_schedule, control_build["dumps"][control_schedule.name])
    output = {"status": "not_run", "label": label,
              "scope": "O-only diagnostic trace comparison plus separate standard-host complete-O summary (PROFILE=info)",
              "sources": {"compile_record": digest(compiled_path), "control_build": digest(control_build_path),
                          "candidate_schedule": digest(candidate_schedule), "control_schedule": digest(control_schedule),
                          "shared_e104_analyzer": digest(QO / "analyze.py"),
                          "analyzer": digest(Path(__file__).resolve())},
              "mapping_method": "Complete ordered native marker-kind / scheduled operation-kind correspondence, all22 operations in every run/cluster, weight/table/TRF→dot and dot→reshape→gather→scale lineage checks.",
              "limitations": [
                  "Marker intervals can include dependency and queue waits; they are not HBM bandwidth or DM-bank counters.",
                  "Every event subtraction stays within one launch and one cluster's Task clock. Across launches only relative windows or durations are compared.",
                  "Overlapping durations and their medians are never summed into predicted savings.",
                  "Compiler cycles have no fixed conversion to hardware cycles. Diagnostic hosts/profile modes differ from the whole benchmark host.",
                  "All three runs and both clusters are retained. Six observations are not six independent benchmark allocations.",
                  "Whole-kernel summary is separate from profile windows; neither is an official competition score."],
              "jobs": {}}
    maps = {"status": "not_run", "label": label, "sides": {}}
    package_path = HERE / f"package-{label}.json"
    if not package_path.exists():
        assert not result_paths, "Execution results exist without their package record"
        output["pending"] = "Verified profile package and execution results"
        return output, maps
    package = load(package_path)
    assert package["status"] == "ready"
    assert package["task_bytes_unchanged"] and package["profile_name_bytes_only"]
    assert package["control_diagnostic_reuse"]["exact_o_matches_e122"]
    assert_digest(compiled_path, package["compile_record"])
    output["sources"]["package_record"] = digest(package_path)
    diagnostic = ROOT / "moa-dev/experiments/artifacts" / HERE.name / label / "diagnostic"

    def frozen_diagnostic(name):
        frozen = package.get("frozen_files", {}).get(name)
        path = ROOT / frozen["path"] if frozen else diagnostic / name
        assert_digest(path, package["diagnostic_files"][name])
        if frozen:
            assert_digest(path, frozen)
        return path

    for side, schedule in (("control", control_schedule), ("candidate", candidate_schedule)):
        metadata_path = frozen_diagnostic(f"{side}-o.image-profile.json")
        mapping = make_mapping(schedule, metadata_path)
        map_path = frozen_diagnostic(f"{side}-marker-map.json")
        saved = load(map_path)
        assert saved["task_label"] == mapping["task_label"] and saved["spans"] == mapping["spans"]
        assert saved["operations"] == [{"marker": row["marker"], "schedule": row["schedule"]} for row in mapping["operations"]]
        maps["sides"][side] = mapping
    maps["status"] = "mapped_not_run"
    if not result_paths:
        output["pending"] = "Collected successful screen results"
        return output, maps
    for result_path in result_paths:
        report = load(result_path)
        assert report["actual_upload_matches"] and report["screen_complete"]
        assert report["valid"] and report["stage_sequence_matches"] and report["correctness_passes"] == 57
        assert report["job"]["status"] in {"SUCCEEDED", "COMPLETED"} and report["job"]["exit_code"] == 0
        screen_path = ROOT / report["build_record"]["path"]
        assert_digest(screen_path, report["build_record"])
        screen = load(screen_path)
        assert screen["build_label"] == label
        assert_digest(package_path, screen["candidate_package"])
        assert {k: v for k, v in report["package_files"].items() if k != "job.yaml"} == screen["package_files"]
        assert_digest(ROOT / report["log"], report["log_digest"])
        job = report["job"]["id"]
        assert str(job) not in output["jobs"], "Duplicate job input"
        stages = report["stages"]
        sides = {}
        for side in ("control", "candidate"):
            info = stages[f"{side}_info"]
            assert info["valid"] and info["complete"] and info["correctness_passes"] == 3
            sides[side] = validate_trace(stages[f"{side}_trace"], maps["sides"][side], job, side)
            sides[side]["profile_info_cycles"] = cycles_of(info)
        output["jobs"][str(job)] = {"source_result": digest(result_path), "sides": sides,
                                    "profile_comparison": compare_profiles(sides["control"], sides["candidate"]),
                                    "whole_o_summary": whole_summary(stages)}
        assert output["jobs"][str(job)]["whole_o_summary"]["candidate_change_percent"] == report["whole_o_comparison"]["candidate_change_percent"]
    output["status"] = maps["status"] = "analyzed"
    return output, maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", nargs="?", default="m08-pair4-v1")
    parser.add_argument("--results", type=Path, nargs="*", help="Explicit collected result JSONs; default: this candidate's *-job-*-results.json")
    parser.add_argument("--check", action="store_true", help="Check saved analysis without rewriting")
    args = parser.parse_args()
    paths = args.results if args.results is not None else sorted(HERE.glob("*-job-*-results.json"))
    output, maps = build_analysis(args.label, paths)
    for name, value in ((f"profile-analysis-{args.label}.json", output), (f"marker-source-map-{args.label}.json", maps)):
        path = HERE / name
        if args.check:
            assert load(path) == value, f"Regenerate {path}"
        else:
            path.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "jobs": list(output["jobs"]),
                      "mapped_sides": list(maps["sides"])}, indent=2))


if __name__ == "__main__":
    main()
