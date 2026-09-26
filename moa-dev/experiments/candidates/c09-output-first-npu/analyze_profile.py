#!/usr/bin/env python3
"""Validate frozen C09's first hardware trace, separately from whole-O timings.

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
spec = importlib.util.spec_from_file_location("c09_e104_o_analysis", QO / "analyze.py")
REFERENCE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REFERENCE)
SELECTED = {
    "control": REFERENCE.CONFIG[KERNEL]["selected"],
    "candidate": {
        "stage0_table_constant_dma": "5", "stage0_table_prepare": "24",
        "input_hbm": "25", "stage0_weight": "27", "input_trf": "32",
        "stage0_dot_reduce_commit": "37", "stage1_table_constant_dma": "38",
        "stage1_table_prepare": "41", "stage1_weight": "42", "stage1_dot_reduce_commit": "53",
        "channel_scale_load": "54", "channel_scale_vrf": "58",
        "post_rms_weight_load": "59", "post_rms_weight_vrf": "65",
        "residual_load": "66", "residual_vrf": "72", "f32_result_head_gather": "81",
        "channel_scale_bf16": "89", "rms_square": "94", "rms_sum": "99", "rms_sqrt": "104",
        "rms_scalar_transfer": "107", "rms_scalar_vrf": "111", "rms_normalize": "115",
        "residual_add": "120", "output_store": "123",
    },
}
HISTORICAL = HERE.parent / "c09-output-pipeline"
HISTORICAL_IMAGE_SHA256 = "fe90ed758c27fe40a11e78ff5132e7b98152fc8cd79208d49c22053672859495"


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


def make_mapping(schedule_path, metadata_path, side):
    schedule, metadata = load(schedule_path), load(metadata_path)
    assert len(metadata["spans"]) == 1
    spans = sorted(metadata["spans"][0], key=lambda row: row["begin_marker"])
    assert len({row["label"] for row in spans}) == len(spans)
    markers = [row for row in spans if row["name"] not in REFERENCE.WRAPPERS]
    operations = [node for node in schedule["instructions"] if REFERENCE.operation_kind(node)]
    expected_count = {"control": 22, "candidate": 26}[side]
    assert len(markers) == len(operations) == expected_count
    assert [REFERENCE.operation_kind(node) for node in operations] == [row["name"] for row in markers]
    nodes = {node["index"]: node for node in schedule["instructions"]}
    assert {node["index"] for node in operations} == set(SELECTED[side].values())
    if side == "control":
        prerequisites = [(p, "26") for p in ("16", "13", "21")]
        edges = [("26", "47"), ("47", "54"), ("54", "58"), ("58", "62")]
        gather = "54"
    else:
        prerequisites = [(p, "37") for p in ("27", "24", "32")]
        prerequisites += [(p, "53") for p in ("42", "41", "32")]
        edges = [("37", "43"), ("43", "49"), ("49", "53"),
                 ("53", "73"), ("73", "74"), ("74", "81"), ("81", "85"), ("85", "89")]
        gather = "81"
        assert ">(0)" in nodes["27"]["description"] and ">(60)" in nodes["42"]["description"]
    for producer, consumer in prerequisites + edges:
        assert nodes[producer]["output_tensors"][0] in nodes[consumer]["input_tensors"], (side, producer, consumer)
    assert nodes[gather]["description"].endswith("result.to_dm(&mut device.tdma)")
    task, = [row for row in spans if row["name"] == "Task"]
    names = {index: name for name, index in SELECTED[side].items()}
    return {"source_schedule": digest(schedule_path), "source_metadata": digest(metadata_path),
            "compiler_makespan": max(n["lifetime"]["end"] for n in schedule["instructions"]),
            "task_label": task["label"], "spans": spans, "mapped_operation_count": expected_count,
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


def dependency_observation(rows, task, side):
    def begin(index):
        return rows[index]["begin"]

    def end(index):
        return rows[index]["end"]

    def duration(index):
        return end(index)-begin(index)

    if side == "control":
        weights, dots, tables = ["16"], ["26"], ["13"]
        input_trf, gather, scale = "21", "54", "62"
        rms_start, rms_end, residual_vrf, residual_add, store = "67", "88", "46", "93", "96"
    else:
        weights, dots, tables = ["27", "42"], ["37", "53"], ["24", "41"]
        input_trf, gather, scale = "32", "81", "89"
        rms_start, rms_end, residual_vrf, residual_add, store = "94", "115", "72", "120", "123"
    first_weight, final_weight = weights[0], weights[-1]
    first_dot, final_dot = dots[0], dots[-1]
    readiness = []
    for i, (weight, dot, table) in enumerate(zip(weights, dots, tables)):
        ready = {"weight": end(weight), "input_trf": end(input_trf), "table_prepare": end(table)}
        if i:
            ready["previous_dot"] = end(dots[i-1])
        assert max(ready.values()) <= begin(dot), (side, i, "prerequisite after dot start")
        readiness.append({"stage": i, "latest_observed_boundary": max(ready, key=ready.get),
                          "boundaries_from_task": {name: value-task["begin"] for name, value in ready.items()},
                          "dot_begin_minus_latest_boundary": begin(dot)-max(ready.values()),
                          "scope": "Observed readiness boundaries; mutable-view Core/Cluster bookkeeping is not separately attributed."})
    windows = {
        "input_ready_from_task": end(input_trf)-task["begin"],
        "input_ready_to_first_dot_begin": begin(first_dot)-end(input_trf),
        "first_table_ready_to_first_dot_begin": begin(first_dot)-end(tables[0]),
        "first_weight_end_to_first_dot_begin": begin(first_dot)-end(first_weight),
        "final_weight_end_to_final_dot_begin": begin(final_dot)-end(final_weight),
        "first_weight_begin_from_task": begin(first_weight)-task["begin"],
        "final_weight_end_from_task": end(final_weight)-task["begin"],
        "first_dot_begin_from_task": begin(first_dot)-task["begin"],
        "final_dot_end_from_task": end(final_dot)-task["begin"],
        "weight_supply_envelope": end(final_weight)-begin(first_weight),
        "projection_first_weight_begin_to_final_dot_end": end(final_dot)-begin(first_weight),
        "projection_first_weight_begin_to_gather_end": end(gather)-begin(first_weight),
        "final_dot_end_to_gather_begin": begin(gather)-end(final_dot),
        "final_dot_end_to_task_end_tail": task["end"]-end(final_dot),
        "f32_head_gather_interval": duration(gather),
        "gather_begin_from_task": begin(gather)-task["begin"],
        "gather_end_from_task": end(gather)-task["begin"],
        "f32_head_gather_end_to_scale_begin": begin(scale)-end(gather),
        "channel_scale_begin_from_task": begin(scale)-task["begin"],
        "scale_begin_to_task_end": task["end"]-begin(scale),
        "rms_begin_to_rms_end": end(rms_end)-begin(rms_start),
        "output_store_end_to_task_end": task["end"]-end(store),
        "residual_vrf_ready_to_residual_add": begin(residual_add)-end(residual_vrf),
        "weight_interval_sum_diagnostic_only": sum(duration(w) for w in weights),
        "dot_interval_sum_diagnostic_only": sum(duration(d) for d in dots),
    }
    for i, (weight, dot, table) in enumerate(zip(weights, dots, tables)):
        windows.update({
            f"stage{i}_weight_interval": duration(weight),
            f"stage{i}_weight_begin_from_task": begin(weight)-task["begin"],
            f"stage{i}_weight_end_from_task": end(weight)-task["begin"],
            f"stage{i}_dot_interval": duration(dot),
            f"stage{i}_dot_begin_from_task": begin(dot)-task["begin"],
            f"stage{i}_dot_end_from_task": end(dot)-task["begin"],
            f"stage{i}_table_prepare_interval": duration(table),
            f"stage{i}_table_ready_to_dot_begin": begin(dot)-end(table),
        })
    if side == "candidate":
        windows.update({
            "stage0_table_dma_interval": duration("5"),
            "stage1_table_dma_interval": duration("38"),
            "first_dot_second_weight_overlap": REFERENCE.overlap(rows["37"], rows["42"]),
            "first_dot_second_table_dma_overlap": REFERENCE.overlap(rows["37"], rows["38"]),
            "first_dot_second_table_prepare_overlap": REFERENCE.overlap(rows["37"], rows["41"]),
            "second_weight_begin_minus_first_weight_end": begin("42")-end("27"),
            "second_weight_begin_minus_first_dot_begin": begin("42")-begin("37"),
            "second_weight_begin_minus_second_table_dma_end": begin("42")-end("38"),
            "first_dot_end_minus_second_table_prepare_end": end("37")-end("41"),
            "second_dot_begin_minus_first_dot_end": begin("53")-end("37"),
        })
    else:
        windows["stage0_table_dma_interval"] = duration("4")
    return {"windows": windows, "stage_readiness": readiness}


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
            dependencies = dependency_observation(rows, task, side)
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
                                 "task_cycles": duration, "observed_engine_count": mapping["mapped_operation_count"],
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
    # Stage0 is all120 rows in control but only the first60 in C09. Do not
    # present a half-stage/full-stage duration delta as an optimization gain.
    comparable = [name for name in control["observations"][0]["windows"]
                  if not name.startswith("stage") and name in candidate["observations"][0]["windows"]]
    pairs = []
    for a, b in zip(control["observations"], candidate["observations"]):
        assert (a["run"], a["cluster"]) == (b["run"], b["cluster"])
        pairs.append({"run": a["run"], "cluster": a["cluster"],
                      "task_cycles_delta": b["task_cycles"]-a["task_cycles"],
                      "window_deltas": {name: b["windows"][name]-a["windows"][name] for name in comparable}})
    return {"scope": "Candidate-minus-control complete-projection boundary/aggregate differences from separate launches at the same run/cluster index. Half-stage versus full-stage durations are not directly compared. Diagnostic interval sums are resource statistics, not latency or savings. No shared absolute clock is assumed.",
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
    historical_build_path = HISTORICAL / "build-c09-two-stage-v1-repro.json"
    historical_build = load(historical_build_path)
    assert compiled["exact_image"] == historical_build["exact_image"]
    assert compiled["exact_image"]["sha256"] == HISTORICAL_IMAGE_SHA256
    assert compiled["source_inputs"]["src/device/sliding/projection.rs"] == historical_build["source_inputs"]["src/device/sliding/projection.rs"]
    candidate_schedule = HERE / "evidence" / label / "schedule.json"
    assert_digest(candidate_schedule, compiled["schedule"])
    control_schedule = QO / "evidence/profile-v1/sliding_attention_output.json"
    control_build_path = QO / "build-profile-v1.json"
    control_build = load(control_build_path)
    assert_digest(control_schedule, control_build["dumps"][control_schedule.name])
    output = {"status": "not_run", "label": label,
              "scope": "O-only diagnostic trace comparison plus separate standard-host complete-O summary (PROFILE=info)",
              "sources": {"compile_record": digest(compiled_path), "control_build": digest(control_build_path),
                          "historical_c09_build": digest(historical_build_path),
                          "candidate_schedule": digest(candidate_schedule), "control_schedule": digest(control_schedule),
                          "shared_e104_analyzer": digest(QO / "analyze.py"),
                          "analyzer": digest(Path(__file__).resolve())},
              "frozen_c09_identity": {"exact_image_matches_historical": True, "projection_source_matches_historical": True,
                                      "exact_image": compiled["exact_image"]},
              "mapping_method": "Complete ordered native marker-kind / scheduled operation-kind correspondence: control22 / C09 candidate26 operations in every run/cluster. Both stage-specific weight/table/TRF→dot lineages and mutable output→reshape→gather→scale lineage are checked.",
              "limitations": [
                  "Marker intervals can include dependency and queue waits; they are not HBM bandwidth or DM-bank counters.",
                  "Every event subtraction stays within one launch and one cluster's Task clock. Across launches only relative windows or durations are compared.",
                  "Overlapping durations and their medians are never summed into predicted savings.",
                  "weight_interval_sum_diagnostic_only and dot_interval_sum_diagnostic_only deliberately sum resource intervals; they are not latency. Projection boundary windows account for overlap without adding component durations.",
                  "C09 first dot covers60 rows per slice; control's only dot covers120. Stage-specific intervals are reported separately, not compared as equal work.",
                  "Final-dot→gather gap contains mutable-output handoff/other waits. Generic Core/Cluster wrappers are preserved but not assigned to one SDK operation or causal cost.",
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
        mapping = make_mapping(schedule, metadata_path, side)
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
    parser.add_argument("label", nargs="?", default="c09-first-npu-v1")
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
