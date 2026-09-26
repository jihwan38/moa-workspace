#!/usr/bin/env python3
"""Map the unchanged E104 QKV/O trace to its complete scheduled engine sequence.

Runtime markers contain operation kinds, not source locations. The correspondence
below checks the whole kind sequence, then the observed issue order and presence
sets in every run/cluster. Intervals include dependency/queue waits. All windows
use one cluster's Task clock; neither overlapping durations nor compiler cycles
are converted into predicted speedups.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EVIDENCE = HERE / "evidence/profile-v1"
WRAPPERS = {"Task", "Core", "Cluster"}
CONFIG = {
    "sliding_project_qkv": {"stage": "qkv", "operations": 76, "passes": 9,
        "cluster_absent": {0: set(), 1: {"97", "107", "331", "374"}},
        "selected": {
            "input_hbm": "14", "pre_rms_square": "22", "pre_rms_reduce": "28",
            "pre_rms_finish": "73", "activation_leader_transfer": "79",
            "activation_ring32_broadcast": "96", "kv_input_trf": "134", "q_input_trf": "138",
            "q_weight": "125", "k_weight": "145", "v_weight": "204",
            "q_scale_vrf": "124", "k_scale_vrf": "51", "v_scale_vrf": "182",
            "k_table_constant_dma": "52", "k_table_prepare": "76",
            "q_table_constant_dma": "139", "q_table_prepare": "144",
            "v_table_constant_dma": "183", "v_table_prepare": "220",
            "q_dot_scale": "151", "k_dot_scale": "163", "v_dot_scale": "328",
            "q_post_dot_switch": "156", "k_post_dot_switch": "189", "v_post_dot_switch": "340",
            "q_head_gather": "166", "k_head_gather": "198", "v_head_gather": "344",
            "q_head_rms_begin": "197", "k_head_rms_begin": "232", "v_head_rms_begin": "354",
            "q_rope_gather_finish": "322", "k_rope_gather_finish": "311",
            "q_output_store": "329", "k_cache_scatter": "331", "v_cache_scatter": "374",
            "v_norm_finish": "368", "v_cache_layout_transfer": "371"}},
    "sliding_attention_output": {"stage": "output", "operations": 22, "passes": 3,
        "cluster_absent": {0: set(), 1: set()},
        "selected": {
            "table_constant_dma": "4", "table_prepare": "13", "input_hbm": "14",
            "weight": "16", "input_trf": "21", "dot_and_16slice_reduce": "26",
            "channel_scale_load": "28", "channel_scale_vrf": "32",
            "post_rms_weight_load": "33", "post_rms_weight_vrf": "39",
            "residual_load": "40", "residual_vrf": "46", "f32_result_head_gather": "54",
            "channel_scale_bf16": "62", "rms_square": "67", "rms_sum": "72",
            "rms_sqrt": "77", "rms_scalar_transfer": "80", "rms_scalar_vrf": "84",
            "rms_normalize": "88", "residual_add": "93", "output_store": "96"}},
}


def digest(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def operation_kind(node):
    # Keep this classifier identical to e104-weight-path/analyze.py.
    if node["tpe"].startswith("Dma"):
        return "DMA"
    if node["tpe"] == "Main":
        return "Renegade::TuExec"
    if node["tpe"] == "Sub" and node["contexts"] != ["Noop"]:
        source = node["description"]
        if ".to_trf()" in source:
            return "Renegade::StoTrf"
        if not source.strip():
            return "Renegade::StoTab"
        if "fetch_cast" in source or "vector_" in source:
            return "Renegade::TuExec"
        return "Renegade::StoVrf"
    return None


def stats(values):
    return {"count": len(values), "median": statistics.median(values),
            "range": [min(values), max(values)]}


def overlap(a, b):
    return max(0, min(a["end"], b["end"]) - max(a["begin"], b["begin"]))


def mapped_kernel(kernel, config):
    schedule_path = EVIDENCE / f"{kernel}.json"
    metadata_path = EVIDENCE / f"{kernel}.image-profile.json"
    schedule = json.loads(schedule_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    assert len(metadata["spans"]) == 1
    spans = sorted(metadata["spans"][0], key=lambda row: row["begin_marker"])
    markers = [row for row in spans if row["name"] not in WRAPPERS]
    operations = [node for node in schedule["instructions"] if operation_kind(node)]
    assert len(markers) == len(operations) == config["operations"]
    assert [operation_kind(n) for n in operations] == [s["name"] for s in markers]
    tensors = {row["index"]: row for row in schedule["tensors"]}
    named = {index: name for name, index in config["selected"].items()}
    mapping = {marker["label"]: {
        "marker": marker, "schedule": node, "semantic_name": named.get(node["index"]),
        "scheduled_output_allocations": [tensors[str(i)] for i in node["output_tensors"] if str(i) in tensors],
        "measurements": {}}
        for marker, node in zip(markers, operations)}
    # K/V are separate calls to project_one_kv_matrix, not identical-task aliases.
    # Check weight -> dot -> result lineage before assigning their names.
    nodes = {node["index"]: node for node in operations}
    if kernel == "sliding_project_qkv":
        for weight, dot, switch, gather in [("125", "151", "156", "166"),
                                           ("145", "163", "189", "198"),
                                           ("204", "328", "340", "344")]:
            assert nodes[weight]["output_tensors"][0] in nodes[dot]["input_tensors"]
            assert nodes[dot]["output_tensors"][0] in nodes[switch]["input_tensors"]
            # Reshape between Switch and gather has its own alias tensor.
            assert nodes[gather]["description"].endswith("output.to_dm(&mut device.tdma)")
        for table, dot in [("76", "163"), ("144", "151"), ("220", "328")]:
            assert nodes[table]["output_tensors"][0] in nodes[dot]["input_tensors"]
    else:
        assert nodes["16"]["output_tensors"][0] in nodes["26"]["input_tensors"]
        assert nodes["13"]["output_tensors"][0] in nodes["26"]["input_tensors"]
    return {"source_schedule": digest(schedule_path), "source_metadata": digest(metadata_path),
            "compiler_makespan": max(n["lifetime"]["end"] for n in schedule["instructions"]),
            "mapped_operation_count": len(mapping), "operations": mapping}, spans


def dependency_observation(kernel, rows, task):
    def begin(index):
        return rows[index]["begin"]

    def end(index):
        return rows[index]["end"]

    def duration(index):
        return end(index) - begin(index)

    if kernel == "sliding_project_qkv":
        q_ready = {"q_weight": end("125"), "q_input_trf": end("138"),
                   "q_scale_vrf": end("124"), "q_table_prepare": end("144")}
        assert max(q_ready.values()) <= begin("151")
        assert end("96") <= begin("151")
        # No Main operation lies inside the observed post-broadcast/pre-Q-dot gap.
        windows = {
            "task_to_broadcast_end": end("96") - task["begin"],
            "pre_rms_end_to_q_input_ready": end("138") - end("73"),
            "q_input_ready_to_q_dot_begin": begin("151") - end("138"),
            "q_weight_end_minus_q_input_ready": end("125") - end("138"),
            "broadcast_end_to_q_dot_begin_main_gap": begin("151") - end("96"),
            "q_weight_broadcast_overlap": overlap(rows["125"], rows["96"]),
            "q_weight_end_to_q_dot_begin": begin("151") - end("125"),
            "q_table_dma_interval": duration("139"),
            "q_table_dma_q_weight_overlap": overlap(rows["139"], rows["125"]),
            "q_weight_end_to_q_table_dma_end": end("139") - end("125"),
            "q_table_prepare_interval": duration("144"),
            "q_dot_begin_minus_latest_prerequisite": begin("151") - max(q_ready.values()),
            "q_dot_begin_from_task": begin("151") - task["begin"],
            "k_weight_end_to_k_dot_begin": begin("163") - end("145"),
            "q_switch_end_to_k_dot_begin": begin("163") - end("156"),
            "v_weight_end_to_v_dot_begin": begin("328") - end("204"),
            "q_rope_end_to_v_dot_begin": begin("328") - end("322"),
            "v_weight_end_minus_q_rope_end": end("204") - end("322"),
            "q_head_rms_to_q_rope_end": end("322") - begin("197"),
            "v_dot_end_to_task_end_tail": task["end"] - end("328"),
            "v_head_gather_end_to_v_rms_begin": begin("354") - end("344"),
            "v_norm_end_to_task_end": task["end"] - end("368"),
            "q_table_weight_trf_last_ready": max(q_ready.values()) - task["begin"],
        }
        latest = max(q_ready, key=q_ready.get)
        v_ready = {"v_weight": end("204"), "q_rope_main_finish": end("322"),
                   "v_table_prepare": end("220"), "v_scale_vrf": end("182"),
                   "kv_input_trf": end("134")}
        assert max(v_ready.values()) <= begin("328")
        extra = {"q_dot_latest_local_prerequisite": latest,
                 "v_dot_latest_local_readiness_boundary": max(v_ready, key=v_ready.get)}
    else:
        ready = {"weight": end("16"), "input_trf": end("21"), "table_prepare": end("13")}
        assert max(ready.values()) <= begin("26")
        windows = {
            "weight_interval": duration("16"),
            "input_ready_from_task": end("21") - task["begin"],
            "input_ready_to_dot_begin": begin("26") - end("21"),
            "weight_end_to_dot_begin": begin("26") - end("16"),
            "table_ready_to_dot_begin": begin("26") - end("13"),
            "dot_begin_from_task": begin("26") - task["begin"],
            "dot_and_16slice_reduce_interval": duration("26"),
            "dot_end_to_task_end_tail": task["end"] - end("26"),
            "f32_head_gather_interval": duration("54"),
            "f32_head_gather_end_to_scale_begin": begin("62") - end("54"),
            "scale_begin_to_task_end": task["end"] - begin("62"),
            "rms_begin_to_rms_end": end("88") - begin("67"),
            "output_store_end_to_task_end": task["end"] - end("96"),
            "residual_vrf_ready_to_residual_add": begin("93") - end("46"),
        }
        extra = {"dot_latest_local_prerequisite": max(ready, key=ready.get)}
    return {"windows": windows, **extra}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify saved reports without rewriting them')
    args = parser.parse_args()
    paths = sorted(HERE.glob("qo-profile-screen-v1-job-*-results.json"))
    assert paths, "Collect at least one completed profile screen first"
    reports = [json.loads(path.read_text()) for path in paths]
    for report in reports:
        assert report["actual_upload_matches"] and report["screen_complete"]
        assert report["job"]["status"] == "SUCCEEDED" and report["job"]["exit_code"] == 0
        for side in ("control_before", "control_after"):
            assert report["stages"][side]["exit_code"] == 0
            assert all(k["correctness"] == "passed" for k in report["stages"][side]["kernels"].values())

    maps, analyses = {}, {}
    for kernel, config in CONFIG.items():
        mapped, metadata_spans = mapped_kernel(kernel, config)
        mapping = mapped["operations"]
        indexed = {v["schedule"]["index"]: label for label, v in mapping.items()}
        labelled = {s["label"]: s for s in metadata_spans}
        task_label, = [s["label"] for s in metadata_spans if s["name"] == "Task"]
        observations, jobs = [], {}
        for path, report in zip(paths, reports):
            job = str(report["job"]["id"])
            for level in ("info", "trace"):
                stage = report["stages"][f"{config['stage']}_{level}"]
                assert stage["exit_code"] == 0 and stage["complete"]
                assert stage["correctness_passes"] == config["passes"]
                assert len(stage["cycles"]) == 1 and len(stage["cycles"][0]["values"]) == 3
            stage = report["stages"][f"{config['stage']}_trace"]
            profile = stage["spans"]
            task_durations = []
            for run in range(3):
                cluster_durations = []
                for cluster in range(2):
                    samples = [s for s in profile if s["run"] == run and s["cluster"] == cluster]
                    by_label = {s["name"]: s for s in samples}
                    assert len(samples) == len(by_label)
                    assert set(by_label).issubset(labelled)
                    present = set(by_label) & set(mapping)
                    expected = set(mapping) - {indexed[i] for i in config["cluster_absent"][cluster]}
                    assert present == expected, (kernel, job, run, cluster)
                    ordered = [by_label[label] for label in mapping if label in present]
                    assert [s["begin"] for s in ordered] == sorted(s["begin"] for s in ordered)
                    task = by_label[task_label]
                    assert all(task["begin"] <= s["begin"] <= s["end"] <= task["end"] for s in samples)
                    cluster_durations.append(task["end"] - task["begin"])
                    rows = {index: by_label[label] for index, label in indexed.items() if label in by_label}
                    if kernel == "sliding_project_qkv":
                        assert not any(s["begin"] < rows["151"]["begin"] and s["end"] > rows["96"]["end"]
                            for label, s in by_label.items() if label in mapping and mapping[label]["schedule"]["tpe"] == "Main")
                    observation = {"job": int(job), "run": run, "cluster": cluster,
                        "task_cycles": cluster_durations[-1], "observed_engine_count": len(present),
                        "absent_schedule_indices": sorted(config["cluster_absent"][cluster], key=int),
                        **dependency_observation(kernel, rows, task),
                        "selected_relative_intervals": {
                            name: {"begin": rows[index]["begin"] - task["begin"],
                                   "end": rows[index]["end"] - task["begin"]}
                            for name, index in config["selected"].items() if index in rows},
                        "unassigned_wrapper_intervals": [
                            {"label": s["name"], "kind": labelled[s["name"]]["name"],
                             "begin": s["begin"] - task["begin"], "end": s["end"] - task["begin"]}
                            for s in samples if s["name"] not in mapping]}
                    observations.append(observation)
                task_durations.append(max(cluster_durations))
            assert task_durations == stage["cycles"][0]["values"]
            for label, record in mapping.items():
                samples = [s for s in profile if s["name"] == label]
                by_run = [max(s["end"] - s["begin"] for s in samples if s["run"] == run) for run in range(3)]
                record["measurements"][job] = {
                    "cycles_by_run_cluster": [{"run": s["run"], "cluster": s["cluster"],
                                                "cycles": s["end"] - s["begin"]} for s in samples],
                    "max_observed_cluster_cycles_by_run": by_run,
                    "median_max_observed_cluster_cycles": statistics.median(by_run)}
            jobs[job] = {"source": digest(path), "log": report["log"], "log_digest": report["log_digest"],
                "control_before": report["stages"]["control_before"]["kernels"][kernel],
                "control_after": report["stages"]["control_after"]["kernels"][kernel],
                "profile_info": report["stages"][f"{config['stage']}_info"]["cycles"][0],
                "profile_trace": stage["cycles"][0], "task_marker_max_clusters_matches_reported_cycles": True}
        window_names = observations[0]["windows"]
        summaries = {}
        for window in window_names:
            summaries[window] = {
                "all_run_clusters": stats([o["windows"][window] for o in observations]),
                "by_job": {job: stats([o["windows"][window] for o in observations if str(o["job"]) == job]) for job in jobs},
                "by_cluster": {str(cluster): stats([o["windows"][window] for o in observations if o["cluster"] == cluster]) for cluster in range(2)}}
        analyses[kernel] = {
            "mapped_operation_count": mapped["mapped_operation_count"],
            "compiler_makespan": mapped["compiler_makespan"], "jobs": jobs,
            "selected": {name: {"schedule_index": index, "label": indexed[index],
                          "source": mapping[indexed[index]]["schedule"]["description"],
                          "measurements": mapping[indexed[index]]["measurements"]}
                         for name, index in config["selected"].items()},
            "window_summary": summaries, "run_cluster_observations": observations}
        maps[kernel] = mapped

    output = {
        "scope": "Unchanged E104 QKV/O device task bytes; two three-input trace screens; all observed clusters retained",
        "mapping_method": "Whole scheduled engine-kind sequence matches native marker-kind sequence (QKV 76, O 22). Present-marker issue order and exact cluster-specific presence sets are checked in every run. This is ordered metadata correspondence, not runtime-emitted source annotations.",
        "source_identity": digest(HERE / "build-profile-v1.json"),
        "interpretation": [
            "QKV cluster 1 does not emit the two RoPE HBM gathers or the two cache scatters. Cluster 0 emits all 76 engine markers; cluster 1 emits 72. This stable asymmetry is retained, not imputed.",
            "QKV's Q input is already ready before Q weight and table readiness. The post-broadcast Main gap and delayed Q table preparation are measured dependency windows, not a removable broadcast-duration estimate.",
            "The Q constant-table DMA marker overlaps Q weight DMA for most of its interval. Its duration is not the isolated service time of a 4096-byte transfer.",
            "O's first dot waits on weight readiness after input and lookup table are ready. Head gather, normalization and residual form a measured post-dot tail.",
            "QKV table delay and input slack confirm prior static hypotheses with runtime evidence. SDK 0.8.1 exposes no prepared-table handle; shared-TRF and interleaved-KV are existing investigated families, not newly enabled fixes.",
            "O supply/compute tiling is an existing C09 family, not a new proposal: c09-output-pipeline/two-stage-v1 already overlapped weight/dot for 1825 compiler cycles but increased static makespan 0.71%; it was not run on NPU. This profile alone does not justify repeating it. A successor needs evidence that added lookup/setup/task/partial-output costs are removed while preserving FP32 reduction and BF16 rounding boundaries, before any NPU screen.",
            "O Main/Sub epilogue overlap is also an existing C12 family; c12-output-main-sub passed basic NPU correctness but regressed O 2.46%. The measured post-dot tail does not by itself reopen that same implementation.",
            "These profiles rank readiness mechanisms; they do not prove HBM bandwidth saturation, DM bank conflicts, an exact critical path, or candidate speedup. All unprofiled controls and their variation remain visible.",
        ],
        "limitations": [
            "Per-operation intervals may include queue/dependency waits and cross-cluster synchronization. Overlapping durations are never summed as runtime shares or savings.",
            "All event differences are within one cluster's Task clock. Across clusters only duration statistics are compared; no synchronized absolute time is assumed.",
            "Compiler makespan and measured device cycles have no assumed fixed conversion. Named readiness boundaries are partial evidence, not a reconstructed hardware causal graph.",
            "Window medians use every run/cluster, including run 0, and expose per-job and per-cluster ranges. They are diagnostic statistics, not official kernel benchmark medians.",
            "Profile instrumentation and changing controls prohibit a stable overhead or optimization claim. Job 104933 QKV control-after includes the 123912-cycle sample.",
            "Generic Task/Cluster/Core wrappers are retained without source ownership. Q/K/V identities follow weight/dot dataflow and the frozen project_key_value K-then-V caller.",
        ],
        "kernels": analyses,
    }
    for name, value in [('marker-source-map.json', maps), ('profile-analysis.json', output)]:
        path = HERE / name
        if args.check:
            assert json.loads(path.read_text()) == value, f'Regenerate {path}'
        else:
            path.write_text(json.dumps(value, indent=2) + "\n")
    for kernel, analysis in analyses.items():
        print(kernel, "mapped", analysis["mapped_operation_count"], "compiler makespan", analysis["compiler_makespan"])
        for job, data in analysis["jobs"].items():
            print(" job", job, "controls", data["control_before"]["median_cycles"], data["control_after"]["median_cycles"],
                  "trace", data["profile_trace"]["median"])
        for name, summary in analysis["window_summary"].items():
            print(" ", name, summary["all_run_clusters"])


if __name__ == "__main__":
    main()
