#!/usr/bin/env python3
"""Match all 120 scheduled engine operations to ordered ABI profile markers.

The runtime exposes unnamed operation kinds, not source locations. Mapping is
an explicit ordered correspondence, checked across the entire engine sequence,
not a duration-based guess. CPU/Cluster wrapper markers are kept separate.
"""
import json
from pathlib import Path
import statistics
from image_profile import digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / "evidence/profile-v1"


def operation_kind(node):
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


def union(intervals):
    total = 0
    end = None
    for begin, stop in sorted(intervals):
        if end is None or begin >= end:
            total += stop-begin
        elif stop > end:
            total += stop-end
        end = max(end or stop, stop)
    return total


def main():
    schedule = json.loads((OUT / "schedule.json").read_text())
    metadata = json.loads((HERE / "profile-v1-image-profile.json").read_text())
    assert len(metadata["spans"]) == 1
    spans = sorted(metadata["spans"][0], key=lambda s:s["begin_marker"])
    markers = [s for s in spans if s["name"] not in {"Task", "Core", "Cluster"}]
    operations = [n for n in schedule["instructions"] if operation_kind(n)]
    assert len(markers) == len(operations) == 120
    assert [operation_kind(n) for n in operations] == [s["name"] for s in markers]
    labelled = {s["label"]:s for s in spans}
    mapping = {s["label"]:{"marker":s, "schedule":n} for s,n in zip(markers,operations)}
    indexed = {v["schedule"]["index"]:label for label,v in mapping.items()}
    summaries = {}
    front_end = []
    paths = sorted(HERE.glob("e104-profile-screen-v1-job-*-results.json"))
    for path in paths:
        report = json.loads(path.read_text())
        assert report["actual_upload_matches"] and report["screen_complete"]
        stage = report["stages"]["probe_trace"]
        assert stage["exit_code"] == 0 and stage["complete"] and stage["correctness_passes"] == 3
        profile = stage["spans"]
        for run in range(3):
            for cluster in range(2):
                rows = [s for s in profile if s["run"]==run and s["cluster"]==cluster]
                assert len(rows) == len({s["name"] for s in rows}) == 128
                assert set(mapping).issubset({s["name"] for s in rows})
                # Marker order and observed issue/start order agree for all 120
                # engine operations; ties have no forced order.
                ordered = sorted([s for s in rows if s["name"] in mapping],key=lambda s:labelled[s["name"]]["begin_marker"])
                assert [s["begin"] for s in ordered] == sorted(s["begin"] for s in ordered)
                by_label = {s["name"]:s for s in rows}
                norm = by_label[indexed["140"]]
                replication = by_label[indexed["150"]]
                amax = by_label[indexed["161"]]
                weight = by_label[indexed["112"]]
                assert norm["end"] <= replication["begin"] < weight["end"] <= replication["end"] <= amax["begin"]
                assert not any(s["begin"] < amax["begin"] and s["end"] > norm["end"]
                    for s in rows if s["name"] in mapping and mapping[s["name"]]["schedule"]["tpe"]=="Main")
                origin = next(s["begin"] for s in rows if labelled[s["name"]]["name"]=="Task")
                front_end.append({"job":report["job"]["id"],"run":run,"cluster":cluster,
                    "main_gap_cycles":amax["begin"]-norm["end"],
                    "replication_interval_cycles":replication["end"]-replication["begin"],
                    "weight_finishes_after_replication_begins":weight["end"]-replication["begin"],
                    "replication_finishes_after_weight":replication["end"]-weight["end"],
                    "relative_intervals":{name:{"begin":s["begin"]-origin,"end":s["end"]-origin}
                        for name,s in (("pre_rms_final",norm),("gate_weight_load",weight),("activation_replication",replication),("activation_amax",amax))}})
        controls = [report["stages"][name]["kernels"]["decoder_feedforward"]["median_cycles"] for name in ("control_before","control_after")]
        summaries[str(report["job"]["id"])] = {"control_ffn_medians":controls,
            "profile_level_medians":{name:report["stages"][name]["cycles"][0]["median"] for name in ("probe_info","probe_debug","probe_trace")},
            "source":str(path.relative_to(ROOT)), "log":report["log"],
            "trace_vs_control_percent":100*(stage["cycles"][0]["median"]/statistics.median(controls)-1)}
        for label, record in mapping.items():
            samples = [s for s in profile if s["name"]==label]
            record.setdefault("measurements",{})[str(report["job"]["id"])] = {
                "cycles_by_run_cluster":[{"run":s["run"],"cluster":s["cluster"],"cycles":s["end"]-s["begin"]} for s in samples],
                "max_cluster_cycles_by_run":[max(s["end"]-s["begin"] for s in samples if s["run"]==r) for r in range(3)],
                "median_max_cluster_cycles":statistics.median(max(s["end"]-s["begin"] for s in samples if s["run"]==r) for r in range(3))}
    assert paths
    selected = {
        "gate_weight_load":"112", "up_weight_load":"274", "down_weight_load":"383",
        "gate_fp8_materialization":"229", "up_fp8_materialization":"359", "down_fp8_materialization":"523",
        "gate_dot_input_scale_partial":"287", "up_dot_input_scale_partial":"391", "down_lookup_dot_partial":"563",
        "activation_initial_replication":"150", "activation_three_term_broadcast":"261",
        "activation_leader_transfer":"250", "down_activation_transfer":"524", "down_activation_broadcast":"552",
        "gate_block_scale_load":"251", "up_block_scale_load":"362", "down_block_scale_load":"531"}
    selected_records = {name:{"label":indexed[node],**mapping[indexed[node]]} for name,node in selected.items()}
    output = {"scope":"Whole E104 profile; original NPU task bytes unchanged; profile level trace",
        "mapping_method":"All 120 non-Core/non-Cluster scheduled engine kinds match all 120 native profile kinds in marker order; issue/start order also agrees for all measured runs/clusters. This is ordered metadata correspondence, not source annotations emitted by the runtime.",
        "source_schedule":digest((OUT/"schedule.json").read_bytes()), "mapped_operation_count":120,
        "jobs":summaries,"selected":selected_records,"front_end_dependency_observations":front_end,
        "front_end_main_gap_median":statistics.median(x["main_gap_cycles"] for x in front_end),
        "front_end_main_gap_range":[min(x["main_gap_cycles"] for x in front_end),max(x["main_gap_cycles"] for x in front_end)],
        "limitations":["Profile intervals can include queue/dependency waits; they are not isolated data-movement or ALU service times.",
                       "Overlapping durations cannot be added as runtime shares or possible speedups.",
                       "Clock conversion from compiler cycles to device cycles is not assumed.",
                       "Trace is an instrumented execution; apparent faster full-kernel times are not an optimization claim.",
                       "Generic Cluster/Core wrappers are not assigned source operations; they are retained in raw logs."]}
    (HERE/"marker-source-map.json").write_text(json.dumps(mapping,indent=2)+"\n")
    (HERE/"profile-analysis.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(summaries,indent=2))
    print('Front-end Main gap:',output['front_end_main_gap_median'],output['front_end_main_gap_range'])
    for name,rec in selected_records.items():
        print(name,'static',rec['schedule']['lifetime']['end']-rec['schedule']['lifetime']['begin'],
              {job:val['median_max_cluster_cycles'] for job,val in rec['measurements'].items()})


if __name__ == "__main__":
    main()
