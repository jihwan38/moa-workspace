#!/usr/bin/env python3
"""Read preserved job322975 evidence; print deterministic JSON, never run NPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ELF_SHA = "c0835bbf4fcd57cc2611b3c39deb566ec26a107efa157f2854d9420f161e1ab1"
LOCK_SHA = "a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
    return json.loads(path.read_text())
def fields(line):
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", line))
def aggregate(values):
    assert values and all(math.isfinite(x) and x > 0 for x in values)
    return {"samples": values, "min": min(values), "max": max(values), "mean": statistics.mean(values)}

def analyze(raw, verify_local):
    raw_hashes = load(raw / "raw-files.json")
    for name, expected in raw_hashes.items():
        path = raw / name
        assert sha(path) == expected["sha256"] and path.stat().st_size == expected["bytes"]
    assert load(raw / "arena-records.json") == []  # no per-task trace payload in this collection
    status = load(raw / "arena-status.json")
    run = load(raw / "run.json")
    inputs = load(raw / "source-hashes.json")
    build_inputs = load(raw / "build-source-hashes.json")
    build = load(HERE / "a003-build-manifest.json")
    registry = load(HERE / "a003-embedded-registry.json")
    assert status["id"] == 322975 and status["status"] == "SUCCEEDED" and status["exit_code"] == 0
    assert status["duration_sec"] == 9.610338484
    assert len(inputs) == 402 and inputs == build_inputs
    assert inputs["Cargo.lock"] == LOCK_SHA
    assert run["submitted_artifact"]["sha256"] == build["captured_elf"]["sha256"] == registry["elf_sha256"] == ELF_SHA
    assert run["command"][run["command"].index("--timeout") + 1] == "70"
    assert "FURIOSA_OPT_PROFILE=1" in run["command"]
    assert registry["record_count"] == 4
    images = {x["path"]: x for x in registry["records"]}
    for x in build["embedded_new_bin_parity"]:
        assert x["embedded_equal"] and images[x["path"]]["sha256"] == x["sha256"]
        assert sha(HERE.parents[1] / x["new_bin"]) == x["sha256"]
    if verify_local:
        repo = HERE.parents[1]
        for name, expected in inputs.items():
            assert sha(repo / name) == expected, name
        for artifact in [run["submitted_artifact"], build["captured_elf"]]:
            assert sha(Path(artifact["path"])) == ELF_SHA
    lines = (raw / "arena.log").read_text().splitlines()
    def records(prefix):
        return [fields(line) for line in lines if line.startswith(prefix + " ")]
    final, = records("PASS_I4_GROUP960_PROJECTION")
    expected_counts = {"layer": 0, "output_bf16_values": 35_389_440,
                       "independent_i64_F64_samples": 3072, "candidate_permutation_values": 17_694_720,
                       "immutable_bytes_compared": 562_092_040}
    for name, value in expected_counts.items():
        assert int(final[name]) == value
    assert final["down_resource_HOLD_preserved"] == final["no_full_FFN_or_model_quality_claim"] == "true"
    exact = records("I4_GROUP960_EXACT")
    assert [int(x["round"]) for x in exact] == [0, 1]
    assert sum(int(x["output_bf16_values"]) for x in exact) == expected_counts["output_bf16_values"]
    assert all(x["all_output_bits"] == x["finite"] == x["device_and_supplied_sources_immutable"] == "true" for x in exact)
    samples = records("I4_GROUP960_F64")
    assert {(int(x["round"]), x["projection"]) for x in samples} == {(r, p) for r in [0, 1] for p in ["UP", "GATE", "DOWN"]}
    assert len(samples) == 6 and all(int(x["samples"]) == 512 for x in samples)
    assert sum(int(x["samples"]) for x in samples) == expected_counts["independent_i64_F64_samples"]
    weights = records("I4_GROUP960_WEIGHT")
    assert len(weights) == 3 and {x["name"] for x in weights} == {"UP", "GATE", "DOWN"}
    assert all(int(x["values"]) == 58_982_400 and int(x["scales"]) == 61_440 for x in weights)
    timed = records("I4_GROUP960_TIME")
    assert len(timed) == 8
    for r in [0, 1]:
        group = [x for x in timed if int(x["round"]) == r]
        assert [int(x["block"]) for x in group] == [0, 1, 2, 3]
        assert [x["candidate"] for x in group] == ["true", "false", "false", "true"]
    for x in timed:
        assert int(x["samples"]) == 8
        assert x["scope"] == "projection_with_all_output_stores"
        assert x["excludes"] == "activation_quant_pack_GeGLU_upload_download"
        assert x["reference_extra_row_mul"] == "unity"
    timings = {}
    for stage in ["upgate", "down"]:
        c = aggregate([float(x[stage + "_us"]) for x in timed if x["candidate"] == "true"])
        r = aggregate([float(x[stage + "_us"]) for x in timed if x["candidate"] == "false"])
        timings[stage] = {"candidate_us": c, "intended_E4_reference_us": r,
                          "latency_reduction_percent_vs_intended_reference": 100 * (1 - c["mean"] / r["mean"]),
                          "reference_over_candidate_ratio": r["mean"] / c["mean"],
                          "awaited_calls_per_arm": 32}
    startup = records("I4_GROUP960_STARTUP")
    assert len(startup) == 2
    assert int(startup[0]["values"]) == 176_947_200 and int(startup[0]["scale_groups"]) == 184_320
    assert startup[1]["down_resource_status"] == "HOLD_245684_gt_240000"
    prepared = records("I4_GROUP960_INPUT")
    assert len(prepared) == 4
    prep_by_round = []
    for r in [0, 1]:
        pair = [x for x in prepared if int(x["round"]) == r]
        assert len(pair) == 2
        prep_by_round.append({"round": r, "cpu_fixture_quantize_pack_ms": float(pair[0]["cpu_fixture_quantize_pack_ms"]),
                              "input_upload_output_poison_ms": float(pair[1]["input_upload_output_poison_ms"])})
    markers = [line for line in lines if line.startswith("PROFILE_I4_GROUP960_")]
    assert len(markers) == 4
    return {"schema_version": 1, "job_id": status["id"], "status": status["status"],
            "job_duration_seconds": status["duration_sec"], "build_source_commit": build["source_commit"],
            "submitted_evidence_commit": run["commit"], "ELF_sha256": ELF_SHA,
            "source_input_count": len(inputs), "build_submit_source_hashes_equal": True,
            "Cargo.lock_sha256": LOCK_SHA, "embedded_image_count": 4,
            "raw_input_hashes": raw_hashes, "correctness_counts": expected_counts,
            "sample_radius_records": samples, "weight_code_fingerprints": weights,
            "timings": timings, "cpu_weight_reconstruct_quantize_audit_ms": float(startup[0]["cpu_load_reconstruct_quantize_audit_ms"]),
            "weight_upload_ms": float(startup[1]["weight_upload_ms"]), "input_preparation": prep_by_round,
            "estimated_device_payload_bytes_not_allocator_measurement": int(startup[1]["estimated_peak_live_device_payload_bytes"]),
            "profile_markers": markers, "profile_trace_payload_collected": False,
            "decision": "Intended-code layer0 projection correctness PASS; DOWN resource HOLD remains; no host/model promotion",
            "limits": ["Reference is dense E4 integer-coded per960 arithmetic, not current production FFN or original BF16 accuracy.",
                       "Reference retains an extra exact unity row-scale multiplication.",
                       "Projection timing excludes CPU/device quantization, packing, GeGLU, allocation, upload and readback.",
                       "Provided layer0 only; synthetic deterministic activations and one row permutation.",
                       "No complete FFN/model accuracy/serving throughput or official-score inference.",
                       "Four profiling markers were collected but no per-task trace payload appears in preserved arena.log or records."]}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=HERE / "actual-322975")
    p.add_argument("--verify-local-artifacts", action="store_true")
    args = p.parse_args()
    print(json.dumps(analyze(args.raw_dir, args.verify_local_artifacts), indent=2, sort_keys=True))
if __name__ == "__main__":
    main()
