#!/usr/bin/env python3
"""Validate saved host-only diagnostic records; no model execution or network.

Usage: python3 experiments/int4-rational-weight-audit/analyze_actual.py
Reads the frozen raw/ evidence and writes actual-result.json beside this script.
"""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASK64 = (1 << 64) - 1
METHODS = ("coarse_power2_row", "rational_row", "coarse_power2_block960", "rational_block960")
KEYS = [(layer, projection) for layer in (0, 6, 47)
        for projection in ("up_proj", "gate_proj", "down_proj")]


def mix(x):
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK64
    return x ^ (x >> 31)


def selected_rows(rows, layer):
    return [0, rows // 2 - 1, rows // 2, rows - 1] + [
        mix(pick + layer * 313) % rows for pick in range(4, 64)]


def records(tag):
    prefix = tag + " "
    return [json.loads(line[len(prefix):]) for line in LOG.splitlines()
            if line.startswith(prefix)]


def one(tag):
    values = records(tag)
    assert len(values) == 1, (tag, len(values))
    return values[0]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    assert isinstance(value, (int, float)) and math.isfinite(value), value


LOG = (HERE / "raw/arena.log").read_text()
weights = records("A4_RATIONAL_WEIGHT")
done = one("A4_RATIONAL_DONE")
source = one("A4_RATIONAL_SOURCE")
oracle = one("A4_RATIONAL_ORACLE")
status = json.loads((HERE / "raw/arena-status.json").read_text())
assert status["id"] == 322813 and status["status"] == "SUCCEEDED" and status["exit_code"] == 0
assert [(r["layer"], r["projection"]) for r in weights] == KEYS
assert oracle["finite_patterns_checked"] == 65280 and oracle["explicit_boundary_cases"] == 13
assert oracle["reciprocal_seven_f32_bits"] == "3e124925"
assert source["layers"] == [0, 6, 47] and source["sampled_rows_per_projection"] == 64

old = {}
for line in (HERE / "raw/r1194-arena.log").read_text().splitlines():
    if line.startswith("A4P2_WEIGHT "):
        r = dict(field.split("=", 1) for field in line.split()[1:])
        key = (int(r["layer"]), r["projection"])
        assert key not in old
        old[key] = r
assert list(old) == KEYS

table, differences = [], []
source_values = zero_count = negative_zero_count = subnormal_count = 0
groups = {method: 0 for method in METHODS}
histograms = {method: [0] * 15 for method in METHODS}
chain = 0xcbf29ce484222325
for r in weights:
    key = (r["layer"], r["projection"])
    rows, width = (3840, 15360) if key[1] == "down_proj" else (15360, 3840)
    n = 64 * width
    prefix = f"model.language_model.layers.{key[0]}.mlp.{key[1]}"
    assert r["sampled_rows"] == selected_rows(rows, key[0])
    assert len(r["sampled_rows"]) == 64 and all(0 <= x < rows for x in r["sampled_rows"])
    assert r["values"] == n
    assert r["packed_tensor"] == {"name": prefix + ".weight_packed", "dtype": "U8",
                                   "shape": [rows, width // 2], "bytes": rows * width // 2}
    assert r["scale_tensor"] == {"name": prefix + ".weight_scale", "dtype": "F8_E4M3",
                                  "shape": [rows, width // 16], "bytes": rows * width // 16}
    for field in ("finite_min", "finite_max", "host_elapsed_ms_including_decode_and_all_four_audits"):
        finite(r[field])
    assert r["finite_min"] <= 0 <= r["finite_max"]
    counts = [r[k] for k in ("negative_zero_count", "positive_zero_count", "ftz_subnormal_count")]
    assert all(isinstance(x, int) and 0 <= x <= n for x in counts) and sum(counts) <= n
    source_values += n
    negative_zero_count += counts[0]
    zero_count += counts[0] + counts[1]
    subnormal_count += counts[2]
    for field in ("sampled_source_fnv1a64", "sampled_bf16_fnv1a64"):
        assert len(r[field]) == 16 and format(int(r[field], 16), "016x") == r[field]
    for byte in prefix.encode() + int(r["sampled_source_fnv1a64"], 16).to_bytes(8, "little"):
        chain = ((chain ^ byte) * 0x100000001b3) & MASK64
    for method in METHODS:
        m = r[method]
        assert m["groups"] == (64 * width // 960 if method.endswith("block960") else 64)
        assert m["values"] == n
        h = m["code_histogram_minus7_through7"]
        assert len(h) == 15 and all(isinstance(x, int) and x >= 0 for x in h) and sum(h) == n
        assert m["code_min"] == next(i - 7 for i, x in enumerate(h) if x)
        assert m["code_max"] == max(i - 7 for i, x in enumerate(h) if x)
        assert h[7] >= sum(counts)
        for field in ("relative_l2", "squared_error", "squared_reference", "max_absolute_error",
                      "scale_min", "scale_max", "clamped_amax_min", "clamped_amax_max"):
            finite(m[field])
            assert m[field] >= 0
        assert 0 < m["scale_min"] <= m["scale_max"]
        assert 2**-112 <= m["clamped_amax_min"] <= m["clamped_amax_max"]
        assert m["squared_reference"] == r["coarse_power2_row"]["squared_reference"] > 0
        assert abs(math.sqrt(m["squared_error"] / m["squared_reference"]) - m["relative_l2"]) <= 1e-15
        groups[method] += m["groups"]
        histograms[method] = [a + b for a, b in zip(histograms[method], h)]
    previous = old[key]
    assert int(previous["sampled_rows"]) == 64 and int(previous["values"]) == n
    deltas = []
    for method, oldfield in (("coarse_power2_row", "row_relative_l2"),
                             ("coarse_power2_block960", "block960_relative_l2")):
        difference = abs(r[method]["relative_l2"] - float(previous[oldfield]))
        assert difference <= 1e-12, (key, method, difference)
        deltas.append(difference)
        differences.append(difference)
    table.append({"layer": key[0], "projection": key[1], "values": n,
                  "unique_selected_rows": len(set(r["sampled_rows"])),
                  **{method: r[method]["relative_l2"] for method in METHODS},
                  "coarse_r1194_absolute_differences_row_then_block": deltas})

assert done["status"] == "PASS provided-weight diagnostic completed"
assert done["projections"] == len(weights) == 9
assert done["sampled_rows"] == 9 * 64 == 576
assert done["source_values"] == source_values == 4423680
assert done["quantized_code_checks"] == 4 * source_values == 17694720
assert done["sampled_source_chain_fnv1a64"] == format(chain, "016x")
assert groups == {"coarse_power2_row": 576, "rational_row": 576,
                  "coarse_power2_block960": 4608, "rational_block960": 4608}
assert all(sum(h) == source_values for h in histograms.values())
assert subnormal_count == negative_zero_count == 0

result = {
    "status": "PASS independently validated saved actual diagnostic records",
    "job": 322813,
    "validation": {
        "nine_ordered_records_and_one_done": True, "same_r1194_row_selection": True,
        "tensor_headers_and_lengths": True, "histograms_bounds_group_counts": True,
        "finite_metric_and_zero_count_consistency": True,
        "l2_recomputed_from_reported_error_and_norm": True,
        "sampled_source_fingerprint_chain_consistency": True,
        "coarse_r1194_l2_tolerance": 1e-12,
        "coarse_r1194_max_absolute_difference": max(differences),
        "coarse_r1194_comparisons": len(differences),
    },
    "counts": {"projections": 9, "sampled_rows_including_duplicates": 576,
               "source_values": source_values, "quantized_code_checks": 4 * source_values,
               "positive_zero_values": zero_count, "negative_zero_values": negative_zero_count,
               "subnormal_values": subnormal_count, "nonzero_finite_values": source_values - zero_count,
               "groups_by_method": groups, "code_histograms_by_method": histograms},
    "relative_l2_ranges": {method: {"min": min(r[method]["relative_l2"] for r in weights),
                                   "max": max(r[method]["relative_l2"] for r in weights)} for method in METHODS},
    "per_projection": table,
    "observations": {"rational_lower_than_coarse_in_all_nine_rows": all(
        r["rational_row"]["relative_l2"] < r["coarse_power2_row"]["relative_l2"] for r in weights),
        "rational_lower_than_coarse_in_all_nine_blocks": all(
        r["rational_block960"]["relative_l2"] < r["coarse_power2_block960"]["relative_l2"] for r in weights)},
    "timing_scope_only": {"arena_process_seconds": status["duration_sec"],
                          "reported_total_host_ms": done["elapsed_ms"],
                          "not_a_model_or_npu_speed_measurement": True},
    "input_provenance": source,
    "raw_sha256": {str(p.relative_to(HERE)): sha(p) for p in sorted((HERE / "raw").iterdir()) if p.is_file()},
    "limits": [
        "Provided static weights only: BF16(F4*per16 E4), before global; no actual activation or task/model accuracy proof",
        "Row and block960 are separate quantization experiments; no combined whole-model result",
        "The observed lower sampled L2 is not a guarantee over unsampled weights or task accuracy",
        "This validator checks saved report consistency and r1194 equality; it does not reread model weights",
        "Input FNV fingerprints are noncryptographic and sampled-only, not full model hashes",
        "65280 finite selfchecks establish code range at maxfinite A, not exhaustive (x,A) equality",
        "r742 synthetic0.1319 is excluded; old DivF failures remain unchanged",
        "Exact p20 reciprocal/FP32-product proof3055874a is a separate future implementation opportunity, not a new weight-accuracy result",
    ],
}
out = HERE / "actual-result.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"result": result["status"], "r1194_max_difference": max(differences),
                  "ranges": result["relative_l2_ranges"], "output": str(out)}, indent=2))
