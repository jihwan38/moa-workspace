#!/usr/bin/env python3
"""Replay frozen job323281 evidence. Read-only; never runs a build or ELF."""
import contextlib
import io
import json
import math
from pathlib import Path

import audit_build

HERE = Path(__file__).resolve().parent
RAW = HERE / "actual-323281"
PINNED = {
    "arena.log": "dbd521efe4fd15bb68e74d954e28f271ac3223c6362e0939ccb2959416a242d5",
    "arena-status.json": "1639a51fcf62d528ce7839c476eb8c688e52110eb60007983af2053945d250c7",
    "arena-records.json": "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570",
    "run.json": "8c30c7f85892d50102533bb69b9b1e13db0d092891d3c87bcff609e69feded66",
    "source-hashes.json": "55268e1c77d1de87c20b7d11bf3350e283c03fd22bcf9193c14ec55f43697681",
    "source.diff": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "output.log": "ae22fc06e9a81a4a2395235971595fbfc9fa531b68835089d5fca33bb09ad323",
}
P, H, L = 512, 3840, 15360


def read_json(path):
    return json.loads(path.read_text())


def finite(value):
    if isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, list):
        for v in value:
            finite(v)
    elif isinstance(value, float):
        assert math.isfinite(value)


def fields(text):
    result = {}
    for token in text.split():
        key, value = token.split("=", 1)
        assert key not in result, key
        if value in ("true", "false"):
            value = value == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        result[key] = value
    return result


def metric(value, count):
    assert value["values"] == count
    assert 0 <= value["reconstructed_zeros"] <= count
    assert value["sse"] >= 0 and value["reference_ss"] > 0 and value["maxabs"] >= 0
    assert math.isclose(value["relative_l2"], math.sqrt(value["sse"] / value["reference_ss"]), rel_tol=1e-14)


def main():
    for name, digest in PINNED.items():
        assert audit_build.sha(RAW / name) == digest, name
    status, run = read_json(RAW / "arena-status.json"), read_json(RAW / "run.json")
    assert status["id"] == 323281 and status["status"] == "SUCCEEDED" and status["exit_code"] == 0
    assert run["name"] == "int4-channel-balance-actual-20261026"
    assert run["commit"] == audit_build.SOURCE and run["exit_code"] == 0
    assert run["result"] == "command_succeeded"
    assert (RAW / "source.diff").read_bytes() == b""
    expected_untracked = {
        "?? experiments/int4-channel-balance-diagnostic/" + p for p in
        ("a003-device-build/", "audit_build.py", "build-handoff.md", "build-result.json", "registry_audit.py")
    }
    assert set(run["status"].splitlines()) == expected_untracked
    # Recheck all source/dependency and nineteen prior/fresh embedded image bindings.
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        audit_build.main()
    build = json.loads(output.getvalue())
    assert build == read_json(HERE / "build-result.json")
    source_hashes = read_json(RAW / "source-hashes.json")
    assert source_hashes == read_json(HERE / "a003-device-build/source-hashes.json")
    assert len(source_hashes) == 423
    artifact = run["submitted_artifact"]
    assert artifact["sha256"] == build["captured_artifact"]["sha256"]
    assert artifact["size"] == build["captured_artifact"]["size"]
    for name in ("path", "source_path"):
        path = Path(artifact[name])
        assert audit_build.sha(path) == artifact["sha256"] and path.stat().st_size == artifact["size"]

    lines = (RAW / "arena.log").read_text().splitlines()
    at = 0

    def take(marker):
        nonlocal at
        prefix = marker + " "
        assert at < len(lines) and lines[at].startswith(prefix), (at, marker)
        payload = lines[at][len(prefix):]
        at += 1
        value = json.loads(payload) if payload.startswith("{") else fields(payload)
        finite(value)
        return value

    gamma = take("I4_BALANCE_GAMMA")
    assert gamma["source"] == "model.language_model.layers.0.pre_feedforward_layernorm.weight"
    assert gamma["dtype"] == "BF16" and gamma["shape"] == [H] and gamma["channels"] == H
    assert gamma["gamma_exact_inverse"] is True and gamma["identity_zero_subnormal"] == 0
    assert (gamma["negative"], gamma["min_exponent"], gamma["max_exponent"]) == (298, -18, 5)
    weight_proofs, weights = [], []
    for name in ("up_proj", "gate_proj"):
        proof = take("I4_BALANCE_WEIGHT_PROOF")
        assert proof["projection"] == name and proof["values"] == H * L
        assert proof["failures"] == 0 and proof["all_value_BF16_inverse"] is True
        assert proof["original_zeros"] + proof["original_nonzeros"] == H * L
        m = take("I4_BALANCE_WEIGHT_METRIC")
        assert m["projection"] == name and m["global_unchanged"] is True
        assert [x[0] for x in m["source"]] == [
            f"model.language_model.layers.0.mlp.{name}.{suffix}" for suffix in
            ("weight_packed", "weight_scale", "weight_global_scale")]
        for coordinate in ("original_coordinates", "balanced_coordinates", "balanced_reconstruction_restored_coordinates"):
            metric(m[coordinate], H * L)
        assert m["original_coordinates"]["reference_ss"] == m["balanced_reconstruction_restored_coordinates"]["reference_ss"]
        weight_proofs.append(proof)
        weights.append(m)
    startup = take("I4_BALANCE_STARTUP")
    assert startup["includes_original_and_balanced_and_metrics"] is True and startup["not_serving_startup"] is True
    owners = take("I4_BALANCE_OWNERS")
    assert owners == {
        "gamma_prime_bytes": H * 2, "additional_balanced_packed_bytes": 3 * H * L // 2,
        "additional_balanced_dense_reference_bytes": 3 * H * L,
        "additional_balanced_scale_bytes": 3 * H * L // 960 * 4 + 4,
        "unchanged_DOWN_copied": True, "no_model_capacity_claim": True,
    }
    steps = []
    for row_order in range(2):
        for step in range(2):
            coordinate = {"round": row_order, "step": step}

            def check_coordinate(value):
                assert all(value[k] == v for k, v in coordinate.items())

            prefix = take("I4_BALANCE_PREFIX_PROOF")
            check_coordinate(prefix)
            assert prefix["values"] == P * H and prefix["failures"] == prefix["nonnormal_nonzero"] == 0
            assert prefix["exact_F64_inverse_and_BF16_bits"] is True
            products = []
            for name in ("up_proj", "gate_proj"):
                proof = take("I4_BALANCE_PRODUCT_PROOF")
                check_coordinate(proof)
                assert proof["projection"] == name and proof["dots"] == P and proof["products"] == P * H
                assert proof["exact_term_and_same_order_F64_sum"] is True and proof["normal_or_zero_F32_products"] is True
                products.append(proof)
            intended = []
            for arm in ("balanced", "unbalanced"):
                start = take("I4_BALANCE_INTENDED_BEGIN")
                assert start == {**coordinate, "arm": arm}
                arm_activations, samples = [], []
                for stage, projections, width in (("prefix", ("UP", "GATE"), H), ("hidden", ("DOWN",), L)):
                    activation = take("I4_GROUPED_COMPLETE_ACTIVATION")
                    check_coordinate(activation)
                    assert activation["stage"] == stage and activation["values"] == P * width
                    assert 0 <= activation["zero_codes"] <= P * width
                    assert 0 <= activation["bf16_ftz_zeros"] <= P * width
                    assert math.isclose(activation["zero_fraction"], activation["zero_codes"] / (P * width), abs_tol=1e-12)
                    arm_activations.append(activation)
                    for name in projections:
                        sample = take("I4_GROUPED_COMPLETE_F64")
                        check_coordinate(sample)
                        assert sample["projection"] == name and sample["samples"] == P
                        assert sample["oracle"] == "i64_local_F64_envelope" and sample["max_preBF16_radius"] >= 0
                        samples.append(sample)
                intended.append({"arm": arm, "activation_metrics": arm_activations, "sample_checks": samples})
            activation = take("I4_BALANCE_ACTIVATION")
            check_coordinate(activation)
            for key in ("original_coordinates", "balanced_coordinates", "balanced_reconstruction_restored_coordinates"):
                metric(activation[key], P * H)
            assert activation["original_coordinates"]["reference_ss"] == activation["balanced_reconstruction_restored_coordinates"]["reference_ss"]
            drift = []
            for stage, count in (("joined", 2 * P * L), ("hidden", P * L), ("down", P * H), ("residual", P * H)):
                for comparison in ("balanced_vs_q020", "unbalanced_vs_q020", "balanced_vs_unbalanced"):
                    value = take("I4_BALANCE_DRIFT")
                    check_coordinate(value)
                    assert value["comparison"] == comparison + "_" + stage and value["bf16_values"] == count
                    assert 0 <= value["changed"] <= count and value["relative_l2"] >= 0 and value["maxabs"] >= 0
                    assert value["matched_incoming_inputs"] is True and value["no_model_quality_claim"] is True
                    assert value["common_feedback"] == "original production residual"
                    drift.append(value)
            passed = take("I4_BALANCE_STEP_PASS")
            assert passed == {**coordinate, **{k: True for k in (
                "matched_incoming", "original_production_feedback", "inverse_and_unquantized_products_passed",
                "intended_code_parity", "no_task_quality_claim")}}
            steps.append({**coordinate, "prefix_proof": prefix, "product_proofs": products,
                          "intended_references": intended, "activation": activation, "drift": drift})
    final = take("PASS_I4_CHANNEL_BALANCE_DIAGNOSTIC")
    # Source-loop counts; permutation is a heterogeneous element count, not bytes/BF16 only.
    candidate_bytes = 13 * P * (H + L) // 2 + 8 * P * (H // 960 + L // 960)
    production_bytes = 5 * P * (H + L) + 8 * P
    assert final == {
        "layer": 0, "prefix_inverse_values": 4 * P * H, "exact_unquantized_products": 8 * P * H,
        "intended_bf16_values": 8 * 5 * P * (H + L), "sampled_quantized_outputs": 8 * 3 * P,
        "permutation_values": 4 * (4 * P * (H + L) + 2 * P * (H // 960 + L // 960)),
        "immutable_bytes_compared": 1198677136,
        "reuse_bytes": 4 * (2 * candidate_bytes + production_bytes + 2 * P * L),
        "no_timing_gate": True, "no_task_quality_or_global_bitexactness_claim": True,
    }
    assert at == len(lines), "Unparsed log lines (including possible failure or extra result)"
    # Device permutation is checked by the binary; this additionally cross-checks aggregates.
    for a, b in zip(steps[:2], steps[2:]):
        for x, y in zip(a["drift"], b["drift"]):
            assert x["comparison"] == y["comparison"] and x["changed"] == y["changed"] and x["maxabs"] == y["maxabs"]
            assert abs(x["relative_l2"] - y["relative_l2"]) <= 2e-12
        for key in ("original_coordinates", "balanced_coordinates", "balanced_reconstruction_restored_coordinates"):
            x, y = a["activation"][key], b["activation"][key]
            assert x["values"] == y["values"] and x["reconstructed_zeros"] == y["reconstructed_zeros"]
            assert x["maxabs"] == y["maxabs"] and abs(x["relative_l2"] - y["relative_l2"]) <= 2e-12
    residuals = []
    for s in steps:
        d = {x["comparison"]: x for x in s["drift"]}
        balanced = d["balanced_vs_q020_residual"]["relative_l2"]
        unbalanced = d["unbalanced_vs_q020_residual"]["relative_l2"]
        assert balanced > unbalanced
        residuals.append({"round": s["round"], "step": s["step"], "unbalanced": unbalanced, "balanced": balanced})
    result = {
        "decision": "NO_PROMOTION_FIXED_GAMMA_POLICY",
        "execution": status, "source_commit": run["commit"], "functional_source": "52d821e2",
        "source_inputs": 423, "same_prior_actual323213_device_images": 19,
        "source_input_sha256": PINNED["source-hashes.json"], "submitted_artifact": artifact,
        "raw_files": PINNED, "run_record": run,
        "build_replay": {"result": "PASS", "script_sha256": audit_build.sha(HERE / "audit_build.py"),
                         "result_sha256": audit_build.sha(HERE / "build-result.json")},
        "gamma": gamma, "weight_inverse_proofs": weight_proofs, "weight_metrics": weights,
        "cpu_diagnostic_startup": startup, "additional_layer0_diagnostic_owners": owners,
        "steps": steps, "final_counts": final,
        "total_transformed_weight_values": sum(x["values"] for x in weight_proofs),
        "total_unquantized_dot_samples": sum(x["dots"] for s in steps for x in s["product_proofs"]),
        "residual_relative_l2_on_matched_production_inputs": residuals,
        "scope": [
            "Provided layer0 weights/norms and synthetic512-row attention/residual inputs; two row orders and two matched original-production feedback steps.",
            "All exact inverse/range, sampled unquantized products, intended-code parity, permutation, reuse and immutable-owner checks passed.",
            "Original-coordinate UP/GATE reconstruction and all four final residual comparisons worsen; this deterministic policy is not promoted.",
            "Transformed-coordinate scalar activation error is not original-coordinate error or a dot-error decomposition. Restored step0 activation worsens; step1 L2 slightly improves but max error and final residual worsen.",
            "No timed complete-chain speed test, full-model logits, task quality, global equivalence or serving-capacity proof. CPU diagnostic time includes decoding, inverse auditing, both encoders and metrics; device duration includes all oracle/readback work.",
            "Permutation counter combines BF16, group metadata and unpacked-code elements; immutable/reuse byte totals include repeated comparisons, not unique memory footprint.",
            "Preserves the prior grouped complete-FFN timing HOLD and every original source/control; no new policy or runtime activation.",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
