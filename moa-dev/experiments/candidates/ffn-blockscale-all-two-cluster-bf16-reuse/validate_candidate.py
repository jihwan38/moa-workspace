#!/usr/bin/env python3
"""Check scalar exactness, indexing and a conditional overflow bound; no NPU claims."""

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCALAR_PROOF = HERE.parent / "ffn-blockscale-after-dot/prove_products.py"
SPEC = importlib.util.spec_from_file_location("scalar_proof", SCALAR_PROOF)
FORMATS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FORMATS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    current = (HERE / "mlp.rs").read_text()
    original = (HERE / "mlp.e12-parent.rs").read_text()
    separator = "pub(crate) type DownCluster"
    assert current.split(separator)[0] == original.split(separator)[0]
    assert current.count(".fetch_table_lookup::<bf16>()") == 3
    assert current.count(".fetch_table_lookup::<f8e4m3>()") == 3
    down = current.split("pub(crate) fn project_down", 1)[1]
    stages = [".contract_time::<m![H % 60 = 12, L / 16 % 120]>()",
              ".vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)",
              ".vector_intra_slice_reduce::<L,",
              ".vector_inter_slice_reduce::<DownRows,",
              ".cast::<bf16, m![1 # 16]>()"]
    offsets = [down.index(stage) for stage in stages]
    assert offsets == sorted(offsets)

    rows = [cluster * 1920 + row_group * 60 + tile * 12 + packet * 4 + lane
            for cluster in range(2) for row_group in range(32)
            for tile in range(5) for packet in range(3) for lane in range(4)]
    columns = [column_slice * 1920 + block * 16 + lane
               for column_slice in range(8) for block in range(120) for lane in range(16)]
    assert rows == list(range(3840)) and columns == list(range(15360))
    assert [s * 120 + b for s in range(8) for b in range(120)] == list(range(960))
    assert all((s * 1920 + b * 16 + lane) // 16 == s * 120 + b
               for s in range(8) for b in range(120) for lane in range(16))

    # Exhaust format codes, independently of fixture bands or activation values.
    fp4 = [FORMATS.exact_fp4(code) for code in range(16)]
    fp8 = [FORMATS.exact_fp8(code) for code in range(256)]
    checked = 0
    for q in fp4:
        assert FORMATS.f32bits(float(q)) & 0xFFFF == 0
        for scale in fp8:
            if scale is None:
                continue
            product = q * scale
            converted = FORMATS.f32(float(product))
            assert Fraction.from_float(converted) == product
            assert FORMATS.f32bits(converted) & 0xFFFF == 0
            checked += 1
    assert checked == 4064

    qmax = max(abs(q) for q in fp4)
    smax = max(abs(s) for s in fp8 if s is not None)
    bf16_max = Fraction(255) * 2**120
    f32_max = Fraction(2**24 - 1) * 2**104
    u = Fraction(1, 2**24)
    # Overestimates product/add depth even for sequential accumulation.
    operation_bound = 2 * 15360 + 1024
    assert operation_bound * u < Fraction(1, 2)
    # Thus gamma_n = n*u/(1-n*u) < 1. Absolute summation error is < S.
    coefficient_bound = 15360 * qmax * smax
    exponent = 0
    while 2 * coefficient_bound * 2**(exponent + 1) < bf16_max:
        exponent += 1
    safe_activation = Fraction(2**exponent)
    rounded_sum_bound = 2 * coefficient_bound * safe_activation
    # Include a pessimistic accumulation of subnormal rounding error.
    subnormal_error_bound = Fraction(operation_bound, 2**149) * 2 * smax
    assert rounded_sum_bound + subnormal_error_bound < bf16_max <= f32_max
    unscaled_block_bound = 2 * 16 * qmax * safe_activation
    assert unscaled_block_bound + subnormal_error_bound < f32_max

    # The conditional bound cannot be replaced by "all finite BF16 inputs".
    # Sixteen nonzero activations, q=1, scale=min positive FP8; others are zero.
    activation = Fraction(2**124)
    scale = min(s for s in fp8 if s is not None and s > 0)
    unscaled_dot = 16 * activation
    original_dot = 16 * activation * scale
    assert activation < bf16_max and unscaled_dot > f32_max
    assert original_dot < bf16_max
    assert FORMATS.f32bits(float(activation)) & 0xFFFF == 0
    assert FORMATS.f32bits(float(original_dot)) & 0xFFFF == 0

    return {
        "candidate_mlp_sha256": digest(HERE / "mlp.rs"),
        "parent_mlp_sha256": digest(HERE / "mlp.e12-parent.rs"),
        "scalar_formula_module_sha256": digest(SCALAR_PROOF),
        "installed_decoder_table_validation": "../ffn-blockscale-after-dot/numerical-proof.json",
        "upgate_and_geglu_prefix_byte_identical": True,
        "logical_rows_covered_once": len(rows),
        "logical_columns_covered_once": len(columns),
        "scale_blocks_per_row": 960,
        "down_scale_vrf_bytes_per_slice": 12 * 120 * 4,
        "down_packed_and_bf16_bytes_per_slice": 60 * 1920 // 2 + 60 * 1920 * 2,
        "finite_scalar_products_exact_in_bf16": checked,
        "fp4_to_bf16_exact_codes": len(fp4),
        "exact_real_identity": "sum_b,j ((s_b*q_bj)*x_bj) = sum_b s_b*sum_j(q_bj*x_bj)",
        "identity_reason": "Every old dequantized coefficient s_b*q_bj is BF16-exact; distributivity then applies over exact reals for arbitrary activations.",
        "fp32_bit_identity_claimed": False,
        "rounding_counterexample_reference": "../ffn-blockscale-after-dot/numerical-proof.json",
        "conditional_no_overflow": {
            "scope": "One down projection before its global scale; all inputs/scales finite, |x| bounded by the reported maximum, IEEE round-to-nearest FP32 arithmetic.",
            "maximum_absolute_activation": str(safe_activation),
            "maximum_absolute_activation_power_of_two": exponent,
            "all_finite_fp4_and_fp8_scale_codes_covered": True,
            "max_q": str(qmax), "max_local_scale": str(smax),
            "absolute_sum_coefficient_bound": str(coefficient_bound),
            "operation_count_bound": operation_bound,
            "gamma_bound_less_than": 1,
            "rounded_scaled_sum_bound": str(rounded_sum_bound),
            "bf16_max_finite": str(bf16_max),
            "unscaled_block_dot_bound": str(unscaled_block_bound),
            "subnormal_error_allowance": str(subnormal_error_bound),
            "runtime_bounds_checked": False,
            "competition_fixture_bands_used": False,
            "excluded_claims": ["NaN/Inf equivalence", "hardware flush-to-zero behavior", "FP32 bitwise equality", "overflow after a later global scale"]
        },
        "unconditional_nonoverflow_counterexample": {
            "nonzero_activation_count": 16,
            "activation": "2^124", "fp4_weight": 1, "fp8_scale": str(scale),
            "unscaled_16_term_dot": str(unscaled_dot),
            "original_scaled_dot": str(original_dot),
            "unscaled_dot_exceeds_f32_max": True,
            "original_result_is_finite_bf16": True,
            "observed_on_hardware": False,
            "reason": "A small scale applied after a large unscaled dot can overflow where per-weight scaling would remain finite."
        },
        "hardware_correctness_validated": False
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate()
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))
