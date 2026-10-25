#!/usr/bin/env python3
"""CPU proof for a new rational BF16 A4 rule; no NPU performance claim."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct


FLOOR_BITS = (127 - 112) << 7
FLOOR = Fraction(1, 1 << 112)


def exact_bf16(bits):
    exponent = (bits >> 7) & 255
    assert exponent != 255, "nonfinite input is outside the contract"
    if exponent == 0:
        return Fraction(0)  # RNGD BF16 FTZ policy, including signed zero.
    magnitude = Fraction(128 + (bits & 127)) * Fraction(2) ** (exponent - 134)
    return -magnitude if bits & 0x8000 else magnitude


def oracle(bits, amax_bits):
    magnitude = abs(exact_bf16(bits))
    amax = max(FLOOR, exact_bf16(amax_bits))
    assert magnitude <= amax
    ratio = 7 * magnitude / amax
    nearest = (2 * ratio.numerator + ratio.denominator) // (2 * ratio.denominator)
    assert 0 <= nearest <= 7
    return -nearest if bits & 0x8000 else nearest


def integer_codes(bits, amax_bits):
    magnitude = bits & 0x7fff
    assert magnitude < 0x7f80 and 0 <= amax_bits < 0x7f80
    amax_bits = max(FLOOR_BITS, amax_bits)
    assert magnitude <= amax_bits
    exponent = magnitude >> 7
    if exponent == 0:
        return 0
    delta = (amax_bits >> 7) - exponent
    if delta >= 5:
        return 0
    numerator = 14 * (128 + (magnitude & 127))
    denominator = (128 + (amax_bits & 127)) << delta
    code = sum(numerator >= (2 * k + 1) * denominator for k in range(7))
    return -code if bits & 0x8000 else code


def parallel_threshold_codes(bits, amax_bits):
    """Equivalent eight-lane form using a bounded signed arithmetic shift."""
    magnitude = bits & 0x7fff
    assert magnitude < 0x7f80 and 0 <= amax_bits < 0x7f80
    amax_bits = max(FLOOR_BITS, amax_bits)
    assert magnitude <= amax_bits
    delta = min(5, (amax_bits >> 7) - (magnitude >> 7))
    numerator = 14 * (128 + (magnitude & 127))
    denominator = (128 + (amax_bits & 127)) << delta
    # The eighth (15*denominator) threshold cannot pass. For delta>=5,
    # no threshold passes, including the apparent mantissa of FTZ inputs.
    code = -sum(((2 * k + 1) * denominator - 1 - numerator) >> 31
                for k in range(8))
    return -code if bits & 0x8000 else code


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


normalized = 0
ties = 0
minimum_gap = Fraction(1)
gap_witness = None
max_threshold = 0
# Homogeneity removes the absolute exponent for all nonzero code decisions.
# Every normalized mantissa is visited; delta>=5 is strictly below 1/2.
for delta in range(5):
    for ma in range(128, 256):
        for mx in range(128, 256):
            if delta == 0 and mx > ma:
                continue
            amax_bits = (127 << 7) | (ma - 128)
            positive = ((127 - delta) << 7) | (mx - 128)
            for sign in (0, 0x8000):
                bits = positive | sign
                assert integer_codes(bits, amax_bits) == oracle(bits, amax_bits)
                assert parallel_threshold_codes(bits, amax_bits) == oracle(bits, amax_bits)
                normalized += 1
            ratio = Fraction(7 * mx, ma << delta)
            nearest_threshold_index = max(0, min(6, ratio.numerator // ratio.denominator))
            gap = abs(ratio - Fraction(2 * nearest_threshold_index + 1, 2))
            if gap == 0:
                ties += 1
            elif gap < minimum_gap:
                minimum_gap = gap
                gap_witness = {"mx": mx, "ma": ma, "exponent_difference": delta}
            max_threshold = max(max_threshold, 13 * (ma << delta))

assert Fraction(7 * 255, 128 << 5) < Fraction(1, 2)
boundary_cases = 0
scale_cases = 0
one_seventh = f32(1 / 7)
largest_scale_relative_error = Fraction(0)
# This visits every finite bit pattern with its own row maximum, plus several
# independent fixed row maxima; the normalized proof above covers all others.
for magnitude in range(0x7f80):
    maxima = {max(magnitude, FLOOR_BITS)}
    maxima.update(a for a in (FLOOR_BITS, 0x3f80, 0x7f7f) if magnitude <= a)
    for amax_bits in maxima:
        for sign in (0, 0x8000):
            bits = magnitude | sign
            assert integer_codes(bits, amax_bits) == oracle(bits, amax_bits)
            assert parallel_threshold_codes(bits, amax_bits) == oracle(bits, amax_bits)
            boundary_cases += 1
    if magnitude >= FLOOR_BITS:
        amax = exact_bf16(magnitude)
        # BF16 (8 significant bits) times an F32 constant (24 bits) is exact
        # in Python binary64 before the explicit single binary32 rounding.
        scale = f32(float(amax) * one_seventh)
        bits = f32_bits(scale)
        assert 0 < ((bits >> 23) & 255) < 255
        error = abs(Fraction.from_float(scale) / (amax / 7) - 1)
        largest_scale_relative_error = max(largest_scale_relative_error, error)
        scale_cases += 1

bias = Fraction(1, 1 << 16)
base = Path(__file__).resolve().parent
report = {
    "contract": json.loads((base / "plan.json").read_text())["new_contract"],
    "proof_program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "normalized_signed_cases": normalized,
    "finite_bit_pattern_boundary_cases": boundary_cases,
    "mathematical_half_ties_positive": ties,
    "minimum_non_tie_distance_from_half": str(minimum_gap),
    "minimum_non_tie_distance_float": float(minimum_gap),
    "minimum_gap_witness": gap_witness,
    "maximum_integer_threshold": max_threshold,
    "maximum_clamped_eight_lane_threshold": 15 * 255 * (1 << 5),
    "scale_cases": scale_cases,
    "maximum_scale_relative_error": float(largest_scale_relative_error),
    "bias_alternative": {
        "candidate_magnitude_bias": str(bias),
        "sufficient_total_quotient_and_bias_add_error_bound_strict": str(min(bias, minimum_gap - bias)),
        "hardware_error_bound_established": False,
        "note": "Not an implementation or proof for DivF. Quotient approximation plus bias-add rounding must be bounded below both bias and gap-minus-bias; old tests are unchanged.",
    },
    "result": "CPU_CODE_RULE_PROVED_FOR_FINITE_BF16_WITH_FTZ_AND_FLOOR",
    "device_compiled": False,
    "device_correctness": False,
    "model_quality": False,
    "performance_gain": None,
}
print(json.dumps(report, indent=2))
