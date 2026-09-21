#!/usr/bin/env python3
"""Exhaust all finite FP4 x FP8 codes, independently of competition input bands.

Stdlib only. Reads the installed 0.8.1 decoder tables, checks every table entry
against independent format formulas, then proves each rational product fits BF16.
This proves weight dequantization is exact; it does NOT prove reassociated sums
are bit-identical. A separate small FP32 tree model records a counterexample.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct
import subprocess

STD = "/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.8.1/src"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def from_bits(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def bf16bits(value):
    bits = f32bits(value)
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def exact_fp4(code):
    exponent, mantissa = (code >> 1) & 3, code & 1
    magnitude = Fraction(mantissa, 2) if exponent == 0 else Fraction(2 + mantissa, 2) * 2 ** (exponent - 1)
    return -magnitude if code & 8 else magnitude


def exact_fp8(code):
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return None
    magnitude = Fraction(mantissa, 512) if exponent == 0 else Fraction(8 + mantissa, 8) * Fraction(2) ** (exponent - 7)
    return -magnitude if code & 128 else magnitude


def tree_sum(values):
    while len(values) > 1:
        values = [f32(values[i] + values[i + 1]) for i in range(0, len(values), 2)]
    return values[0]


def rounding_example(weights, scales):
    rng = random.Random(0xB10C5CA1E)
    first = None
    for trial in range(20000):
        codes = [rng.randrange(16) for _ in range(16)]
        xb = [bf16bits(rng.uniform(-1, 1)) for _ in range(16)]
        x = [from_bits(bits << 16) for bits in xb]
        sc = rng.choice([code for code, value in enumerate(scales) if math.isfinite(value) and value != 0])
        scale = scales[sc]
        q = [weights[code] for code in codes]
        old = tree_sum([f32(f32(w * scale) * value) for w, value in zip(q, x)])
        new = f32(scale * tree_sum([f32(w * value) for w, value in zip(q, x)]))
        if f32bits(old) != f32bits(new):
            example = {"trial": trial, "fp4_codes": codes, "activation_bf16_bits": xb,
                       "scale_fp8_code": sc, "scale": scale, "scale_before_tree_f32": old,
                       "scale_after_tree_f32": new, "absolute_difference": abs(old - new),
                       "before_bf16_bits": bf16bits(old), "after_bf16_bits": bf16bits(new)}
            if first is None:
                first = example
            if bf16bits(old) != bf16bits(new):
                return {"first_f32_counterexample": first, "bf16_counterexample": example,
                        "tested_blocks": trial + 1,
                        "scope": "CPU balanced FP32 tree only; not a model of the complete RNGD accumulation pipeline."}
    return {"first_f32_counterexample": first, "bf16_counterexample": None, "tested_blocks": 20000,
            "scope": "CPU balanced FP32 tree only; failure to find a BF16 difference is not a proof."}


def prove(container):
    source = {}
    for name in ("scalar.rs", "float/f8e4.rs"):
        source[name] = subprocess.check_output(["docker", "exec", container, "cat", STD + "/" + name], text=True)
    f4_text = source["scalar.rs"].split("const F4E2M1_TO_F8E4:", 1)[1].split("= [", 1)[1].split("];", 1)[0]
    f8_text = source["float/f8e4.rs"].split("const F8E4_TO_F32:", 1)[1].split("= &[", 1)[1].split("];", 1)[0]
    f4_table = [int(value, 16) for value in re.findall(r"0x[0-9a-fA-F]+", f4_text)]
    f8_table = [int(value, 16) for value in re.findall(r"0x[0-9a-fA-F]+", f8_text)]
    assert len(f4_table) == 16 and len(f8_table) == 256, "Unexpected decoder tables"
    scales = [from_bits(bits) for bits in f8_table]
    weights = [scales[code] for code in f4_table]
    for code, value in enumerate(weights):
        assert Fraction.from_float(value) == exact_fp4(code)
        assert (f32bits(value) >> 31) == (code >> 3)
    for code, value in enumerate(scales):
        expected = exact_fp8(code)
        assert math.isnan(value) if expected is None else Fraction.from_float(value) == expected

    count = zero_count = max_precision = 0
    values = []
    failures = []
    for weight_code, weight in enumerate(weights):
        for scale_code, scale in enumerate(scales):
            if not math.isfinite(scale):
                continue
            count += 1
            exact = exact_fp4(weight_code) * exact_fp8(scale_code)
            rounded = f32(weight * scale)
            bits = f32bits(rounded)
            # First prove no F32 rounding; then BF16 is exact iff no discarded bits.
            if Fraction.from_float(rounded) != exact or (bits & 0xFFFF):
                failures.append({"fp4": weight_code, "fp8": scale_code})
            if exact == 0:
                zero_count += 1
                assert bits >> 31 == ((weight_code >> 3) ^ (scale_code >> 7))
            else:
                numerator = abs(exact.numerator)
                numerator >>= (numerator & -numerator).bit_length() - 1
                max_precision = max(max_precision, numerator.bit_length())
                values.append(abs(exact))
    assert not failures
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {name: {"installed_path": STD + "/" + name,
                              "sha256": hashlib.sha256(text.encode()).hexdigest()}
                       for name, text in source.items()},
            "coverage": {"fp4_codes": 16, "fp8_all_codes": 256,
                         "fp8_finite_codes": sum(math.isfinite(x) for x in scales),
                         "finite_products_checked": count, "signed_zero_products_checked": zero_count,
                         "skipped_nonfinite_codes": [i for i, value in enumerate(scales) if not math.isfinite(value)],
                         "competition_input_bands_used": False},
            "all_finite_products_exact_in_bf16": not failures,
            "max_required_significand_bits": max_precision, "bf16_available_significand_bits": 8,
            "minimum_nonzero_product": str(min(values)), "maximum_absolute_product": str(max(values)),
            "weight_dequantization_bf16_rounding_changes_any_finite_product": False,
            "dot_product_reassociation_bit_identity_proven": False,
            "rounding_counterexample": rounding_example(weights, scales)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="moa2026-dev")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prove(args.container)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))
