"""CPU FP32/BF16 rounding probe, not an RNGD arithmetic or reduction oracle."""
import json
import numpy as np


def bf16_bits(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16
    # Preserve nonfinite classes instead of applying finite rounding to NaNs.
    return np.where(np.isfinite(x), rounded, bits >> 16).astype(np.uint16)


def from_bf16(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def paths(x, rms, gamma):
    reciprocal = np.float32(np.float32(1) / rms)
    direct = np.float32(np.float32(x / rms) * gamma)
    multiply = np.float32(np.float32(x * reciprocal) * gamma)
    return direct, multiply


with np.errstate(all="ignore"):
    codes = np.arange(65536, dtype=np.uint32)
    x = from_bf16(codes[(codes & 0x7F80) != 0x7F80])
    # rms >= sqrt(EPS); finite FP32 mean_square implies rms <= sqrt(FLT_MAX).
    rms_values = np.geomspace(np.sqrt(np.float32(1e-6)), np.sqrt(np.finfo(np.float32).max), 64, dtype=np.float32)
    gammas = np.asarray([0.5, 1, 1.125, 1.5, 1.75, 2], dtype=np.float32)
    stats = {"bf16_input_codes": int(x.size), "rms_values": int(rms_values.size), "gamma_values": int(gammas.size), "finite_output_pairs": 0, "fp32_different_pairs": 0, "bf16_different_pairs": 0, "nonfinite_class_mismatches": 0, "max_relative_bf16_difference_for_normal_reference": 0.0}
    examples = []
    for rms in rms_values:
        for gamma in gammas:
            a, b = paths(x, rms, gamma)
            valid = np.isfinite(a) & np.isfinite(b)
            stats["finite_output_pairs"] += int(np.count_nonzero(valid))
            stats["fp32_different_pairs"] += int(np.count_nonzero(valid & (a != b)))
            stats["nonfinite_class_mismatches"] += int(np.count_nonzero(np.isfinite(a) != np.isfinite(b)))
            ac, bc = bf16_bits(a), bf16_bits(b)
            changed = valid & (ac != bc)
            stats["bf16_different_pairs"] += int(np.count_nonzero(changed))
            ar, br = from_bf16(ac), from_bf16(bc)
            normal = np.isfinite(ar) & np.isfinite(br) & (np.abs(ar) >= np.finfo(np.float32).tiny)
            if np.any(normal):
                rel = np.abs(ar[normal].astype(np.float64) - br[normal]) / np.abs(ar[normal])
                stats["max_relative_bf16_difference_for_normal_reference"] = max(stats["max_relative_bf16_difference_for_normal_reference"], float(np.max(rel)))
            if len(examples) < 4 and np.any(changed):
                i = int(np.flatnonzero(changed)[0])
                examples.append({"x": float(x[i]), "rms": float(rms), "gamma": float(gamma), "direct_bf16": float(ar[i]), "reciprocal_bf16": float(br[i])})

    rng = np.random.default_rng(20260921)
    actual = {"vectors": 0, "values": 0, "bf16_different_values": 0, "max_abs_difference": 0.0}
    for exponent in (-20, -6, -2, 0, 2, 6, 12, 18):
        for _ in range(8):
            v = from_bf16(bf16_bits(rng.normal(size=3840).astype(np.float32) * np.float32(10.0 ** exponent)))
            gamma = from_bf16(bf16_bits(rng.uniform(0.5, 2.0, size=3840).astype(np.float32)))
            # Both paths consume the same rounded rms. NumPy's reduction tree
            # is deliberately not presented as the RNGD eight-slice tree.
            rms = np.sqrt(np.float32(np.mean(np.float32(v * v), dtype=np.float32) + np.float32(1e-6)))
            a, b = paths(v, rms, gamma)
            ac, bc = bf16_bits(a), bf16_bits(b)
            actual["vectors"] += 1
            actual["values"] += int(v.size)
            actual["bf16_different_values"] += int(np.count_nonzero(ac != bc))
            actual["max_abs_difference"] = max(actual["max_abs_difference"], float(np.max(np.abs(from_bf16(ac) - from_bf16(bc)))))

    infinity = np.float32(np.inf)
    zero_bits = infinity.view(np.uint32) & np.uint32(0)
    clear_plus_one = np.float32(zero_bits.view(np.float32) + np.float32(1))
    assert clear_plus_one == np.float32(1)
    assert np.float32(clear_plus_one / infinity) == np.float32(0)

print(json.dumps({"model": "NumPy FP32 operations plus BF16 round-to-nearest-even; no NPU correctness claim", "exhaustive_x_sampled_rms_gamma": stats, "rounding_difference_examples": examples, "synthetic_rms_vectors": actual, "infinite_rms_bitwise_clear": "PASS; 1/Inf=0, avoids Inf*0 while forming numerator"}, indent=2))
