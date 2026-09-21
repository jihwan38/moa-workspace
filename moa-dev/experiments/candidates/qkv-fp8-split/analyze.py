"""Independent E4M3 decomposition checks; NumPy CPU, no model/test fixtures."""
import json
import sys
import numpy as np


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    y = rounded.view(np.float32)
    return np.where(np.abs(x) < np.finfo(np.float32).tiny, np.copysign(0.0, x), y).astype(np.float32)


# Independently enumerate IEEE E4M3 finite positive bit patterns, then nearest-even.
codes = np.arange(127, dtype=np.uint8)
exponents, mantissas = codes >> 3, codes & 7
positive = np.where(exponents == 0, mantissas.astype(np.float64) * 2.0**-9,
                    (1.0 + mantissas.astype(np.float64) / 8.0) * np.exp2(exponents.astype(int) - 7))


def fp8(x):
    x = np.asarray(x, dtype=np.float32)
    a = np.abs(x).astype(np.float64)
    upper = np.searchsorted(positive, a, side="left").clip(0, 126)
    lower = np.maximum(upper - 1, 0)
    du, dl = np.abs(positive[upper] - a), np.abs(a - positive[lower])
    use_upper = (du < dl) | ((du == dl) & ((upper & 1) == 0))
    return np.copysign(positive[np.where(use_upper, upper, lower)], x).astype(np.float32)


def scale(x):
    a = np.float32(max(float(np.max(np.abs(x))), 2.0**-119))
    e = np.array(a.view(np.uint32) & np.uint32(0x7F800000), dtype=np.uint32).view(np.float32)
    return np.float32(e * np.float32(2.0**-7))


def split(x):
    s = scale(x)
    z = (x / s).astype(np.float32)
    hi = fp8(z)
    lo = fp8(np.float32(16.0) * (z - hi))
    # Add before rescaling: s*hi can overflow for near-max finite BF16.
    rebuilt = (s * (hi + np.float32(1.0 / 16.0) * lo)).astype(np.float32)
    return s, z, hi, lo, rebuilt


def summarize(name, x, weights):
    x = bf16(x)
    s, z, hi, lo, rebuilt = split(x)
    error = np.abs(x.astype(np.float64) - rebuilt.astype(np.float64))
    exact_dot = weights.astype(np.float64) @ x.astype(np.float64)
    reconstructed_dot = weights.astype(np.float64) @ rebuilt.astype(np.float64)
    baseline_fp32 = weights @ x
    p_hi, p_lo = weights @ hi, weights @ lo
    split_fp32 = (p_hi + p_lo * np.float32(1 / 16)) * s
    reference_bf16, split_bf16 = bf16(baseline_fp32), bf16(split_fp32)
    bound = abs(float(s)) * 2.0**-14
    dot_bound = np.sum(np.abs(weights.astype(np.float64)), axis=1) * bound
    result = {
        "name": name, "scale": float(s), "amax": float(np.max(np.abs(x))),
        "max_abs_normalized": float(np.max(np.abs(z))),
        "max_abs_hi": float(np.max(np.abs(hi))), "max_abs_lo": float(np.max(np.abs(lo))),
        "nonexact_input_count": int(np.count_nonzero(x != rebuilt)),
        "input_max_abs_error": float(np.max(error)), "input_error_bound": bound,
        "bound_pass": bool(np.all(error <= bound)),
        "normal_binade_exact": bool(np.all((np.abs(z) < 2.0**-6) | (x == rebuilt))),
        "exact_arithmetic_dot_max_abs_error": float(np.max(np.abs(exact_dot - reconstructed_dot))),
        "dot_l1_bound_pass": bool(np.all(np.abs(exact_dot - reconstructed_dot) <= dot_bound + 1e-12)),
        "fp32_two_dot_max_abs_error": float(np.max(np.abs(baseline_fp32.astype(np.float64) - split_fp32))),
        "bf16_dot_mismatches": int(np.count_nonzero(reference_bf16 != split_bf16)),
        "dot_outputs": len(exact_dot),
    }
    if name == "adversarial_cancellation":
        result["first_exact_dot"] = float(exact_dot[0])
        result["first_split_dot"] = float(split_fp32[0])
    return result


def main():
    rng = np.random.default_rng(20260921)
    h, outputs = 3840, 64
    weights = fp8(rng.normal(size=(outputs, h)).astype(np.float32))
    cases = []
    for std in (2.0**-20, 1.0, 2.0**20):
        cases.append(summarize(f"gaussian_std_{std:g}", rng.normal(size=h) * std, weights))
    cases.append(summarize("zeros", np.zeros(h), weights))
    cases.append(summarize("log_uniform_40_bits", rng.choice([-1, 1], h) * np.exp2(rng.uniform(-20, 20, h)), weights))
    x = rng.normal(size=h); x[0] = 2.0**20
    cases.append(summarize("single_outlier_2pow20", x, weights))
    x = np.zeros(h, np.float32); x[:3] = [1, 1, 2.0**-22]
    w = np.zeros((1, h), np.float32); w[0, :3] = [1, -1, 1]
    cases.append(summarize("adversarial_cancellation", x, w))

    # All BF16 encodings, no fixture distribution. Restrict normalized |z| <=256.
    bits = np.arange(65536, dtype=np.uint32) << 16
    all_bf16 = bits.view(np.float32)
    finite = all_bf16[np.isfinite(all_bf16)]
    z = finite[np.abs(finite) <= 256]
    hi = fp8(z); lo = fp8(16 * (z - hi)); rebuilt = hi + lo / 16
    err = np.abs(z.astype(np.float64) - rebuilt.astype(np.float64))
    exhaustive = {
        "count": len(z), "nonexact": int(np.count_nonzero(z != rebuilt)),
        "max_abs_error": float(np.max(err)), "bound": 2.0**-14,
        "bound_pass": bool(np.all(err <= 2.0**-14)),
        "exact_for_abs_at_least_2pow_minus6": bool(np.all((np.abs(z) < 2.0**-6) | (z == rebuilt))),
        "max_abs_hi": float(np.max(np.abs(hi))), "max_abs_lo": float(np.max(np.abs(lo))),
    }
    scale_extremes = []
    for vals in [[0.0], [2.0**-126], [float.fromhex("0x1.fep+127")], [2.0**-126, float.fromhex("0x1.fep+127")]]:
        x = bf16(vals); s, z, hi, lo, rebuilt = split(x)
        scale_extremes.append({"input": list(map(float, x)), "s": float(s),
                               "normalized": list(map(float, z)), "reconstructed": list(map(float, rebuilt)),
                               "finite": bool(np.isfinite(rebuilt).all())})
    result = {"model": "Independent nearest-even saturating E4M3 table; BF16 RNE+FTZ; NumPy CPU matmul is not RNGD reduction order",
              "h": h, "outputs": outputs, "seed": 20260921,
              "exhaustive_scaled_bf16": exhaustive, "sample_cases": cases, "scale_extremes": scale_extremes}
    assert exhaustive["bound_pass"] and exhaustive["exact_for_abs_at_least_2pow_minus6"]
    assert all(c["bound_pass"] and c["normal_binade_exact"] and c["dot_l1_bound_pass"] for c in cases)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
