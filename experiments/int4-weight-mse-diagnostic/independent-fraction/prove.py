#!/usr/bin/env python3
"""Independent exact-rational contract checks, not model calibration or quality."""
from fractions import Fraction as F
from pathlib import Path
from bisect import bisect_left
from collections import Counter
import hashlib
import json
import random
import struct

UNIT = 1 << 38


def power(n):
    return F(1 << n) if n >= 0 else F(1, 1 << -n)


def rn32_bits(x):
    """Software nearest-even conversion; all nonzero arguments are normal."""
    sign = 0x80000000 if x < 0 else 0
    x = abs(x)
    if not x:
        return sign
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if x < power(e):
        e -= 1
    assert -126 <= e <= 127
    y = x * power(23 - e)
    q, r = divmod(y.numerator, y.denominator)
    q += 2 * r > y.denominator or (2 * r == y.denominator and q % 2 == 1)
    if q == 1 << 24:
        q >>= 1
        e += 1
    return sign | ((e + 127) << 23) | (q - (1 << 23))


def from32(bits):
    e, m = (bits >> 23) & 255, bits & 0x7fffff
    assert e != 255
    x = F(m) * power(-149) if e == 0 else F((1 << 23) + m) * power(e - 150)
    return -x if bits >> 31 else x


def bf16(x):
    bits = rn32_bits(x)
    assert bits & 65535 == 0
    return bits >> 16


def e4(code):
    e, m = code >> 3, code & 7
    assert 0 <= code < 127
    return F(m, 512) if e == 0 else F(8 + m) * power(e - 10)


F4 = [F(0), F(1, 2), F(1), F(3, 2), F(2), F(3), F(4), F(6)]
DOMAIN = sorted({a * e4(b) for a in F4 for b in range(127)})
assert len(DOMAIN) == 238 and DOMAIN[1] == F(1, 1024) and DOMAIN[-1] == 2688
assert all(from32(bf16(x) << 16) == x for x in DOMAIN)
UNITS = [int(x * UNIT) for x in DOMAIN]
assert all(x * UNIT == i for x, i in zip(DOMAIN, UNITS))
R7 = from32(rn32_bits(F(1, 7)))
assert rn32_bits(F(1, 7)) == 0x3e124925


def nearest(x, scale):
    t = abs(x) / scale
    n = min(7, (2 * t.numerator + t.denominator) // (2 * t.denominator))
    return -n if x < 0 else n


def original(values):
    a = max(max(map(abs, values)), power(-112))
    return [nearest(x, a / 7) for x in values], rn32_bits(a * R7)


def scales(s0):
    return [rn32_bits(from32(s0) * F(j, 32)) for j in range(8, 33)]


def sse_fraction(values, codes, scale):
    return sum((x - q * scale) ** 2 for x, q in zip(values, codes))


def histogram_score(values, scale):
    """Separate prefix-sum/bucket formula to cross-check scalar Fractions."""
    counts = Counter(abs(x) for x in values)
    c, sx, ss = [0], [0], [0]
    for x, u in zip(DOMAIN, UNITS):
        n = counts[x]
        c.append(c[-1] + n)
        sx.append(sx[-1] + n * u)
        ss.append(ss[-1] + n * u * u)
    su = scale * UNIT
    assert su.denominator == 1
    su = int(su)
    doubled = [2 * u for u in UNITS]
    cuts = [0] + [bisect_left(doubled, (2 * q - 1) * su) for q in range(1, 8)] + [len(DOMAIN)]
    total = 0
    for q in range(8):
        lo, hi = cuts[q:q + 2]
        n, x, xx = c[hi] - c[lo], sx[hi] - sx[lo], ss[hi] - ss[lo]
        term = xx - 2 * q * su * x + q * q * su * su * n
        assert 0 <= term < 1 << 112
        total += term
    assert total < 1 << 112
    return total


def fixture(name, values):
    assert len(values) == 960
    codes, bits = original(values)
    baseline = sse_fraction(values, codes, from32(bits))
    best, best_codes, best_bits, choice = baseline, codes, bits, 0
    if any(values):
        for i, b in enumerate(scales(bits), 1):
            s = from32(b)
            q = [nearest(x, s) for x in values]
            error = sse_fraction(values, q, s)
            assert error * UNIT * UNIT == histogram_score(values, s)
            if error < best:
                best, best_codes, best_bits, choice = error, q, b, i
    assert best <= baseline
    packed = bytes((best_codes[i] & 15) | ((best_codes[i+1] & 15) << 4) for i in range(0, 960, 2))
    for i, q in enumerate(best_codes):
        n = (packed[i // 2] >> (4 * (i % 2))) & 15
        assert (n - 16 if n >= 8 else n) == q
    assert (best * UNIT * UNIT).denominator == (baseline * UNIT * UNIT).denominator == 1
    return {"name": name, "bf16_bits": [bf16(x) for x in values],
            "original_scale_bits": bits, "selected_scale_bits": best_bits,
            "selected_index": choice, "packed_hex": packed.hex(),
            "baseline_sse_units": str(int(baseline * UNIT * UNIT)),
            "selected_sse_units": str(int(best * UNIT * UNIT))}


def main():
    pairs, ties, different = 0, 0, []
    min_scale, max_scale = None, F(0)
    for a in DOMAIN[1:]:
        s0 = rn32_bits(a * R7)
        for j, b in zip(range(8, 33), scales(s0)):
            s = from32(b)
            # Product has <=29 significant bits, exactly representable in binary64.
            machine = struct.unpack('<I', struct.pack('<f', float(from32(s0) * F(j, 32))))[0]
            assert machine == b
            assert (s * UNIT).denominator == 1
            min_scale = s if min_scale is None else min(min_scale, s)
            max_scale = max(max_scale, s)
            su = int(s * UNIT)
            for x in DOMAIN:
                if x > a:
                    break
                q = nearest(x, s)
                threshold_q = sum(2 * int(x * UNIT) >= (2 * k - 1) * su for k in range(1, 8))
                assert q == threshold_q and nearest(-x, s) == -q
                assert abs(x - q * s) < 8192
                t = x / s
                ties += t.denominator == 2 and 0 <= t < 7
                pairs += 1
        for x in DOMAIN:
            if x > a:
                break
            if nearest(x, a / 7) != nearest(x, from32(s0)):
                if len(different) < 8:
                    different.append({"amax": str(a), "value": str(x), "scale_bits": s0,
                                      "rational_code": nearest(x, a/7), "rerounded_code": nearest(x, from32(s0))})
    assert different, "The original code+scale candidate is necessary."
    rng = random.Random(20261026)
    groups = [fixture("zero", [F(0)] * 960),
              fixture("signed-smallest", [DOMAIN[1], -DOMAIN[1]] * 480),
              fixture("max-dense", [DOMAIN[-1], -DOMAIN[-1]] * 480),
              fixture("amax21-rational-tie", [F(21)] + [F(9, 2), F(-9, 2)] * 479 + [F(0)]),
              fixture("single-max-outlier", [DOMAIN[-1]] + [F(1)] * 959)]
    # Signed zero has the same rational value; its original I4 code is also0.
    groups[0]["bf16_bits"] = [0, 0x8000] * 480
    for i in range(11):
        limit = 1 + i * (len(DOMAIN)-2) // 10
        values = [DOMAIN[limit]] + [rng.choice(DOMAIN[:limit+1]) * rng.choice([-1, 1]) for _ in range(959)]
        groups.append(fixture(f"fixed-domain-{i}", values))
    report = {"scope": "Independent provided-format arithmetic proof and synthetic CPU vectors; no model data or task/throughput claim.",
              "domain_magnitudes": len(DOMAIN), "domain_bf16_bits": [bf16(x) for x in DOMAIN],
              "nonzero_amax": len(DOMAIN)-1, "scales_per_amax": 25,
              "exhaustive_nonnegative_pairs": pairs, "exact_half_ties": ties,
              "minimum_scale": str(min_scale), "maximum_scale": str(max_scale),
              "integer_grid_exponent": -38, "conservative_group_bound_bits": 112,
              "bound": "960*(8192*2^38)^2 < 2^112; original and clipped q*s errors bounded by8192.",
              "original_rational_vs_stored_scale_examples": different,
              "fixtures": groups,
              "limitations": ["The SSE objective is exact real q*stored-scale, not RN32(q*scale).",
                              "Neither activation reconstruction, output drift nor full-model loss is monotone.",
                              "All-zero group keeps its original floor-based scale; the2^-38 grid is used only for nonzero groups."]}
    target = Path(__file__).with_name("fraction-vectors.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["domain_magnitudes", "nonzero_amax", "exhaustive_nonnegative_pairs", "exact_half_ties", "minimum_scale", "maximum_scale"]}))
    print(json.dumps({"fixtures": len(groups), "path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
