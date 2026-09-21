"""Independent FP32 order sensitivity probe; neither order is an RNGD oracle."""
import json
import math
import random
import struct


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bf16(x):
    b = bits(x)
    return struct.unpack("<f", struct.pack("<I", (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000))[0]


def sequential(values):
    value = 0.0
    for x in values:
        value = f32(value + x)
    return value


def balanced(values):
    while len(values) > 1:
        values = [f32(values[i] + values[i + 1]) for i in range(0, len(values), 2)]
    return values[0]


def main():
    rng = random.Random(20260921)
    sum_differences = rms_differences = final_differences = max_sum_ulps = 0
    max_relative_sum_error = max_relative_rms_error = max_bf16_absolute_error = 0.0
    for case in range(4096):
        center = rng.randint(-100, 100)
        values = [f32(rng.uniform(0.5, 1.0) * 2.0 ** (center + rng.randint(-3, 3)))
                  for _ in range(8)]
        if case % 2:
            values = [f32(rng.uniform(0.5, 1.0) * 2.0 ** rng.randint(-120, 120))
                      for _ in range(8)]
        a, b = sequential(values), balanced(values)
        sum_differences += a != b
        max_sum_ulps = max(max_sum_ulps, abs(bits(a) - bits(b)))
        max_relative_sum_error = max(max_relative_sum_error, abs(a - b) / max(a, b))
        ra, rb = f32(math.sqrt(f32(a + f32(1e-6)))), f32(math.sqrt(f32(b + f32(1e-6))))
        rms_differences += ra != rb
        max_relative_rms_error = max(max_relative_rms_error, abs(ra - rb) / max(ra, rb))
        for _ in range(8):
            x, gamma = bf16(rng.uniform(-2, 2) * ra), bf16(rng.uniform(0.75, 1.25))
            ya, yb = bf16(f32(f32(x / ra) * gamma)), bf16(f32(f32(x / rb) * gamma))
            final_differences += ya != yb
            max_bf16_absolute_error = max(max_bf16_absolute_error, abs(ya - yb))
    # Pure coordinate proof: each cluster has exactly eight scalar partials.
    for cluster in range(2):
        pairs = [(cluster, s, 0) for s in range(8)]
        gathered = [(cluster, 0, s) for _, s, _ in pairs]
        assert len(set(gathered)) == 8
        assert [p[2] for p in gathered] == list(range(8))
    print(json.dumps({
        "seed": 20260921, "cases": 4096, "bf16_final_outputs": 32768,
        "sequential_vs_balanced_sum_differences": sum_differences,
        "max_sum_ulps": max_sum_ulps, "max_relative_sum_difference": max_relative_sum_error,
        "rms_differences": rms_differences, "max_relative_rms_difference": max_relative_rms_error,
        "bf16_final_differences": final_differences, "max_bf16_absolute_difference": max_bf16_absolute_error,
        "two_cluster_eight_scalar_coordinate_check": "PASS",
        "limitation": "Hypothetical FP32 orders; this does not reproduce either RNGD reduction implementation or prove NPU tolerance."
    }, indent=2))


if __name__ == "__main__":
    main()
