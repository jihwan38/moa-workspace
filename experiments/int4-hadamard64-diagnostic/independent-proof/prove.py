#!/usr/bin/env python3
"""Independent fixed H64 sign/orthogonality and binary32 order vectors.

This is a numeric-format/algorithm proof, not model data or an accuracy claim.
Rows x and w use the same R = D H64 / 8, so (xR)(wR)^T = x w^T.
The inverse is R^T, not R: signs are before H in forward and after H in inverse.
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import struct

HERE = Path(__file__).resolve().parent
MASK = (1 << 64) - 1
SEED = 0x243F6A8885A308D3


def sign(channel):
    z = ((channel ^ SEED) + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    z ^= z >> 31
    return -1 if z >> 63 else 1


def h(k, j):
    return -1 if (k & j).bit_count() & 1 else 1


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def from_bits(x):
    return struct.unpack('<f', struct.pack('<I', x))[0]


def bf16_bits(x):
    b = bits(x)
    assert b & 0x7F800000 != 0x7F800000
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF


def direct(v, base):
    # Independent dense rational matrix, not the butterfly implementation.
    return [sum(Fraction(v[k]) * sign(base + k) * h(k, j)
                for k in range(64)) / 8 for j in range(64)]


def butterfly(v, base):
    a = [f32(x * sign(base + i)) for i, x in enumerate(v)]
    stride = 1
    while stride < 64:
        for start in range(0, 64, 2 * stride):
            for j in range(stride):
                x, y = a[start + j], a[start + j + stride]
                a[start + j] = f32(x + y)
                a[start + j + stride] = f32(x - y)
        stride *= 2
    return [f32(x * 0.125) for x in a]


def main():
    gram = [[sum(h(k, i) * h(k, j) for k in range(64))
             for j in range(64)] for i in range(64)]
    assert gram == [[64 if i == j else 0 for j in range(64)] for i in range(64)]
    vectors = []
    for case in range(16):
        base = [0, 64, 896, 960, 3776, 3840, 15296, 15360][case % 8]
        if case == 0:
            x = [0.0] * 64
        elif case == 1:
            x = [1.0] * 64
        elif case == 2:
            x = [1024.0 if i == 31 else 0.0 for i in range(64)]
        elif case == 3:
            x = [-128.0 if i & 1 else 128.0 for i in range(64)]
        elif case == 4:
            x = [2.0 ** 30 if i == 0 else (-1.0 if i & 1 else 1.0) * 2.0 ** -30 for i in range(64)]
        else:
            x = [from_bits(bf16_bits(f32((((i * 73 + case * 127) % 509) - 254) *
                                            2.0 ** ((case % 7) - 10))) << 16) for i in range(64)]
        w = [from_bits(bf16_bits(f32((((i * 41 + case * 19) % 241) - 120) / 64)) << 16) for i in range(64)]
        exact_x, exact_w = direct(x, base), direct(w, base)
        assert sum(a * b for a, b in zip(exact_x, exact_w)) == sum(Fraction(a) * Fraction(b) for a, b in zip(x, w))
        assert sum(a * a for a in exact_x) == sum(Fraction(a) ** 2 for a in x)
        inverse = [sign(base + k) * sum(exact_x[j] * h(k, j) for j in range(64)) / 8 for k in range(64)]
        assert inverse == list(map(Fraction, x))
        actual = butterfly(x, base)
        # Six correctly rounded add/sub stages and exact normal power-of-two scale.
        bound = Fraction(6, (1 << 24) - 6) * sum(abs(Fraction(a)) for a in x) / 8
        assert all(abs(Fraction(a) - e) <= bound for a, e in zip(actual, exact_x))
        assert all(math.isfinite(a) for a in actual)
        vectors.append({'case': case, 'base_channel': base,
                        'input_f32_bits': list(map(bits, x)),
                        'signs': [sign(base + i) for i in range(64)],
                        'rotated_f32_bits': list(map(bits, actual)),
                        'rotated_bf16_bits': list(map(bf16_bits, actual)),
                        'dense_exact_f64': list(map(float, exact_x)),
                        'max_f32_absolute_error': float(max(abs(Fraction(a) - e) for a, e in zip(actual, exact_x))),
                        'f32_absolute_error_bound': float(bound)})
    result = {'seed_hex': hex(SEED), 'dimension': 64, 'normalization': '1/8',
              'forward': 'x D H64 / 8; six F32 butterfly stages, then F32 scale, one BF16 rounding',
              'sign': 'SplitMix64(channel XOR seed), high bit selects minus',
              'scope': 'Exact-real local dot/norm/inverse identity; F32 algorithm vectors on fixed inputs only. No model quality/speed claim.',
              'gram_entries': 4096, 'vectors': vectors}
    path = HERE / 'vectors.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    record = {'result': 'PASS', 'gram_entries': 4096, 'vector_count': len(vectors),
              'vector_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'proof_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / 'proof-record.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
