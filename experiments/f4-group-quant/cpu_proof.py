#!/usr/bin/env python3
"""Independent Fraction oracle. Run with a compiled, std-only Rust driver."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

LEVELS = list(map(F, [0, F(1, 2), 1, F(3, 2), 2, 3, 4, 6]))
FINITE = [b for b in range(256) if b & 127 != 127]


def p2(e):
    return F(2) ** e


def decode(b):
    m, e = b & 7, (b >> 3) & 15
    value = F(m, 8) * p2(-6) if e == 0 else (1 + F(m, 8)) * p2(e - 7)
    return -value if b & 128 else value


def exponent_for(value):
    value = abs(value)
    if not value:
        return 0
    e = 0
    while 6 * p2(e) < value:
        e += 1
    while 6 * p2(e - 1) >= value:
        e -= 1
    return e


def nearest(b, e):
    target = abs(decode(b)) / p2(e)
    # Minimum distance, then even code. E2M1 nearest ties involve adjacent codes.
    code = min(range(8), key=lambda i: (abs(LEVELS[i] - target), i % 2))
    return code | (8 if b & 128 else 0)


def f32_bits(value, negative_zero=False):
    if negative_zero and not value:
        return 0x80000000
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def offset(row, col, cols):
    return (((row // 240) * (cols // 960) + col // 960) * 240 + row % 240) * 960 + col % 960


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    driver = args.driver.resolve()
    checked = {'finite_e4_patterns': len(FINITE), 'scale_exponents': 19}
    table = {(b, e): nearest(b, e) for e in range(-11, 8) for b in FINITE}
    decoded = {b: decode(b) for b in FINITE}
    exact_e4 = {(abs(decode(b)), bool(b & 128)): b for b in FINITE}
    # Zero's two signs are distinct dictionary keys.
    dense = [exact_e4[(LEVELS[c & 7], bool(c & 8))] for c in range(16)]
    with tempfile.TemporaryDirectory(prefix='f4-group960-proof-') as name:
        work = Path(name)
        prefix = work / 'table'
        subprocess.run([str(driver), 'tables', str(prefix)], check=True)
        got = prefix.with_suffix('.codes').read_bytes()
        for e in range(-11, 8):
            for b in range(256):
                want = table[b, e] if b in FINITE else 255
                assert got[(e + 11) * 256 + b] == want, ('code', b, e, want)
        checked['exact_finite_code_scale_pairs'] = len(table)
        checked['rejected_nonfinite_code_scale_pairs'] = 2 * 19
        got_e = struct.unpack('<256i', prefix.with_suffix('.exponents').read_bytes())
        got_d = struct.unpack('<256I', prefix.with_suffix('.decoded').read_bytes())
        for b in range(256):
            if b not in FINITE:
                assert got_e[b] == 999 and got_d[b] == 0xffffffff
                continue
            assert got_e[b] == exponent_for(decoded[b]), ('scale', b)
            assert got_d[b] == f32_bits(decoded[b], bool(b & 128)), ('decode', b)
        got_f4 = struct.unpack('<16I', prefix.with_suffix('.f4').read_bytes())
        for c in range(16):
            assert got_f4[c] == f32_bits((-1 if c & 8 else 1) * LEVELS[c & 7], bool(c & 8))

        # Distinct canonical rows, two row blocks and two K960 blocks. Vary each
        # group's maximum, with zero/subnormal/extreme and all finite E4 codes.
        rows, cols = 480, 1920
        canonical = bytearray(rows * cols)
        for r in range(rows):
            for kb in range(cols // 960):
                cap = (r * 31 + kb * 47) % 127
                for k in range(960):
                    mag = (r * 17 + kb * 41 + k * 53) % (cap + 1)
                    canonical[r * cols + kb * 960 + k] = mag | (128 if (r + kb + k) % 2 else 0)
                canonical[r * cols + kb * 960] = cap

        def run_fixture(data, label):
            blocked = bytearray(len(data))
            for r in range(rows):
                for k in range(cols):
                    blocked[offset(r, k, cols)] = data[r * cols + k]
            path = work / (label + '.input')
            path.write_bytes(blocked)
            prefix = work / label
            subprocess.run([str(driver), 'quantize', str(rows), str(cols), str(path), str(prefix)], check=True)
            assert path.read_bytes() == blocked
            packed = prefix.with_suffix('.packed').read_bytes()
            expanded = prefix.with_suffix('.dense').read_bytes()
            scales = struct.unpack('<' + str(len(blocked) // 960) + 'I', prefix.with_suffix('.scales').read_bytes())
            assert len(packed) == len(data) // 2 and len(expanded) == len(data)
            canonical_codes = bytearray(len(data))
            canonical_scales = []
            for r in range(rows):
                for kb in range(cols // 960):
                    start = r * cols + kb * 960
                    maximum = max(abs(decoded[b]) for b in data[start:start + 960])
                    e = exponent_for(maximum)
                    g = offset(r, kb * 960, cols) // 960
                    assert scales[g] == f32_bits(p2(e)), ('group-scale', label, r, kb)
                    canonical_scales.append(scales[g])
                    for k in range(960):
                        source = data[start + k]
                        expected = table[source, e]
                        i = offset(r, kb * 960 + k, cols)
                        actual = (packed[i // 2] >> (4 * (i % 2))) & 15
                        assert actual == expected, ('packed', label, r, kb, k)
                        assert expanded[i] == dense[expected], ('dense', label, r, kb, k)
                        canonical_codes[start + k] = actual
            return canonical_codes, canonical_scales, hashlib.sha256(packed).hexdigest()

        codes, scales, digest = run_fixture(canonical, 'original')
        # True same-data permutation: reverse all rows and every K960 block's
        # local K order. Group membership is unchanged, including signs/zeros.
        permuted = bytearray(len(canonical))
        for r in range(rows):
            for kb in range(cols // 960):
                source = (rows - 1 - r) * cols + kb * 960
                permuted[r * cols + kb * 960:r * cols + (kb + 1) * 960] = canonical[source:source + 960][::-1]
        codes_p, scales_p, digest_p = run_fixture(permuted, 'permuted')
        for r in range(rows):
            for kb in range(cols // 960):
                a, b = r * cols + kb * 960, (rows - 1 - r) * cols + kb * 960
                assert codes_p[a:a + 960] == codes[b:b + 960][::-1]
                assert scales_p[r * 2 + kb] == scales[(rows - 1 - r) * 2 + kb]
        checked.update(blocked_fixture_values=2 * rows * cols, blocked_fixture_groups=2 * rows * (cols // 960),
                       packed_bytes=rows * cols, fixture_rows=rows, fixture_cols=cols,
                       original_packed_sha256=digest, permuted_packed_sha256=digest_p,
                       true_same_data_permutation=True)
        # Both NaN signs are rejected; shape validation also precedes allocation.
        for bad in (127, 255):
            path = work / 'bad.input'
            raw = bytearray(240 * 960)
            raw[1919] = bad
            path.write_bytes(raw)
            result = subprocess.run([str(driver), 'quantize', '240', '960', str(path), str(work / 'bad')], capture_output=True)
            assert result.returncode != 0 and b'NonFinite' in result.stderr
        for r, c in ((0, 960), (239, 960), (240, 959), (240, 0)):
            result = subprocess.run([str(driver), 'quantize', str(r), str(c), str(path), str(work / 'bad')], capture_output=True)
            assert result.returncode != 0 and b'InvalidShape' in result.stderr
        checked['nonfinite_group_rejections'] = 2
        checked['invalid_shape_rejections'] = 4
    checked.update(status='PASS', oracle='Python Fraction nearest-code and while-loop scale selection, compared against compiled Rust helper',
                   scope='CPU helper only; no provided-weight statistics, device/model correctness, task quality or performance claim')
    args.output.write_text(json.dumps(checked, indent=2) + '\n')
    print(json.dumps(checked, indent=2))


if __name__ == '__main__':
    main()
