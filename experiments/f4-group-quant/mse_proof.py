#!/usr/bin/env python3
"""Independent exact Fraction oracle for all dyadic candidates and group fitting."""
import argparse
import collections
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
from cpu_proof import FINITE, LEVELS, decode, exponent_for, f32_bits, offset, p2

SCALES = lambda e: [p2(e)] + [F(m, 8) * p2(e - d) for d in (1, 2, 3) for m in range(8, 16)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    driver = args.driver.resolve()
    values = {b: abs(decode(b)) for b in FINITE}
    codes, scores, denominators = {}, {}, {}
    for base in range(-11, 8):
        fractions = []
        for c, scale in enumerate(SCALES(base)):
            errors = []
            for b in range(127):
                want = min(range(8), key=lambda i: (abs(values[b] - LEVELS[i] * scale), i % 2))
                codes[base, c, b] = want
                codes[base, c, b | 128] = want | 8
                errors.append((values[b] - LEVELS[want] * scale) ** 2)
            fractions.append(errors)
        # Derive a common exact SSE denominator from Fraction results, without
        # the Rust helper's E4-unit/bit-shift formula.
        common = math.lcm(*(x.denominator for row in fractions for x in row))
        denominators[base] = common
        scores[base] = [[x.numerator * (common // x.denominator) for x in row] for row in fractions]

    with tempfile.TemporaryDirectory(prefix='f4-mse-proof-') as directory:
        work = Path(directory)
        table = work / 'table'
        subprocess.run([str(driver), 'mse-tables', str(table)], check=True)
        got = table.with_suffix('.codes').read_bytes()
        scales = struct.unpack('<475I', table.with_suffix('.scales').read_bytes())
        for base in range(-11, 8):
            for c, scale in enumerate(SCALES(base)):
                assert scales[(base + 11) * 25 + c] == f32_bits(scale)
                for b in range(256):
                    expected = codes[base, c, b] if b in FINITE else 255
                    assert got[((base + 11) * 25 + c) * 256 + b] == expected, (base, c, b)

        rows, cols = 480, 1920
        source = bytearray(rows * cols)
        for row in range(rows):
            for part in range(2):
                g = row * 2 + part
                cap = (g * 73 + 31) % 127
                for k in range(960):
                    # Alternate broad histograms and concentrated low values
                    # with a rare large value, exercising clipping candidates.
                    local_cap = cap if g % 2 == 0 else cap // 3
                    mag = (k * 53 + g * 17) % (local_cap + 1)
                    source[row * cols + part * 960 + k] = mag | (128 if (g + k) % 2 else 0)
                source[row * cols + part * 960] = cap
        counts = collections.Counter()
        improved = 0
        baseline_sse = F(0)
        fitted_sse = F(0)

        def check(data, name):
            nonlocal improved, baseline_sse, fitted_sse
            raw = bytearray(len(data))
            for row in range(rows):
                for col in range(cols):
                    raw[offset(row, col, cols)] = data[row * cols + col]
            path, prefix = work / (name + '.input'), work / name
            path.write_bytes(raw)
            subprocess.run([str(driver), 'quantize-mse', str(rows), str(cols), str(path), str(prefix)], check=True)
            assert path.read_bytes() == raw
            packed = prefix.with_suffix('.packed').read_bytes()
            scales = struct.unpack('<960I', prefix.with_suffix('.scales').read_bytes())
            error_reports = struct.unpack('<2880d', prefix.with_suffix('.errors').read_bytes())
            restored = bytearray(len(data))
            restored_scales = []
            for row in range(rows):
                for part in range(2):
                    start = row * cols + part * 960
                    group = data[start:start + 960]
                    histogram = collections.Counter(b & 127 for b in group)
                    maximum = max(values[b] for b in histogram)
                    base = exponent_for(maximum)
                    sse = [sum(n * errors[b] for b, n in histogram.items()) for errors in scores[base]]
                    selected = min(range(25), key=lambda c: (sse[c], c))
                    assert sse[selected] <= sse[0]
                    g = offset(row, part * 960, cols) // 960
                    expected_scale = f32_bits(SCALES(base)[selected])
                    assert scales[g] == expected_scale, (name, row, part, selected)
                    norm = sum(n * values[b]**2 for b, n in histogram.items())
                    assert error_reports[3*g] == float(F(sse[0], denominators[base]))
                    assert error_reports[3*g+1] == float(F(sse[selected], denominators[base]))
                    assert error_reports[3*g+2] == float(norm)
                    restored_scales.append(scales[g])
                    counts[selected] += 1
                    improved += sse[selected] < sse[0]
                    baseline_sse += F(sse[0], denominators[base])
                    fitted_sse += F(sse[selected], denominators[base])
                    for k, b in enumerate(group):
                        i = offset(row, part * 960 + k, cols)
                        actual = (packed[i // 2] >> (4 * (i % 2))) & 15
                        assert actual == codes[base, selected, b]
                        restored[start + k] = actual
            return restored, restored_scales, hashlib.sha256(packed).hexdigest()

        a, sa, ha = check(source, 'original')
        permuted = bytearray(len(source))
        for row in range(rows):
            for part in range(2):
                a0 = (rows - 1 - row) * cols + part * 960
                b0 = row * cols + part * 960
                permuted[b0:b0 + 960] = source[a0:a0 + 960][::-1]
        b, sb, hb = check(permuted, 'permuted')
        for row in range(rows):
            for part in range(2):
                a0, b0 = (rows - 1 - row) * cols + part * 960, row * cols + part * 960
                assert a[a0:a0 + 960][::-1] == b[b0:b0 + 960]
                assert sa[(rows - 1 - row) * 2 + part] == sb[row * 2 + part]
        for sign in (127, 255):
            path = work / 'bad.input'
            raw = bytearray(240 * 960)
            raw[1919] = sign
            path.write_bytes(raw)
            result = subprocess.run([str(driver), 'quantize-mse', '240', '960', str(path), str(work / 'bad')], capture_output=True)
            assert result.returncode != 0 and b'NonFinite' in result.stderr
    bound = 960 * (448 * 2**18)**2
    assert bound < 2**64
    result = dict(status='PASS', candidates=25, base_exponents=19, finite_code_decisions=19*25*254,
                  scale_bit_checks=475, nonfinite_code_rejections=19*25*2, fixture_values=2*rows*cols,
                  exact_group_scale_selections=1920, exact_group_error_reports=1920*3, groups_strictly_improved=improved,
                  candidate_counts=dict(counts), synthetic_SSE_ratio=float(fitted_sse/baseline_sse),
                  u64_conservative_bound=bound, u64_limit=2**64,
                  original_packed_sha256=ha, permuted_packed_sha256=hb, true_same_data_permutation=True,
                  scope='Independent Fraction code/error tables and histogram selection. No supplied-weight, device or task-quality claim.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
