#!/usr/bin/env python3
"""Compare bounded OFF/ON summaries; never claim full-vector error/exactness."""
import argparse
import json
import math
from pathlib import Path
import struct


def read(path, enabled):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith('FIXED_E4_ROW '):
            continue
        fields = dict(item.split('=', 1) for item in line.split()[1:])
        assert fields['enabled'] == str(enabled).lower()
        assert fields['nonfinite'] == '0' and fields['values'] == '262144'
        assert fields['cross_program_scope'] == 'summary_and_128_samples_only'
        label = fields['label']
        assert label not in rows, f'duplicate row label {label}'
        fields['samples'] = {int(k): int(v, 16) for k, v in (s.split(':') for s in fields['samples'].split(','))}
        assert len(fields['samples']) == 128
        rows[label] = fields
    assert rows, f'no bounded row reports in {path}'
    return rows


def f32(bits):
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


def compare(off_path, on_path):
    off, on = read(off_path, False), read(on_path, True)
    assert off.keys() == on.keys(), {'missing_on': sorted(off.keys() - on.keys()), 'missing_off': sorted(on.keys() - off.keys())}
    rows = []
    for label, a in off.items():
        b = on[label]
        assert a['samples'].keys() == b['samples'].keys()
        pairs = [(f32(x), f32(b['samples'][i])) for i, x in a['samples'].items()]
        assert all(math.isfinite(x) and math.isfinite(y) for x, y in pairs)
        err = sum((x-y)**2 for x, y in pairs)
        norm = sum(x*x for x, _ in pairs)
        rows.append({
            'label': label,
            'top1_off': int(a['top1']), 'top1_on': int(b['top1']),
            'same_top1': a['top1'] == b['top1'],
            'same_seeded_stochastic': a['seeded_stochastic'] == b['seeded_stochastic'],
            'checksum_fnv64_equal_not_exact_proof': a['checksum_fnv64'] == b['checksum_fnv64'],
            'full_squared_norm_off': float(a['squared_l2']), 'full_squared_norm_on': float(b['squared_l2']),
            'full_sum_off': float(a['sum']), 'full_sum_on': float(b['sum']),
            'sample_count': 128, 'sample_relative_l2': math.sqrt(err / max(norm, 1e-300)),
            'sample_max_abs': max(abs(x-y) for x, y in pairs),
            'sample_bit_differences': sum(x != b['samples'][i] for i, x in a['samples'].items()),
        })
    return {
        'off_log': str(off_path), 'on_log': str(on_path), 'rows': rows,
        'row_count': len(rows), 'top1_changes': sum(not x['same_top1'] for x in rows),
        'mean_sample_relative_l2': sum(x['sample_relative_l2'] for x in rows) / len(rows),
        'max_sample_relative_l2': max(x['sample_relative_l2'] for x in rows),
        'scope': 'Full-vocabulary finite/top1/norm summaries; error only over128 fixed samples. No all-element L2 or cross-program exactness claim. No public quality/performance conclusion.',
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('off'); p.add_argument('on'); p.add_argument('--output')
    args = p.parse_args()
    result = json.dumps(compare(args.off, args.on), indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(result)
    else:
        print(result, end='')
