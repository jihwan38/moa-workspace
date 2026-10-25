#!/usr/bin/env python3
"""Replay the two frozen LUT quantizer gates; no device/network/build commands."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
EXPECTED_ELF = '43664ed77db36c0493c92874b6d07ed88e9d44423a1d13c4f7a3820245229b7c'
EXPECTED_INPUTS = '55a3cb431b60aebebdf6455b999fe4acffbb22fd1a70e5f527324e3aaedbada9'
SCOPE = 'whole_I32_codes_and_scale_no_pack_H15360_or_model'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(line):
    return dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)


def select(lines, prefix):
    return [tokens(line) for line in lines if line.startswith(prefix + ' ')]


def integer_map(row, keys):
    return {key: int(row[key]) for key in keys}


def stats(rows):
    samples = sum(int(r['samples']) for r in rows)
    values = [float(r['us']) for r in rows]
    return {
        'blocks': len(rows), 'completed_calls': samples,
        'weighted_mean_us': sum(float(r['us']) * int(r['samples']) for r in rows) / samples,
        'minimum_block_mean_us': min(values), 'maximum_block_mean_us': max(values),
        'median_block_mean_us': statistics.median(values),
        'note': 'Distribution is of completed-call block means, not per-call percentiles.',
        'all_blocks': rows,
    }


def analyze(root):
    build = json.loads((HERE / 'device-build.json').read_text())
    build_run = root / 'runs/a013-rational-lut-device-build'
    inputs_path = build_run / 'source-hashes.json'
    inputs = json.loads(inputs_path.read_text())
    assert sha(inputs_path) == EXPECTED_INPUTS
    assert len(inputs) == 399
    assert all((WORKTREE / p).is_file() and sha(WORKTREE / p) == h for p, h in inputs.items())
    assert {str(p.relative_to(WORKTREE)) for p in (WORKTREE / 'src').rglob('*') if p.is_file()} == {p for p in inputs if p.startswith('src/')}
    elf_path = build_run / 'built/target/release/test_int4_rational_lut'
    assert sha(elf_path) == EXPECTED_ELF == build['artifact']['sha256']
    elf = elf_path.read_bytes()
    images = []
    for item in build['embedded_images']:
        original_worktree = Path(build['fresh_kernel_output_directory']).parents[2]
        image = (WORKTREE / Path(item['path']).relative_to(original_worktree)).read_bytes()
        assert hashlib.sha256(image).hexdigest() == item['sha256']
        offset = elf.find(image)
        assert offset == item['elf_exact_image_offset'] >= 0
        images.append({'function': item['name'], 'sha256': item['sha256'], 'elf_byte_offset': offset})
    assert len(images) == 3
    references = []
    result = {
        'result': 'STRICT_DEVICE_FIXTURE_PASS',
        'lineage': {'compiled_source_commit': build['source_commit'], 'submitted_evidence_commit': '1b619707ffeae9ccb5dde6e868d2e6232a123ff7',
                    'elf_sha256': EXPECTED_ELF, 'elf_size': len(elf), 'source_input_count': len(inputs),
                    'source_input_manifest_sha256': EXPECTED_INPUTS, 'all_current_inputs_equal': True,
                    'embedded_images': images},
        'scope': ['Finite BF16 FTZ exact-rational fixture codes and exact reconstruction-scale bits.',
                  'Every tested output compared; not an exhaustive test of every possible complete input row.',
                  'Quantizer-only timings include all index/LUT/code/scale work and completed launch waits.',
                  'Setup, upload, oracle and readback excluded symmetrically.',
                  'No H15360 hidden quantization, I4 packing/consumer, complete FFN or model-quality claim.',
                  'No per-operation actual trace is present; resource cycle estimates remain separate.'],
    }
    parsed = {}
    for mode, job in [('b16', 322891), ('p512', 322892)]:
        folder = HERE / 'actual' / mode
        run = json.loads((folder / 'run.json').read_text())
        status = json.loads((folder / 'arena-status.json').read_text())
        assert status['id'] == job and status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
        assert status['duration_sec'] < 70
        assert run['submitted_artifact']['sha256'] == EXPECTED_ELF
        submitted = root / f'runs/rational-lut-{mode}-20261026/submitted/test_int4_rational_lut'
        assert sha(submitted) == EXPECTED_ELF
        assert json.loads((folder / 'source-hashes.json').read_text()) == inputs
        command = run['command']
        assert command[command.index('--arg') + 1] == mode
        assert command[command.index('--timeout') + 1] == '70'
        lines = (folder / 'arena.log').read_text().splitlines()
        allowed = ('RATIONAL_LUT_A4_', 'PASS_RATIONAL_LUT_A4_', 'PROFILE_RATIONAL_LUT_A4_')
        assert all(not line.strip() or line.startswith(allowed) for line in lines), 'unexpected raw log records require review'
        requests = select(lines, 'PASS_RATIONAL_LUT_A4_REQUEST')
        assert len(requests) == 1 and requests[0]['mode'] == mode
        assert json.loads((folder / 'arena-records.json').read_text()) == []
        parsed[mode] = {'exact': select(lines, 'RATIONAL_LUT_A4_EXACT'),
                        'timing': select(lines, 'RATIONAL_LUT_A4_TIME'), 'lines': lines,
                        'status': status}
        for p in sorted(folder.iterdir()):
            if p.is_file():
                original = root / f'runs/rational-lut-{mode}-20261026' / p.name
                assert sha(original) == sha(p)
                references.append({'saved': str(p.relative_to(HERE)), 'original': str(original),
                                   'bytes': p.stat().st_size, 'sha256': sha(p)})
    b = parsed['b16']
    assert len(b['exact']) == 13
    assert {(r['mode'], int(r['case'])) for r in b['exact']} == {('b16', i) for i in range(13)}
    for row in b['exact']:
        assert integer_map(row, ['codes', 'scales', 'code_diff', 'scale_diff']) == {'codes': 61440, 'scales': 16, 'code_diff': 0, 'scale_diff': 0}
    assert len(b['timing']) == 26
    assert {(int(r['case']), int(r['repeat'])) for r in b['timing']} == {(i, j) for i in range(13) for j in range(2)}
    for row in b['timing']:
        assert row['mode'] == 'b16' and row['samples'] == '20' and row['scope'] == SCOPE
    marker = select(b['lines'], 'PASS_RATIONAL_LUT_A4_B16')
    assert len(marker) == 1
    bcounts = integer_map(marker[0], ['codes', 'scales', 'immutable_bytes'])
    assert bcounts == {'codes': 798720, 'scales': 208, 'immutable_bytes': 1650688}
    assert sum(int(r['codes']) for r in b['exact']) == bcounts['codes']
    assert sum(int(r['scales']) for r in b['exact']) == bcounts['scales']
    result['b16'] = {'arena': b['status'], 'counts': bcounts, 'timing': stats(b['timing']),
                     'kernel_calls_source_reconciled': 13 + 520 + 1,
                     'comparison': 'Standalone measurement; no matched old-threshold or power2 timing control in this job.'}
    p = parsed['p512']
    assert len(p['exact']) == 8
    assert {(r['mode'], int(r['case'])) for r in p['exact']} == {(m, i) for m in ['p512_original', 'p512_compact'] for i in range(4)}
    for row in p['exact']:
        assert integer_map(row, ['codes', 'scales', 'code_diff', 'scale_diff']) == {'codes': 1966080, 'scales': 512, 'code_diff': 0, 'scale_diff': 0}
    paired = select(p['lines'], 'RATIONAL_LUT_A4_PAIRED')
    assert len(paired) == 4 and {int(r['case']) for r in paired} == set(range(4))
    assert all(int(r['codes']) == 1966080 and int(r['scales']) == 512 for r in paired)
    assert len(p['timing']) == 16
    assert {(int(r['case']), int(r['block'])) for r in p['timing']} == {(i, j) for i in range(4) for j in range(4)}
    for row in p['timing']:
        assert row['mode'] == 'p512' and row['samples'] == '20' and row['scope'] == SCOPE
        assert row['variant'] == ('original' if int(row['block']) in [0, 3] else 'compact')
    marker = select(p['lines'], 'PASS_RATIONAL_LUT_A4_P512')
    assert len(marker) == 1
    pcounts = integer_map(marker[0], ['candidate_codes', 'original_codes', 'candidate_scales', 'original_scales',
                                     'paired_codes', 'paired_scales', 'permutation_codes', 'permutation_scales', 'immutable_bytes'])
    assert pcounts == {'candidate_codes': 7864320, 'original_codes': 7864320, 'candidate_scales': 2048, 'original_scales': 2048,
                       'paired_codes': 7864320, 'paired_scales': 2048, 'permutation_codes': 3932160, 'permutation_scales': 1024,
                       'immutable_bytes': 15761408}
    original = stats([r for r in p['timing'] if r['variant'] == 'original'])
    compact = stats([r for r in p['timing'] if r['variant'] == 'compact'])
    cases = []
    for case in range(4):
        old = statistics.mean(float(r['us']) for r in p['timing'] if int(r['case']) == case and r['variant'] == 'original')
        new = statistics.mean(float(r['us']) for r in p['timing'] if int(r['case']) == case and r['variant'] == 'compact')
        cases.append({'case': case, 'original_mean_us': old, 'compact_mean_us': new, 'reduction_fraction': 1 - new / old})
    result['p512'] = {'arena': p['status'], 'counts': pcounts, 'timing': {'original': original, 'compact': compact,
                      'reduction_fraction': 1 - compact['weighted_mean_us'] / original['weighted_mean_us'], 'per_case': cases},
                      'kernel_calls_source_reconciled': 8 + 320 + 2,
                      'comparison': 'Same rational arithmetic; original distributed scalar scale store versus compact scale store. Both were strict numeric controls.'}
    result['totals'] = {'cpu_oracle_code_comparisons': bcounts['codes'] + pcounts['candidate_codes'] + pcounts['original_codes'],
                        'cpu_oracle_scale_comparisons': bcounts['scales'] + pcounts['candidate_scales'] + pcounts['original_scales'],
                        'post_all_calls_immutable_bytes': bcounts['immutable_bytes'] + pcounts['immutable_bytes'],
                        'paired_and_permutation_counts_are_additional_comparisons_not_unique_inputs': True}
    assert result['totals']['cpu_oracle_code_comparisons'] == 16527360
    assert result['totals']['cpu_oracle_scale_comparisons'] == 4304
    assert result['totals']['post_all_calls_immutable_bytes'] == 17412096
    result['references'] = references
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=WORKTREE.parent)
    parser.add_argument('--output', type=Path, default=HERE / 'actual-result.json')
    args = parser.parse_args()
    output = analyze(args.root.resolve())
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'result': output['result'], 'totals': output['totals'],
                      'b16_mean_us': output['b16']['timing']['weighted_mean_us'],
                      'p512_original_mean_us': output['p512']['timing']['original']['weighted_mean_us'],
                      'p512_compact_mean_us': output['p512']['timing']['compact']['weighted_mean_us'],
                      'p512_reduction_fraction': output['p512']['timing']['reduction_fraction']}, indent=2))
