#!/usr/bin/env python3
"""Replay frozen hidden15360 quantizer counts/timings and source/image lineage."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
ELF = 'a5a1b579e32f474ae5f505fecf7627ba8014bcd7cea7b2472267c2aaeddad5de'
MANIFEST = '4cf81395c13ae990db3dbe8859dd0c809cf8a66a3cb825e733473ade0f8e2761'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(lines, prefix):
    return [dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)
            for line in lines if line.startswith(prefix + ' ')]


def analyze(root):
    raw = HERE / 'actual'
    source = root / 'runs/rational-hidden-p512-20261026'
    build_run = root / 'runs/a005-rational-hidden-device-build'
    build = json.loads((HERE / 'device-build.json').read_text())
    run = json.loads((raw / 'run.json').read_text())
    status = json.loads((raw / 'arena-status.json').read_text())
    assert status['id'] == 322970 and status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
    assert status['duration_sec'] < 70
    assert run['command'][run['command'].index('--timeout') + 1] == '70'
    assert '--arg' not in run['command']
    assert run['submitted_artifact']['sha256'] == ELF == build['artifact']['sha256']
    elf_path = build_run / 'built/target/release/test_int4_rational_lut_hidden'
    assert sha(elf_path) == ELF == sha(source / 'submitted/test_int4_rational_lut_hidden')
    inputs_file = build_run / 'source-hashes.json'
    assert sha(inputs_file) == MANIFEST
    inputs = json.loads(inputs_file.read_text())
    assert len(inputs) == 402 and inputs == json.loads((raw / 'source-hashes.json').read_text())
    assert all((WORKTREE / p).is_file() and sha(WORKTREE / p) == h for p, h in inputs.items())
    assert {str(p.relative_to(WORKTREE)) for p in (WORKTREE / 'src').rglob('*') if p.is_file()} == {p for p in inputs if p.startswith('src/')}
    assert len(build['embedded_images']) == 1
    image = build['embedded_images'][0]
    image_path = WORKTREE / 'experiments/int4-rational-lut-hidden/kernel-out-a005-device/furiosa-opt-gemma4' / (image['name'] + '.bin')
    blob = image_path.read_bytes()
    assert sha(image_path) == image['sha256']
    assert elf_path.read_bytes().find(blob) == image['embedded_offset'] >= 0
    resource = WORKTREE / 'experiments/int4-rational-lut-hidden/kernel-out-a003/furiosa-opt-gemma4' / (image['name'] + '.bin')
    assert sha(resource) == image['sha256']
    lines = (raw / 'arena.log').read_text().splitlines()
    allowed = ('RATIONAL_LUT_HIDDEN_', 'PROFILE_RATIONAL_LUT_HIDDEN', 'PASS_RATIONAL_LUT_HIDDEN')
    assert all(not x.strip() or x.startswith(allowed) for x in lines)
    exact = rows(lines, 'RATIONAL_LUT_HIDDEN_EXACT')
    assert len(exact) == 4 and {int(r['case']) for r in exact} == set(range(4))
    for r in exact:
        assert {k: int(r[k]) for k in ['rows', 'width', 'codes', 'code_diff', 'scales', 'scale_diff']} == {
            'rows': 512, 'width': 15360, 'codes': 7864320, 'code_diff': 0, 'scales': 512, 'scale_diff': 0}
    repeated = rows(lines, 'RATIONAL_LUT_HIDDEN_POST_REUSE')
    assert len(repeated) == 4 and {int(r['case']) for r in repeated} == set(range(4))
    assert all(int(r['output_bytes']) == 31459328 for r in repeated)
    markers = rows(lines, 'PASS_RATIONAL_LUT_HIDDEN')
    assert len(markers) == 1
    counts = {k: int(markers[0][k]) for k in ['codes', 'scales', 'permutation_codes', 'permutation_scales', 'immutable_bytes', 'post_reuse_output_bytes', 'rows', 'width']}
    assert counts == {'codes': 31457280, 'scales': 2048, 'permutation_codes': 15728640, 'permutation_scales': 1024,
                      'immutable_bytes': 62930944, 'post_reuse_output_bytes': 125837312, 'rows': 512, 'width': 15360}
    assert sum(int(r['codes']) for r in exact) == counts['codes']
    assert sum(int(r['scales']) for r in exact) == counts['scales']
    assert sum(int(r['output_bytes']) for r in repeated) == counts['post_reuse_output_bytes']
    timing = rows(lines, 'RATIONAL_LUT_HIDDEN_TIME')
    assert len(timing) == 8 and {(int(r['case']), int(r['repeat'])) for r in timing} == {(i, j) for i in range(4) for j in range(2)}
    assert all(r['samples'] == '20' and r['scope'] == 'whole_P512_L15360_I32_codes_and_scales_no_pack_consumer_FFN_model' for r in timing)
    values = [float(r['us']) for r in timing]
    samples = sum(int(r['samples']) for r in timing)
    assert samples == 160
    assert json.loads((raw / 'arena-records.json').read_text()) == []
    references = []
    for p in sorted(raw.iterdir()):
        assert sha(p) == sha(source / p.name)
        references.append({'saved': str(p.relative_to(HERE)), 'original': str(source / p.name), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    return {'result': 'STRICT_HIDDEN_QUANTIZER_DEVICE_FIXTURE_PASS', 'arena': status, 'counts': counts,
            'count_scope': 'Initial CPU code/scale comparisons, cross-case permutations, immutable bytes and post-reuse output bytes are separate assertions; do not sum them as unique inputs.',
            'timing': {'completed_calls': samples, 'blocks': len(timing), 'weighted_mean_us': sum(float(r['us']) * int(r['samples']) for r in timing) / samples,
                       'minimum_block_mean_us': min(values), 'maximum_block_mean_us': max(values), 'median_block_mean_us': statistics.median(values),
                       'all_blocks': timing, 'note': 'Completed-call block means, not per-call percentiles. No matched performance control in this job.'},
            'kernel_calls_source_reconciled': 4 + 160 + 1,
            'lineage': {'compiled_source_commit': build['compiled_source_commit'], 'submission_evidence_commit': run['commit'],
                        'elf_sha256': ELF, 'source_input_manifest_sha256': MANIFEST, 'source_input_count': len(inputs), 'all_inputs_rechecked_equal': True,
                        'embedded_kernel_sha256': image['sha256'], 'embedded_offset': image['embedded_offset'], 'resource_build_image_byte_parity': True},
            'scope': ['Explicit full512×15360 quantizer with complete I32 output and exactF32 reconstruction scales.',
                      'Every fixture output compared, including post-reuse stability; not an exhaustive test of every possible input row.',
                      'Full quantization/index/LUT/scale work and awaited completion included in timing; setup/oracle/readback excluded.',
                      'No I4 packing/consumer, supplied projection, completeFFN or model quality/speed claim.',
                      'No per-operation runtime trace present; compiler resource estimate55,439cycles remains separate.'],
            'references': references}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=WORKTREE.parent)
    parser.add_argument('--output', type=Path, default=HERE / 'actual-result.json')
    args = parser.parse_args()
    result = analyze(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'result': result['result'], 'counts': result['counts'], 'mean_us': result['timing']['weighted_mean_us'],
                      'min_us': result['timing']['minimum_block_mean_us'], 'max_us': result['timing']['maximum_block_mean_us']}, indent=2))
