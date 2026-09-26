#!/usr/bin/env python3
"""Normalize six identical M03 benchmark-stage payloads, retaining raw batch provenance."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'moa-dev'))
from measure import parse_log

SOURCES = [
    ('m03-producer-local-ffn', 'm03-native-local-screen', 'candidate_0'),
    ('m05-local-activation-ready', 'm05-local-prepare-screen', 'candidate_1'),
    ('m06-sharded-input-pipeline', 'm06-sharded-packed-screen', 'candidate_1'),
    ('m06-sharded-input-pipeline', 'm06-sharded-dma-screen', 'candidate_1'),
    ('m07-weight-ingress-chunks', 'm07-stream-assembly-screen', 'candidate_1'),
    ('m03-producer-local-ffn', 'm03-adoption-clean-20260927', 'candidate_0'),
]
KERNELS = ('sliding_project_qkv', 'sliding_attention_output', 'decoder_feedforward')
PAYLOAD = ('test_runtime', 'fixtures.safetensors', 'remote_entrypoint.sh')
CANDIDATE_REPORT = 'docs/experiments/e122-m03-adoption-validation.json'
CONTROL_REPORT = 'docs/experiments/e122-m03-e104-paired-control.json'


def digest(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def stage_files(batch, stage):
    return {name: batch['package_files'][stage + '_' + name] for name in PAYLOAD}


def records():
    candidate_runs, control_runs, deltas = [], [], []
    for family, label, stage in SOURCES:
        path = ROOT / 'moa-dev/experiments/candidates' / family / (label + '-results.json')
        batch = json.loads(path.read_text())
        job = batch['job']['id']
        archive_path = ROOT / 'moa-dev/experiments/artifacts' / family / f'job-{job}.tar'
        with tarfile.open(archive_path) as archive:
            members = [m for m in archive if m.isfile()]
            assert len(members) == len({m.name for m in members})
            actual_archive = {m.name: digest(archive.extractfile(m).read()) for m in members}
        assert actual_archive == batch['package_files']
        log_path = ROOT / batch['log']
        raw = log_path.read_bytes()
        assert digest(raw)['sha256'] == batch['log_sha256']
        sections = dict(re.findall(r'^STRUCTURAL_BEGIN (\w+)\n(.*?)^STRUCTURAL_END \1$', raw.decode(), re.M | re.S))
        assert raw.count(b'STRUCTURAL_COMPLETE') == 1
        for name in ('control_before', stage, 'control_after'):
            assert batch['stages'][name]['valid']
            assert re.findall(r'^STRUCTURAL_EXIT ' + name + r' (\d+)$', sections[name], re.M) == ['0']
            assert parse_log(sections[name]) == batch['stages'][name]['kernels']
        before, after = [batch['stages'][name] for name in ('control_before', 'control_after')]
        assert stage_files(batch, 'control_before') == stage_files(batch, 'control_after')
        provenance = {
            'source_batch_report': str(path.relative_to(ROOT)), 'source_batch_report_digest': digest(path.read_bytes()),
            'original_batch_valid': batch['valid'], 'uploaded_archive_path': str(archive_path.relative_to(ROOT)),
            'uploaded_archive_files': batch['package_files'], 'uploaded_archive_verified': True,
            'package_files_scope': 'Canonical three-file benchmark-stage payload; not the full uploaded batch archive.',
        }
        run = {
            'job': batch['job'], 'log': batch['log'], 'log_sha256': batch['log_sha256'],
            'package_files': stage_files(batch, stage), 'valid': True,
            'kernels': batch['stages'][stage]['kernels'], 'comparisons_passed': batch['stages'][stage]['comparisons_passed'],
            'measurement_kind': 'actual_benchmark_stage', 'original_batch_stage': stage,
            'original_benchmark_files': {name: stage + '_' + name for name in PAYLOAD}, **provenance,
        }
        pair_kernels = {}
        for kernel in KERNELS:
            b, a = before['kernels'][kernel], after['kernels'][kernel]
            pair_kernels[kernel] = {
                'correctness': 'passed',
                'median_cycles': statistics.median([b['median_cycles'], a['median_cycles']]),
                'bracketing_process_medians': [b['median_cycles'], a['median_cycles']],
                'output_comparisons_passed': b['output_comparisons_passed'] + a['output_comparisons_passed'],
            }
        control = {
            'job': batch['job'], 'log': batch['log'], 'log_sha256': batch['log_sha256'],
            'package_files': stage_files(batch, 'control_before'), 'valid': True,
            'measurement_kind': 'derived_bracketing_control_reference',
            'kernels': pair_kernels, 'comparisons_passed': before['comparisons_passed'] + after['comparisons_passed'],
            'control_before': before, 'control_after': after,
            'original_benchmark_files': {name: {p: name + '_' + p for p in PAYLOAD} for name in ('control_before', 'control_after')},
            'note': 'median_cycles is the median of two measured process medians, not an additional benchmark process or three-input timing sample.',
            **copy.deepcopy(provenance),
        }
        delta = 100 * (run['kernels']['decoder_feedforward']['median_cycles'] / pair_kernels['decoder_feedforward']['median_cycles'] - 1)
        deltas.append({'job_id': job, 'candidate_ffn_cycles': run['kernels']['decoder_feedforward']['median_cycles'],
                       'bracketing_control_ffn_cycles': pair_kernels['decoder_feedforward']['median_cycles'], 'delta_percent': delta})
        candidate_runs.append(run)
        control_runs.append(control)
    for runs in (candidate_runs, control_runs):
        assert len({run['job']['id'] for run in runs}) == 6
        assert all(run['package_files'] == runs[0]['package_files'] for run in runs)
    for name in ('fixtures.safetensors', 'remote_entrypoint.sh'):
        assert candidate_runs[0]['package_files'][name] == control_runs[0]['package_files'][name]
    output = []
    for label, runs in (('E122 M03 validation', candidate_runs), ('E104 paired reference for E122', control_runs)):
        aggregate = {}
        for kernel in KERNELS:
            values = [run['kernels'][kernel]['median_cycles'] for run in runs]
            aggregate[kernel] = {'median_cycles': statistics.median(values), 'job_medians': values,
                                 'min_cycles': min(values), 'max_cycles': max(values)}
        output.append({'schema_version': 1, 'label': label, 'valid': True, 'runs': runs,
                       'independent_arena_jobs': 6, 'aggregate': aggregate,
                       'method': 'Six distinct Arena allocations of byte-identical benchmark stage host/fixture/runner, including five historical companion observations and one clean prospective comparison. Full raw uploaded archives and logs retained per run. Two historical outer batches failed because M06 failed; these M03/E104 stages passed. Same three basic inputs reused; not six prospectively identical batch orders.',
                       'official_score': False})
    candidate, control = output
    candidate.update(reference=CONTROL_REPORT, candidate_build='moa-dev/experiments/candidates/m03-producer-local-ffn/build-m03-native-local-v1-repro.json',
                     target_kernel='decoder_feedforward', paired_observations=deltas,
                     median_paired_delta_percent=statistics.median(row['delta_percent'] for row in deltas),
                     aggregate_median_delta_percent=100*(candidate['aggregate']['decoder_feedforward']['median_cycles']/control['aggregate']['decoder_feedforward']['median_cycles']-1),
                     control_after_job_medians=[run['control_after']['kernels']['decoder_feedforward']['median_cycles'] for run in control_runs],
                     unchanged_kernel_timing_note='QKV/O images unchanged; their recorded timing variation is not a candidate optimization.',
                     adoption_budget_exception='Recover an already implemented small gain reproduced in six allocations; no parameter tuning. The 5% exploration-budget cutoff is not an automatic adoption prohibition.')
    control['measurement_kind'] = 'derived_bracketing_control_reference'
    control['method'] += ' Per-job median_cycles is the median of before/after control process medians, explicitly derived; the original full three-kernel controls are retained.'
    return {CANDIDATE_REPORT: candidate, CONTROL_REPORT: control}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for name, data in records().items():
        path = ROOT / name
        if args.check:
            assert json.loads(path.read_text()) == data
        else:
            with path.open('x') as stream:
                json.dump(data, stream, indent=2)
                stream.write('\n')
        print(name, data['aggregate']['decoder_feedforward'])
