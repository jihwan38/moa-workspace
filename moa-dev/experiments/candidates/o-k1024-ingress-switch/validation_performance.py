#!/usr/bin/env python3
"""Aggregate three frozen E149-versus-E147 standard-host allocations without running jobs.

Original screen reports remain immutable. Each actual uploaded stage is bound
to the same candidate/E147 host, fixture and runner. Arithmetic probes and extra
input timings are deliberately outside this performance report.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VALIDATION = HERE / 'adoption'
BUILD_LABEL = 'e149-ingress-v1'
LABELS = ('e149-ingress-v1-screen', 'e149-ingress-v1-repeat-a', 'e149-ingress-v1-repeat-b')
CANDIDATE_REPORT = str((VALIDATION / 'e149-adoption-performance.json').relative_to(ROOT))
CONTROL_REPORT = str((VALIDATION / 'e149-e147-paired-control.json').relative_to(ROOT))
sys.path.insert(0, str(HERE))
from package import KERNELS, TARGET, checked_package, digest, file_record, package_files, read, write
from screen import STAGES, verified_batch


def outputs():
    package_path, package, candidate_dir, control_dir = checked_package(BUILD_LABEL)
    normalized = {'candidate': package_files(candidate_dir), 'control': package_files(control_dir)}
    paths = []
    for label in LABELS:
        submission = read(HERE / f'{label}-submission.json')
        paths.append(HERE / f'{label}-job-{submission["job_id"]}-results.json')
    verified = [verified_batch(path) for path in paths]
    assert len({batch['job']['id'] for batch, _ in verified}) == 3
    candidates, controls, paired = [], [], []
    payload = None
    for label, path, (batch, built) in zip(LABELS, paths, verified):
        assert built['label'] == label and built['build_label'] == BUILD_LABEL
        assert built['candidate_package'] == file_record(package_path)
        assert built['control'] == package['control'] and built['graph_gate'] == package['graph_gate']
        assert built['normalized_stage_files'] == normalized
        assert built['stages'] == list(STAGES)
        files = batch['package_files']
        actual_payload = {key: value for key, value in files.items() if key != 'job.yaml'}
        if payload is None:
            payload = actual_payload
        assert actual_payload == payload, 'The three outer batch payloads must also be byte-identical'
        common = {'job': batch['job'], 'log': batch['log'], 'log_sha256': batch['log_digest']['sha256'],
                  'valid': True, 'source_batch_report': str(path.relative_to(ROOT)),
                  'source_batch_report_digest': digest(path.read_bytes()),
                  'uploaded_archive_path': batch['archive']['path'],
                  'uploaded_archive_digest': {k: batch['archive'][k] for k in ('bytes', 'sha256')},
                  'uploaded_archive_files': files, 'uploaded_archive_verified': True,
                  'package_files_scope': 'Canonical three-file benchmark stage; full actual archive remains in uploaded_archive_files.'}
        mappings = {}
        for side in ('candidate', 'control'):
            mappings[side] = {'test_runtime': side + '_test_runtime',
                              'fixtures.safetensors': 'fixtures.safetensors',
                              'remote_entrypoint.sh': 'remote_entrypoint.sh'}
            assert {key: files[value] for key, value in mappings[side].items()} == normalized[side]
        cand = batch['stages']['candidate_whole']
        before = batch['stages']['control_before']
        after = batch['stages']['control_after']
        candidates.append({**common, 'package_files': normalized['candidate'], 'kernels': cand['kernels'],
            'comparisons_passed': 15, 'measurement_kind': 'actual_benchmark_stage',
            'original_batch_stage': 'candidate_whole', 'original_benchmark_files': mappings['candidate']})
        kernels = {kernel: {
            'correctness': 'passed',
            'median_cycles': statistics.median([before['kernels'][kernel]['median_cycles'], after['kernels'][kernel]['median_cycles']]),
            'bracketing_process_medians': [before['kernels'][kernel]['median_cycles'], after['kernels'][kernel]['median_cycles']],
            'output_comparisons_passed': before['kernels'][kernel]['output_comparisons_passed'] + after['kernels'][kernel]['output_comparisons_passed']
        } for kernel in KERNELS}
        controls.append({**common, 'package_files': normalized['control'], 'kernels': kernels,
            'comparisons_passed': 30, 'measurement_kind': 'derived_bracketing_control_reference',
            'control_before': before, 'control_after': after,
            'original_benchmark_files': {stage: mappings['control'] for stage in ('control_before', 'control_after')},
            'note': 'Median of measured before/after process medians. This is a derived reference, not an extra process or independent allocation.'})
        cy, ref = cand['kernels'][TARGET]['median_cycles'], kernels[TARGET]['median_cycles']
        paired.append({'job_id': batch['job']['id'], 'screen_label': label,
            'candidate_o_cycles': cy, 'control_before_o_cycles': before['kernels'][TARGET]['median_cycles'],
            'control_after_o_cycles': after['kernels'][TARGET]['median_cycles'],
            'bracketing_control_o_cycles': ref, 'ratio': cy / ref, 'delta_percent': 100 * (cy / ref - 1)})
    method = ('Three distinct Arena allocations of byte-identical whole-kernel candidate/E147 benchmark stages and outer batch payloads. '
              'Each runs actual E147 before -> frozen E149 O K1024 ingress/Main Switch -> actual E147 after; original three inputs and PROFILE=info. '
              'Uploaded archives and complete raw logs independently rechecked. Inputs, before/after and other kernels are not independent allocations.')

    def report(label, runs):
        assert all(run['package_files'] == runs[0]['package_files'] for run in runs)
        return {'schema_version': 1, 'label': label, 'valid': True, 'runs': runs,
                'independent_arena_jobs': 3, 'method': method,
                'aggregate': {kernel: {
                    'median_cycles': statistics.median(run['kernels'][kernel]['median_cycles'] for run in runs),
                    'job_medians': [run['kernels'][kernel]['median_cycles'] for run in runs],
                    'min_cycles': min(run['kernels'][kernel]['median_cycles'] for run in runs),
                    'max_cycles': max(run['kernels'][kernel]['median_cycles'] for run in runs)
                } for kernel in KERNELS}}

    candidate = report('E149 frozen O K1024 ingress/Main Switch O performance validation', candidates)
    control = report('E147 paired reference for E149 O K1024 ingress/Main Switch', controls)
    median_delta = statistics.median(row['delta_percent'] for row in paired)
    aggregate_delta = 100 * (candidate['aggregate'][TARGET]['median_cycles'] / control['aggregate'][TARGET]['median_cycles'] - 1)
    all_improved = all(row['delta_percent'] < 0 for row in paired)
    candidate.update(reference=CONTROL_REPORT, candidate_build=str((HERE / f'build-{BUILD_LABEL}-repro.json').relative_to(ROOT)),
        candidate_package=str(package_path.relative_to(ROOT)), target_kernel=TARGET,
        paired_observations=paired, median_paired_delta_percent=median_delta,
        aggregate_median_delta_percent=aggregate_delta, all_paired_o_improved=all_improved,
        normalized_stage_payloads=normalized, outer_batch_payloads_identical=True,
        unchanged_kernel_timing_note='QKV/FFN images unchanged E147; their timing variation is not an optimization.',
        accuracy_scope='Three distinct standard inputs reused across allocations;45 candidate output comparisons, including9 O comparisons. Extra6 and boundary oracle checks are separate, required adoption evidence.',
        adoption_policy='Frozen complete-path improvement requires three contemporary allocations plus extra6, actual projection oracle and original whole-O case7. Gain below5% does not block adoption validation; no parameter tuning.',
        adopted=False, official_submission=False)
    summary = {'id': 'E149', 'valid': True, 'independent_allocations': 3,
               'candidate_distinct_standard_inputs': 3, 'candidate_comparisons_passed': 45,
               'candidate_o_comparisons_passed': 9, 'total_standard_comparisons_passed': 135,
               'normalized_stage_payloads': normalized, 'outer_batch_payloads_identical': True,
               'paired_observations': paired, 'median_paired_delta_percent': median_delta,
               'aggregate_median_delta_percent': aggregate_delta, 'all_paired_o_improved': all_improved,
               'candidate_report': CANDIDATE_REPORT, 'control_report': CONTROL_REPORT,
               'builder': file_record(Path(__file__).resolve()), 'method': method,
               'adopted': False, 'official_submission': False,
               'status': 'performance_reproduced_pending_separate_correctness_review' if all_improved else
                         'performance_not_uniform_requires_root_evidence_review',
               'decision_owner': 'Root; this report describes performance only and never adopts automatically.'}
    md = ['# E149 동결 O K1024 ingress/Main Switch O 성능 검증', '',
          '| Job | Candidate O | E147 before / after | Paired 변화율 |', '|---|---:|---|---:|']
    for row in paired:
        md.append(f"| {row['job_id']} | {row['candidate_o_cycles']:,} | {row['control_before_o_cycles']:,} / {row['control_after_o_cycles']:,} | {row['delta_percent']:+.6f}% |")
    md += ['', f'Paired 변화율 중앙값 **{median_delta:+.6f}%**; aggregate 중앙값 비율 **{aggregate_delta:+.6f}%**.',
           '', '세 allocation의 실제 업로드 host·fixture·runner와 바깥 실행 패키지가 byte-identical하다. 원본 세 입력을 반복했으며45개 candidate 비교(O9개), 전체135개 비교가 PASS다. 입력 수나 앞뒤 대조를 독립 allocation으로 세지 않았다.',
           '', '첫 screen 및 기존 보고서는 보존했다. 추가6입력과 별도 projection oracle는 이 성능 집계에 포함하지 않으며 최종 채택 검토에서 따로 확인해야 한다. QKV/FFN timing 변화는 최적화로 해석하지 않는다.',
           '', '5%는 추가 parameter tuning 예산 기준이며 최소 채택 이득이 아니다. 이 기록은 성능 재현 여부만 보고하고 root의 최종 채택 결정을 대신하지 않는다.',
           '', f"상태: `{summary['status']}`. 작성 시점 adopted=false, 공식 제출 없음."]
    return {ROOT / CANDIDATE_REPORT: candidate, ROOT / CONTROL_REPORT: control,
            VALIDATION / 'performance-summary.json': summary}, '\n'.join(md) + '\n'


def report(check=False):
    values, markdown = outputs()
    md_path = VALIDATION / 'performance-summary.md'
    if not check:
        assert all(not path.exists() for path in [*values, md_path]), 'Refusing to overwrite frozen reports'
    for path, value in values.items():
        if check:
            assert read(path) == value, path
        else:
            write(path, value)
    if check:
        assert md_path.read_text() == markdown
    else:
        with md_path.open('x') as stream:
            stream.write(markdown)
    print(json.dumps(values[VALIDATION / 'performance-summary.json'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report(args.check)
