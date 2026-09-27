#!/usr/bin/env python3
"""Write stage-compatible E137 reports and three separate proposed decisions.

The root reviewer owns the final decision. Every positive complete-kernel
screen calls for bounded validation; the 5% tuning budget never vetoes it.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from package import TARGET, file_record, read, write
from screen import STAGES, verified_batch


def decisions(change):
    positive = change < 0
    return {
        'adoption': {
            'status': 'validate_frozen_candidate' if positive else 'not_promoted_from_this_screen',
            'reason': ('A positive complete-kernel screen warrants bounded independent contemporary repeats, extra6 and changed-boundary oracle checks; gain below5% is eligible.'
                       if positive else 'This single allocation did not show a net gain. Preserve it; root decides whether a concrete unresolved variance or implementation issue justifies another check.'),
            'remaining_checks': ['independent contemporary allocations', 'extra6 same-image correctness',
                                 'original BF16 projection-before-RMS oracle plus full-public cancellation case'] if positive else [],
        },
        'further_tuning': {
            'status': 'stop_parameter_exploration' if change > -5 else 'defer_until_validation_and_cost_review',
            'reason': '5% is a provisional additional-tuning budget, not an adoption minimum. No tile sweep is authorized by this report.'},
        'major_research': {
            'status': 'review_complete_dataflow_cost',
            'question': 'Which actual quantization/globalmax/replication/weight dependency limits the whole path, and is there a supported coupled repair? Do not infer attribution from static intervals or close the native family universally.'},
        'decision_owner': 'Root reviewer; these are evidence-based proposed next actions, not an automatic adoption.'}


def outputs(screen, job):
    batch_path = HERE / f'{screen}-job-{job}-results.json'
    batch, built = verified_batch(batch_path)
    assert batch['job']['id'] == job
    prefix = 'docs/experiments/e137-o-native-activation-resident'

    def report(stage, side):
        data = batch['stages'][stage]
        mapping = {'test_runtime': side + '_test_runtime', 'fixtures.safetensors': 'fixtures.safetensors',
                   'remote_entrypoint.sh': 'remote_entrypoint.sh'}
        assert {k: batch['package_files'][v] for k, v in mapping.items()} == built['normalized_stage_files'][side]
        run = {'job': batch['job'], 'log': batch['log'], 'log_sha256': batch['log_digest']['sha256'],
               'valid': True, 'kernels': data['kernels'], 'comparisons_passed': 15,
               'package_files': batch['package_files'], 'benchmark_files': mapping,
               'normalized_stage_files': built['normalized_stage_files'][side],
               'batch_stage': stage, 'executed_host': side + '_test_runtime',
               'source_batch_report': file_record(batch_path), 'uploaded_archive': batch['archive'],
               'uploaded_archive_verified': True}
        return {'schema_version': 1, 'label': 'E137 ' + stage, 'valid': True, 'runs': [run],
                'independent_arena_jobs': 1, 'method': built['method'],
                'aggregate': {kernel: {'median_cycles': value['median_cycles'],
                                      'job_medians': [value['median_cycles']]}
                              for kernel, value in data['kernels'].items()}}

    before = report('control_before', 'control')
    candidate = report('candidate_whole', 'candidate')
    after = report('control_after', 'control')
    b, c, a = [batch['stages'][name]['kernels'][TARGET]['median_cycles'] for name in STAGES]
    bracket = statistics.median([b, a])
    change = 100 * (c / bracket - 1)
    assert batch['whole_o_comparison'] == {
        'control_before': b, 'candidate': c, 'control_after': a,
        'bracketing_median': bracket, 'candidate_change_percent': change}
    candidate.update(candidate_build=str((HERE / f'build-{built["build_label"]}-repro.json').relative_to(ROOT)),
        candidate_package=built['candidate_package']['path'], reference=prefix + '-e134-before.json',
        control_after_report=prefix + '-e134-after.json', target_kernel=TARGET,
        change_percent_vs_bracketing_median=change, adopted=False, official_submission=False,
        unchanged_kernel_timing_note='QKV/FFN images are unchanged E134; their observed timing variation is not an optimization.',
        accuracy_scope='Original three distinct inputs; candidate15 comparisons include3 O outputs. Independent nine-case projection and legacy full-public O cancellation are separate validation records; no extra6 or repeated performance allocations are included here.')
    compile_record = read(HERE / f'build-{built["build_label"]}-repro.json')
    decision = {'id': 'E137', 'status': 'screen_positive_pending_validation' if change < 0 else 'screen_no_net_gain',
                'adopted': False, 'official_submission': False, 'source_base_commit': compile_record['base_commit'],
                'comparison_recipe': 'baseline/e134.json', 'candidate_build_label': built['build_label'],
                'job': job, 'independent_allocations': 1, 'candidate_distinct_standard_inputs': 3,
                'total_comparisons_passed': 45, 'candidate_comparisons_passed': 15,
                'candidate_o_cycles': c, 'control_before_o_cycles': b, 'control_after_o_cycles': a,
                'bracketing_median': bracket, 'change_percent': change,
                'candidate_cycles_by_run': candidate['runs'][0]['kernels'][TARGET]['cycles_by_run'],
                'static_candidate_cycles': compile_record['static_cycles'],
                'graph_gate': built['graph_gate'], 'source_batch_report': file_record(batch_path),
                'added_raw_f32_write_bytes': 0, 'added_raw_f32_valid_read_bytes': 0,
                'extra_scale_consumer_tasks': 0,
                'validation_scope': 'See separate validation/ records; no historical native result inherited.',
                'scope': 'Complete native E4 activation-resident O with weighted four spatial terms, fused Term/K16 F32 reduction, and common scale after canonical gather. All quantization, replication, TRF/coefficient loads and final public operations included; no internal NPU attribution collected.',
                'decisions': decisions(change)}
    return {ROOT / (prefix + '-e134-before.json'): before,
            ROOT / (prefix + '-e134-after.json'): after,
            ROOT / (prefix + '.json'): candidate, HERE / 'screen-decision.json': decision}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('screen')
    parser.add_argument('job', type=int)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for path, value in outputs(args.screen, args.job).items():
        if args.check:
            assert read(path) == value, path
        else:
            write(path, value)
    print('E137 three standard-stage reports and separate proposed decisions verified')
