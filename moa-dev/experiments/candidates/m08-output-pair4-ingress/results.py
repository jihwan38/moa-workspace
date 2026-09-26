#!/usr/bin/env python3
"""Record the official whole-kernel stages separately from diagnostic profiles."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
O = 'sliding_attention_output'


def outputs(screen, job):
    batch_path = HERE / f'{screen}-job-{job}-results.json'
    batch = json.loads(batch_path.read_text())
    assert batch['valid'] and batch['actual_upload_matches'] and batch['correctness_passes'] == 57
    assert hashlib.sha256((ROOT / batch['log']).read_bytes()).hexdigest() == batch['log_digest']['sha256']

    def report(stage, side):
        data = batch['stages'][stage]
        assert data['valid'] and data['correctness_passes'] == 15
        run = {key: batch[key] for key in ('job', 'log', 'package_files')}
        run.update(valid=True, log_sha256=batch['log_digest']['sha256'],
                   kernels=data['kernels'], comparisons_passed=15,
                   batch_stage=stage, executed_host=side + '_test_runtime',
                   benchmark_files={'test_runtime': side + '_test_runtime',
                                    'fixtures.safetensors': 'fixtures.safetensors',
                                    'remote_entrypoint.sh': 'remote_entrypoint.sh'})
        return {'schema_version': 1, 'valid': True, 'runs': [run], 'independent_arena_jobs': 1,
                'method': 'Official original three-input hosts in one allocation; E122 before and after. Diagnostic info/trace timings excluded from this report.',
                'aggregate': {kernel: {'median_cycles': value['median_cycles'],
                                       'job_medians': [value['median_cycles']]}
                              for kernel, value in data['kernels'].items()}}

    before = report('control_before', 'control')
    candidate = report('candidate_whole', 'candidate')
    after = report('control_after', 'control')
    controls = [r['aggregate'][O]['median_cycles'] for r in (before, after)]
    cycles = candidate['aggregate'][O]['median_cycles']
    reference = statistics.median(controls)
    change = 100 * (cycles / reference - 1)
    assert change == batch['whole_o_comparison']['candidate_change_percent']
    before_path = 'docs/experiments/e125-m08-e122-before.json'
    after_path = 'docs/experiments/e125-m08-e122-after.json'
    candidate.update(candidate_build=str((HERE / 'build-m08-pair4-v1-repro.json').relative_to(ROOT)),
                     candidate_package=str((HERE / 'package-m08-pair4-v1.json').relative_to(ROOT)),
                     reference=before_path, control_after_report=after_path,
                     change_percent_vs_bracketing_median=change, adopted=False,
                     accuracy_scope='Three original inputs only. Nine candidate O comparisons across info/trace/whole reuse the same three inputs; no expanded-input validation yet.')
    status = ('promising_requires_validation' if change <= -5 else
              'closed_measured_regression' if change > 0 else 'closed_below_exploration_screen')
    decision = {
        'id': 'E125', 'candidate': 'M08', 'status': status, 'adopted': False,
        'source_base_commit': 'c3553ab87046fb4d59c10149a82517f8440bba27',
        'comparison_recipe': 'baseline/e122.json', 'job': job, 'independent_allocations': 1,
        'scope': 'One local producer/consumer layout replacement in the historical row-interleave family.',
        'candidate_whole_o': cycles, 'control_whole_o': controls,
        'bracketing_median': reference, 'change_percent': change,
        'candidate_cycles_by_run': candidate['runs'][0]['kernels'][O]['cycles_by_run'],
        'total_comparisons_passed': 57, 'official_whole_comparisons_passed': 45,
        'diagnostic_o_comparisons_passed': 12, 'candidate_distinct_inputs': 3,
        'compile_attempts': 1,
        'static': {'control': 27239, 'candidate': 28259,
                   'weight_interval_unchanged': 13383, 'dot_reduce_interval_unchanged': 5063,
                   'gather_control': 470, 'gather_candidate': 1490},
        'descriptor_gate': 'Exact one-pass15MiB weight coverage, BF16 input q ownership,1920x8B F32 gather,16K F32 sum and scale-before-BF16 verified.',
        'changed_cost': 'HBM row traversal changes with no extra weight copy; direct F32 gather changes32x480B to1920x8B at equal15360B payload.',
        'diagnostic_choice': 'Full O existing engine markers preserve concurrent conditions; no artificial isolated roundtrip or free preprocessing.',
        'profile_analysis': str((HERE / 'profile-analysis-m08-pair4-v1.json').relative_to(ROOT)),
        'official_submission': False,
        'next_gate': ('Independent contemporary repeat and expanded correctness; no adoption from this single screen.'
                      if change <= -5 else 'No parameter sweep, extra-input campaign or adoption for this prototype; preserve measured result and current E122.')}
    return {ROOT / before_path: before, ROOT / after_path: after,
            ROOT / 'docs/experiments/e125-m08-output-pair4.json': candidate,
            HERE / 'decision.json': decision}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('screen')
    parser.add_argument('job', type=int)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for path, value in outputs(args.screen, args.job).items():
        if args.check:
            assert json.loads(path.read_text()) == value, path
        else:
            with path.open('x') as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
    print('E125 whole-host reports and exploration decision verified')
