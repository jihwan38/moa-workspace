#!/usr/bin/env python3
"""Write stage-compatible E151 reports and three separate proposed decisions.

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
                                 'original rawdown9 production helper and full-FFN regression on actual new source/image'] if positive else [],
        },
        'further_tuning': {
            'status': 'stop_parameter_exploration' if change > -5 else 'defer_until_validation_and_cost_review',
            'reason': '5% is a provisional additional-tuning budget, not an adoption minimum. No tile sweep is authorized by this report.'},
        'major_research': {
            'status': 'review_complete_dataflow_cost',
            'question': 'Does full-row low2 scale ingress and in-stream Sub Switch-to-VRF advance the measured scale delivery boundary after the last down native, including all added F32 routing and bitmap/setup? No overlap-window savings are presumed.'},
        'decision_owner': 'Root reviewer; these are evidence-based proposed next actions, not an automatic adoption.'}


def outputs(screen, job):
    batch_path = HERE / f'{screen}-job-{job}-results.json'
    batch, built = verified_batch(batch_path)
    assert batch['job']['id'] == job
    prefix = str((HERE / 'normalized/e151-ffn-down-scale-ingress-switch').relative_to(ROOT))

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
        return {'schema_version': 1, 'label': 'E151 ' + stage, 'valid': True, 'runs': [run],
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
    assert batch['whole_ffn_comparison'] == {
        'control_before': b, 'candidate': c, 'control_after': a,
        'bracketing_median': bracket, 'candidate_change_percent': change}
    candidate.update(candidate_build=str((HERE / f'build-{built["build_label"]}-repro.json').relative_to(ROOT)),
        candidate_package=built['candidate_package']['path'], reference=prefix + '-e150-before.json',
        control_after_report=prefix + '-e150-after.json', target_kernel=TARGET,
        change_percent_vs_bracketing_median=change, adopted=False, official_submission=False,
        unchanged_kernel_timing_note='QKV/O images are unchanged E150; their observed timing variation is not an optimization.',
        accuracy_scope='Original three distinct standard inputs; candidate15 comparisons include3 FFN outputs. Separate root-owned rawdown9 production helper and full-FFN regression suites establish their own finite scope; results are not inherited here.')
    compile_record = read(HERE / f'build-{built["build_label"]}-repro.json')
    decision = {'id': 'E151', 'status': 'screen_positive_pending_validation' if change < 0 else 'screen_no_net_gain',
                'adopted': False, 'official_submission': False, 'source_base_commit': compile_record['base_commit'],
                'comparison_recipe': 'baseline/e150.json', 'candidate_build_label': built['build_label'],
                'job': job, 'independent_allocations': 1, 'candidate_distinct_standard_inputs': 3,
                'total_comparisons_passed': 45, 'candidate_comparisons_passed': 15,
                'candidate_ffn_cycles': c, 'control_before_ffn_cycles': b, 'control_after_ffn_cycles': a,
                'bracketing_median': bracket, 'change_percent': change,
                'candidate_cycles_by_run': candidate['runs'][0]['kernels'][TARGET]['cycles_by_run'],
                'static_candidate_cycles': compile_record['static_cycles'],
                'graph_gate': built['graph_gate'], 'source_batch_report': file_record(batch_path),
                'down_weight_hbm_read_bytes':29491200,'down_weight_dm_write_bytes':29491200,'down_weight_dm_read_bytes':29491200,
                'down_partial_f32_write_bytes':14745600,'down_partial_f32_read_bytes':14745600,
                'down_weight_native_streams':2,'additional_down_streams_vs_e150':0,
                'down_scale_hbm_bytes':3686400,'additional_f32_switch_payload_bytes':14745600,'scale_sub_switch_streams':5,'consumer_scale_vrf_bytes_per_slice':5760,
                'cost_scope':'Original logical weight/partial/scale bytes remain. Replaces five scale Sub loads with full-row ingress and five Sub Switch-to-original-VRF streams, adding14.0625MiB logical F32 routing and bitmap/setup. Whole timings include all costs; no overlap-span or bus-stall savings are presumed.',
                'scope':'Only down scale ingress ownership and Sub routing into original five12 scale VRFs change. Original E150 two weight/native stages, F32 partials, weighted quantizer/broadcast/TRF, scale/reduce arithmetic, epilogue and up/gate remain. QKV/O images unchanged; original reference boundary checks are separate.',
                'decisions': decisions(change)}
    return {ROOT / (prefix + '-e150-before.json'): before,
            ROOT / (prefix + '-e150-after.json'): after,
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
            path.parent.mkdir(parents=True,exist_ok=True)
            write(path, value)
    print('E151 three standard-stage reports and separate proposed decisions verified')
