#!/usr/bin/env python3
"""Synthetic EOS/replay mutations plus actual completed continuation evidence.

No model execution. Each negative checks its intended gate's exact error.
"""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True
import collect_model_continuation as collector

R = collector.ROOT
N = collector.N


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def choice(token, value=8.0, runner=999, runner_value=7.0, ties=1):
    return {'selected_id': token, 'selected_logit': value, 'selected_logit_bits': bits(value),
            'runner_up_id': runner, 'runner_up_logit': runner_value, 'runner_up_logit_bits': bits(runner_value),
            'margin': value - runner_value, 'tie_count': ties,
            'rank_order': 'f32_total_cmp', 'tie_rule': 'public_sample_unstable_order'}


def top(token, value=8.0, runner=999, runner_value=7.0):
    return {'top1_id': token, 'top1_logit': value, 'top2_id': runner, 'top2_logit': runner_value,
            'top1_margin': value - runner_value, 'top1_tie_count': 1, 'tie_rule': 'lowest_token_id'}


def stats(reference, candidate):
    # Reference has chosen=8, runner999=7 and all other logits zero. For a
    # differing candidate, raise its previously-zero token to9. All formulas
    # below cover the full N-sized virtual arrays without allocating them.
    differs = reference != candidate
    error = 9.0 if differs else 0.0
    return {'elements_compared': N, 'max_abs': error, 'mean_abs': error / N,
            'rmse': error / math.sqrt(N), 'bit_equal_count': N - int(differs),
            'max_abs_index': candidate if differs else 0,
            'reference_at_max_abs': 0.0, 'actual_at_max_abs': error,
            'reference_top': top(reference),
            'actual_top': top(candidate, 9.0, reference, 8.0) if differs else top(reference),
            'diagnostic_absolute_error_counts': [
                {'absolute_error_at_most': t, 'count': N - int(t < error), 'count_above': int(t < error)}
                for t in [0, 1e-6, 1e-4, 1e-3, .01, .1, .5, 1, 2, 5]],
            'accuracy_acceptance_threshold': None, 'accuracy_pass': None}


def event(name, mode, **fields):
    return {'schema': collector.SCHEMA, 'event': name, 'mode': mode,
            'accuracy_acceptance_threshold': None, 'accuracy_pass': None, **fields}


def synthetic_events():
    ids = [[1000 + i] * length for i, length in enumerate([14, 21, 19, 28])]
    selections = [[1], [20, 106], list(range(30, 38)), [40, 41, 50]]
    reasons = ['eos', 'eos', 'length', 'eos']
    counts = list(map(len, selections))
    result = []
    for mode in ('emit', 'compare'):
        result.append(event('continuation_inputs_validated', mode, case_count=4,
                            prompts=collector.PROMPTS, token_ids=copy.deepcopy(ids),
                            total_prompt_tokens=82, guarded_total_tokens=114,
                            max_new_tokens=8, max_total_tokens=128, guard_before_device_new=True,
                            thinking=False, add_generation_prompt=True,
                            sampler={'temperature': 1.0, 'top_k': 1, 'top_p': 1.0},
                            banned_tokens=[258882, 258883], eos_tokens=[1, 106, 50],
                            termination_policy=collector.TERMINATION, replay_policy=collector.REPLAY,
                            reference_all_checkpoints_validated=mode == 'compare',
                            reference_parse_seconds=.01, elapsed_seconds=.1))
        for case, selected in enumerate(selections):
            first = None
            for step, token in enumerate(selected):
                checkpoint = event('reference_checkpoint_written' if mode == 'emit' else 'checkpoint_logits_compared', mode,
                                   case_index=case, step=step, position=len(ids[case]) - 1 + step,
                                   element_count=N, nonfinite_count=0,
                                   checkpoint_file=f'case-{case}-step-{step}.json', fresh_workspace_for_case=True,
                                   timings={'decode_and_read_seconds': .1, 'checkpoint_seconds': .2, 'elapsed_suite_seconds': 1.0},
                                   consumed_token_ids=ids[case] + selected[:step],
                                   reference_selected_token=token, reference_selected_is_eos=token in collector.EOS,
                                   reference_legal_choice=choice(token))
                if mode == 'compare':
                    # An early candidate EOS must not truncate the 8-checkpoint
                    # forced-original case. Another case disagrees later.
                    candidate = 1 if (case, step) == (2, 0) else 42 if (case, step) == (3, 1) else token
                    if candidate != token and first is None:
                        first = step
                    checkpoint.update(candidate_selected_token=candidate,
                                      candidate_selected_is_eos=candidate in collector.EOS,
                                      candidate_legal_choice=choice(candidate, 9.0, token, 8.0) if candidate != token else choice(token),
                                      selected_token_match=candidate == token,
                                      first_legal_greedy_disagreement_step=first,
                                      reference_choice_candidate_rank=2 if candidate != token else 1,
                                      replay_policy=collector.REPLAY, comparison=stats(token, candidate))
                result.append(checkpoint)
            result.append(event('reference_case_complete' if mode == 'emit' else 'comparison_case_complete', mode,
                                case_index=case, checkpoint_count=len(selected),
                                completion_tokens=len(selected) - int(reasons[case] == 'eos'),
                                forward_tokens=len(ids[case]) + len(selected) - 1,
                                finish_reason=reasons[case], reference_selected_tokens=list(selected),
                                first_legal_greedy_disagreement_step=first, total_elements=len(selected) * N,
                                prompt=collector.PROMPTS[case], token_ids=ids[case], consumed_token_ids=ids[case] + selected[:-1],
                                fresh_workspace=True, replay_policy=collector.REPLAY,
                                timings={'workspace_seconds': .1, 'prefill_seconds': .1, 'case_seconds': 1., 'elapsed_suite_seconds': 4.}))
        result.append(event('continuation_reference_complete' if mode == 'emit' else 'continuation_comparison_complete', mode,
                            completed_cases=4, total_prompt_tokens=82, guarded_total_tokens=114,
                            max_total_tokens=128, max_new_tokens=8, total_checkpoints=sum(counts),
                            total_forward_tokens=82 + sum(counts) - 4, total_elements=sum(counts) * N,
                            case_checkpoint_counts=counts, case_finish_reasons=reasons,
                            elements_per_checkpoint=N, manifest_file='manifest.json',
                            termination_policy=collector.TERMINATION, replay_policy=collector.REPLAY,
                            first_legal_greedy_disagreement_steps=[None] * 4 if mode == 'emit' else [None, None, 0, 1],
                            reference_parse_seconds=.01, load_seconds=1.0, total_seconds=5.0))
    return result


def synthetic_lines(events):
    # Reuse actual, independently verified asset-report shape; do not invent
    # successful weight hashing or require the remote 9.3-GB payload locally.
    saved = json.loads((R / 'docs/experiments/e55-model-prefix-suite.json').read_text())
    assets = saved['asset_snapshots']
    scope = collector.validate_events(events)['scope']
    files = {'reference/manifest.json'} | {
        f'reference/case-{case}-step-{step}.json'
        for case, count in enumerate(scope['case_checkpoint_counts']) for step in range(count)}
    hashes = [hashlib.sha256(name.encode()).hexdigest() + '  ' + name for name in sorted(files)]
    emit = [json.dumps(item) for item in events if item['mode'] == 'emit']
    compare = [json.dumps(item) for item in events if item['mode'] == 'compare']
    manifest = next(line for line in hashes if line.endswith('  reference/manifest.json'))
    return ['MODEL_CONTINUATION_PAIR_BEGIN', 'MODEL_CONTINUATION_ASSETS_BEFORE_BEGIN', json.dumps(assets[0]),
            'MODEL_CONTINUATION_ASSETS_BEFORE_END', *emit,
            'MODEL_CONTINUATION_REFERENCE_MANIFEST_SHA256', manifest,
            'MODEL_CONTINUATION_REFERENCE_FILES_SHA256_BEGIN', *hashes, 'MODEL_CONTINUATION_REFERENCE_FILES_SHA256_END',
            *compare,
            'MODEL_CONTINUATION_REFERENCE_FILES_AFTER_SHA256_BEGIN', *hashes, 'MODEL_CONTINUATION_REFERENCE_FILES_AFTER_SHA256_END',
            'MODEL_CONTINUATION_ASSETS_AFTER_BEGIN', json.dumps(assets[1]), 'MODEL_CONTINUATION_ASSETS_AFTER_END',
            'MODEL_CONTINUATION_PAIR_STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT', 'MODEL_CONTINUATION_PAIR_EXIT=0']


def synthetic_package(directory, payloads=None, job_yaml=None, member_kind=None):
    """Small arbitrary bytes exercise archive attribution, not ELF validation."""
    directory = Path(directory)
    payloads = payloads or {name: ('synthetic payload: ' + name).encode() for name in collector.MEMBERS}
    files = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
             for name, data in payloads.items()}
    package = {'files': files}
    archive = directory / 'submitted.tar'
    job_yaml = job_yaml or b'name: synthetic-continuation\nentrypoint: ./remote_entrypoint.sh\nargs: []\nenv: {}\ntimeout_sec: 70\n'
    with tarfile.open(archive, 'w') as tar:
        for name, data in {**payloads, 'job.yaml': job_yaml}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            if name == member_kind:
                info.type = tarfile.SYMTYPE
                info.linkname = 'elsewhere'
                info.size = 0
                tar.addfile(info)
            else:
                tar.addfile(info, io.BytesIO(data))
    lines = [files[name]['sha256'] + '  ./' + name for name in sorted(collector.MEMBERS - {'remote_entrypoint.sh'})]
    return archive, package, lines


def main():
    passed = []

    def reject(label, operation, message):
        try:
            operation()
        except ValueError as error:
            if str(error) != message:
                raise AssertionError(f'{label}: wrong gate: {error}') from error
        else:
            raise AssertionError('Incorrectly accepted ' + label)
        passed.append(label)

    original = synthetic_events()
    validated = collector.validate_events(original)
    if validated['scope']['case_checkpoint_counts'] != [1, 2, 8, 3]:
        raise AssertionError('variable EOS counts')
    if [c['first_legal_greedy_disagreement_step'] for c in validated['case_summaries']] != [None, None, 0, 1]:
        raise AssertionError('forced replay first disagreements')
    passed.append('synthetic_variable_EOS_and_candidate_early_EOS_positive')
    collector.legal_choice(choice(42, 0.0, 43, -0.0, 1), 42)
    passed.append('signed_zero_distinct_total_cmp_zero_margin_positive')
    collector.legal_choice(choice(43, 8.0, 42, 8.0, 2), 43)
    passed.append('unstable_tie_allows_nonlowest_token_positive')

    def mutate_event(label, predicate, mutation, error):
        data = copy.deepcopy(original)
        item = next(item for item in data if predicate(item))
        mutation(item)
        reject(label, lambda: collector.validate_events(data), error)

    checkpoint = lambda x: x['event'] == 'checkpoint_logits_compared' and x['case_index'] == 2 and x['step'] == 1
    first_checkpoint = lambda x: x['event'] == 'checkpoint_logits_compared' and x['case_index'] == 2 and x['step'] == 0
    mutate_event('wrong_consumed_reference_history', checkpoint,
                 lambda x: x['consumed_token_ids'].__setitem__(-1, 1), 'forced reference history mismatch')
    mutate_event('candidate_EOS_flag_wrong', first_checkpoint, lambda x: x.update(candidate_selected_is_eos=False), 'candidate EOS flag mismatch')
    mutate_event('wrong_replay_policy', checkpoint, lambda x: x.update(replay_policy='candidate_tokens'), 'wrong candidate replay policy')
    mutate_event('first_disagreement_erased', checkpoint, lambda x: x.update(first_legal_greedy_disagreement_step=None), 'incorrect first legal greedy disagreement')
    mutate_event('truncated_comparison', checkpoint, lambda x: x['comparison'].update(elements_compared=N - 1), 'truncated logit comparison')
    mutate_event('nonfinite_report', checkpoint, lambda x: x.update(nonfinite_count=1), 'invalid checkpoint position/finite/shape')
    mutate_event('arbitrary_accuracy_PASS', checkpoint, lambda x: x.update(accuracy_pass=True), 'unexpected accuracy verdict')
    mutate_event('wrong_final_forward_count', lambda x: x['event'] == 'continuation_comparison_complete',
                 lambda x: x.update(total_forward_tokens=999), 'continuation final counters mismatch')
    mutate_event('wrong_length_finish', lambda x: x['event'] == 'comparison_case_complete' and x['case_index'] == 2,
                 lambda x: x.update(finish_reason='eos'), 'case completion metadata/history mismatch')
    mutate_event('wrong_completion_tokens_includes_EOS', lambda x: x['event'] == 'reference_case_complete' and x['case_index'] == 0,
                 lambda x: x.update(completion_tokens=1), 'case completion counters mismatch')
    mutate_event('candidate_wrong_reference_choice', checkpoint,
                 lambda x: x['reference_legal_choice'].update(runner_up_id=998), 'candidate reference choice differs from emitted reference')
    mutate_event('rank_outside_allowed_vocabulary', checkpoint,
                 lambda x: x.update(reference_choice_candidate_rank=N), 'invalid reference choice candidate rank')
    mutate_event('rank_one_disagreement_without_tie', first_checkpoint,
                 lambda x: x.update(reference_choice_candidate_rank=1), 'different rank-one reference choice requires a legal tie')
    mutate_event('duplicated_step', checkpoint, lambda x: x.update(step=0), 'checkpoint case/step is duplicated, missing or reordered')
    input_compare = lambda x: x['event'] == 'continuation_inputs_validated' and x['mode'] == 'compare'
    case_complete = lambda x: x['event'] == 'comparison_case_complete' and x['case_index'] == 2
    final = lambda x: x['event'] == 'continuation_comparison_complete'
    mutate_event('chat_thinking_policy', input_compare, lambda x: x.update(thinking=True), 'wrong chat policy')
    mutate_event('sampler_changed', input_compare, lambda x: x['sampler'].update(top_k=2), 'wrong public sampler policy')
    mutate_event('EOS_array_changed', input_compare, lambda x: x.update(eos_tokens=[1, 50, 106]), 'wrong banned/EOS token policy')
    mutate_event('reference_not_prevalidated', input_compare, lambda x: x.update(reference_all_checkpoints_validated=False),
                 'reference checkpoints were not validated before device creation')
    mutate_event('checkpoint_wrong_filename', checkpoint, lambda x: x.update(checkpoint_file='case-2-step-0.json'), 'wrong checkpoint filename')
    mutate_event('checkpoint_reuses_workspace', checkpoint, lambda x: x.update(fresh_workspace_for_case=False), 'checkpoint did not use fresh case workspace')
    mutate_event('checkpoint_infinite_timing', checkpoint, lambda x: x['timings'].update(checkpoint_seconds=float('inf')), 'invalid finite timing')
    mutate_event('case_wrong_consumed_history', case_complete, lambda x: x['consumed_token_ids'].__setitem__(-1, 1), 'case completed with wrong prompt/feed history')
    mutate_event('case_reuses_workspace', case_complete, lambda x: x.update(fresh_workspace=False), 'case workspace/replay policy mismatch')
    mutate_event('final_boolean_count', final, lambda x: x['case_checkpoint_counts'].__setitem__(0, True), 'continuation final case/guard mismatch')
    mutate_event('final_wrong_manifest', final, lambda x: x.update(manifest_file='other.json'), 'continuation final file/shape/policy mismatch')
    mutate_event('final_wrong_first_disagreements', final, lambda x: x['first_legal_greedy_disagreement_steps'].__setitem__(2, None), 'continuation final first-disagreement list mismatch')
    mutate_event('final_negative_load_time', final, lambda x: x.update(load_seconds=-1), 'invalid finite timing')
    no_case = [x for x in original if not (x.get('case_index') == 3 and x['mode'] == 'compare')]
    reject('missing_entire_last_case', lambda: collector.validate_events(no_case), 'case has no checkpoints or exceeds budget')
    truncated = original[:-1]
    reject('missing_final_completion', lambda: collector.validate_events(truncated), 'continuation events are incomplete')
    bad = choice(42, 0.0, 43, -0.0, 2)
    reject('signed_zero_wrong_tie_count', lambda: collector.legal_choice(bad, 42), 'total_cmp tie count disagrees with runner-up')
    bad = choice(42); bad['selected_logit_bits'] ^= 1
    reject('wrong_f32_bits', lambda: collector.legal_choice(bad, 42), 'legal logit bits disagree with finite f32 value')
    reject('banned_choice', lambda: collector.legal_choice(choice(258882), 258882), 'selected token is invalid or banned')
    reject('nonmaximal_choice', lambda: collector.legal_choice(choice(42, 6.0, 43, 7.0), 42), 'selected token is below legal runner-up')

    lines = synthetic_lines(original)
    collector.validate_log(lines)
    passed.append('synthetic_full_marker_hash_asset_log_positive')
    missing = list(lines)
    del missing[missing.index('MODEL_CONTINUATION_REFERENCE_FILES_AFTER_SHA256_BEGIN') + 1]
    reject('missing_after_reference_hash', lambda: collector.validate_log(missing), 'reference checkpoint file set mismatch')
    changed = list(lines)
    i = changed.index('MODEL_CONTINUATION_REFERENCE_FILES_AFTER_SHA256_BEGIN') + 1
    changed[i] = '0' * 64 + changed[i][64:]
    reject('reference_changed_during_replay', lambda: collector.validate_log(changed), 'reference files changed during candidate replay')
    changed = list(lines)
    i = changed.index('MODEL_CONTINUATION_REFERENCE_MANIFEST_SHA256') + 1
    changed[i] = '0' * 64 + changed[i][64:]
    reject('manifest_hash_not_inventory', lambda: collector.validate_log(changed), 'standalone manifest hash disagrees with reference file inventory')
    missing = [line for line in lines if line != 'MODEL_CONTINUATION_PAIR_EXIT=0']
    reject('missing_runner_exit', lambda: collector.validate_log(missing), 'missing/duplicate marker: MODEL_CONTINUATION_PAIR_EXIT=0')
    asset_partial = list(lines)
    i = asset_partial.index('MODEL_CONTINUATION_ASSETS_BEFORE_BEGIN') + 1
    asset = json.loads(asset_partial[i])
    asset['files'][0]['full_file']['status'] = 'incomplete'
    asset_partial[i] = json.dumps(asset)
    reject('partial_asset_identity', lambda: collector.validate_log(asset_partial), 'incomplete before-run asset identity: model.safetensors')

    with tempfile.TemporaryDirectory(prefix='moa-continuation-collector-test-') as directory:
        temp = Path(directory)
        status = {'id': 999999, 'name': 'synthetic-continuation', 'status': 'SUCCEEDED', 'exit_code': 0}
        archive, package, executable_lines = synthetic_package(temp)
        collector.verify_archive(archive, package, status, executable_lines)
        passed.append('synthetic_archive_four_payloads_three_executed_hashes_positive')
        report_path, log_path, package_path = temp / 'status.json', temp / 'log.txt', temp / 'manifest.json'
        report_path.write_text(json.dumps(status))
        log_path.write_text('\n'.join(executable_lines + lines) + '\n')
        package_path.write_text(json.dumps(package))
        result = collector.collect(report_path, log_path, archive, package_path)
        if result['total_checkpoints'] != 14 or result['total_logits_compared'] != N * 14 or result['accuracy_pass'] is not None:
            raise AssertionError('end-to-end synthetic collector scope')
        passed.append('synthetic_collection_complete_no_accuracy_verdict_positive')
        for state, exit_code in [('RUNNING', 0), ('SUCCEEDED', 1), ('SUCCEEDED', False)]:
            report_path.write_text(json.dumps({**status, 'status': state, 'exit_code': exit_code}))
            reject(f'job_status_{state}_exit_{exit_code}', lambda: collector.collect(report_path, log_path, archive, package_path),
                   'job has not succeeded with exit0')
        report_path.write_text(json.dumps(status))
        reject('missing_executed_digest', lambda: collector.verify_archive(archive, package, status, executable_lines[:-1]), 'executed binary digest missing/duplicated')
        reject('duplicated_executed_digest', lambda: collector.verify_archive(archive, package, status, executable_lines + [executable_lines[0]]), 'executed binary digest missing/duplicated')
        bad_manifest = copy.deepcopy(package)
        bad_manifest['files']['probe_model_e55']['sha256'] = '0' * 64
        reject('actual_payload_not_manifest', lambda: collector.verify_archive(archive, bad_manifest, status, executable_lines), 'uploaded file mismatch')
        changed = {name: ('synthetic payload: ' + name).encode() for name in collector.MEMBERS}
        changed['probe_model_e55'] += b' replacement'
        archive, changed_package, _ = synthetic_package(temp, changed)
        reject('replacement_payload_manifest_agree_but_execution_differs',
               lambda: collector.verify_archive(archive, changed_package, status, executable_lines), 'executed binary digest missing/duplicated')
        archive, package, executable_lines = synthetic_package(temp, job_yaml=b'name: synthetic-continuation\nentrypoint: ./remote_entrypoint.sh\nargs: []\nenv: {}\ntimeout_sec: 71\n')
        reject('actual_job_timeout_changed', lambda: collector.verify_archive(archive, package, status, executable_lines), 'wrong actual job config')
        archive, package, executable_lines = synthetic_package(temp, member_kind='remote_entrypoint.sh')
        reject('symlink_payload', lambda: collector.verify_archive(archive, package, status, executable_lines), 'non-regular archive member')
        archive, package, executable_lines = synthetic_package(temp)
        with tarfile.open(archive, 'a') as tar:
            info = tarfile.TarInfo('unexpected'); info.size = 1
            tar.addfile(info, io.BytesIO(b'x'))
        reject('extra_archive_member', lambda: collector.verify_archive(archive, package, status, executable_lines), 'unexpected archive members')
    reject('duplicate_JSON_key', lambda: collector.parse_json('{"mode":"emit","mode":"compare"}'), 'duplicate JSON key')
    reject('nonstandard_NaN_JSON', lambda: collector.parse_json('{"seconds":NaN}'), 'nonstandard JSON constant: NaN')

    actual_reports = []
    for path in sorted((R / 'docs/experiments').glob('e55-model-continuation*.json')):
        saved = json.loads(path.read_text())
        if 'job' not in saved or 'log' not in saved or 'submitted_archive' not in saved:
            continue
        # Related adapters keep distinct event contracts and their own tests.
        if not saved.get('continuation_events') or saved['continuation_events'][0].get('schema') != collector.SCHEMA:
            continue
        log = Path(saved['log']['path'])
        status = log.with_name('status-' + str(saved['job']['id']) + '.json')
        actual = collector.collect(status, log, Path(saved['submitted_archive']['path']))
        if actual['accuracy_pass'] is not None:
            raise AssertionError('Actual result asserted accuracy')
        actual_reports.append(str(path.relative_to(R)))
        passed.append('actual_job_' + str(saved['job']['id']))
        actual_events = actual['continuation_events']
        without_final = [item for item in actual_events if item['event'] != 'continuation_comparison_complete']
        reject('actual_truncated_final_' + str(saved['job']['id']),
               lambda: collector.validate_events(without_final), 'continuation events are incomplete')
        nonfinite = copy.deepcopy(actual_events)
        next(item for item in nonfinite if item['event'] == 'checkpoint_logits_compared')['nonfinite_count'] = 1
        reject('actual_nonfinite_checkpoint_' + str(saved['job']['id']),
               lambda: collector.validate_events(nonfinite), 'invalid checkpoint position/finite/shape')
        wrong_total = copy.deepcopy(actual_events)
        next(item for item in wrong_total if item['event'] == 'continuation_comparison_complete')['total_elements'] -= 1
        reject('actual_incomplete_element_total_' + str(saved['job']['id']),
               lambda: collector.validate_events(wrong_total), 'continuation final counters mismatch')
        false_verdict = copy.deepcopy(actual_events)
        next(item for item in false_verdict if item['event'] == 'continuation_comparison_complete')['accuracy_pass'] = True
        reject('actual_arbitrary_accuracy_PASS_' + str(saved['job']['id']),
               lambda: collector.validate_events(false_verdict), 'unexpected accuracy verdict')
        continued = next((item for item in actual_events if item['event'] == 'checkpoint_logits_compared' and item['step'] > 0), None)
        if continued is not None:
            wrong_history = copy.deepcopy(actual_events)
            target = next(item for item in wrong_history if item['event'] == 'checkpoint_logits_compared'
                          and item['case_index'] == continued['case_index'] and item['step'] == continued['step'])
            target['consumed_token_ids'][-1] = (target['consumed_token_ids'][-1] + 1) % N
            reject('actual_changed_consumed_history_' + str(saved['job']['id']),
                   lambda: collector.validate_events(wrong_history), 'forced reference history mismatch')
        actual_lines = log.read_text().splitlines()
        wrong_phase = list(actual_lines)
        event_index = next(i for i, line in enumerate(wrong_phase) if line.startswith('{')
                           and json.loads(line).get('event') == 'checkpoint_logits_compared')
        moved = wrong_phase.pop(event_index)
        wrong_phase.insert(wrong_phase.index('MODEL_CONTINUATION_REFERENCE_MANIFEST_SHA256'), moved)
        reject('actual_comparison_outside_runner_phase_' + str(saved['job']['id']),
               lambda: collector.validate_log(wrong_phase), 'continuation event outside its runner phase')
    print(json.dumps({'status': 'PASS', 'test_count': len(passed), 'tests': passed,
                      'actual_reports_tested': actual_reports, 'actual_evidence_available': bool(actual_reports),
                      'npu_run': False}, indent=2))


if __name__ == '__main__':
    main()
