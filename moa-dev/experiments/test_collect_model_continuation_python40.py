#!/usr/bin/env python3
"""Python40 adapter tests; reuse old helpers and test observed-prefix linkage.

The first eight synthetic checkpoints reuse actual 71329 host reports. Later
checkpoints are virtual finite arrays, explicitly not new model execution.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
import collect_model_continuation_python40 as adapter
import test_collect_model_continuation as old_tests

core = adapter.core
CONFIG = adapter.CONFIG
N = core.N


def prior_report():
    return core.parse_json(adapter.PRIOR_REPORT.read_text())


def synthetic_events(count=40, eos=False, diverge_step=12):
    prior = prior_report()
    old = prior['case_summaries'][3]['checkpoint_comparisons']
    prefix = old[0]['consumed_token_ids']
    selected = [old[step]['reference_selected_token'] if step < 8 else 20000 + step for step in range(count)]
    if eos:
        selected[-1] = 106
    events = []
    for mode in ('emit', 'compare'):
        input_event = copy.deepcopy(next(item for item in prior['continuation_events']
                                         if item['event'] == 'continuation_inputs_validated' and item['mode'] == mode))
        input_event.update(schema=CONFIG.schema, case_count=1, prompts=list(CONFIG.prompts), token_ids=[prefix],
                           total_prompt_tokens=len(prefix), guarded_total_tokens=len(prefix) + 40,
                           max_new_tokens=40)
        events.append(input_event)
        first = None
        for step, token in enumerate(selected):
            if step < 8 and not (eos and step == count - 1):
                item = copy.deepcopy(next(item for item in prior['continuation_events']
                                          if item['event'] == ('reference_checkpoint_written' if mode == 'emit' else 'checkpoint_logits_compared')
                                          and item['case_index'] == 3 and item['step'] == step))
                item.update(schema=CONFIG.schema, case_index=0, checkpoint_file=f'case-0-step-{step}.json')
            else:
                item = old_tests.event('reference_checkpoint_written' if mode == 'emit' else 'checkpoint_logits_compared', mode,
                    schema=CONFIG.schema, case_index=0, step=step, position=len(prefix) - 1 + step,
                    checkpoint_file=f'case-0-step-{step}.json', consumed_token_ids=prefix + selected[:step],
                    element_count=N, nonfinite_count=0, fresh_workspace_for_case=True,
                    reference_selected_token=token, reference_selected_is_eos=token in core.EOS,
                    reference_legal_choice=old_tests.choice(token),
                    timings={'decode_and_read_seconds': .01, 'checkpoint_seconds': .02, 'elapsed_suite_seconds': 10.0 + step})
                if mode == 'compare':
                    candidate = 1 if step == diverge_step else token
                    if candidate != token and first is None:
                        first = step
                    item.update(candidate_selected_token=candidate, candidate_selected_is_eos=candidate in core.EOS,
                                candidate_legal_choice=old_tests.choice(candidate, 9.0, token, 8.0) if candidate != token else old_tests.choice(token),
                                selected_token_match=candidate == token, first_legal_greedy_disagreement_step=first,
                                reference_choice_candidate_rank=2 if candidate != token else 1,
                                replay_policy=core.REPLAY, comparison=old_tests.stats(token, candidate))
            events.append(item)
        events.append(old_tests.event('reference_case_complete' if mode == 'emit' else 'comparison_case_complete', mode,
            schema=CONFIG.schema, case_index=0, prompt=CONFIG.prompts[0], token_ids=prefix,
            checkpoint_count=count, completion_tokens=count - int(eos), forward_tokens=len(prefix) + count - 1,
            finish_reason='eos' if eos else 'length', reference_selected_tokens=selected,
            consumed_token_ids=prefix + selected[:-1], first_legal_greedy_disagreement_step=first,
            total_elements=count * N, fresh_workspace=True, replay_policy=core.REPLAY,
            timings={'workspace_seconds': .1, 'prefill_seconds': .2, 'case_seconds': 2., 'elapsed_suite_seconds': 3.}))
        events.append(old_tests.event('continuation_reference_complete' if mode == 'emit' else 'continuation_comparison_complete', mode,
            schema=CONFIG.schema, completed_cases=1, total_prompt_tokens=len(prefix), guarded_total_tokens=len(prefix) + 40,
            max_total_tokens=128, max_new_tokens=40, elements_per_checkpoint=N, total_checkpoints=count,
            total_forward_tokens=len(prefix) + count - 1, total_elements=count * N,
            case_checkpoint_counts=[count], case_finish_reasons=['eos' if eos else 'length'],
            first_legal_greedy_disagreement_steps=[first], termination_policy=core.TERMINATION, replay_policy=core.REPLAY,
            manifest_file='manifest.json', reference_parse_seconds=input_event['reference_parse_seconds'], load_seconds=1., total_seconds=5.))
    return events


def synthetic_lines(events):
    assets = prior_report()['asset_snapshots']
    count = core.validate_events(events, CONFIG)['scope']['case_checkpoint_counts'][0]
    names = ['reference/manifest.json'] + [f'reference/case-0-step-{step}.json' for step in range(count)]
    hashes = [hashlib.sha256(name.encode()).hexdigest() + '  ' + name for name in names]
    mark = lambda suffix: CONFIG.marker_prefix + '_' + suffix
    return [mark('PAIR_BEGIN'), mark('ASSETS_BEFORE_BEGIN'), json.dumps(assets[0]), mark('ASSETS_BEFORE_END'),
            *[json.dumps(item) for item in events if item['mode'] == 'emit'],
            mark('REFERENCE_MANIFEST_SHA256'), hashes[0], mark('REFERENCE_FILES_SHA256_BEGIN'), *hashes,
            mark('REFERENCE_FILES_SHA256_END'), *[json.dumps(item) for item in events if item['mode'] == 'compare'],
            mark('REFERENCE_FILES_AFTER_SHA256_BEGIN'), *hashes, mark('REFERENCE_FILES_AFTER_SHA256_END'),
            mark('ASSETS_AFTER_BEGIN'), json.dumps(assets[1]), mark('ASSETS_AFTER_END'),
            mark('PAIR_STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT'), mark('PAIR_EXIT=0')]


def main():
    passed = []

    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        passed.append(label)

    def reject(label, function, expected):
        try:
            function()
        except ValueError as error:
            if str(error) != expected:
                raise AssertionError(f'{label}: wrong gate: {error}') from error
        else:
            raise AssertionError('Incorrectly accepted ' + label)
        passed.append(label)

    events = synthetic_events()
    result = core.validate_events(events, CONFIG)
    check('forty_checkpoints_with_late_candidate_EOS_forced_replay',
          result['scope']['total_checkpoints'] == 40 and result['scope']['total_forward_tokens'] == 67
          and result['scope']['guarded_total_tokens'] == 68
          and result['case_summaries'][0]['first_legal_greedy_disagreement_step'] == 12)
    eos = core.validate_events(synthetic_events(18, True), CONFIG)
    check('reference_EOS_stops_before_forty', eos['scope']['case_checkpoint_counts'] == [18]
          and eos['scope']['case_finish_reasons'] == ['eos'])
    first_eos = core.validate_events(synthetic_events(1, True), CONFIG)
    check('reference_EOS_on_first_checkpoint', first_eos['scope']['case_checkpoint_counts'] == [1])

    def mutate(label, predicate, change, expected):
        data = copy.deepcopy(events)
        change(next(item for item in data if predicate(item)))
        reject(label, lambda: core.validate_events(data, CONFIG), expected)

    inputs = lambda x: x['event'] == 'continuation_inputs_validated' and x['mode'] == 'compare'
    step39 = lambda x: x['event'] == 'checkpoint_logits_compared' and x['step'] == 39
    mutate('legacy_schema_rejected', inputs, lambda x: x.update(schema=core.SCHEMA), 'wrong event schema/mode')
    mutate('four_case_guard_not_reused', inputs, lambda x: x.update(guarded_total_tokens=188), 'continuation token budget mismatch')
    mutate('boolean_single_case_rejected', inputs, lambda x: x.update(case_count=True), 'wrong continuation prompt suite')
    mutate('old_max_eight_rejected', inputs, lambda x: x.update(max_new_tokens=8), 'wrong continuation guard')
    mutate('case_index_one_rejected', step39, lambda x: x.update(case_index=1), 'checkpoint case/step is duplicated, missing or reordered')
    mutate('step_forty_rejected', step39, lambda x: x.update(step=40), 'checkpoint case/step is duplicated, missing or reordered')
    mutate('late_first_disagreement_erased', step39, lambda x: x.update(first_legal_greedy_disagreement_step=None), 'incorrect first legal greedy disagreement')
    mutate('late_wrong_history', step39, lambda x: x['consumed_token_ids'].__setitem__(-1, 2), 'forced reference history mismatch')
    mutate('late_nonfinite', step39, lambda x: x.update(nonfinite_count=1), 'invalid checkpoint position/finite/shape')
    mutate('late_accuracy_PASS', step39, lambda x: x.update(accuracy_pass=True), 'unexpected accuracy verdict')
    reject('non_EOS_stopped_at_old_eight', lambda: core.validate_events(synthetic_events(8), CONFIG), 'non-EOS case stopped before length budget')

    lines = synthetic_lines(events)
    data = core.validate_log(lines, CONFIG)
    check('new_markers_and_full_step39_inventory', len(data['reference_file_hashes']) == 41)
    reject('default_collector_rejects_python40_markers', lambda: core.validate_log(lines), 'missing/duplicate marker: MODEL_CONTINUATION_PAIR_BEGIN')
    check('two_digit_filename_valid', bool(core.hash_lines(['a' * 64 + '  reference/case-0-step-39.json'], CONFIG)))
    for filename in ['case-0-step-40.json', 'case-1-step-0.json', 'case-0-step-039.json']:
        reject('invalid_filename_' + filename, lambda filename=filename: core.hash_lines(['a' * 64 + '  reference/' + filename], CONFIG),
               'invalid/duplicate reference file hash')
    changed = list(lines)
    index = changed.index(CONFIG.marker_prefix + '_REFERENCE_FILES_AFTER_SHA256_BEGIN') + 40
    changed[index] = '0' * 64 + changed[index][64:]
    reject('late_reference_file_changed', lambda: core.validate_log(changed, CONFIG), 'reference files changed during candidate replay')

    observed = {'case_summaries': result['case_summaries'], 'asset_identities_before': prior_report()['asset_identities_before']}
    linked = adapter.link_prior_observations(observed)
    check('prior_first_eight_fields_and_assets_match', linked['all_first_eight_reported_fields_equal']
          and linked['same_asset_content_hashes_and_lengths'] and linked['available_checkpoint_count'] == 8)
    different = copy.deepcopy(observed)
    different['case_summaries'][0]['checkpoint_comparisons'][0]['comparison']['rmse'] += .001
    linked = adapter.link_prior_observations(different)
    check('changed_stat_recorded_without_accuracy_verdict', linked['status'] == 'FIRST_EIGHT_REPORTED_FIELDS_DIFFER'
          and linked['checkpoints'][0]['different_fields'] == ['comparison.rmse'] and linked['accuracy_pass'] is None)
    different = copy.deepcopy(observed)
    different['asset_identities_before']['model.safetensors']['sha256'] = '0' * 64
    linked = adapter.link_prior_observations(different)
    check('different_assets_observation_preserved', linked['status'] == 'DIFFERENT_ASSETS'
          and not linked['same_asset_content_hashes_and_lengths'])
    shorter = {'case_summaries': first_eos['case_summaries'], 'asset_identities_before': observed['asset_identities_before']}
    linked = adapter.link_prior_observations(shorter)
    check('fewer_than_eight_is_observation', linked['status'] == 'FEWER_THAN_EIGHT_NEW_CHECKPOINTS'
          and linked['available_checkpoint_count'] == 1 and not linked['all_first_eight_reported_fields_equal'])

    with tempfile.TemporaryDirectory(prefix='moa-python40-collector-test-') as directory:
        temp = Path(directory)
        archive, package, executable_lines = old_tests.synthetic_package(temp)
        status = temp / 'status.json'; log = temp / 'log.txt'; manifest = temp / 'manifest.json'
        status.write_text(json.dumps({'id': 999999, 'name': 'synthetic-continuation', 'status': 'SUCCEEDED', 'exit_code': 0}))
        log.write_text('\n'.join(executable_lines + lines) + '\n')
        manifest.write_text(json.dumps(package))
        complete = adapter.collect(status, log, archive, manifest)
        check('full_adapter_archive_collection_positive', complete['distinct_prompts'] == 1
              and complete['total_logits_compared'] == 40 * N and complete['accuracy_pass'] is None
              and complete['prior_python_first_eight_observations']['all_first_eight_reported_fields_equal'])
        stale = temp / 'stale-prior.json'
        stale.write_bytes(adapter.PRIOR_REPORT.read_bytes() + b'\n')
        reject('changed_frozen_prior_report', lambda: adapter.link_prior_observations(observed, stale),
               'prior continuation report differs from frozen 71329 evidence')

    actual_reports = []
    for path in sorted((core.ROOT / 'docs/experiments').glob('e55-model-continuation-python40*.json')):
        saved = core.parse_json(path.read_text())
        if 'job' not in saved or 'log' not in saved or 'submitted_archive' not in saved:
            continue
        log = Path(saved['log']['path'])
        status = log.with_name('status-' + str(saved['job']['id']) + '.json')
        actual = adapter.collect(status, log, Path(saved['submitted_archive']['path']))
        check('actual_job_' + str(saved['job']['id']), actual['accuracy_pass'] is None)
        actual_reports.append(str(path.relative_to(core.ROOT)))
        actual_events = actual['continuation_events']
        actual_checkpoints = actual['case_summaries'][0]['checkpoint_comparisons']
        suffix = '_' + str(saved['job']['id'])
        if saved['job']['id'] == 71362:
            check('actual_71362_fourteen_EOS_checkpoints_and_prefix_link', actual['case_checkpoint_counts'] == [14]
                  and actual['case_finish_reasons'] == ['eos'] and actual['total_logits_compared'] == 14 * N
                  and actual['prior_python_first_eight_observations']['all_first_eight_reported_fields_equal'])

        def mutate_actual(label, predicate, change, expected):
            data = copy.deepcopy(actual_events)
            change(next(item for item in data if predicate(item)))
            reject(label + suffix, lambda: core.validate_events(data, CONFIG), expected)

        late = next((item for item in actual_checkpoints if item['step'] >= 8), None)
        if late is not None:
            late_predicate = lambda x: x['event'] == 'checkpoint_logits_compared' and x['step'] == late['step']
            mutate_actual('actual_late_history_changed', late_predicate,
                          lambda x: x['consumed_token_ids'].__setitem__(-1, (x['consumed_token_ids'][-1] + 1) % N), 'forced reference history mismatch')
            mutate_actual('actual_late_nonfinite', late_predicate, lambda x: x.update(nonfinite_count=1), 'invalid checkpoint position/finite/shape')
            mutate_actual('actual_late_partial_logits', late_predicate,
                          lambda x: x['comparison'].update(elements_compared=N - 1), 'truncated logit comparison')
            no_tail = [item for item in actual_events if not (item['event'] == 'checkpoint_logits_compared' and item['step'] >= 8)]
            reject('actual_six_late_checkpoints_omitted' + suffix,
                   lambda: core.validate_events(no_tail, CONFIG), 'non-EOS case stopped before length budget')
        final_checkpoint = actual_checkpoints[-1]
        if final_checkpoint['reference_selected_is_eos']:
            last_step = final_checkpoint['step']
            last_predicate = lambda x: x['event'] == 'checkpoint_logits_compared' and x['step'] == last_step
            mutate_actual('actual_final_reference_EOS_flag_false', last_predicate,
                          lambda x: x.update(reference_selected_is_eos=False), 'reference EOS flag mismatch')
            mutate_actual('actual_EOS_counted_as_completion', lambda x: x['event'] == 'comparison_case_complete',
                          lambda x: x.update(completion_tokens=x['checkpoint_count']), 'case completion counters mismatch')
            mutate_actual('actual_EOS_wrong_length_finish', lambda x: x['event'] == 'comparison_case_complete',
                          lambda x: x.update(finish_reason='length'), 'case completion metadata/history mismatch')
            if last_step + 1 < CONFIG.max_new:
                after_eos = copy.deepcopy(actual_events)
                extra = copy.deepcopy(next(item for item in after_eos if item['event'] == 'reference_checkpoint_written' and item['step'] == last_step))
                extra.update(step=last_step + 1, position=extra['position'] + 1, checkpoint_file=f'case-0-step-{last_step + 1}.json',
                             consumed_token_ids=extra['consumed_token_ids'] + [extra['reference_selected_token']])
                insertion = next(i for i, item in enumerate(after_eos) if item['event'] == 'reference_case_complete')
                after_eos.insert(insertion, extra)
                reject('actual_checkpoint_after_reference_EOS' + suffix, lambda: core.validate_events(after_eos, CONFIG), 'checkpoint follows reference EOS')
        tied = next((item for item in actual_checkpoints if item['candidate_legal_choice']['tie_count'] >= 2), None)
        if tied is not None:
            tie_predicate = lambda x: x['event'] == 'checkpoint_logits_compared' and x['step'] == tied['step']
            mutate_actual('actual_legal_tie_count_falsified', tie_predicate,
                          lambda x: x['candidate_legal_choice'].update(tie_count=1), 'total_cmp tie count disagrees with runner-up')
            mutate_actual('actual_tied_runner_bits_changed', tie_predicate,
                          lambda x: x['candidate_legal_choice'].update(runner_up_logit_bits=x['candidate_legal_choice']['runner_up_logit_bits'] ^ 1),
                          'legal logit bits disagree with finite f32 value')
            # A different equal-maximal choice is permitted by the sampler's
            # unstable tie rule. This modified log is a parser test, not a new
            # observed run or assertion that the other choice occurred.
            alternative = copy.deepcopy(actual_events)
            for item in alternative:
                if item['mode'] != 'compare':
                    continue
                if item['event'] == 'checkpoint_logits_compared' and item['step'] >= tied['step']:
                    item['first_legal_greedy_disagreement_step'] = tied['step']
                    if item['step'] == tied['step']:
                        choice = item['candidate_legal_choice']
                        for field in ('id', 'logit', 'logit_bits'):
                            choice['selected_' + field], choice['runner_up_' + field] = choice['runner_up_' + field], choice['selected_' + field]
                        item.update(candidate_selected_token=choice['selected_id'], candidate_selected_is_eos=choice['selected_id'] in core.EOS,
                                    selected_token_match=False, reference_choice_candidate_rank=1)
                elif item['event'] == 'comparison_case_complete':
                    item['first_legal_greedy_disagreement_step'] = tied['step']
                elif item['event'] == 'continuation_comparison_complete':
                    item['first_legal_greedy_disagreement_steps'] = [tied['step']]
            checked = core.validate_events(alternative, CONFIG)
            check('actual_tie_alternative_legal_choice_parser_positive' + suffix,
                  checked['case_summaries'][0]['first_legal_greedy_disagreement_step'] == tied['step'])
    print(json.dumps({'status': 'PASS', 'test_count': len(passed), 'tests': passed,
                      'actual_reports_tested': actual_reports, 'actual_evidence_available': bool(actual_reports),
                      'npu_run': False}, indent=2))


if __name__ == '__main__':
    main()
