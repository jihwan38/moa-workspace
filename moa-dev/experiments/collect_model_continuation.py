#!/usr/bin/env python3
"""Collect complete forced-reference continuation evidence, never accuracy PASS.

Validate the frozen host's reported statistics and actual uploaded payload.
Reference arrays stay remote; this collector does not recompute their errors.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import tarfile

import collect_model_prefix_suite as common

ROOT = common.ROOT
N = 262144
SCHEMA = 'moa-model-continuation-v1'
PROMPTS = common.PROMPTS
MAX_NEW = 8
MAX_TOTAL = 128
BANNED = {258882, 258883}
EOS = {1, 106, 50}
REPLAY = 'reference_tokens_even_if_candidate_diverges_or_stops'
TERMINATION = 'reference_original_eos_or_max_new_tokens'
PACKAGE = 'moa-dev/experiments/candidates/model-continuation-pair-e55/package-manifest.json'
MEMBERS = {'probe_model_original', 'probe_model_e55', 'moa-model-asset-inspector', 'remote_entrypoint.sh'}
require = common.require


@dataclass(frozen=True)
class CollectorConfig:
    schema: str = SCHEMA
    prompts: tuple[str, ...] = tuple(PROMPTS)
    max_new: int = MAX_NEW
    max_total: int = MAX_TOTAL
    marker_prefix: str = 'MODEL_CONTINUATION'
    package: str = PACKAGE
    scope_intro: str = 'Four short prompts with 1..8 reference-defined checkpoints each'


DEFAULT_CONFIG = CollectorConfig()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'duplicate JSON key')
        value[key] = item
    return value


def parse_json(text):
    def reject(value):
        raise ValueError('nonstandard JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject)


def integer(value, minimum=0, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def null_verdict(value):
    require('accuracy_pass' in value and 'accuracy_acceptance_threshold' in value
            and value['accuracy_pass'] is None and value['accuracy_acceptance_threshold'] is None,
            'unexpected accuracy verdict')


def token_ids(ids, nonempty=True):
    require(isinstance(ids, list) and (bool(ids) or not nonempty)
            and all(integer(i, 0, N - 1) for i in ids), 'invalid token IDs')


def integer_fields(value, expected, message):
    require(all(integer(value[key]) and value[key] == number for key, number in expected.items()), message)


def timing_fields(value, names):
    require(all(common.finite_nonnegative(value[key]) for key in names), 'invalid finite timing')


def disagreement(value, expected, message, max_new=MAX_NEW):
    require((value is None or integer(value, 0, max_new - 1)) and value == expected, message)


def f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', common.as_f32(value)))[0]


def total_order(bits):
    return (~bits & 0xffffffff) if bits & 0x80000000 else bits ^ 0x80000000


def legal_choice(choice, selected_id):
    require(integer(selected_id, 0, N - 1) and selected_id not in BANNED,
            'selected token is invalid or banned')
    require(choice['selected_id'] == selected_id and integer(choice['selected_id'], 0, N - 1),
            'legal choice selected ID mismatch')
    runner = choice['runner_up_id']
    require(integer(runner, 0, N - 1) and runner not in BANNED and runner != selected_id,
            'invalid legal runner-up ID')
    for label in ('selected', 'runner_up'):
        bits = choice[label + '_logit_bits']
        require(integer(bits, 0, 0xffffffff) and bits == f32_bits(choice[label + '_logit']),
                'legal logit bits disagree with finite f32 value')
    selected_bits, runner_bits = choice['selected_logit_bits'], choice['runner_up_logit_bits']
    require(total_order(selected_bits) >= total_order(runner_bits), 'selected token is below legal runner-up')
    ties = choice['tie_count']
    require(integer(ties, 1, N - len(BANNED)), 'invalid legal tie count')
    require((selected_bits == runner_bits) == (ties >= 2), 'total_cmp tie count disagrees with runner-up')
    require(choice['rank_order'] == 'f32_total_cmp' and choice['tie_rule'] == 'public_sample_unstable_order',
            'wrong legal ordering/tie semantics')
    difference = common.as_f32(choice['selected_logit']) - common.as_f32(choice['runner_up_logit'])
    require(common.finite_nonnegative(choice['margin'])
            and math.isclose(difference, choice['margin'], rel_tol=1e-12, abs_tol=1e-12),
            'legal margin disagrees with logits')


def comparison_stats(value):
    null_verdict(value)
    integer_fields(value, {'elements_compared': N}, 'truncated logit comparison')
    common.check_top(value['reference_top'])
    common.check_top(value['actual_top'])
    require(all(common.finite_nonnegative(value[k]) for k in ('max_abs', 'mean_abs', 'rmse'))
            and value['max_abs'] + 1e-12 >= value['rmse'] >= value['mean_abs'] - 1e-12,
            'inconsistent finite error statistics')
    require(integer(value['max_abs_index'], 0, N - 1), 'maximum-error token index out of range')
    at_max = abs(common.as_f32(value['actual_at_max_abs']) - common.as_f32(value['reference_at_max_abs']))
    require(math.isclose(at_max, value['max_abs'], rel_tol=1e-12, abs_tol=1e-12),
            'maximum error disagrees with reported values')
    require(integer(value['bit_equal_count'], 0, N), 'invalid bit-equal count')
    thresholds = [0, 1e-6, 1e-4, 1e-3, .01, .1, .5, 1, 2, 5]
    counts = value['diagnostic_absolute_error_counts']
    require(isinstance(counts, list) and len(counts) == len(thresholds), 'missing diagnostic thresholds')
    previous = 0
    for item, threshold in zip(counts, thresholds):
        require(item['absolute_error_at_most'] == threshold and integer(item['count'], previous, N)
                and integer(item['count_above'], 0, N) and item['count_above'] == N - item['count'],
                'invalid threshold counts')
        require((item['count'] == N) == (threshold >= value['max_abs']),
                'threshold count contradicts maximum error')
        previous = item['count']
    require(value['bit_equal_count'] <= counts[0]['count'], 'bit-equal count exceeds zero-error count')


def validated_inputs(event, config=DEFAULT_CONFIG):
    count = len(config.prompts)
    require(integer(event['case_count']) and event['case_count'] == count and event['prompts'] == list(config.prompts),
            'wrong continuation prompt suite')
    require(event['max_new_tokens'] == config.max_new and event['max_total_tokens'] == config.max_total
            and event['guard_before_device_new'] is True, 'wrong continuation guard')
    require(event['thinking'] is False and event['add_generation_prompt'] is True,
            'wrong chat policy')
    sampler = event['sampler']
    require(set(sampler) == {'temperature', 'top_k', 'top_p'} and integer(sampler['top_k'])
            and sampler['top_k'] == 1 and type(sampler['temperature']) in (int, float)
            and type(sampler['top_p']) in (int, float) and sampler['temperature'] == sampler['top_p'] == 1,
            'wrong public sampler policy')
    token_ids(event['banned_tokens'])
    token_ids(event['eos_tokens'])
    require(event['banned_tokens'] == [258882, 258883] and event['eos_tokens'] == [1, 106, 50],
            'wrong banned/EOS token policy')
    require(event['termination_policy'] == TERMINATION and event['replay_policy'] == REPLAY,
            'wrong termination/replay policy')
    require(event['reference_all_checkpoints_validated'] is (event['mode'] == 'compare'),
            'reference checkpoints were not validated before device creation')
    timing_fields(event, ('reference_parse_seconds', 'elapsed_seconds'))
    ids = event['token_ids']
    require(isinstance(ids, list) and len(ids) == count, 'wrong tokenized case count')
    for case_ids in ids:
        token_ids(case_ids)
        require(len(case_ids) + config.max_new <= 32768, 'runtime context guard exceeded')
    total = sum(map(len, ids))
    guarded = total + config.max_new * count
    require(event['total_prompt_tokens'] == total and event['guarded_total_tokens'] == guarded
            and guarded <= config.max_total, 'continuation token budget mismatch')
    return ids, total, guarded


def validate_events(events, config=DEFAULT_CONFIG):
    """Validate variable EOS-complete sequences independently of package I/O."""
    for event in events:
        require(event['schema'] == config.schema and event['mode'] in ('emit', 'compare'), 'wrong event schema/mode')
        null_verdict(event)
    cursor = 0
    phases = {}
    reference_cases = []
    summaries = []

    def take(name, mode):
        nonlocal cursor
        require(cursor < len(events), 'continuation events are incomplete')
        item = events[cursor]
        require(item['event'] == name and item['mode'] == mode, 'continuation event order mismatch')
        cursor += 1
        return item

    for mode in ('emit', 'compare'):
        input_event = take('continuation_inputs_validated', mode)
        ids, prompt_total, guarded = validated_inputs(input_event, config)
        if mode == 'compare':
            require(ids == phases['emit']['token_ids'], 'candidate prompt IDs differ from reference')
        counts, reasons, first_steps, forward_total = [], [], [], 0
        for case_index, prefix in enumerate(ids):
            selected = []
            checkpoints = []
            first_disagreement = None
            checkpoint_event = 'reference_checkpoint_written' if mode == 'emit' else 'checkpoint_logits_compared'
            while cursor < len(events) and events[cursor]['event'] == checkpoint_event:
                item = take(checkpoint_event, mode)
                step = len(selected)
                require(step < config.max_new and integer(item['case_index']) and integer(item['step'])
                        and item['case_index'] == case_index and item['step'] == step,
                        'checkpoint case/step is duplicated, missing or reordered')
                integer_fields(item, {'position': len(prefix) - 1 + step, 'element_count': N, 'nonfinite_count': 0},
                               'invalid checkpoint position/finite/shape')
                require(item['checkpoint_file'] == f'case-{case_index}-step-{step}.json',
                        'wrong checkpoint filename')
                require(item['fresh_workspace_for_case'] is True, 'checkpoint did not use fresh case workspace')
                timing_fields(item['timings'], ('decode_and_read_seconds', 'checkpoint_seconds', 'elapsed_suite_seconds'))
                token_ids(item['consumed_token_ids'])
                require(item['consumed_token_ids'] == prefix + selected, 'forced reference history mismatch')
                token = item['reference_selected_token']
                legal_choice(item['reference_legal_choice'], token)
                require(item['reference_selected_is_eos'] is (token in EOS), 'reference EOS flag mismatch')
                require(not selected or selected[-1] not in EOS, 'checkpoint follows reference EOS')
                if mode == 'compare':
                    expected_steps = reference_cases[case_index]['checkpoints']
                    require(step < len(expected_steps), 'candidate has extra checkpoint')
                    expected = expected_steps[step]
                    require(token == expected['reference_selected_token']
                            and item['reference_legal_choice'] == expected['reference_legal_choice'],
                            'candidate reference choice differs from emitted reference')
                    candidate = item['candidate_selected_token']
                    legal_choice(item['candidate_legal_choice'], candidate)
                    require(item['candidate_selected_is_eos'] is (candidate in EOS), 'candidate EOS flag mismatch')
                    match = candidate == token
                    require(item['selected_token_match'] is match, 'selected-token match flag mismatch')
                    if not match and first_disagreement is None:
                        first_disagreement = step
                    disagreement(item['first_legal_greedy_disagreement_step'], first_disagreement,
                                 'incorrect first legal greedy disagreement', config.max_new)
                    rank = item['reference_choice_candidate_rank']
                    require(integer(rank, 1, N - len(BANNED)) and (not match or rank == 1),
                            'invalid reference choice candidate rank')
                    require(match or rank != 1 or item['candidate_legal_choice']['tie_count'] >= 2,
                            'different rank-one reference choice requires a legal tie')
                    require(item['replay_policy'] == REPLAY, 'wrong candidate replay policy')
                    comparison_stats(item['comparison'])
                selected.append(token)
                checkpoints.append(item)
            require(1 <= len(selected) <= config.max_new, 'case has no checkpoints or exceeds budget')
            count = len(selected)
            reason = 'eos' if selected[-1] in EOS else 'length'
            require(reason == 'eos' or count == config.max_new, 'non-EOS case stopped before length budget')
            if mode == 'compare':
                require(count == len(reference_cases[case_index]['checkpoints']), 'candidate omitted reference checkpoints')
            end = take('reference_case_complete' if mode == 'emit' else 'comparison_case_complete', mode)
            require(integer(end['case_index']) and integer(end['checkpoint_count'])
                    and end['case_index'] == case_index and end['checkpoint_count'] == count
                    and end['reference_selected_tokens'] == selected and end['finish_reason'] == reason,
                    'case completion metadata/history mismatch')
            token_ids(end['reference_selected_tokens'])
            token_ids(end['token_ids'])
            token_ids(end['consumed_token_ids'])
            require(end['prompt'] == config.prompts[case_index] and end['token_ids'] == prefix
                    and end['consumed_token_ids'] == prefix + selected[:-1], 'case completed with wrong prompt/feed history')
            require(end['fresh_workspace'] is True and end['replay_policy'] == REPLAY,
                    'case workspace/replay policy mismatch')
            timing_fields(end['timings'], ('workspace_seconds', 'prefill_seconds', 'case_seconds', 'elapsed_suite_seconds'))
            completion_tokens = count - int(reason == 'eos')
            forward = len(prefix) + count - 1
            integer_fields(end, {'completion_tokens': completion_tokens, 'forward_tokens': forward,
                                 'total_elements': count * N}, 'case completion counters mismatch')
            disagreement(end['first_legal_greedy_disagreement_step'], first_disagreement,
                         'case first disagreement mismatch', config.max_new)
            if mode == 'emit':
                reference_cases.append({'checkpoints': checkpoints, 'finish_reason': reason})
            else:
                summaries.append({'case_index': case_index, 'prompt': config.prompts[case_index],
                                  'prompt_token_count': len(prefix), 'checkpoint_count': count,
                                  'completion_tokens': completion_tokens, 'forward_tokens': forward,
                                  'finish_reason': reason, 'reference_selected_tokens': selected,
                                  'first_legal_greedy_disagreement_step': first_disagreement,
                                  'checkpoint_comparisons': checkpoints})
            counts.append(count)
            reasons.append(reason)
            first_steps.append(first_disagreement)
            forward_total += forward
        final = take('continuation_reference_complete' if mode == 'emit' else 'continuation_comparison_complete', mode)
        integer_fields(final, {'completed_cases': len(config.prompts), 'total_prompt_tokens': prompt_total,
                               'guarded_total_tokens': guarded, 'max_total_tokens': config.max_total,
                               'max_new_tokens': config.max_new}, 'continuation final case/guard mismatch')
        require(isinstance(final['case_checkpoint_counts'], list)
                and all(integer(count, 1, config.max_new) for count in final['case_checkpoint_counts'])
                and final['case_checkpoint_counts'] == counts
                and final['case_finish_reasons'] == reasons, 'continuation final case/guard mismatch')
        require(final['elements_per_checkpoint'] == N and final['manifest_file'] == 'manifest.json'
                and final['termination_policy'] == TERMINATION and final['replay_policy'] == REPLAY,
                'continuation final file/shape/policy mismatch')
        require(isinstance(final['first_legal_greedy_disagreement_steps'], list)
                and len(final['first_legal_greedy_disagreement_steps']) == len(config.prompts),
                'continuation final first-disagreement list mismatch')
        for reported, actual in zip(final['first_legal_greedy_disagreement_steps'], first_steps):
            disagreement(reported, actual, 'continuation final first-disagreement list mismatch', config.max_new)
        timing_fields(final, ('reference_parse_seconds', 'load_seconds', 'total_seconds'))
        require(final['reference_parse_seconds'] == input_event['reference_parse_seconds'],
                'reference parse timing changed in final event')
        integer_fields(final, {'total_checkpoints': sum(counts), 'total_forward_tokens': forward_total,
                              'total_elements': sum(counts) * N}, 'continuation final counters mismatch')
        phases[mode] = {'token_ids': ids, 'total_prompt_tokens': prompt_total, 'guarded_total_tokens': guarded,
                        'case_checkpoint_counts': counts, 'case_finish_reasons': reasons,
                        'total_checkpoints': sum(counts), 'total_forward_tokens': forward_total,
                        'total_elements': sum(counts) * N}
    require(cursor == len(events), 'unexpected trailing continuation events')
    require(phases['emit'] == phases['compare'], 'reference/candidate continuation scope differs')
    return {'scope': phases['compare'], 'case_summaries': summaries}


def unique_marker(lines, marker):
    require(lines.count(marker) == 1, 'missing/duplicate marker: ' + marker)
    return lines.index(marker)


def hash_lines(lines, config=DEFAULT_CONFIG):
    result = {}
    cases = '|'.join(map(str, range(len(config.prompts))))
    steps = '|'.join(map(str, range(config.max_new)))
    pattern = rf'([0-9a-f]{{64}})  (reference/(?:manifest|case-(?:{cases})-step-(?:{steps}))\.json)'
    for line in lines:
        match = re.fullmatch(pattern, line)
        require(match is not None and match[2] not in result, 'invalid/duplicate reference file hash')
        result[match[2]] = match[1]
    return result


def validate_log(lines, config=DEFAULT_CONFIG):
    markers = [
        'MODEL_CONTINUATION_PAIR_BEGIN', 'MODEL_CONTINUATION_ASSETS_BEFORE_BEGIN',
        'MODEL_CONTINUATION_ASSETS_BEFORE_END', 'MODEL_CONTINUATION_REFERENCE_MANIFEST_SHA256',
        'MODEL_CONTINUATION_REFERENCE_FILES_SHA256_BEGIN', 'MODEL_CONTINUATION_REFERENCE_FILES_SHA256_END',
        'MODEL_CONTINUATION_REFERENCE_FILES_AFTER_SHA256_BEGIN', 'MODEL_CONTINUATION_REFERENCE_FILES_AFTER_SHA256_END',
        'MODEL_CONTINUATION_ASSETS_AFTER_BEGIN', 'MODEL_CONTINUATION_ASSETS_AFTER_END',
        'MODEL_CONTINUATION_PAIR_STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT', 'MODEL_CONTINUATION_PAIR_EXIT=0',
    ]
    positions = {marker: unique_marker(lines, marker.replace('MODEL_CONTINUATION', config.marker_prefix, 1))
                 for marker in markers}
    require(list(positions.values()) == sorted(positions.values()), 'runner markers are out of order')
    records = [(i, parse_json(line)) for i, line in enumerate(lines) if line.startswith('{')]
    assets = [(i, item) for i, item in records if item.get('schema') == common.ASSET_SCHEMA]
    events = [(i, item) for i, item in records if item.get('schema') == config.schema]
    require(len(assets) == 2 and len(records) == len(events) + 2, 'unexpected/missing JSON records')
    for (line, _), moment in zip(assets, ('BEFORE', 'AFTER')):
        require(positions['MODEL_CONTINUATION_ASSETS_' + moment + '_BEGIN'] < line
                < positions['MODEL_CONTINUATION_ASSETS_' + moment + '_END'], 'asset snapshot outside its runner phase')
    for line, event in events:
        left, right = (('MODEL_CONTINUATION_ASSETS_BEFORE_END', 'MODEL_CONTINUATION_REFERENCE_MANIFEST_SHA256')
                       if event['mode'] == 'emit' else
                       ('MODEL_CONTINUATION_REFERENCE_FILES_SHA256_END', 'MODEL_CONTINUATION_REFERENCE_FILES_AFTER_SHA256_BEGIN'))
        require(positions[left] < line < positions[right], 'continuation event outside its runner phase')
    identity = common.checked_assets(assets[0][1], assets[1][1])
    validated = validate_events([item for _, item in events], config)
    expected_files = {'reference/manifest.json'} | {
        f'reference/case-{case}-step-{step}.json'
        for case, count in enumerate(validated['scope']['case_checkpoint_counts']) for step in range(count)}
    before = hash_lines(lines[positions['MODEL_CONTINUATION_REFERENCE_FILES_SHA256_BEGIN'] + 1:
                             positions['MODEL_CONTINUATION_REFERENCE_FILES_SHA256_END']], config)
    after = hash_lines(lines[positions['MODEL_CONTINUATION_REFERENCE_FILES_AFTER_SHA256_BEGIN'] + 1:
                            positions['MODEL_CONTINUATION_REFERENCE_FILES_AFTER_SHA256_END']], config)
    require(set(before) == set(after) == expected_files, 'reference checkpoint file set mismatch')
    require(before == after, 'reference files changed during candidate replay')
    index = positions['MODEL_CONTINUATION_REFERENCE_MANIFEST_SHA256']
    standalone = hash_lines(lines[index + 1:index + 2], config)
    require(standalone == {'reference/manifest.json': before['reference/manifest.json']},
            'standalone manifest hash disagrees with reference file inventory')
    return {**validated, 'events': [item for _, item in events], 'assets': [item for _, item in assets],
            'asset_identity': identity, 'reference_file_hashes': before}


def verify_archive(archive_path, package, status, lines):
    require(set(package['files']) == MEMBERS, 'wrong continuation package file inventory')
    archive = common.fingerprint(archive_path)
    archive['members'] = []
    with tarfile.open(archive_path) as tar:
        members = tar.getmembers()
        require(len(members) == 5 and {m.name for m in members} == MEMBERS | {'job.yaml'},
                'unexpected archive members')
        for member in members:
            require(member.isfile(), 'non-regular archive member')
            data = tar.extractfile(member).read()
            digest = hashlib.sha256(data).hexdigest()
            archive['members'].append({'name': member.name, 'bytes': len(data), 'sha256': digest, 'mode': oct(member.mode)})
            if member.name == 'job.yaml':
                expected = [f"name: {status['name']}", 'entrypoint: ./remote_entrypoint.sh',
                            'args: []', 'env: {}', 'timeout_sec: 70']
                actual = data.decode().splitlines()
                require(len(actual) == 5 and set(actual) == set(expected), 'wrong actual job config')
            else:
                entry = package['files'][member.name]
                require(len(data) == entry['bytes'] and digest == entry['sha256'], 'uploaded file mismatch')
                if member.name != 'remote_entrypoint.sh':
                    possible = [digest + '  ./' + member.name + suffix for suffix in ('', '.exec')]
                    require(sum(lines.count(line) for line in possible) == 1, 'executed binary digest missing/duplicated')
    return archive


def collect(status_path, log_path, archive_path, package_path=None, config=DEFAULT_CONFIG):
    status = parse_json(status_path.read_text())
    require(status['status'] == 'SUCCEEDED' and type(status['exit_code']) is int and status['exit_code'] == 0,
            'job has not succeeded with exit0')
    lines = log_path.read_text().splitlines()
    data = validate_log(lines, config)
    package_path = ROOT / config.package if package_path is None else package_path
    package = parse_json(package_path.read_text())
    archive = verify_archive(archive_path, package, status, lines)
    known = common.read_json(ROOT / 'docs/experiments/e55-arena-model-identity.json')['identities']
    matches_known = all(data['asset_identity'][name]['sha256'] == item['sha256']
                        and data['asset_identity'][name]['size_bytes'] == item['bytes'] for name, item in known.items())
    from datetime import datetime, timezone
    scope = data['scope']
    return {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'FORCED_REFERENCE_CONTINUATION_COMPARED_NO_ACCURACY_VERDICT', 'job': status,
        'collector': common.fingerprint(Path(__file__).resolve()),
        'package_manifest': str(package_path), 'package_manifest_fingerprint': common.fingerprint(package_path),
        'submitted_archive': archive, 'uploaded_package_matches_manifest': True,
        'log': common.fingerprint(log_path), 'asset_snapshots': data['assets'],
        'asset_identities_before': data['asset_identity'], 'assets_match_separate_job_71224': matches_known,
        'asset_identity_scope': 'Full hash before; weight stat/header and metadata full hashes after. No second full weight hash or proof against unobservable mutation.',
        'reference_file_sha256_before_and_after': data['reference_file_hashes'],
        'reference_files_unchanged_by_reported_hash': True, 'reference_json_downloaded': False,
        'continuation_events': data['events'], 'case_summaries': data['case_summaries'],
        'distinct_prompts': len(config.prompts), 'total_prompt_tokens': scope['total_prompt_tokens'],
        'guarded_total_tokens': scope['guarded_total_tokens'],
        'total_forward_tokens_per_model': scope['total_forward_tokens'],
        'total_checkpoints': scope['total_checkpoints'], 'total_logits_compared': scope['total_elements'],
        'case_checkpoint_counts': scope['case_checkpoint_counts'], 'case_finish_reasons': scope['case_finish_reasons'],
        'first_legal_greedy_disagreement_steps': [c['first_legal_greedy_disagreement_step'] for c in data['case_summaries']],
        'replay_policy': REPLAY, 'accuracy_acceptance_threshold': None, 'accuracy_pass': None,
        'official_evaluation': False,
        'scope': config.scope_intro + ', full finite arrays, original-token forced replay after candidate disagreement/EOS. Public greedy sampler uses total_cmp and its unspecified unstable tie order. No free-running answer accuracy, independent oracle, long context, multimodal result, official score or controlled performance claim. Statistics are validated host reports, not independently recomputed arrays.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('status', 'log', 'archive'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.status, args.log, args.archive), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
