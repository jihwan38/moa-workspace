#!/usr/bin/env python3
"""Validate terminal four-prefix evidence and emit a report; never an accuracy PASS.

Checks reported statistics and provenance; does not download/recompute logits.
Run from the workspace root with --status, --log and --archive paths.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ['Hi', 'What is 2 + 2?', '대한민국의 수도는?',
           'Complete the Python expression: len([1, 2, 3])']
N = 262144
SCHEMA = 'moa-model-prefix-suite-v1'
ASSET_SCHEMA = 'MOA_MODEL_ASSET_INSPECTOR_RUST_V1'
PACKAGE = 'moa-dev/experiments/candidates/model-prefix-suite-e55/package-manifest.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def reject_constant(value):
        raise ValueError('nonstandard JSON constant: ' + value)
    return json.loads(path.read_text(), parse_constant=reject_constant)


def fingerprint(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def as_f32(value):
    require(type(value) in (int, float) and math.isfinite(value), 'nonfinite/non-numeric logit')
    result = struct.unpack('<f', struct.pack('<f', value))[0]
    require(math.isfinite(result), 'logit outside finite f32 range')
    return result


def check_top(top):
    require(all(type(top[key]) is int and 0 <= top[key] < N for key in ('top1_id', 'top2_id'))
            and top['top1_id'] != top['top2_id'], 'invalid top token IDs')
    require(type(top['top1_tie_count']) is int and 1 <= top['top1_tie_count'] <= N
            and top['tie_rule'] == 'lowest_token_id', 'invalid top tie summary')
    a, b = (as_f32(top[key]) for key in ('top1_logit', 'top2_logit'))
    require(a >= b and finite_nonnegative(top['top1_margin'])
            and math.isclose(a - b, top['top1_margin'], rel_tol=1e-12, abs_tol=1e-12),
            'inconsistent top margin')


def checked_assets(before, after):
    require(before['header_only'] is False and after['header_only'] is True,
            'asset inspection order/mode mismatch')
    require(before['selector'] == after['selector'], 'model selector changed')
    require(before['embedded_contract_sha256'] == after['embedded_contract_sha256'],
            'header contract changed')
    require(before['embedded_contract_sha256'] ==
            '8e2b9346ff1f4ee3d58581326c85bb89f2dca5c97cb6c5e2b1fd9c3260754055',
            'unexpected frozen header contract')
    left = {item['basename']: item for item in before['files']}
    right = {item['basename']: item for item in after['files']}
    names = {'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
             'chat_template.jinja', 'chat_template.json'}
    require(len(before['files']) == len(after['files']) == 5 and set(left) == set(right) == names,
            'missing or duplicate asset reports')
    identity = {}
    for name in sorted(names):
        a, b = left[name], right[name]
        if name == 'chat_template.json' and a['status'] == b['status'] == 'missing':
            continue
        require(a['file_unchanged_by_stat'] is True and b['file_unchanged_by_stat'] is True,
                'asset changed during inspection: ' + name)
        require(a['stat_before'] == a['stat_after'] == b['stat_before'] == b['stat_after'],
                'asset size/mtime changed across model runs: ' + name)
        require(a['status'] == 'complete' and a['full_file']['status'] == 'complete',
                'incomplete before-run asset identity: ' + name)
        sha = a['full_file']['sha256']
        require(isinstance(sha, str) and re.fullmatch(r'[0-9a-f]{64}', sha) is not None,
                'invalid full digest: ' + name)
        require(a['full_file']['bytes_hashed'] == a['stat_before']['size_bytes'],
                'full digest byte count mismatch: ' + name)
        require(a['full_file']['sha256_scope'] == 'entire_file', 'wrong digest scope')
        if name == 'model.safetensors':
            require(b['status'] == 'header_only' and b['full_file']['status'] == 'skipped'
                    and b['full_file']['sha256'] is None and b['full_file']['bytes_hashed'] == 0,
                    'unexpected after-run weight hash mode')
            require(a['header']['status'] == b['header']['status'] == 'complete'
                    and a['header']['sha256'] == b['header']['sha256'], 'weight header changed')
            for entry in (a, b):
                h = entry['header_contract']
                require(h['status'] == 'pass' and h['loader_contract_pass'] is True
                        and h['semantic_dtype_pass'] is True
                        and h['required_tensor_count'] == h['required_tensors_found'] == 1149,
                        'required header conditions not satisfied')
        else:
            require(b['status'] == 'complete' and b['full_file']['status'] == 'complete'
                    and b['full_file']['sha256'] == sha
                    and b['full_file']['bytes_hashed'] == b['stat_before']['size_bytes']
                    and b['full_file']['sha256_scope'] == 'entire_file',
                    'metadata contents/hash extent changed: ' + name)
        identity[name] = {**a['stat_before'], 'sha256': sha}
    return identity


def collect(status_path, log_path, archive_path):
    status = read_json(status_path)
    require(status['status'] == 'SUCCEEDED' and status['exit_code'] == 0,
            'job has not succeeded with exit0')
    lines = log_path.read_text().splitlines()
    for marker in ('MODEL_PREFIX_SUITE_PAIR_EXIT=0',
                   'MODEL_PREFIX_SUITE_PAIR_STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT'):
        require(lines.count(marker) == 1, 'missing/duplicate completion marker')
    records = [json.loads(line) for line in lines if line.startswith('{')]
    assets = [item for item in records if item.get('schema') == ASSET_SCHEMA]
    events = [item for item in records if item.get('schema') == SCHEMA]
    require(len(assets) == 2 and len(records) == len(events) + 2, 'unexpected JSON records')
    expected = (['suite_inputs_validated'] + ['reference_case_ready'] * 4
                + ['suite_reference_written', 'suite_inputs_validated']
                + ['case_logits_compared'] * 4 + ['suite_comparison_complete'])
    require([item.get('event') for item in events] == expected, 'suite incomplete or reordered')
    for event in events:
        verdict = event['comparison'] if event['event'] == 'case_logits_compared' else event
        require('accuracy_pass' in verdict and 'accuracy_acceptance_threshold' in verdict
                and verdict['accuracy_pass'] is None and verdict['accuracy_acceptance_threshold'] is None,
                'unexpected accuracy verdict in event')
    require(records[0] == assets[0] and records[-1] == assets[1], 'asset snapshot order mismatch')
    identity = checked_assets(*assets)
    inputs = [item for item in events if item['event'] == 'suite_inputs_validated']
    require(inputs[0] == inputs[1], 'original and E55 inputs/guards differ')
    inp = inputs[0]
    require(inp['case_count'] == 4 and inp['prompts'] == PROMPTS
            and inp['max_total_tokens'] == 128 and inp['guard_before_device_new'] is True,
            'wrong suite inputs')
    ids = inp['token_ids']
    require(len(ids) == 4 and all(v and all(type(t) is int and 0 <= t < N for t in v) for v in ids),
            'invalid token IDs')
    total = sum(map(len, ids))
    require(total == inp['total_token_count'] and total <= 128, 'token budget mismatch')
    originals = [item for item in events if item['event'] == 'reference_case_ready']
    comparisons = [item for item in events if item['event'] == 'case_logits_compared']
    summaries = []
    for i, (old, new) in enumerate(zip(originals, comparisons)):
        for item in (old, new):
            require(item['case_index'] == i and item['prompt'] == PROMPTS[i]
                    and item['token_ids'] == ids[i] and item['element_count'] == N
                    and item['nonfinite_count'] == 0 and item['fresh_workspace'] is True,
                    'case identity/finite/workspace mismatch')
        c = new['comparison']
        require(new['reference_token_ids_match'] is True and c['elements_compared'] == N
                and c['accuracy_pass'] is None and c['accuracy_acceptance_threshold'] is None,
                'wrong comparison scope/verdict')
        require(c['reference_top'] == old['top'], 'reference top summary mismatch')
        check_top(c['reference_top'])
        check_top(c['actual_top'])
        require(all(finite_nonnegative(c[key]) for key in ('max_abs', 'mean_abs', 'rmse'))
                and c['max_abs'] + 1e-12 >= c['rmse'] >= c['mean_abs'] - 1e-12,
                'inconsistent error statistics')
        require(type(c['max_abs_index']) is int and 0 <= c['max_abs_index'] < N,
                'maximum-error token index out of range')
        at_max = abs(as_f32(c['actual_at_max_abs']) - as_f32(c['reference_at_max_abs']))
        require(math.isclose(at_max, c['max_abs'], rel_tol=1e-12, abs_tol=1e-12),
                'maximum error disagrees with reported values')
        require(type(c['bit_equal_count']) is int and 0 <= c['bit_equal_count'] <= N,
                'invalid bit-equal count')
        thresholds = [0, 1e-6, 1e-4, 1e-3, .01, .1, .5, 1, 2, 5]
        counts = c['diagnostic_absolute_error_counts']
        require(len(counts) == len(thresholds), 'missing thresholds')
        previous = 0
        for count, threshold in zip(counts, thresholds):
            require(count['absolute_error_at_most'] == threshold
                    and type(count['count']) is int and type(count['count_above']) is int
                    and previous <= count['count'] <= N and count['count_above'] == N - count['count'],
                    'invalid threshold counts')
            previous = count['count']
            require(threshold < c['max_abs'] or count['count'] == N,
                    'threshold count contradicts maximum error')
        require(c['bit_equal_count'] <= counts[0]['count'], 'bit-equal count exceeds zero-error count')
        old_time, new_time = (item['timings']['decode_and_read_seconds'] for item in (old, new))
        require(finite_nonnegative(old_time) and finite_nonnegative(new_time) and new_time > 0,
                'invalid diagnostic timing')
        summaries.append({'case_index': i, 'prompt': PROMPTS[i], 'token_count': len(ids[i]),
                          'max_abs': c['max_abs'], 'mean_abs': c['mean_abs'], 'rmse': c['rmse'],
                          'bit_equal_count': c['bit_equal_count'],
                          'reference_top': c['reference_top'], 'e55_top': c['actual_top'],
                          'top1_same': c['reference_top']['top1_id'] == c['actual_top']['top1_id'],
                          'original_decode_and_read_seconds': old_time,
                          'e55_decode_and_read_seconds': new_time,
                          'one_run_decode_and_read_ratio': old_time / new_time})
    for item in (events[5], events[-1]):
        require(item['completed_cases'] == 4 and item['total_token_count'] == total
                and item['elements_per_case'] == N and item['total_elements'] == 4 * N
                and item['accuracy_pass'] is None and item['accuracy_acceptance_threshold'] is None,
                'missing/invalid suite completion')
    marker = 'MODEL_PREFIX_SUITE_REFERENCE_SHA256'
    require(lines.count(marker) == 1, 'missing reference hash')
    match = re.fullmatch(r'([0-9a-f]{64})  reference.json', lines[lines.index(marker) + 1])
    require(match is not None, 'invalid reference hash line')
    package = read_json(ROOT / PACKAGE)
    archive = fingerprint(archive_path)
    archive['members'] = []
    with tarfile.open(archive_path) as tar:
        members = tar.getmembers()
        require(len(members) == 5 and {m.name for m in members} == set(package['files']) | {'job.yaml'},
                'unexpected archive members')
        for member in members:
            require(member.isfile(), 'non-regular archive member')
            data = tar.extractfile(member).read()
            digest = hashlib.sha256(data).hexdigest()
            archive['members'].append({'name': member.name, 'bytes': len(data),
                                       'mode': oct(member.mode), 'sha256': digest})
            if member.name != 'job.yaml':
                entry = package['files'][member.name]
                require(len(data) == entry['bytes'] and digest == entry['sha256'], 'uploaded file mismatch')
                if member.name != 'remote_entrypoint.sh':
                    possible = [digest + '  ./' + member.name + suffix for suffix in ('', '.exec')]
                    require(sum(lines.count(line) for line in possible) == 1,
                            'executed binary digest missing/duplicated in log')
            else:
                expected_yaml = [f"name: {status['name']}", 'entrypoint: ./remote_entrypoint.sh',
                                 'args: []', 'env: {}', 'timeout_sec: 70']
                yaml_lines = data.decode().splitlines()
                require(len(yaml_lines) == 5 and set(yaml_lines) == set(expected_yaml),
                        'unexpected entrypoint/args/env/timeout in actual job config')
    known = read_json(ROOT / 'docs/experiments/e55-arena-model-identity.json')['identities']
    matches_known = all(identity[name]['sha256'] == item['sha256']
                        and identity[name]['size_bytes'] == item['bytes'] for name, item in known.items())
    return {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'FOUR_REAL_MODEL_PREFIXES_COMPARED_NO_ACCURACY_VERDICT', 'job': status,
            'package_manifest': PACKAGE, 'collector': fingerprint(Path(__file__).resolve()),
            'submitted_archive': archive, 'uploaded_package_matches_manifest': True,
            'log': fingerprint(log_path), 'asset_snapshots': assets, 'asset_identities_before': identity,
            'assets_match_separate_job_71224': matches_known,
            'asset_identity_scope': 'Full weight hash before model runs; weight size/mtime/header unchanged after. Metadata full hashes unchanged before/after. No second full weight hash or proof against unobservable external mutation.',
            'suite_events': events, 'case_summaries': summaries, 'total_prompt_tokens': total,
            'total_logits_compared': 4 * N, 'reference_file_sha256_reported_in_job': match.group(1),
            'reference_json_downloaded': False, 'accuracy_acceptance_threshold': None, 'accuracy_pass': None,
            'official_evaluation': False,
            'scope': 'Four short fixed prefixes, fresh Workspace each, finite whole-array comparison. No sampling, generated answer scoring, long-context/page/ring boundary or multimodal coverage. Sequential uncontrolled timings are not official performance. Collector checks reported statistics and provenance, not the raw logits independently.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('status', 'log', 'archive'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.status, args.log, args.archive), indent=2, ensure_ascii=False,
                     allow_nan=False))


if __name__ == '__main__':
    main()
