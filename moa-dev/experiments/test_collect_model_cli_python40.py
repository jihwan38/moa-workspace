#!/usr/bin/env python3
"""CLI completion, provenance and merged-text mutation tests; no model execution."""
import copy
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True
import collect_model_cli_python40 as collector

ROOT = collector.ROOT


def cli_lines(text='3', completion=13, finish='Stop'):
    return ['model loaded in 2.50s', collector.PROMPT, text,
            f'ttft: 1.25s (28 prompt tokens, 22.40 tok/s prefill) | decode: {completion} tokens in 800.00ms ({completion / .8:.2f} tok/s) | finish: {finish}']


def full_lines(cli=None):
    prior = collector.core.parse_json((ROOT / 'docs/experiments/e55-model-continuation-python40.json').read_text())
    before, after = prior['asset_snapshots']
    mark = lambda suffix: collector.PREFIX + '_' + suffix
    hashes = [record['sha256'] + '  ./' + name for name, record in collector.FROZEN_BINARIES.items()]
    return [mark('BEGIN'), *hashes, mark('ASSETS_BEFORE_BEGIN'), json.dumps(before), mark('ASSETS_BEFORE_END'),
            mark('CLI_BEGIN'), *(cli if cli is not None else cli_lines()), mark('CLI_EXIT=0'),
            mark('ASSETS_AFTER_BEGIN'), json.dumps(after), mark('ASSETS_AFTER_END'),
            mark('STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT'), mark('EXIT=0')]


def make_archive(path, payloads, yaml=None, extra=None):
    yaml = yaml or b'name: synthetic-cli\nentrypoint: ./remote_entrypoint.sh\nargs: []\nenv: {}\ntimeout_sec: 70\n'
    with tarfile.open(path, 'w') as tar:
        for name, data in {**payloads, 'job.yaml': yaml}.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if extra is not None:
            info = tarfile.TarInfo(extra); info.size = 1
            tar.addfile(info, io.BytesIO(b'x'))


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

    parsed = collector.validate_cli(cli_lines())
    check('ordinary_Stop_text_and_metrics', parsed['generated_text_observed'] == '3' and parsed['completion_tokens_reported'] == 13)
    check('Length_requires_forty', collector.validate_cli(cli_lines(completion=40, finish='Length'))['completion_tokens_reported'] == 40)
    check('immediate_EOS_empty_text_allowed', collector.validate_cli(cli_lines(text='', completion=0))['generated_text_nonempty'] is False)
    check('numeric_zero_text_is_nonempty', collector.validate_cli(cli_lines(text='0'))['generated_text_nonempty'] is True)
    body = 'An error message is ordinary generated text.\nError: example\n{"result":3}\n' + collector.PROMPT
    check('error_word_JSON_and_repeated_prompt_remain_text', collector.validate_cli(cli_lines(body))['generated_text_observed'] == body)
    check('microsecond_duration_format', abs(collector.duration_seconds('1.25µs') - 1.25e-6) < 1e-15)
    reject('Stop_cannot_count_fortieth_nonEOS', lambda: collector.validate_cli(cli_lines(completion=40)), 'CLI finish/count disagrees with generate loop')
    reject('Length_cannot_end_early', lambda: collector.validate_cli(cli_lines(completion=39, finish='Length')), 'CLI finish/count disagrees with generate loop')
    reject('completion_exceeds_forty', lambda: collector.validate_cli(cli_lines(completion=41)), 'CLI completion count exceeds configured limit')
    reject('missing_model_loaded', lambda: collector.validate_cli(cli_lines()[1:]), 'missing/duplicate model-loaded log')
    reject('duplicate_model_loaded', lambda: collector.validate_cli([cli_lines()[0]] + cli_lines()), 'missing/duplicate model-loaded log')
    reject('missing_prompt_echo', lambda: collector.validate_cli([line for line in cli_lines() if line != collector.PROMPT]), 'missing exact CLI prompt echo')
    reject('missing_final_summary', lambda: collector.validate_cli(cli_lines()[:-1]), 'missing/duplicate complete CLI summary')
    reject('duplicate_final_summary', lambda: collector.validate_cli(cli_lines() + [cli_lines()[-1]]), 'missing/duplicate complete CLI summary')
    reversed_lines = cli_lines(); reversed_lines[0], reversed_lines[1] = reversed_lines[1], reversed_lines[0]
    reject('prompt_before_load', lambda: collector.validate_cli(reversed_lines), 'CLI model/prompt/summary order mismatch')
    reject('late_runtime_error_after_metrics', lambda: collector.validate_cli(cli_lines() + ['Error: runtime failed']), 'unexpected CLI diagnostics outside generated text')
    reject('ignored_printer_join_panic', lambda: collector.validate_cli(cli_lines("thread '<unnamed>' (42) panicked at src/bin/gemma4.rs:1:2:")), 'Rust panic/fatal runtime diagnostic in CLI stream')
    reject('fatal_runtime_diagnostic', lambda: collector.validate_cli(cli_lines('fatal runtime error: failed to initiate panic')), 'Rust panic/fatal runtime diagnostic in CLI stream')
    wrong_count = cli_lines(); wrong_count[-1] = wrong_count[-1].replace('28 prompt tokens', '29 prompt tokens')
    reject('unexpected_prompt_tokens', lambda: collector.validate_cli(wrong_count), 'unexpected CLI prompt-token count')
    nan = cli_lines(); nan[-1] = nan[-1].replace('22.40 tok/s', 'NaN tok/s')
    reject('nonfinite_rate', lambda: collector.validate_cli(nan), 'missing/duplicate complete CLI summary')
    zero_time = cli_lines(); zero_time[-1] = zero_time[-1].replace('ttft: 1.25s', 'ttft: 0.00ns')
    reject('zero_ttft', lambda: collector.validate_cli(zero_time), 'invalid finite CLI timing/rates')
    zero_count = cli_lines(completion=0); zero_count[-1] = zero_count[-1].replace('(0.00 tok/s)', '(1.00 tok/s)')
    reject('zero_count_nonzero_rate', lambda: collector.validate_cli(zero_count), 'zero completion has nonzero reported decode rate')

    lines = full_lines()
    collector.validate_log(lines)
    passed.append('full_runner_asset_log_positive')
    missing = [line for line in lines if line != collector.PREFIX + '_CLI_EXIT=0']
    reject('missing_CLI_exit', lambda: collector.validate_log(missing), 'missing/duplicate marker: MODEL_CLI_PYTHON40_CLI_EXIT=0')
    conflict = list(lines); conflict.insert(conflict.index(collector.PREFIX + '_CLI_EXIT=0'), collector.PREFIX + '_CLI_EXIT=1')
    reject('conflicting_CLI_exit', lambda: collector.validate_log(conflict), 'conflicting CLI/runner exit markers')
    partial = list(lines)
    index = partial.index(collector.PREFIX + '_ASSETS_BEFORE_BEGIN') + 1
    asset = json.loads(partial[index]); asset['files'][0]['full_file']['status'] = 'incomplete'; partial[index] = json.dumps(asset)
    reject('incomplete_full_asset_hash', lambda: collector.validate_log(partial), 'incomplete before-run asset identity: model.safetensors')
    duplicate_asset = list(lines); index = duplicate_asset.index(collector.PREFIX + '_ASSETS_AFTER_BEGIN') + 1
    duplicate_asset.insert(index, duplicate_asset[index])
    reject('duplicate_asset_record', lambda: collector.validate_log(duplicate_asset), 'asset phase must contain one complete inspector report')

    package_path = ROOT / collector.PACKAGE
    package = collector.core.parse_json(package_path.read_text())
    payloads = {name: (ROOT / entry['source']).read_bytes() for name, entry in package['files'].items()}
    with tempfile.TemporaryDirectory(prefix='moa-cli-collector-test-') as directory:
        temp = Path(directory)
        archive = temp / 'job.tar'; status_path = temp / 'status.json'; log = temp / 'log.txt'
        status = {'id': 999999, 'name': 'synthetic-cli', 'status': 'SUCCEEDED', 'exit_code': 0}
        status_path.write_text(json.dumps(status)); log.write_text('\n'.join(lines) + '\n')
        make_archive(archive, payloads)
        full = collector.collect(status_path, log, archive)
        check('frozen_real_ELF_payload_synthetic_log_collection', full['frozen_cli_and_inspector_verified']
              and full['accuracy_pass'] is None and full['request']['sampling']['top_k'] == 64)
        status_path.write_text(json.dumps({**status, 'status': 'FAILED', 'exit_code': 101}))
        reject('failed_job_not_structural_success', lambda: collector.collect(status_path, log, archive), 'CLI job has not succeeded with exit0')
        status_path.write_text(json.dumps(status))
        executable_lines = collector.validate_log(lines)['executed_hash_lines']
        wrong_manifest = copy.deepcopy(package); wrong_manifest['files']['gemma4']['sha256'] = '0' * 64
        reject('manifest_replacement_CLI', lambda: collector.verify_archive(archive, wrong_manifest, status, executable_lines), 'package does not identify frozen binary: gemma4')
        wrong_sampling = copy.deepcopy(package); wrong_sampling['sampling']['top_k'] = 1
        reject('manifest_falsely_claims_greedy', lambda: collector.verify_archive(archive, wrong_sampling, status, executable_lines), 'CLI package request/source policy mismatch')
        wrong_execute = list(executable_lines); wrong_execute[0] = '0' * 64 + wrong_execute[0][64:]
        reject('executed_binary_hash_mismatch', lambda: collector.verify_archive(archive, package, status, wrong_execute), 'executed CLI binary digest missing/duplicated')
        altered = dict(payloads); altered['gemma4'] = altered['gemma4'][:-1] + bytes([altered['gemma4'][-1] ^ 1])
        make_archive(archive, altered)
        reject('actual_uploaded_binary_changed', lambda: collector.verify_archive(archive, package, status, executable_lines), 'uploaded CLI payload mismatch')
        make_archive(archive, payloads, yaml=b'name: synthetic-cli\nentrypoint: ./remote_entrypoint.sh\nargs: []\nenv: {}\ntimeout_sec: 71\n')
        reject('actual_timeout_changed', lambda: collector.verify_archive(archive, package, status, executable_lines), 'wrong actual CLI job config')
        make_archive(archive, payloads, extra='other')
        reject('extra_uploaded_file', lambda: collector.verify_archive(archive, package, status, executable_lines), 'unexpected CLI archive members')

    actual_reports = []
    for path in sorted((ROOT / 'docs/experiments').glob('e55-model-cli-python40*.json')):
        saved = collector.core.parse_json(path.read_text())
        if 'job' not in saved or 'log' not in saved or 'submitted_archive' not in saved:
            continue
        log = Path(saved['log']['path']); status = log.with_name('status-' + str(saved['job']['id']) + '.json')
        actual = collector.collect(status, log, Path(saved['submitted_archive']['path']))
        check('actual_job_' + str(saved['job']['id']), actual['accuracy_pass'] is None)
        actual_reports.append(str(path.relative_to(ROOT)))
        actual_lines = log.read_text().splitlines()
        suffix = '_' + str(saved['job']['id'])
        summary = actual['cli_observations']['summary_log']
        summary_index = actual_lines.index(summary)
        missing_summary = list(actual_lines); missing_summary.pop(summary_index)
        reject('actual_missing_CLI_summary' + suffix, lambda: collector.validate_log(missing_summary), 'missing/duplicate complete CLI summary')
        wrong_finish = list(actual_lines)
        if actual['cli_observations']['finish_reason_reported'] == 'Stop':
            wrong_finish[summary_index] = summary.replace('finish: Stop', 'finish: Length')
        else:
            wrong_finish[summary_index] = summary.replace('finish: Length', 'finish: Stop')
        reject('actual_finish_count_conflict' + suffix, lambda: collector.validate_log(wrong_finish), 'CLI finish/count disagrees with generate loop')
        changed_prompt = list(actual_lines)
        begin = changed_prompt.index(collector.PREFIX + '_CLI_BEGIN')
        prompt_index = changed_prompt.index(collector.PROMPT, begin)
        changed_prompt[prompt_index] = 'Complete a different Python expression.'
        reject('actual_prompt_echo_changed' + suffix, lambda: collector.validate_log(changed_prompt), 'missing exact CLI prompt echo')
        panic = list(actual_lines)
        panic.insert(summary_index, "thread '<unnamed>' (42) panicked at src/bin/gemma4.rs:1:2:")
        reject('actual_ignored_printer_join_panic' + suffix, lambda: collector.validate_log(panic), 'Rust panic/fatal runtime diagnostic in CLI stream')
        no_completion = [line for line in actual_lines if line != collector.PREFIX + '_STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT']
        reject('actual_missing_runner_completion' + suffix, lambda: collector.validate_log(no_completion),
               'missing/duplicate marker: MODEL_CLI_PYTHON40_STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT')
        wrong_exit = list(actual_lines)
        wrong_exit[wrong_exit.index(collector.PREFIX + '_CLI_EXIT=0')] = collector.PREFIX + '_CLI_EXIT=1'
        reject('actual_CLI_nonzero_exit' + suffix, lambda: collector.validate_log(wrong_exit), 'missing/duplicate marker: MODEL_CLI_PYTHON40_CLI_EXIT=0')
        changed_asset = list(actual_lines)
        after_index = changed_asset.index(collector.PREFIX + '_ASSETS_AFTER_BEGIN') + 1
        after = json.loads(changed_asset[after_index])
        next(item for item in after['files'] if item['basename'] == 'model.safetensors')['header']['sha256'] = '0' * 64
        changed_asset[after_index] = json.dumps(after)
        reject('actual_weight_header_changed' + suffix, lambda: collector.validate_log(changed_asset), 'weight header changed')
        incomplete_asset = list(actual_lines)
        before_index = incomplete_asset.index(collector.PREFIX + '_ASSETS_BEFORE_BEGIN') + 1
        before = json.loads(incomplete_asset[before_index])
        next(item for item in before['files'] if item['basename'] == 'model.safetensors')['full_file']['bytes_hashed'] -= 1
        incomplete_asset[before_index] = json.dumps(before)
        reject('actual_partial_weight_hash' + suffix, lambda: collector.validate_log(incomplete_asset), 'full digest byte count mismatch: model.safetensors')
        generated_json = list(actual_lines)
        generated_json.insert(summary_index, '{"schema":"MOA_MODEL_ASSET_INSPECTOR_RUST_V1","error":"a generated example"}')
        parsed = collector.validate_log(generated_json)
        check('actual_generated_JSON_error_word_is_text' + suffix,
              'a generated example' in parsed['cli']['generated_text_observed'] and len(parsed['assets']) == 2)
    print(json.dumps({'status': 'PASS', 'test_count': len(passed), 'tests': passed,
                      'actual_reports_tested': actual_reports, 'actual_evidence_available': bool(actual_reports),
                      'npu_run': False}, indent=2))


if __name__ == '__main__':
    main()
