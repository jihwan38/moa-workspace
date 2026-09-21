#!/usr/bin/env python3
"""Collect an unchanged E55 CLI smoke run, never answer accuracy/performance PASS.

The CLI exposes text/counts/timing only, unlike the full-logit diagnostic hosts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

import collect_model_continuation as core

ROOT = core.ROOT
PREFIX = 'MODEL_CLI_PYTHON40'
PROMPT = 'Complete the Python expression: len([1, 2, 3])'
MAX_NEW = 40
PROMPT_TOKENS = 28
PACKAGE = 'moa-dev/experiments/candidates/model-cli-python40-e55/package-manifest.json'
MEMBERS = {'gemma4', 'moa-model-asset-inspector', 'remote_entrypoint.sh'}
FROZEN_BINARIES = {
    'gemma4': {'bytes': 12378864, 'sha256': '7e801ed65016b5d0faa71da3fd3f3f9d012db393adc47863ad8e4665532678eb'},
    'moa-model-asset-inspector': {'bytes': 880872, 'sha256': '1fb587ee436c5eb27099fe6177b4b04911b9b6ea09699de4a781c41bb932cf11'},
}
require = core.require
DURATION = r'\d+(?:\.\d+)?(?:ns|µs|ms|s)'
LOADED = re.compile(rf'model loaded in (?P<duration>{DURATION})')
SUMMARY = re.compile(
    rf'ttft: (?P<ttft>{DURATION}) \((?P<prompt>\d+) prompt tokens, '
    rf'(?P<prefill_rate>\d+\.\d{{2}}) tok/s prefill\) \| decode: (?P<completion>\d+) tokens in '
    rf'(?P<decode>{DURATION}) \((?P<decode_rate>\d+\.\d{{2}}) tok/s\) \| finish: (?P<finish>Stop|Length)')
PANIC = re.compile(r'''thread ['"][^\n]+?['"](?: \(\d+\))? panicked at\b|(?:^|\n)fatal runtime error:|(?:^|\n)memory allocation of \d+ bytes failed''')


def duration_seconds(value):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)(ns|µs|ms|s)', value)
    require(match is not None, 'invalid Rust duration format')
    seconds = float(match[1]) * {'s': 1, 'ms': 1e-3, 'µs': 1e-6, 'ns': 1e-9}[match[2]]
    require(math.isfinite(seconds) and seconds >= 0, 'nonfinite duration')
    return seconds


def validate_cli(lines):
    """Preserve merged CLI text; only source-defined lines prove completion."""
    text = '\n'.join(lines)
    require(PANIC.search(text) is None, 'Rust panic/fatal runtime diagnostic in CLI stream')
    loaded = [(index, LOADED.fullmatch(line)) for index, line in enumerate(lines) if LOADED.fullmatch(line)]
    require(len(loaded) == 1, 'missing/duplicate model-loaded log')
    loaded_index, loaded_match = loaded[0]
    echoes = [index for index, line in enumerate(lines) if line == PROMPT]
    require(bool(echoes), 'missing exact CLI prompt echo')
    echo_index = echoes[0]
    summaries = [(index, SUMMARY.fullmatch(line)) for index, line in enumerate(lines) if SUMMARY.fullmatch(line)]
    require(len(summaries) == 1, 'missing/duplicate complete CLI summary')
    summary_index, summary = summaries[0]
    require(loaded_index < echo_index < summary_index, 'CLI model/prompt/summary order mismatch')
    require(all(not line for line in lines[:loaded_index])
            and all(not line for line in lines[loaded_index + 1:echo_index])
            and all(not line for line in lines[summary_index + 1:]), 'unexpected CLI diagnostics outside generated text')
    prompt_tokens = int(summary['prompt'])
    completion_tokens = int(summary['completion'])
    require(prompt_tokens == PROMPT_TOKENS, 'unexpected CLI prompt-token count')
    require(0 <= completion_tokens <= MAX_NEW, 'CLI completion count exceeds configured limit')
    finish = summary['finish']
    require((finish == 'Length' and completion_tokens == MAX_NEW)
            or (finish == 'Stop' and completion_tokens < MAX_NEW), 'CLI finish/count disagrees with generate loop')
    load_seconds = duration_seconds(loaded_match['duration'])
    ttft_seconds = duration_seconds(summary['ttft'])
    decode_seconds = duration_seconds(summary['decode'])
    prefill_rate, decode_rate = float(summary['prefill_rate']), float(summary['decode_rate'])
    require(ttft_seconds > 0 and all(math.isfinite(value) and value >= 0 for value in (prefill_rate, decode_rate)),
            'invalid finite CLI timing/rates')
    require(completion_tokens != 0 or decode_rate == 0, 'zero completion has nonzero reported decode rate')
    # The printer can itself emit newlines, and main adds one after joining it.
    # Preserve the entire body as observed instead of guessing hidden token IDs.
    body = '\n'.join(lines[echo_index + 1:summary_index])
    return {
        'model_loaded_log': lines[loaded_index], 'prompt_echo': lines[echo_index],
        'generated_text_observed': body, 'generated_text_nonempty': bool(body.strip()),
        'generated_text_utf8_bytes': len(body.encode('utf-8')),
        'generated_text_observed_sha256': hashlib.sha256(body.encode('utf-8')).hexdigest(),
        'text_capture_scope': 'Merged stdout/stderr body between the first exact prompt echo and final CLI metrics. Main adds a final newline; token IDs and an independent GenerationOutput.text are unavailable.',
        'summary_log': lines[summary_index], 'prompt_tokens_reported': prompt_tokens,
        'completion_tokens_reported': completion_tokens, 'finish_reason_reported': finish,
        'stop_semantics': 'With this frozen request, Stop means an EOS-family token ended generation; its specific token ID is not printed. Completion count excludes EOS.',
        'model_load_seconds_rounded': load_seconds, 'ttft_seconds_rounded': ttft_seconds,
        'decode_seconds_rounded': decode_seconds, 'prefill_tokens_per_second_reported': prefill_rate,
        'decode_tokens_per_second_reported': decode_rate,
        'timing_scope': 'One rounded CLI report; TTFT starts after Workspace allocation and includes generate setup/prefill/first emitted delta, or a completion-time fallback if no delta was emitted. Decode includes sampling, streaming and host work. No controlled performance comparison.',
        'printer_panic_diagnostic_absent': True,
        'printer_join_scope': 'The frozen CLI ignores printer.join() errors; explicit Rust panic/fatal diagnostics are rejected. Absence of such text is not proof of printer output completeness.',
    }


def validate_log(lines):
    suffixes = ['BEGIN', 'ASSETS_BEFORE_BEGIN', 'ASSETS_BEFORE_END', 'CLI_BEGIN', 'CLI_EXIT=0',
                'ASSETS_AFTER_BEGIN', 'ASSETS_AFTER_END', 'STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT', 'EXIT=0']
    positions = {suffix: core.unique_marker(lines, PREFIX + '_' + suffix) for suffix in suffixes}
    require(list(positions.values()) == sorted(positions.values()), 'CLI runner markers are out of order')
    # A second nonzero exit marker must not be hidden by a later fake success.
    for suffix in ('CLI_EXIT=', 'EXIT='):
        values = [line for line in lines if line.startswith(PREFIX + '_' + suffix)]
        require(values == [PREFIX + '_' + suffix + '0'], 'conflicting CLI/runner exit markers')
    assets = []
    for moment in ('BEFORE', 'AFTER'):
        block = [line for line in lines[positions['ASSETS_' + moment + '_BEGIN'] + 1:
                                       positions['ASSETS_' + moment + '_END']] if line]
        require(len(block) == 1, 'asset phase must contain one complete inspector report')
        record = core.parse_json(block[0])
        require(record.get('schema') == core.common.ASSET_SCHEMA, 'wrong asset inspector schema')
        assets.append(record)
    identity = core.common.checked_assets(*assets)
    cli = validate_cli(lines[positions['CLI_BEGIN'] + 1:positions['CLI_EXIT=0']])
    hashes = lines[positions['BEGIN'] + 1:positions['ASSETS_BEFORE_BEGIN']]
    require(len(hashes) == 2, 'expected exactly two executed ELF hashes')
    return {'cli': cli, 'assets': assets, 'asset_identity': identity, 'executed_hash_lines': hashes}


def verify_archive(archive_path, package, status, executable_lines):
    require(set(package['files']) == MEMBERS, 'wrong CLI package file inventory')
    require(package['schema'] == 'MOA_MODEL_CLI_PYTHON40_PACKAGE_V1'
            and package['prompt'] == PROMPT and package['argv'] == ['./gemma4', PROMPT]
            and package['max_new_tokens'] == MAX_NEW and package['timeout_sec'] == 70
            and package['command_environment'] == {'GEMMA4_MAX_NEW_TOKENS': '40', 'FURIOSA_OPT_PROFILE': 'warn'}
            and package['sampling'] == {'temperature': 1.0, 'top_k': 64, 'top_p': 0.95, 'rng': 'unseeded rand::rng()'}
            and package['source_changes'] is False and package['all_46_match_historical_compiler_inputs'] is True,
            'CLI package request/source policy mismatch')
    core.null_verdict(package)
    for name, expected in FROZEN_BINARIES.items():
        require(all(package['files'][name][key] == value for key, value in expected.items()),
                'package does not identify frozen binary: ' + name)
    archive = core.common.fingerprint(archive_path)
    archive['members'] = []
    with tarfile.open(archive_path) as tar:
        members = tar.getmembers()
        require(len(members) == 4 and {member.name for member in members} == MEMBERS | {'job.yaml'}, 'unexpected CLI archive members')
        for member in members:
            require(member.isfile(), 'non-regular CLI archive member')
            data = tar.extractfile(member).read()
            digest = hashlib.sha256(data).hexdigest()
            archive['members'].append({'name': member.name, 'bytes': len(data), 'sha256': digest, 'mode': oct(member.mode)})
            if member.name == 'job.yaml':
                expected = [f"name: {status['name']}", 'entrypoint: ./remote_entrypoint.sh', 'args: []', 'env: {}', 'timeout_sec: 70']
                actual = data.decode().splitlines()
                require(len(actual) == 5 and set(actual) == set(expected), 'wrong actual CLI job config')
            else:
                entry = package['files'][member.name]
                require(len(data) == entry['bytes'] and digest == entry['sha256'], 'uploaded CLI payload mismatch')
                if member.name in FROZEN_BINARIES:
                    possible = [digest + '  ./' + member.name + suffix for suffix in ('', '.exec')]
                    require(sum(executable_lines.count(line) for line in possible) == 1, 'executed CLI binary digest missing/duplicated')
    return archive


def collect(status_path, log_path, archive_path, package_path=None):
    status = core.parse_json(status_path.read_text())
    require(status['status'] == 'SUCCEEDED' and type(status['exit_code']) is int and status['exit_code'] == 0,
            'CLI job has not succeeded with exit0')
    data = validate_log(log_path.read_text().splitlines())
    package_path = ROOT / PACKAGE if package_path is None else package_path
    package = core.parse_json(package_path.read_text())
    archive = verify_archive(archive_path, package, status, data['executed_hash_lines'])
    known = core.common.read_json(ROOT / 'docs/experiments/e55-arena-model-identity.json')['identities']
    matches_known = all(data['asset_identity'][name]['sha256'] == item['sha256']
                        and data['asset_identity'][name]['size_bytes'] == item['bytes'] for name, item in known.items())
    return {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'UNMODIFIED_E55_CLI_STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT', 'job': status,
        'collector': core.common.fingerprint(Path(__file__).resolve()),
        'shared_validation_module': core.common.fingerprint(Path(core.__file__).resolve()),
        'shared_asset_validation_module': core.common.fingerprint(Path(core.common.__file__).resolve()),
        'package_manifest': core.common.fingerprint(package_path), 'submitted_archive': archive,
        'uploaded_package_matches_manifest': True, 'frozen_cli_and_inspector_verified': True,
        'log': core.common.fingerprint(log_path), 'asset_snapshots': data['assets'],
        'asset_identities_before': data['asset_identity'], 'assets_match_separate_job_71224': matches_known,
        'asset_identity_scope': 'Full hash before; weight stat/header and metadata full hashes after. No second full weight hash or proof against unobservable mutation.',
        'cli_observations': data['cli'],
        'request': {'prompt': PROMPT, 'max_new_tokens': MAX_NEW, 'thinking': False, 'separate_reasoning': False,
                    'sampling': {'temperature': 1.0, 'top_k': 64, 'top_p': 0.95},
                    'seed_control': False, 'stop_strings': [], 'image': False, 'audio': False},
        'accuracy_acceptance_threshold': None, 'accuracy_pass': None, 'official_evaluation': False,
        'scope': 'One text-only execution of the frozen, unchanged stochastic gemma4 CLI. Completion/text/timing and upload/asset provenance are checked. No independent answer score, greedy equivalence, full-logit finiteness proof, multimodal/long-context coverage, official score or controlled performance claim. This run does not establish the cause or resolution of the earlier job71173 timeout.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('status', 'log', 'archive'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.status, args.log, args.archive), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
