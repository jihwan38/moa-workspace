#!/usr/bin/env python3
"""One bounded allocation: actual E128 -> frozen FFN candidate -> actual E128.

Original three-input hosts, fixture and runner. No compilation or diagnostic
host. Independent repeats use new labels and the same immutable stage bytes.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / 'moa-dev/experiments/artifacts' / HERE.name
sys.path.insert(0, str(HERE))
from package import (TARGET, KERNELS, checked_package, clean_label, digest,
                     file_record, package_files, read, write)
sys.path.insert(0, str(ROOT / 'moa-dev'))
from measure import cli, parse_log

STAGES = ('control_before', 'candidate_whole', 'control_after')


def prepare(label, build_label):
    clean_label(label)
    path = HERE / f'{label}-build.json'
    assert not path.exists(), path
    package_path, packaged, candidate, control = checked_package(build_label)
    out = ARTIFACTS / label
    out.mkdir(parents=True, exist_ok=False)
    for side, source in [('control', control), ('candidate', candidate)]:
        (out / f'{side}_test_runtime').write_bytes((source / 'test_runtime').read_bytes())
    for name in ('fixtures.safetensors', 'remote_entrypoint.sh'):
        assert (control / name).read_bytes() == (candidate / name).read_bytes()
        (out / name).write_bytes((control / name).read_bytes())
    runner = ['#!/bin/sh', 'set -u', 'cd "$(dirname "$0")" || exit 1', 'any_failed=0']
    for name in STAGES:
        side = 'candidate' if name == 'candidate_whole' else 'control'
        runner += [f"printf 'DIAG_STAGE_BEGIN {name}\\n'",
                   f'mkdir {name} || exit 1',
                   f'cp {side}_test_runtime {name}/test_runtime || exit 1',
                   f'cp fixtures.safetensors remote_entrypoint.sh {name}/ || exit 1',
                   f'if FURIOSA_OPT_PROFILE=info sh {name}/remote_entrypoint.sh; then code=0; else code=$?; any_failed=1; fi',
                   f"printf 'DIAG_STAGE_EXIT {name} %s\\n' \"$code\"",
                   f"printf 'DIAG_STAGE_END {name}\\n'"]
    runner += ["printf 'DIAG_SCREEN_COMPLETE\\n'", 'exit "$any_failed"']
    (out / 'run.sh').write_text('\n'.join(runner) + '\n')
    (out / 'run.sh').chmod(0o755)
    write(path, {'schema_version': 1, 'label': label, 'build_label': build_label,
                 'candidate_package': file_record(package_path), 'control': packaged['control'],
                 'graph_gate': packaged['graph_gate'], 'stages': list(STAGES),
                 'timeout_seconds': 70, 'expected_comparisons': 45,
                 'builder': file_record(Path(__file__).resolve()),
                 'package_dir': str(out.relative_to(ROOT)),
                 'package_files': {p.name: digest(p.read_bytes()) for p in sorted(out.iterdir())},
                 'normalized_stage_files': {'candidate': package_files(candidate), 'control': package_files(control)},
                 'method': 'One allocation, three serial original three-input benchmark hosts: actual E128 before -> candidate whole FFN -> actual E128 after. PROFILE=info; no diagnostic host. Before/after and inputs are not independent allocations.'})
    print(f'Prepared {label}: three stages / 45 expected comparisons; no submission')


def checked_screen(label):
    path = HERE / f'{clean_label(label)}-build.json'
    saved = read(path)
    package_path, packaged, candidate, control = checked_package(saved['build_label'])
    assert file_record(package_path) == saved['candidate_package']
    assert saved['control'] == packaged['control'] and saved['graph_gate'] == packaged['graph_gate']
    assert saved['stages'] == list(STAGES) and saved['timeout_seconds'] == 70
    assert saved['normalized_stage_files'] == {'candidate': package_files(candidate), 'control': package_files(control)}
    out = ROOT / saved['package_dir']
    assert {p.name: digest(p.read_bytes()) for p in out.iterdir()} == saved['package_files']
    for side in ('candidate', 'control'):
        mapping = {'test_runtime': side + '_test_runtime', 'fixtures.safetensors': 'fixtures.safetensors',
                   'remote_entrypoint.sh': 'remote_entrypoint.sh'}
        assert {key: saved['package_files'][value] for key, value in mapping.items()} == saved['normalized_stage_files'][side]
    return path, saved, out


def submit(label):
    path, saved, out = checked_screen(label)
    submission = HERE / f'{label}-submission.json'
    assert not submission.exists(), submission
    remote = '/tmp/moa-joint-direct-screen/' + label
    subprocess.run(['docker', 'exec', 'moa2026-dev', 'mkdir', '-p', remote], check=True)
    subprocess.run(['docker', 'cp', str(out) + '/.', 'moa2026-dev:' + remote], check=True)
    for name, expected in saved['package_files'].items():
        actual = subprocess.check_output(['docker', 'exec', 'moa2026-dev', 'sha256sum', remote + '/' + name], text=True).split()[0]
        assert actual == expected['sha256'], name
    write(submission, {'attempted_at_utc': datetime.now(timezone.utc).isoformat(), 'build_record': file_record(path)})
    response = cli('submit', *(remote + '/' + name for name in saved['package_files']),
                   '--entrypoint', 'run.sh', '--timeout', saved['timeout_seconds'], '--name', 'beef-' + label)
    value = read(submission)
    value['response'] = response
    jobs = re.findall(r'submitted job (\d+)', response)
    if len(jobs) == 1:
        value['job_id'] = int(jobs[0])
    submission.write_text(json.dumps(value, indent=2) + '\n')
    assert len(jobs) == 1, response
    print(response, end='')


def parse_stages(raw):
    sections = re.findall(r'^DIAG_STAGE_BEGIN (\w+)\n(.*?)^DIAG_STAGE_END \1$', raw, re.M | re.S)
    names = [name for name, _ in sections]
    values = {}
    for name in STAGES:
        parts = [part for observed, part in sections if observed == name]
        partial = re.search(r'^DIAG_STAGE_BEGIN ' + name + r'\n(.*?)(?=^DIAG_STAGE_BEGIN |\Z)', raw, re.M | re.S)
        content = parts[0] if len(parts) == 1 else (partial[1] if partial else '')
        exits = re.findall(r'^DIAG_STAGE_EXIT ' + name + r' (\d+)$', content, re.M)
        try:
            assert len(parts) == 1, 'Missing, incomplete or duplicate stage'
            assert exits == ['0'], exits
            kernels = parse_log(content)
            assert set(kernels) == set(KERNELS)
            assert sum(k['output_comparisons_passed'] for k in kernels.values()) == 15
            values[name] = {'valid': True, 'complete': True, 'exit_code': 0,
                            'correctness_passes': 15, 'kernels': kernels}
        except (AssertionError, ValueError, KeyError) as error:
            comparisons = re.findall(r'^\[([^]]+)\].*-> (PASS|FAIL)$', content, re.M)
            values[name] = {'valid': False, 'error': repr(error), 'stage_started': partial is not None,
                            'complete': len(parts) == 1, 'observed_exit_codes': exits,
                            'observed_comparisons': [{'label': label.strip(), 'status': status} for label, status in comparisons]}
    return values, names


def archive_files(path):
    with tarfile.open(path) as archive:
        members = [m for m in archive if m.isfile()]
        assert len(members) == len({m.name for m in members}), 'Duplicate archive entries'
        return {m.name: digest(archive.extractfile(m).read()) for m in members}


def collect(label, job=None):
    clean_label(label)
    build_path = HERE / f'{label}-build.json'
    saved = read(build_path)
    original = read(HERE / f'{label}-submission.json')['job_id']
    job = original if job is None else job
    path = HERE / f'{label}-job-{job}-results.json'
    assert not path.exists(), path
    status = json.loads(cli('status', job))
    assert status['status'] in {'SUCCEEDED', 'COMPLETED', 'FAILED', 'CANCELLED', 'CANCELED'}, status
    raw = cli('logs', job)
    evidence = HERE / 'evidence'
    evidence.mkdir(exist_ok=True)
    log_path = evidence / f'arena-{job}.log'
    with log_path.open('x') as stream:
        stream.write(raw)
    stages, names = parse_stages(raw)
    result = {'job': status, 'log': str(log_path.relative_to(ROOT)), 'log_digest': digest(raw.encode()),
              'build_record': file_record(build_path), 'method': saved['method'], 'stages': stages,
              'completed_stage_names': names, 'stage_sequence_matches': names == list(STAGES),
              'screen_complete': len(re.findall(r'^DIAG_SCREEN_COMPLETE$', raw, re.M)) == 1,
              'actual_upload_matches': False, 'valid': False}
    try:
        remote = f'/tmp/moa-joint-direct-screen/{label}/job-{job}.tar'
        cli('download', job, '--out', remote)
        archive = ARTIFACTS / f'job-{job}.tar'
        assert not archive.exists(), archive
        subprocess.run(['docker', 'cp', 'moa2026-dev:' + remote, str(archive)], check=True)
        files = archive_files(archive)
        result.update(package_files=files, archive=file_record(archive))
        assert {k: v for k, v in files.items() if k != 'job.yaml'} == saved['package_files']
        result['actual_upload_matches'] = True
        result['valid'] = (status['status'] in {'SUCCEEDED', 'COMPLETED'} and status['exit_code'] == 0
                           and result['stage_sequence_matches'] and result['screen_complete']
                           and all(stage['valid'] for stage in stages.values()))
        if result['valid']:
            result['correctness_passes'] = sum(stage['correctness_passes'] for stage in stages.values())
            assert result['correctness_passes'] == 45
            before, candidate, after = [stages[name]['kernels'][TARGET]['median_cycles'] for name in STAGES]
            bracket = statistics.median([before, after])
            result['whole_ffn_comparison'] = {'control_before': before, 'candidate': candidate,
                'control_after': after, 'bracketing_median': bracket,
                'candidate_change_percent': 100 * (candidate / bracket - 1)}
    except Exception as error:
        result.update(valid=False, error=repr(error))
    write(path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {'stages', 'package_files'}}, indent=2))
    if not result['valid']:
        raise SystemExit('Saved failed/partial evidence; inspect before any further allocation')


def verified_batch(path):
    batch = read(path)
    assert batch['valid'] and batch['actual_upload_matches'] and batch['correctness_passes'] == 45
    build_path = ROOT / batch['build_record']['path']
    assert file_record(build_path) == batch['build_record']
    built = read(build_path)
    raw = (ROOT / batch['log']).read_bytes()
    assert digest(raw) == batch['log_digest']
    archive = ROOT / batch['archive']['path']
    assert file_record(archive) == batch['archive']
    files = archive_files(archive)
    assert files == batch['package_files']
    assert {k: v for k, v in files.items() if k != 'job.yaml'} == built['package_files']
    stages, names = parse_stages(raw.decode())
    assert names == list(STAGES) and stages == batch['stages']
    assert len(re.findall(r'^DIAG_SCREEN_COMPLETE$', raw.decode(), re.M)) == 1
    assert batch['job']['status'] in {'SUCCEEDED', 'COMPLETED'} and batch['job']['exit_code'] == 0
    for side in ('candidate', 'control'):
        mapping = {'test_runtime': side + '_test_runtime', 'fixtures.safetensors': 'fixtures.safetensors',
                   'remote_entrypoint.sh': 'remote_entrypoint.sh'}
        assert {k: files[v] for k, v in mapping.items()} == built['normalized_stage_files'][side]
    return batch, built


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'status', 'collect', 'check'])
    parser.add_argument('label')
    parser.add_argument('build_label', nargs='?')
    parser.add_argument('--job', type=int)
    args = parser.parse_args()
    clean_label(args.label)
    if args.action == 'prepare':
        assert args.build_label and args.job is None
        prepare(args.label, args.build_label)
    elif args.action == 'collect':
        assert args.build_label is None
        collect(args.label, args.job)
    else:
        assert args.build_label is None and args.job is None
        if args.action == 'submit':
            submit(args.label)
        elif args.action == 'status':
            print(cli('status', read(HERE / f'{args.label}-submission.json')['job_id']), end='')
        else:
            checked_screen(args.label)
            print('Screen package and exact E128 control verified')
