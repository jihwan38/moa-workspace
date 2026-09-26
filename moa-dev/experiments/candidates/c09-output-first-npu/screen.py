#!/usr/bin/env python3
"""One allocation: exact E122 whole controls, paired O profiles, C09 whole test."""
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
from package import checked_control, digest, file_record, write
sys.path.insert(0, str(ROOT / 'moa-dev'))
from measure import cli, parse_log

O = 'sliding_attention_output'
STAGES = [
    {'name': 'control_before', 'kind': 'whole', 'side': 'control', 'level': 'info'},
    {'name': 'control_info', 'kind': 'diagnostic', 'side': 'control', 'level': 'info'},
    {'name': 'control_trace', 'kind': 'diagnostic', 'side': 'control', 'level': 'trace'},
    {'name': 'candidate_info', 'kind': 'diagnostic', 'side': 'candidate', 'level': 'info'},
    {'name': 'candidate_trace', 'kind': 'diagnostic', 'side': 'candidate', 'level': 'trace'},
    {'name': 'candidate_whole', 'kind': 'whole', 'side': 'candidate', 'level': 'info'},
    {'name': 'control_after', 'kind': 'whole', 'side': 'control', 'level': 'info'},
]


def prepare(label, build_label):
    assert re.fullmatch(r'[a-z0-9-]{1,70}', label)
    record_path = HERE / f'{label}-build.json'
    assert not record_path.exists()
    package_path = HERE / f'package-{build_label}.json'
    package = json.loads(package_path.read_text())
    assert package['status'] == 'ready'
    assert package['task_bytes_unchanged'] and package['profile_name_bytes_only']
    _, control_build, control_dir, control_record = checked_control()
    assert package['control'] == control_record
    candidate_dir = ARTIFACTS / build_label / 'package'
    diagnostic_dir = ARTIFACTS / build_label / 'diagnostic'
    assert {p.name: digest(p.read_bytes()) for p in candidate_dir.iterdir()} == package['package_files']
    for name, expected in package['diagnostic_files'].items():
        assert digest((diagnostic_dir / name).read_bytes()) == expected, name
    # Recheck the gate before preparing a run, not merely before host compilation.
    subprocess.run([sys.executable, str(HERE / 'verify_descriptors.py'), build_label, '--check'],
                   cwd=ROOT, check=True)
    gate_path = HERE / f'descriptor-analysis-{build_label}.json'
    assert file_record(gate_path) == package['descriptor_gate']
    assert json.loads(gate_path.read_text())['passed'] is True
    out = ARTIFACTS / label
    out.mkdir(parents=True, exist_ok=False)
    for side, folder in [('control', control_dir), ('candidate', candidate_dir)]:
        (out / f'{side}_test_runtime').write_bytes((folder / 'test_runtime').read_bytes())
        for name in ('fixtures.safetensors', 'remote_entrypoint.sh'):
            assert (folder / name).read_bytes() == (control_dir / name).read_bytes()
        (out / f'{side}_probe').write_bytes((diagnostic_dir / f'{side}-probe').read_bytes())
    for name in ('fixtures.safetensors', 'remote_entrypoint.sh'):
        (out / name).write_bytes((control_dir / name).read_bytes())
    runner = ['#!/bin/sh', 'set -u', 'cd "$(dirname "$0")" || exit 1', 'any_failed=0']
    for stage in STAGES:
        name, side = stage['name'], stage['side']
        runner += [f"printf 'DIAG_STAGE_BEGIN {name}\\n'"]
        if stage['kind'] == 'whole':
            # Official runner is byte-identical; the isolated cwd gives it its
            # expected filenames without modifying any uploaded host or fixture.
            runner += [f'mkdir {name} || exit 1',
                       f'cp {side}_test_runtime {name}/test_runtime || exit 1',
                       f'cp fixtures.safetensors remote_entrypoint.sh {name}/ || exit 1',
                       f'if FURIOSA_OPT_PROFILE=info sh {name}/remote_entrypoint.sh; then code=0; else code=$?; any_failed=1; fi']
        else:
            runner += [f'chmod +x {side}_probe || exit 1',
                       f'if FURIOSA_OPT_PROFILE={stage["level"]} DIAG_KERNEL={O} ./{side}_probe; then code=0; else code=$?; any_failed=1; fi']
        runner += [f"printf 'DIAG_STAGE_EXIT {name} %s\\n' \"$code\"",
                   f"printf 'DIAG_STAGE_END {name}\\n'"]
    runner += ["printf 'DIAG_SCREEN_COMPLETE\\n'", 'exit "$any_failed"']
    (out / 'run.sh').write_text('\n'.join(runner) + '\n')
    (out / 'run.sh').chmod(0o755)
    write(record_path, {'label': label, 'build_label': build_label,
                       'candidate_package': file_record(package_path), 'control': control_record,
                       'stages': STAGES, 'timeout_seconds': 70,
                       'expected_output_comparisons': 57,
                       'method': 'One allocation, seven serial processes; three original inputs per stage. Whole measurements use official hosts; profile info/trace use same diagnostic host source. Not independent repetitions.',
                       'profile_scope': package['control_diagnostic_reuse']['scope'],
                       'package_files': {p.name: digest(p.read_bytes()) for p in sorted(out.iterdir())}})
    print(f'Prepared {label}: 7 stages / 57 expected comparisons; no submission')


def submit(label):
    saved = json.loads((HERE / f'{label}-build.json').read_text())
    out = ARTIFACTS / label
    assert {p.name: digest(p.read_bytes()) for p in out.iterdir()} == saved['package_files']
    remote = '/tmp/moa-c09-screen/' + label
    subprocess.run(['docker', 'exec', 'moa2026-dev', 'mkdir', '-p', remote], check=True)
    subprocess.run(['docker', 'cp', str(out) + '/.', 'moa2026-dev:' + remote], check=True)
    for name, expected in saved['package_files'].items():
        actual = subprocess.check_output(['docker', 'exec', 'moa2026-dev', 'sha256sum', remote + '/' + name], text=True).split()[0]
        assert actual == expected['sha256'], name
    path = HERE / f'{label}-submission.json'
    write(path, {'attempted_at_utc': datetime.now(timezone.utc).isoformat()})
    response = cli('submit', *(remote + '/' + n for n in saved['package_files']),
                   '--entrypoint', 'run.sh', '--timeout', saved['timeout_seconds'], '--name', 'beef-' + label)
    record = json.loads(path.read_text())
    record['response'] = response
    jobs = re.findall(r'submitted job (\d+)', response)
    if len(jobs) == 1:
        record['job_id'] = int(jobs[0])
    path.write_text(json.dumps(record, indent=2) + '\n')
    assert len(jobs) == 1, response
    print(response, end='')


def parse_diagnostic(content, stage, marker_map):
    expected = {f'{O} run{run}' for run in range(3)}
    found = re.findall(r'^\[([^]]+)\].*within tol=\s*([\d.]+)%\s*-> (PASS|FAIL)$', content, re.M)
    assert len(found) == 3 and {label.strip() for label, _, _ in found} == expected
    assert all(float(tol) == 100 and status == 'PASS' for _, tol, status in found)
    assert content.count('DIAG_CHECK_COMPLETE selected=1 runs=3 failures=0') == 1
    matches = re.findall(r'median cycles=(\d+) \(of 3 runs: (\[[^]]+\])\)', content)
    assert len(matches) == 1
    median, values = matches[0]
    cycles = json.loads(values)
    assert len(cycles) == 3 and min(cycles) > 0 and statistics.median(cycles) == int(median)
    spans = [json.loads(s) for s in re.findall(r'^DIAG_SPAN (.+)$', content, re.M)]
    assert spans and all(s['kernel'] == O for s in spans)
    assert {s['run'] for s in spans} == {0, 1, 2}
    labels = {s['label']: s for s in marker_map['spans']}
    operations = [row['marker']['label'] for row in marker_map['operations']]
    task_label = marker_map['task_label']
    task_cycles = []
    observations = []
    for run in range(3):
        cluster_cycles = []
        for cluster in range(2):
            samples = [s for s in spans if s['run'] == run and s['cluster'] == cluster]
            by_label = {s['name']: s for s in samples}
            assert len(by_label) == len(samples)
            assert set(by_label).issubset(labels)
            task = by_label[task_label]
            assert all(task['begin'] <= s['begin'] <= s['end'] <= task['end'] for s in samples)
            if stage['level'] == 'trace':
                assert set(by_label) & set(operations) == set(operations), (stage['name'], run, cluster)
                starts = [by_label[name]['begin'] for name in operations]
                assert starts == sorted(starts)
                observations.append({'run': run, 'cluster': cluster,
                    'task_cycles': task['end'] - task['begin'],
                    'operations': [{
                        'schedule_index': row['schedule']['index'],
                        'marker_label': row['marker']['label'],
                        'kind': row['marker']['name'],
                        'begin_from_task': by_label[row['marker']['label']]['begin'] - task['begin'],
                        'end_from_task': by_label[row['marker']['label']]['end'] - task['begin'],
                        'duration': by_label[row['marker']['label']]['end'] - by_label[row['marker']['label']]['begin']}
                        for row in marker_map['operations']]})
            cluster_cycles.append(task['end'] - task['begin'])
        task_cycles.append(max(cluster_cycles))
    assert task_cycles == cycles
    return {'valid': True, 'correctness_passes': 3, 'complete': True,
            'cycles': {'median': int(median), 'values': cycles}, 'spans': spans,
            'trace_observations': observations, 'task_marker_matches_reported_cycles': True}


def collect(label, job=None):
    saved = json.loads((HERE / f'{label}-build.json').read_text())
    original = json.loads((HERE / f'{label}-submission.json').read_text())['job_id']
    job = original if job is None else job
    result_path = HERE / f'{label}-job-{job}-results.json'
    assert not result_path.exists(), result_path
    status = json.loads(cli('status', job))
    assert status['status'] in {'SUCCEEDED', 'COMPLETED', 'FAILED', 'CANCELLED', 'CANCELED'}, status
    raw = cli('logs', job)
    evidence = HERE / 'evidence'
    evidence.mkdir(exist_ok=True)
    log_path = evidence / f'arena-{job}.log'
    with log_path.open('x') as stream:
        stream.write(raw)
    result = {'job': status, 'log': str(log_path.relative_to(ROOT)), 'log_digest': digest(raw.encode()),
              'build_record': file_record(HERE / f'{label}-build.json'), 'method': saved['method'],
              'stages': {}, 'actual_upload_matches': False, 'valid': False}
    try:
        remote = f'/tmp/moa-c09-screen/{label}/job-{job}.tar'
        cli('download', job, '--out', remote)
        archive_path = ARTIFACTS / f'job-{job}.tar'
        subprocess.run(['docker', 'cp', 'moa2026-dev:' + remote, str(archive_path)], check=True)
        with tarfile.open(archive_path) as archive:
            members = [m for m in archive if m.isfile()]
            assert len({m.name for m in members}) == len(members)
            files = {m.name: digest(archive.extractfile(m).read()) for m in members}
        assert {k: v for k, v in files.items() if k != 'job.yaml'} == saved['package_files']
        result.update(actual_upload_matches=True, package_files=files, archive=file_record(archive_path))
        sections = re.findall(r'^DIAG_STAGE_BEGIN (\w+)\n(.*?)^DIAG_STAGE_END \1$', raw, re.M | re.S)
        observed_names = [name for name, _ in sections]
        result['screen_complete'] = raw.count('DIAG_SCREEN_COMPLETE') == 1
        result['stage_sequence_matches'] = observed_names == [s['name'] for s in saved['stages']]
        result['completed_stage_names'] = observed_names
        diagnostic = ARTIFACTS / saved['build_label'] / 'diagnostic'
        package = json.loads((HERE / f'package-{saved["build_label"]}.json').read_text())
        assert file_record(HERE / f'package-{saved["build_label"]}.json') == saved['candidate_package']
        maps = {}
        for side in ('control', 'candidate'):
            path = diagnostic / f'{side}-marker-map.json'
            assert digest(path.read_bytes()) == package['diagnostic_files'][path.name]
            maps[side] = json.loads(path.read_text())
        for stage in saved['stages']:
            name = stage['name']
            matches = [content for observed, content in sections if observed == name]
            if len(matches) != 1:
                partial = re.search(r'^DIAG_STAGE_BEGIN ' + name + r'\n(.*?)(?=^DIAG_STAGE_BEGIN |\Z)', raw, re.M | re.S)
                partial_spans, malformed = [], 0
                for encoded in re.findall(r'^DIAG_SPAN (.+)$', partial[1] if partial else '', re.M):
                    try:
                        partial_spans.append(json.loads(encoded))
                    except json.JSONDecodeError:
                        malformed += 1
                result['stages'][name] = {'valid': False,
                    'error': 'Missing, incomplete or duplicated stage',
                    'stage_started': partial is not None, 'completed_sections': len(matches),
                    'spans': partial_spans, 'malformed_span_lines': malformed}
                continue
            content = matches[0]
            try:
                exits = re.findall(r'^DIAG_STAGE_EXIT ' + name + r' (\d+)$', content, re.M)
                assert exits == ['0'], exits
                if stage['kind'] == 'whole':
                    value = {'valid': True, 'kernels': parse_log(content), 'correctness_passes': 15}
                else:
                    value = parse_diagnostic(content, stage, maps[stage['side']])
                result['stages'][name] = {'exit_code': 0, **value}
            except (AssertionError, ValueError, KeyError) as error:
                # Keep all emitted spans on any attribution/correctness failure.
                result['stages'][name] = {'valid': False, 'error': repr(error),
                    'spans': [json.loads(s) for s in re.findall(r'^DIAG_SPAN (.+)$', content, re.M)]}
        result['valid'] = (status['status'] in {'SUCCEEDED', 'COMPLETED'} and status['exit_code'] == 0
                           and result['screen_complete'] and result['stage_sequence_matches']
                           and all(s['valid'] for s in result['stages'].values()))
        if result['valid']:
            result['correctness_passes'] = sum(s['correctness_passes'] for s in result['stages'].values())
            assert result['correctness_passes'] == 57
            before = result['stages']['control_before']['kernels'][O]['median_cycles']
            after = result['stages']['control_after']['kernels'][O]['median_cycles']
            candidate = result['stages']['candidate_whole']['kernels'][O]['median_cycles']
            result['whole_o_comparison'] = {'control_before': before, 'candidate': candidate,
                'control_after': after, 'bracketing_median': statistics.median([before, after]),
                'candidate_change_percent': 100 * (candidate / statistics.median([before, after]) - 1),
                'scope': 'One allocation; official full-three-kernel hosts only. Diagnostic timings are not promotion measurements.'}
    except Exception as error:
        result.update(valid=False, error=repr(error))
    write(result_path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {'stages', 'package_files'}}, indent=2))
    if not result['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'collect'])
    parser.add_argument('label')
    parser.add_argument('--build', default='c09-first-npu-v1')
    parser.add_argument('--job', type=int)
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]{1,70}', args.label)
    if args.action == 'prepare':
        assert args.job is None
        prepare(args.label, args.build)
    elif args.action == 'submit':
        assert args.job is None
        submit(args.label)
    else:
        collect(args.label, args.job)
