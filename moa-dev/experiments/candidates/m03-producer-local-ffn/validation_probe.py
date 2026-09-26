#!/usr/bin/env python3
"""Build/execute an isolated M03 projection oracle; never a performance experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / 'moa-dev/experiments/artifacts/m03-producer-local-ffn'
TARGET = '/tmp/moa-m03-validation-probe/target'
sys.path.insert(0, str(ROOT / 'moa-dev'))
from measure import cli


def digest(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def record(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def run(command, log=None):
    if log:
        with log.open('x') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def build(label):
    source_label = 'm03-native-local-v1'
    source_record = json.loads((HERE / f'build-{source_label}-repro.json').read_text())
    source = ARTIFACTS / source_label / 'crate'
    original = (source / 'src/device/shared/mlp.rs').read_bytes()
    assert digest(original) == source_record['source_inputs']['src/device/shared/mlp.rs']
    assert original == (HERE / 'native-local-v1/mlp.rs').read_bytes()
    append = (HERE / 'validation/probe_module.rs').read_bytes()
    files = {
        'src/lib.rs': (HERE / 'validation/probe_lib.rs').read_bytes(),
        'src/stream_probe.rs': (HERE / 'validation/stream_probe.rs').read_bytes(),
        'src/device/mlp.rs': original + append,
        'src/bin/probe_stream.rs': (HERE / 'validation/probe_stream.rs').read_bytes(),
    }
    for name in ('src/axes.rs', 'src/device/layout.rs', 'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'):
        files[name] = (source / name).read_bytes()
        assert digest(files[name]) == source_record['source_inputs'][name]
    out = ARTIFACTS / label
    out.mkdir(parents=True, exist_ok=False)
    crate = out / 'crate'
    evidence = HERE / 'evidence' / label
    evidence.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        for base in (crate, evidence):
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    remote = '/tmp/moa-m03-validation-probe/' + label
    data = {
        'source_build': str((HERE / f'build-{source_label}-repro.json').relative_to(ROOT)),
        'source_base_commit': source_record['base_commit'],
        'production_mlp_prefix': digest(original), 'diagnostic_append': digest(append),
        'production_body_unchanged': True,
        'source_inputs': {k: digest(v) for k, v in files.items()},
        'compiler_target_directory': TARGET,
        'scope': 'Seven full-shape independent projection cases, including cross-cluster cancellation; separate diagnostic images, no performance claim.',
        'cases': list(range(7)), 'expected_output_elements': 241920,
    }
    try:
        run(['docker', 'exec', 'moa2026-dev', 'mkdir', '-p', remote])
        run(['docker', 'cp', str(crate) + '/.', 'moa2026-dev:' + remote])
        archive = subprocess.check_output(['docker', 'exec', '--workdir', remote, 'moa2026-dev', 'tar', '-cf', '-', *files])
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            assert {m.name: digest(tar.extractfile(m).read()) for m in tar if m.isfile()} == data['source_inputs']
        data['source_inputs_verified_in_container'] = True
        env = ['docker', 'exec', '--workdir', remote, '-e', 'CARGO_NET_OFFLINE=true', '-e', 'CARGO_TARGET_DIR=' + TARGET,
               '-e', 'FURIOSA_OPT_OUT_DIR=' + remote + '/kernels', 'moa2026-dev']
        run(env + ['cargo', 'furiosa-opt', 'build', '--release', '--locked', '--bin', 'probe_stream'], out / 'build.log')
        data['build_exit_code'] = 0
        package = out / 'package'
        package.mkdir()
        run(['docker', 'cp', 'moa2026-dev:' + TARGET + '/release/probe_stream', str(package / 'probe_stream')])
        (package / 'run.sh').write_text('#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nchmod +x probe_stream\nexec ./probe_stream\n')
        for path in package.iterdir():
            path.chmod(0o755)
        run(['docker', 'cp', 'moa2026-dev:' + remote + '/kernels', str(out / 'kernels')])
        host = (package / 'probe_stream').read_bytes()
        data['images'] = {}
        for path in (out / 'kernels').rglob('*.bin'):
            content = path.read_bytes()
            assert host.count(content) == 1
            data['images'][path.name] = dict(digest(content), host_occurrences=1)
        assert len(data['images']) == 2
        data['package_files'] = {p.name: digest(p.read_bytes()) for p in package.iterdir()}
        run(['docker', 'cp', str(package), 'moa2026-dev:' + remote + '/package'])
        run(['docker', 'exec', 'moa2026-dev', remote + '/package/probe_stream', '--oracle-only'], out / 'oracle.log')
        data['oracle_exit_code'] = 0
        print((out / 'oracle.log').read_text(), end='')
    except subprocess.CalledProcessError as error:
        data.update(failed_exit_code=error.returncode, failed_command=error.cmd)
        raise
    finally:
        data['logs'] = {}
        for name in ('build.log', 'oracle.log'):
            path = out / name
            if path.exists():
                shutil.copyfile(path, evidence / name)
                data['logs'][name] = digest(path.read_bytes())
        record(HERE / f'build-{label}.json', data)


def submit(label):
    data = json.loads((HERE / f'build-{label}.json').read_text())
    assert data['build_exit_code'] == data['oracle_exit_code'] == 0
    out = ARTIFACTS / label / 'package'
    assert {p.name: digest(p.read_bytes()) for p in out.iterdir()} == data['package_files']
    remote = '/tmp/moa-m03-validation-probe/' + label + '/package'
    for name, value in data['package_files'].items():
        actual = subprocess.check_output(['docker', 'exec', 'moa2026-dev', 'sha256sum', remote + '/' + name], text=True).split()[0]
        assert actual == value['sha256']
    path = HERE / f'{label}-submission.json'
    record(path, {'attempted_at_utc': datetime.now(timezone.utc).isoformat()})
    response = cli('submit', *(remote + '/' + n for n in data['package_files']), '--entrypoint', 'run.sh', '--timeout', 70, '--name', 'beef-' + label)
    matches = re.findall(r'submitted job (\d+)', response)
    saved = json.loads(path.read_text())
    saved['response'] = response
    if len(matches) == 1:
        saved['job_id'] = int(matches[0])
    path.write_text(json.dumps(saved, indent=2) + '\n')
    assert len(matches) == 1, response
    print(response, end='')


def collect(label):
    job = json.loads((HERE / f'{label}-submission.json').read_text())['job_id']
    status = json.loads(cli('status', job))
    assert status['status'] in {'SUCCEEDED', 'COMPLETED', 'FAILED', 'CANCELLED', 'CANCELED'}
    log = cli('logs', job)
    path = HERE / 'evidence' / label / f'arena-{job}.log'
    path.write_text(log)
    remote = f'/tmp/moa-m03-validation-probe/job-{job}.tar'
    cli('download', job, '--out', remote)
    local = ARTIFACTS / f'validation-job-{job}.tar'
    run(['docker', 'cp', 'moa2026-dev:' + remote, str(local)])
    with tarfile.open(local) as archive:
        members = [m for m in archive if m.isfile()]
        assert len({m.name for m in members}) == len(members)
        files = {m.name: digest(archive.extractfile(m).read()) for m in members}
    built = json.loads((HERE / f'build-{label}.json').read_text())
    assert {k: v for k, v in files.items() if k != 'job.yaml'} == built['package_files']
    details = re.findall(r'^M03_PROBE case=(\d+) projection=(\w+) checked=(\d+) mismatches=(\d+)$', log, re.M)
    expected = {(str(c), name, str(count)) for c in range(7) for name, count in [('up', 15360), ('gate', 15360), ('down', 3840)]}
    complete = len(details) == 21 and {tuple(d[:3]) for d in details} == expected
    sentinel = re.findall(r'^M03_PROBE_COMPLETE oracle_only=false cases=7 checked=241920 mismatches=(\d+)$', log, re.M)
    valid = complete and sentinel == ['0'] and all(d[3] == '0' for d in details) and status['status'] in {'SUCCEEDED', 'COMPLETED'} and status['exit_code'] == 0
    result = {'job': status, 'build': str((HERE / f'build-{label}.json').relative_to(ROOT)),
              'log': str(path.relative_to(ROOT)), 'log_sha256': digest(log.encode())['sha256'],
              'uploaded_package_verified': True, 'package_files': files, 'details': details,
              'complete': complete, 'valid': valid, 'scope': built['scope']}
    record(HERE / f'{label}-results.json', result)
    print(json.dumps(result, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'submit', 'collect'])
    parser.add_argument('label')
    args = parser.parse_args()
    assert re.fullmatch(r'm03-adoption-probe-[a-z0-9-]+', args.label)
    {'build': build, 'submit': submit, 'collect': collect}[args.action](args.label)
