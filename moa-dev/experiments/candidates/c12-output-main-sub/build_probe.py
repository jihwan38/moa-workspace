#!/usr/bin/env python3
"""Compile a frozen standalone C12 API probe using selected base's Cargo/toolchain inputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = json.loads((HERE / 'candidate.json').read_text())['base_commit']


def digest(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def build(variant, label, entry):
    assert re.fullmatch(r'[a-z0-9-]+', label)
    record_path = HERE / f'build-{label}.json'
    assert not record_path.exists()
    out = ROOT / 'moa-dev/experiments/artifacts/c12-output-main-sub' / label
    out.mkdir(parents=True, exist_ok=False)
    crate = out / 'crate'
    (crate / 'src').mkdir(parents=True)
    for name in ('Cargo.toml', 'rust-toolchain.toml'):
        (crate / name).write_bytes(subprocess.check_output(['git', 'show', BASE + ':furiosa-opt-gemma4-12B/' + name], cwd=ROOT))
    (crate / 'Cargo.lock').write_bytes((ROOT / 'moa-dev/locks/gemma4-Cargo.lock').read_bytes())
    for name in ('lib.rs', 'sub_probe.rs'):
        (crate / 'src' / name).write_bytes((HERE / variant / name).read_bytes())
    inputs = {str(p.relative_to(crate)): digest(p.read_bytes()) for p in sorted(crate.rglob('*')) if p.is_file()}
    remote = '/tmp/moa-c12/' + label
    subprocess.run(['docker', 'exec', 'moa2026-dev', 'mkdir', '-p', remote], check=True)
    subprocess.run(['docker', 'cp', str(crate) + '/.', 'moa2026-dev:' + remote], check=True)
    raw = subprocess.check_output(['docker', 'exec', '--workdir', remote, 'moa2026-dev', 'tar', '-cf', '-', *inputs])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        assert {m.name: digest(archive.extractfile(m).read()) for m in archive if m.isfile()} == inputs
    env = ['docker', 'exec', '--workdir', remote, '-e', 'CARGO_HOME=/home/moa/.cargo',
           '-e', 'CARGO_NET_OFFLINE=true', '-e', 'CARGO_TARGET_DIR=/tmp/moa-c01-probe/target',
           '-e', 'FURIOSA_OPT_OUT_DIR=' + remote + '/kernels', 'moa2026-dev']
    record = {'label': label, 'variant': variant, 'source_base_commit': BASE, 'source_inputs': inputs,
              'source_inputs_verified_in_container': True, 'adopted': False,
              'scope': 'Standalone source-pattern support diagnostic only; no whole O accuracy or hardware performance claim.',
              'builder': digest(Path(__file__).read_bytes()), 'design': digest((HERE / variant / 'intent.json').read_bytes()), 'entry': entry}
    try:
        with (out / 'compile.log').open('w') as log:
            p = subprocess.run(env + ['cargo', 'furiosa-opt', 'compile', 'sub_probe::' + entry, '--exact', '--dump-schedule', 'schedule.json'], stdout=log, stderr=subprocess.STDOUT)
        record['compile_exit_code'] = p.returncode
        if p.returncode == 0:
            subprocess.run(['docker', 'cp', 'moa2026-dev:' + remote + '/schedule.json', str(out / 'schedule.json')], check=True)
            image = remote + '/kernels/furiosa-opt-gemma4/furiosa_opt_gemma4::sub_probe::' + entry + '.bin'
            subprocess.run(['docker', 'cp', 'moa2026-dev:' + image, str(out / 'probe.bin')], check=True)
            record['image'] = digest((out / 'probe.bin').read_bytes())
            record['schedule'] = digest((out / 'schedule.json').read_bytes())
            record['static_cycles'] = max(i['lifetime']['end'] for i in json.loads((out / 'schedule.json').read_text())['instructions'])
    finally:
        evidence = HERE / 'evidence' / label
        evidence.mkdir(parents=True, exist_ok=False)
        record['frozen_files'] = {}
        for rel in inputs:
            dst = evidence / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((crate / rel).read_bytes())
            record['frozen_files'][rel] = str(dst.relative_to(ROOT))
        for name in ('compile.log', 'schedule.json'):
            if (out / name).exists():
                (evidence / name).write_bytes((out / name).read_bytes())
                record[name] = digest((out / name).read_bytes())
                record['frozen_files'][name] = str((evidence / name).relative_to(ROOT))
        record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k in ('label', 'compile_exit_code', 'static_cycles')}, indent=2))
    raise SystemExit(record['compile_exit_code'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant')
    p.add_argument('--label', required=True)
    p.add_argument('--entry', required=True)
    args = p.parse_args()
    build(args.variant, args.label, args.entry)
