#!/usr/bin/env python3
"""Prepare isolated sources, capture actual inputs, and build hosts without NPU execution."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import time

if not __debug__:
    raise RuntimeError('Build provenance checks require assertions; Python -O is forbidden.')
C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55'

def call(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)

call(['python3', str(C / 'prepare_sources.py')])
for variant in ('original', 'e55'):
    scratch = '/tmp/moa-model-prefix-suite-e55/' + variant
    expected = json.loads((C / f'{variant}-source-manifest.json').read_text())['files']
    call(['docker', 'exec', 'moa2026-dev', 'mkdir', '-p', scratch])
    call(['docker', 'cp', str(A / variant / 'crate') + '/.', 'moa2026-dev:' + scratch + '/'])
    code = '''from pathlib import Path
import hashlib,json
p=Path(SCRATCH)
assert {f.name for f in p.iterdir()} <= {'src','Cargo.toml','Cargo.lock','rust-toolchain.toml','target'}
files=sorted([*[f for f in p.joinpath('src').rglob('*') if f.is_file()],p/'Cargo.toml',p/'Cargo.lock',p/'rust-toolchain.toml'])
print(json.dumps({'files':{str(f.relative_to(p)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files},'target_was_empty':not any((p/'target').glob('*'))}))
'''.replace('SCRATCH', repr(scratch))
    actual = json.loads(subprocess.check_output(['docker', 'exec', 'moa2026-dev', 'python3', '-c', code], text=True))
    assert len(expected) == 47 and actual['files'] == expected
    (C / f'{variant}-compiled-source-manifest.json').write_text(json.dumps({
        'variant': variant, 'scratch': scratch, 'files': actual['files'],
        'target_was_empty_before_build': actual['target_was_empty'],
    }, indent=2) + '\n')
    command = f'cd {scratch} && CARGO_HOME=/home/moa/.cargo CARGO_TARGET_DIR={scratch}/target FURIOSA_OPT_OUT_DIR={scratch}/target/furiosa-opt/kernel cargo furiosa-opt build --release --locked --bin probe_model_prefix_suite'
    started = time.monotonic()
    log = A / variant / 'build.log'
    with log.open('w') as out:
        result = subprocess.run(['docker', 'exec', 'moa2026-dev', 'bash', '-lc', command], stdout=out, stderr=subprocess.STDOUT)
    seconds = time.monotonic() - started
    info = {'variant': variant, 'command': command, 'exit_code': result.returncode,
            'elapsed_seconds': seconds, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
            'target_was_empty_before_build': actual['target_was_empty'], 'model_or_npu_executed': False}
    (A / variant / 'build-result.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info), flush=True)
    result.check_returncode()
    call(['docker', 'cp', f'moa2026-dev:{scratch}/target/release/probe_model_prefix_suite', str(A / variant / 'probe_model_prefix_suite')])
    (A / variant / 'kernels').mkdir(exist_ok=True)
    call(['docker', 'cp', f'moa2026-dev:{scratch}/target/furiosa-opt/kernel/furiosa-opt-gemma4/.', str(A / variant / 'kernels') + '/'])
