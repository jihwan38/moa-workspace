#!/usr/bin/env python3
"""Build the isolated control host. Never executes a model or NPU job."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import time

C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-control-full-o-original-e55'
SCRATCH = '/tmp/moa-model-control-full-o-original-e55'


def call(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    call(['python3', str(C / 'prepare_sources.py')])
    expected = json.loads((C / 'source-manifest.json').read_text())['files']
    call(['docker', 'exec', 'moa2026-dev', 'mkdir', '-p', SCRATCH])
    call(['docker', 'cp', str(A / 'crate') + '/.', 'moa2026-dev:' + SCRATCH + '/'])
    code = '''from pathlib import Path
import hashlib,json
p=Path(SCRATCH)
if not {f.name for f in p.iterdir()} <= {'src','Cargo.toml','Cargo.lock','rust-toolchain.toml','target'}:
    raise RuntimeError('unexpected_scratch_input')
files=sorted([*[f for f in p.joinpath('src').rglob('*') if f.is_file()],p/'Cargo.toml',p/'Cargo.lock',p/'rust-toolchain.toml'])
print(json.dumps({'files':{str(f.relative_to(p)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files},'target_was_empty':not any((p/'target').glob('*'))}))
'''.replace('SCRATCH', repr(SCRATCH))
    actual = json.loads(subprocess.check_output(['docker', 'exec', 'moa2026-dev', 'python3', '-c', code], text=True))
    if len(expected) != 47 or actual['files'] != expected:
        raise RuntimeError('actual_compiled_input_mismatch')
    (C / 'compiled-source-manifest.json').write_text(json.dumps({
        'scratch':SCRATCH, 'files':actual['files'], 'target_was_empty_before_build':actual['target_was_empty']
    }, indent=2) + '\n')
    command = ['docker','exec','-w',SCRATCH,'-e','CARGO_HOME=/home/moa/.cargo',
        '-e','CARGO_TARGET_DIR='+SCRATCH+'/target','-e','FURIOSA_OPT_OUT_DIR='+SCRATCH+'/target/furiosa-opt/kernel',
        'moa2026-dev','cargo','furiosa-opt','build','--release','--locked','--bin','probe_model_prefix_suite']
    started = time.monotonic()
    log = A / 'build.log'
    with log.open('w') as out:
        result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
    report = {'command':command,'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-started,
        'completed_at_utc':datetime.now(timezone.utc).isoformat(), 'model_or_npu_executed':False,
        'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(), 'target_was_empty_before_build':actual['target_was_empty']}
    (C / 'build-result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    result.check_returncode()
    call(['docker','cp','moa2026-dev:'+SCRATCH+'/target/release/probe_model_prefix_suite',str(A / 'probe_model_prefix_suite')])
    (A / 'kernels').mkdir(exist_ok=True)
    call(['docker','cp','moa2026-dev:'+SCRATCH+'/target/furiosa-opt/kernel/furiosa-opt-gemma4/.',str(A / 'kernels')+'/'])
    call(['python3', str(C / 'verify_registry.py')])


if __name__ == '__main__':
    main()
