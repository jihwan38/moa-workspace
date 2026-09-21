#!/usr/bin/env python3
"""Build original/E55 continuation hosts in isolated targets; no runtime/NPU calls."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time

C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-continuation-e55'


def call(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('original','e55','all'), default='all')
    args = parser.parse_args()
    call(['python3', str(C / 'prepare_sources.py')])
    variants = ('original','e55') if args.variant == 'all' else (args.variant,)
    for variant in variants:
        scratch = '/tmp/moa-model-continuation-e55/' + variant
        expected = json.loads((C / f'{variant}-source-manifest.json').read_text())['files']
        call(['docker','exec','moa2026-dev','mkdir','-p',scratch])
        call(['docker','cp',str(A / variant / 'crate')+'/.','moa2026-dev:'+scratch+'/'])
        code = '''from pathlib import Path
import hashlib,json
p=Path(SCRATCH)
if not {f.name for f in p.iterdir()} <= {'src','Cargo.toml','Cargo.lock','rust-toolchain.toml','target'}:
    raise RuntimeError('unexpected_top_level_compiler_input')
files=sorted([*[f for f in (p/'src').rglob('*') if f.is_file()],p/'Cargo.toml',p/'Cargo.lock',p/'rust-toolchain.toml'])
print(json.dumps({'files':{str(f.relative_to(p)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files},'target_was_empty':not any((p/'target').glob('*'))}))
'''.replace('SCRATCH', repr(scratch))
        actual = json.loads(subprocess.check_output(['docker','exec','moa2026-dev','python3','-c',code],text=True))
        if len(expected) != 47 or expected != actual['files']:
            raise RuntimeError(variant + '_actual_compiler_input_mismatch')
        (C / f'{variant}-compiled-source-manifest.json').write_text(json.dumps({
            'variant':variant, 'scratch':scratch, 'files':actual['files'],
            'target_was_empty_before_build':actual['target_was_empty'],
        },indent=2)+'\n')
        command = ['docker','exec','-w',scratch,'-e','CARGO_HOME=/home/moa/.cargo',
            '-e','CARGO_TARGET_DIR='+scratch+'/target','-e','FURIOSA_OPT_OUT_DIR='+scratch+'/target/furiosa-opt/kernel',
            'moa2026-dev','cargo','furiosa-opt','build','--release','--locked','--bin','probe_model_continuation']
        log = A / variant / 'build.log'
        started = time.monotonic()
        with log.open('w') as out:
            result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
        report = {'variant':variant,'command':command,'exit_code':result.returncode,
            'elapsed_seconds':time.monotonic()-started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'target_was_empty_before_build':actual['target_was_empty'],
            'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'model_or_npu_executed':False}
        (C / f'{variant}-build-result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        result.check_returncode()
        call(['docker','cp','moa2026-dev:'+scratch+'/target/release/probe_model_continuation',str(A/variant/'probe_model_continuation')])
        (A / variant / 'kernels').mkdir(exist_ok=True)
        call(['docker','cp','moa2026-dev:'+scratch+'/target/furiosa-opt/kernel/furiosa-opt-gemma4/.',str(A/variant/'kernels')+'/'])
    call(['python3',str(C/'verify_source_inputs.py')])
    if all((A/v/'probe_model_continuation').is_file() for v in ('original','e55')):
        call(['python3',str(C/'verify_registry.py')])


if __name__ == '__main__':
    main()
