#!/usr/bin/env python3
"""Build the single-kernel control in its own source/target; never run NPU."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import time

if not __debug__:
    raise RuntimeError('Build provenance assertions require normal Python.')
C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-control-ffn-original-e55'
SCRATCH = '/tmp/moa-model-control-ffn-original-e55'


def inputs():
    code = '''from pathlib import Path
import hashlib,json
p=Path(SCRATCH)
assert {f.name for f in p.iterdir()} <= {'src','Cargo.toml','Cargo.lock','rust-toolchain.toml','target'}
files=sorted([*[f for f in p.joinpath('src').rglob('*') if f.is_file()],p/'Cargo.toml',p/'Cargo.lock',p/'rust-toolchain.toml'])
print(json.dumps({'files':{str(f.relative_to(p)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files},'target_empty':not any((p/'target').glob('*'))}))
'''.replace('SCRATCH', repr(SCRATCH))
    return json.loads(subprocess.check_output(['docker', 'exec', 'moa2026-dev', 'python3', '-c', code], text=True))


def main():
    subprocess.run(['python3', '-B', str(C / 'prepare_sources.py')], check=True)
    subprocess.run(['docker', 'exec', 'moa2026-dev', 'mkdir', '-p', SCRATCH], check=True)
    subprocess.run(['docker', 'cp', str(A / 'crate') + '/.', 'moa2026-dev:' + SCRATCH + '/'], check=True)
    expected = json.loads((C / 'source-manifest.json').read_text())['files']
    before = inputs()
    assert before['files'] == expected and len(expected) == 47
    (C / 'compiled-source-manifest.json').write_text(json.dumps({
        'scratch': SCRATCH, 'files': before['files'], 'target_was_empty_before_build': before['target_empty'],
    }, indent=2) + '\n')
    command = f'cd {SCRATCH} && CARGO_HOME=/home/moa/.cargo CARGO_TARGET_DIR={SCRATCH}/target FURIOSA_OPT_OUT_DIR={SCRATCH}/target/furiosa-opt/kernel cargo furiosa-opt build --release --locked --bin probe_model_prefix_suite'
    started = time.monotonic()
    with (A / 'build.log').open('w') as out:
        result = subprocess.run(['docker', 'exec', 'moa2026-dev', 'bash', '-lc', command], stdout=out, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    after = inputs()
    assert after['files'] == expected, 'compiler inputs changed during build'
    report = {'command': command, 'exit_code': result.returncode, 'elapsed_seconds': elapsed,
              'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'target_was_empty_before_build': before['target_empty'],
              'all_47_inputs_identical_before_and_after': True,
              'log_sha256': hashlib.sha256((A / 'build.log').read_bytes()).hexdigest(),
              'model_or_npu_executed': False}
    (A / 'build-result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    result.check_returncode()
    subprocess.run(['docker', 'cp', 'moa2026-dev:' + SCRATCH + '/target/release/probe_model_prefix_suite', str(A / 'probe_model_prefix_suite')], check=True)
    (A / 'kernels').mkdir(exist_ok=True)
    subprocess.run(['docker', 'cp', 'moa2026-dev:' + SCRATCH + '/target/furiosa-opt/kernel/furiosa-opt-gemma4/.', str(A / 'kernels') + '/'], check=True)
    verified = subprocess.run(['python3', '-B', str(C / 'verify_registry.py')], text=True, capture_output=True)
    (C / 'registry-summary.json').write_text(verified.stdout)
    print(verified.stdout, end='')
    if verified.stderr:
        print(verified.stderr, end='')
    verified.check_returncode()


if __name__ == '__main__':
    main()
