#!/usr/bin/env python3
"""Reproducible isolated host build; source and ten-image gates are mandatory."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
if not __debug__:
    raise RuntimeError('Build integrity requires assertions; Python -O is forbidden.')
sys.dont_write_bytecode = True
from verify_sources import actual_container_inputs
C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-control-qkv-original-e55'
SCRATCH = '/tmp/moa-model-control-qkv-original-e55'

def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)

run(['python3', str(C / 'prepare_sources.py')])
run(['docker', 'exec', 'moa2026-dev', 'mkdir', '-p', SCRATCH])
run(['docker', 'cp', str(A / 'crate') + '/.', 'moa2026-dev:' + SCRATCH + '/'])
actual = actual_container_inputs()
expected = json.loads((C / 'source-manifest.json').read_text())['files']
assert actual == expected and len(actual) == 47
empty = subprocess.check_output(['docker', 'exec', 'moa2026-dev', 'python3', '-c',
        "from pathlib import Path; print(not any(Path('/tmp/moa-model-control-qkv-original-e55/target').glob('*')))"], text=True).strip() == 'True'
(C / 'compiled-source-manifest.json').write_text(json.dumps({
    'files': actual, 'scratch': SCRATCH, 'target_was_empty_before_build': empty,
}, indent=2) + '\n')
command = f'cd {SCRATCH} && CARGO_HOME=/home/moa/.cargo CARGO_NET_OFFLINE=true CARGO_TARGET_DIR={SCRATCH}/target FURIOSA_OPT_OUT_DIR={SCRATCH}/target/furiosa-opt/kernel cargo furiosa-opt build --release --locked --bin probe_model_prefix_suite'
started = time.monotonic()
with (A / 'build.log').open('w') as log:
    result = subprocess.run(['docker', 'exec', 'moa2026-dev', 'bash', '-lc', command], stdout=log, stderr=subprocess.STDOUT)
info = {'command': command, 'exit_code': result.returncode, 'elapsed_seconds': time.monotonic() - started,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'target_was_empty_before_build': empty,
        'build_log_sha256': hashlib.sha256((A / 'build.log').read_bytes()).hexdigest(),
        'npu_executed': False}
(A / 'build-result.json').write_text(json.dumps(info, indent=2) + '\n')
print(json.dumps(info), flush=True)
result.check_returncode()
run(['docker', 'cp', f'moa2026-dev:{SCRATCH}/target/release/probe_model_prefix_suite', str(A / 'probe_model_prefix_suite')])
(A / 'kernels').mkdir(exist_ok=True)
run(['docker', 'cp', f'moa2026-dev:{SCRATCH}/target/furiosa-opt/kernel/furiosa-opt-gemma4/.', str(A / 'kernels') + '/'])
for script, output in [('verify_sources.py', 'source-verification.json'), ('verify_registry.py', 'registry-diff.json')]:
    with (C / output).open('w') as out:
        run(['python3', str(C / script)], stdout=out)
print('SOURCE_AND_TEN_IMAGE_CONTROL_GATES_PASS', flush=True)
