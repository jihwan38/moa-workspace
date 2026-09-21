#!/usr/bin/env python3
"""Read-only verification of the frozen and actual-container 47 compiler inputs."""
from pathlib import Path
import hashlib
import json
import subprocess
if not __debug__:
    raise RuntimeError('Source verification requires assertions; Python -O is forbidden.')
C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-control-qkv-original-e55'
SCRATCH = '/tmp/moa-model-control-qkv-original-e55'

def actual_container_inputs():
    code = '''from pathlib import Path
import hashlib,json
p=Path(SCRATCH)
assert {f.name for f in p.iterdir()} <= {'src','Cargo.toml','Cargo.lock','rust-toolchain.toml','target'}
files=sorted([*[f for f in (p/'src').rglob('*') if f.is_file()],p/'Cargo.toml',p/'Cargo.lock',p/'rust-toolchain.toml'])
print(json.dumps({str(f.relative_to(p)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files}))
'''.replace('SCRATCH', repr(SCRATCH))
    return json.loads(subprocess.check_output(['docker', 'exec', 'moa2026-dev', 'python3', '-c', code], text=True))

def main():
    expected = json.loads((C / 'source-manifest.json').read_text())['files']
    compiled = json.loads((C / 'compiled-source-manifest.json').read_text())['files']
    assert len(expected) == 47 and expected == compiled
    crate = A / 'crate'
    assert {str(f.relative_to(crate)) for f in crate.rglob('*') if f.is_file()} == set(expected)
    for name, info in expected.items():
        f = crate / name
        assert {'bytes': f.stat().st_size, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} == info, name
    assert actual_container_inputs() == expected
    print(json.dumps({'status': 'FROZEN_AND_ACTUAL_47_INPUTS_PASS', 'inputs': 47,
                      'scratch': SCRATCH, 'npu_executed': False}, indent=2))

if __name__ == '__main__':
    main()
