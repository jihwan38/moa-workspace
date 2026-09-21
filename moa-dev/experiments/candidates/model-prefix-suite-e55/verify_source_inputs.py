#!/usr/bin/env python3
"""Read-only verification of all frozen and actual-container compiler inputs."""
from pathlib import Path
import hashlib,json,subprocess
if not __debug__:
    raise RuntimeError('Source verification requires assertions; optimized Python is forbidden.')
C=Path(__file__).resolve().parent
R=C.parents[3]
A=R/'moa-dev/experiments/artifacts/model-prefix-suite-e55'
probe_sha=hashlib.sha256((C/'probe_model_prefix_suite.rs').read_bytes()).hexdigest()
for variant in ['original','e55']:
    expected=json.loads((C/f'{variant}-source-manifest.json').read_text())['files']
    compiled=json.loads((C/f'{variant}-compiled-source-manifest.json').read_text())['files']
    assert len(expected)==47 and expected==compiled
    crate=A/variant/'crate'
    actual_files={str(p.relative_to(crate)) for p in crate.rglob('*') if p.is_file()}
    assert actual_files==set(expected), (variant,'frozen extra or missing files')
    for name,info in expected.items():
        p=crate/name
        assert p.stat().st_size==info['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256'], (variant,name)
    assert expected['src/bin/probe_model_prefix_suite.rs']['sha256']==probe_sha
    code='''from pathlib import Path
import hashlib,json
p=Path('/tmp/moa-model-prefix-suite-e55/VARIANT')
files=sorted([*[p for p in p.joinpath('src').rglob('*') if p.is_file()],p/'Cargo.toml',p/'Cargo.lock',p/'rust-toolchain.toml'])
print(json.dumps({str(f.relative_to(p)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files}))
'''.replace('VARIANT',variant)
    actual=json.loads(subprocess.check_output(['docker','exec','moa2026-dev','python3','-c',code],text=True))
    assert actual==expected, (variant,'actual Docker input mismatch')
print(json.dumps({'status':'SOURCE_INPUT_IDENTITY_PASS','variants':['original','e55'],'inputs_each':47,'identical_probe_sha256':probe_sha,'model_or_npu_executed':False},indent=2))
