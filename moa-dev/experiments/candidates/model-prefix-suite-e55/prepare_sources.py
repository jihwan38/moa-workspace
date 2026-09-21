#!/usr/bin/env python3
"""Copy only audited original/E55 inputs, then add the identical host-driven NPU diagnostic."""
from pathlib import Path
import hashlib,json,shutil
if not __debug__:
    raise RuntimeError("Source preparation requires assertions; Python optimization mode is forbidden.")
C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55'
original = R / 'moa-dev/experiments/artifacts/original-output-boundary-probe-control/crate'
e55 = R / 'moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/crate'
audit = json.loads((R/'moa-dev/experiments/candidates/qkv-boundary-probe-e53/original-source-audit.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert len(audit['source_sha256']) == 45
lock_hash = 'aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09'
assert sha(original/'Cargo.lock') == sha(e55/'Cargo.lock') == lock_hash
assert sha(e55/'src/device/sliding/projection.rs') == '380e57d6502e26086bd6d933ce762f4ad5de18587d52a8ef4aac7fa87e3b9c9f'
for name, expected in audit['source_sha256'].items(): assert sha(original/name) == expected, name
e55_audit=json.loads((R/'moa-dev/experiments/candidates/qkv-bf16-fused-channel-scale-e54/compiled-source-manifest.json').read_text())
assert len(e55_audit)==46
for name, expected in e55_audit.items():
    assert sha(e55/name)==expected['sha256'] and (e55/name).stat().st_size==expected['bytes'], name
for variant, base in [('original',original),('e55',e55)]:
    crate = A / variant / 'crate'
    names = sorted([*audit['source_sha256'], 'Cargo.lock'])
    manifest = {}
    for name in names:
        dest = crate / name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(base/name,dest)
        manifest[name] = {'sha256':sha(dest),'bytes':dest.stat().st_size}
    probe = crate/'src/bin/probe_model_prefix_suite.rs'
    shutil.copyfile(C/'probe_model_prefix_suite.rs',probe)
    manifest['src/bin/probe_model_prefix_suite.rs']={'sha256':sha(probe),'bytes':probe.stat().st_size}
    assert len(list((crate/'src').rglob('*.rs'))) == 44
    actual_files = {str(p.relative_to(crate)) for p in crate.rglob('*') if p.is_file()}
    assert actual_files == set(manifest), ('unexpected_or_missing_crate_files', sorted(actual_files ^ set(manifest)))
    assert not (crate/'src/bin/probe_output.rs').exists()
    (C/f'{variant}-source-manifest.json').write_text(json.dumps({'variant':variant,'original_input_count':46,'added_probe_count':1,'files':manifest},indent=2)+'\n')
    print(variant, crate, sha(probe))
