#!/usr/bin/env python3
"""Build a source-derived, one-kernel sliding-O control; never launch an NPU."""
from pathlib import Path
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
import subprocess
import time

if not __debug__:
    raise RuntimeError('Build checks require assertions; Python -O is forbidden.')
C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts' / C.name
S = R / 'moa-dev/experiments/candidates/model-prefix-suite-e55'
B = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55'
SCRATCH = '/tmp/moa-' + C.name

def fp(data):
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def section(text, start, end):
    assert text.count(start) == text.count(end) == 1
    return text[text.index(start):text.index(end)]

def main():
    inputs = {}
    for variant in ('original', 'e55'):
        manifest = json.loads((S / f'{variant}-source-manifest.json').read_text())['files']
        assert len(manifest) == 47
        inputs[variant] = {}
        for name, expected in manifest.items():
            data = (B / variant / 'crate' / name).read_bytes()
            assert fp(data) == expected, (variant, name)
            inputs[variant][name] = data
    files = inputs['e55'].copy()
    projection = 'src/device/sliding/projection.rs'
    old = inputs['original'][projection].decode()
    new = files[projection].decode()
    marker = 'pub(crate) fn project_output('
    assert old.count(marker) == new.count(marker) == 1
    new = new[:new.index(marker)] + old[old.index(marker):]
    new = new.replace('layout::{Cluster, Replicated}', 'layout::{Cluster, Replicated, Slice}')
    files[projection] = new.encode()
    residual = 'src/device/shared/residual.rs'
    old = inputs['original'][residual].decode()
    original_add = section(old, 'pub(crate) fn add(', 'pub(crate) fn scale_by_layer_gate(')
    files[residual] += ('\n' + original_add.replace('pub(crate) fn add(', 'pub(crate) fn add_original_sliding_control(')).encode()
    ops = 'src/ops.rs'
    old = inputs['original'][ops].decode()
    new = files[ops].decode()
    start = '#[device(chip = 1)]\npub fn sliding_attention_output('
    end = '#[device(chip = 1)]\npub fn full_attention_first_page('
    old_body = section(old, start, end).replace('shared::residual::add(', 'shared::residual::add_original_sliding_control(')
    new = new.replace(section(new, start, end), old_body)
    files[ops] = new.encode()
    changed = sorted(name for name in files if files[name] != inputs['e55'][name])
    assert changed == sorted([projection, residual, ops])
    assert files['src/bin/probe_model_prefix_suite.rs'] == inputs['original']['src/bin/probe_model_prefix_suite.rs']
    patch = ''.join(''.join(difflib.unified_diff(inputs['e55'][name].decode().splitlines(True), files[name].decode().splitlines(True), fromfile='a/'+name, tofile='b/'+name)) for name in changed)
    (C / 'source.patch').write_text(patch)
    crate = A / 'crate'
    for name, data in files.items():
        dest = crate / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    assert {str(p.relative_to(crate)) for p in crate.rglob('*') if p.is_file()} == set(files)
    expected = {name: fp(data) for name, data in sorted(files.items())}
    save(C / 'source-manifest.json', {'files': expected, 'changed_from_e55': changed, 'host_probe_unchanged': True})
    subprocess.run(['docker', 'exec', 'moa2026-dev', 'mkdir', '-p', SCRATCH], check=True)
    subprocess.run(['docker', 'cp', str(crate)+'/.', 'moa2026-dev:'+SCRATCH+'/'], check=True)
    code = '''from pathlib import Path
import hashlib,json
p=Path(SCRATCH)
assert {x.name for x in p.iterdir()} <= {'src','Cargo.toml','Cargo.lock','rust-toolchain.toml','target'}
files=sorted([*[f for f in (p/'src').rglob('*') if f.is_file()],p/'Cargo.toml',p/'Cargo.lock',p/'rust-toolchain.toml'])
print(json.dumps({'files':{str(f.relative_to(p)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files},'target_was_empty':not any((p/'target').glob('*'))}))
'''.replace('SCRATCH', repr(SCRATCH))
    actual = json.loads(subprocess.check_output(['docker', 'exec', 'moa2026-dev', 'python3', '-c', code], text=True))
    assert actual['files'] == expected
    save(C / 'compiled-source-manifest.json', actual)
    command = f'cd {SCRATCH} && CARGO_HOME=/home/moa/.cargo CARGO_TARGET_DIR={SCRATCH}/target FURIOSA_OPT_OUT_DIR={SCRATCH}/target/furiosa-opt/kernel cargo furiosa-opt build --release --locked --bin probe_model_prefix_suite'
    start_time = time.monotonic()
    with (A/'build.log').open('w') as out:
        result = subprocess.run(['docker','exec','moa2026-dev','bash','-lc',command], stdout=out, stderr=subprocess.STDOUT)
    save(C/'build-result.json', {'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start_time,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'command':command,'log':fp((A/'build.log').read_bytes()),'npu_run':False})
    result.check_returncode()
    subprocess.run(['docker','cp',f'moa2026-dev:{SCRATCH}/target/release/probe_model_prefix_suite',str(A/'probe_model_prefix_suite')],check=True)
    (A/'kernels').mkdir(exist_ok=True)
    subprocess.run(['docker','cp',f'moa2026-dev:{SCRATCH}/target/furiosa-opt/kernel/furiosa-opt-gemma4/.',str(A/'kernels')+'/'],check=True)
    spec = importlib.util.spec_from_file_location('registry_verifier', S/'verify_registry.py')
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    control, info = registry.registry(A/'probe_model_prefix_suite')
    baseline, _ = registry.registry(B/'e55/probe_model_prefix_suite')
    original, _ = registry.registry(B/'original/probe_model_prefix_suite')
    target = ('furiosa_opt_gemma4::ops::sliding_attention_output', ())
    assert set(control) == set(baseline) == set(original) and len(control) == 10
    report = {'host':info,'target':target[0],'entries':{name:{**fp(data),'equals_original':data==original[name,key],'equals_e55':data==baseline[name,key]} for (name,key),data in control.items()}, 'npu_run':False}
    report['single_kernel_image_identity_pass'] = control[target] == original[target] and all(data == baseline[name,key] for (name,key),data in control.items() if (name,key) != target)
    for (name,key),data in control.items():
        assert data == (A/'kernels'/(name+'.bin')).read_bytes()
    save(C/'registry-summary.json', report)
    assert report['single_kernel_image_identity_pass'], report
    print(json.dumps({'status':'SOURCE_DERIVED_SINGLE_KERNEL_IMAGE_IDENTITY_PASS','host':info['binary'],'npu_run':False}))

if __name__ == '__main__':
    main()
