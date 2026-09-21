#!/usr/bin/env python3
"""Read all prepared and actual Docker inputs; write only the verification record."""
from pathlib import Path
import hashlib
import json
import subprocess

C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-continuation-e55'


def main():
    host = C / 'probe_model_continuation.rs'
    host_fp = {'sha256':hashlib.sha256(host.read_bytes()).hexdigest(),'bytes':host.stat().st_size}
    variants = []
    for variant in ('original','e55'):
        prepared_path = C / f'{variant}-source-manifest.json'
        compiled_path = C / f'{variant}-compiled-source-manifest.json'
        if not compiled_path.exists():
            continue
        prepared = json.loads(prepared_path.read_text())['files']
        compiled = json.loads(compiled_path.read_text())['files']
        crate = A / variant / 'crate'
        actual = {str(p.relative_to(crate)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
            for p in crate.rglob('*') if p.is_file()}
        if len(actual) != 47 or actual != prepared or prepared != compiled:
            raise RuntimeError(variant + '_prepared_compiled_input_disagreement')
        if actual['src/bin/probe_model_continuation.rs'] != host_fp or 'src/bin/probe_model_prefix_suite.rs' in actual:
            raise RuntimeError(variant + '_host_probe_mismatch')
        scratch = '/tmp/moa-model-continuation-e55/' + variant
        code = '''from pathlib import Path
import hashlib,json
p=Path(SCRATCH)
files=sorted([*[f for f in (p/'src').rglob('*') if f.is_file()],p/'Cargo.toml',p/'Cargo.lock',p/'rust-toolchain.toml'])
print(json.dumps({str(f.relative_to(p)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files}))
'''.replace('SCRATCH',repr(scratch))
        current = json.loads(subprocess.check_output(['docker','exec','moa2026-dev','python3','-c',code],text=True))
        if current != actual:
            raise RuntimeError(variant + '_actual_docker_inputs_changed')
        variants.append(variant)
    if not variants:
        raise RuntimeError('no_compiled_source_manifests')
    report = {'status':'SOURCE_INPUT_IDENTITY_PASS','variants':variants,'inputs_each':47,
        'identical_probe':host_fp,'old_probe_excluded':True,'model_or_npu_executed':False}
    (C/'source-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
