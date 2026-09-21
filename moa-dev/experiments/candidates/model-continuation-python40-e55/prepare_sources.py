#!/usr/bin/env python3
"""Stage frozen original/E55 base46 + the frozen continuation host = exactly47."""
from pathlib import Path
import hashlib
import json
import shutil

C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-continuation-python40-e55'
S = R / 'moa-dev/experiments/candidates/model-prefix-suite-e55'
B = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55'
OLD_PROBE = 'src/bin/probe_model_prefix_suite.rs'
NEW_PROBE = 'src/bin/probe_model_continuation.rs'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fingerprint(path):
    return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes':path.stat().st_size}


def main():
    # A collaborator freezes the actual host before this descriptor is created.
    # Refuse absent/unready input; never compile a placeholder.
    frozen_host = json.loads((C / 'host-source.json').read_text())
    host = C / 'probe_model_continuation.rs'
    require(frozen_host['implementation_ready'] is True and fingerprint(host) == frozen_host['file'], 'host_not_frozen_or_changed')
    require('#[tokio::main]' in host.read_text(), 'missing_real_host_entrypoint')
    for variant in ('original', 'e55'):
        reference = json.loads((S / f'{variant}-source-manifest.json').read_text())['files']
        compiled = json.loads((S / f'{variant}-compiled-source-manifest.json').read_text())['files']
        require(len(reference) == 47 and reference == compiled, 'frozen_base_manifest_disagreement')
        base = B / variant / 'crate'
        actual = {str(p.relative_to(base)):fingerprint(p) for p in base.rglob('*') if p.is_file()}
        require(actual == reference, variant + '_frozen_base_files_changed')
        require(reference[OLD_PROBE]['sha256'] == '07cfaf4646da18a78e03bb386632ac455ccf545d98b858385bded281c3104da4', 'old_suite_host_changed')
        names = sorted(set(reference) - {OLD_PROBE})
        require(len(names) == 46 and NEW_PROBE not in names, 'unexpected_base_file_count')
        crate = A / variant / 'crate'
        for name in names:
            dest = crate / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / name, dest)
        shutil.copyfile(host, crate / NEW_PROBE)
        expected = {name:reference[name] for name in names}
        expected[NEW_PROBE] = fingerprint(host)
        actual = {str(p.relative_to(crate)):fingerprint(p) for p in crate.rglob('*') if p.is_file()}
        require(len(actual) == 47 and actual == expected, variant + '_staged_input_mismatch_or_extra_file')
        require(not (crate / OLD_PROBE).exists(), 'old_probe_must_not_be_staged')
        (C / f'{variant}-source-manifest.json').write_text(json.dumps({
            'variant':variant, 'unchanged_base_input_count':46, 'new_host_count':1, 'input_count':47,
            'excluded_old_probe':OLD_PROBE, 'host_source_sha256':frozen_host['file']['sha256'],
            'base_source_manifest':str((S / f'{variant}-source-manifest.json').relative_to(R)),
            'base_source_manifest_sha256':fingerprint(S / f'{variant}-source-manifest.json')['sha256'],
            'files':expected,
        }, indent=2) + '\n')
        print(json.dumps({'variant':variant,'status':'SOURCE_PREPARED','inputs':47,'unchanged_base_inputs':46,'host':frozen_host['file']}))


if __name__ == '__main__':
    main()
