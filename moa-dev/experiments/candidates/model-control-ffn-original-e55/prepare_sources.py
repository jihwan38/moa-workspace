#!/usr/bin/env python3
"""Freeze E55 with only the original decoder FFN implementation restored."""
from pathlib import Path
import difflib
import hashlib
import json

if not __debug__:
    raise RuntimeError('Provenance assertions require normal Python; -O is forbidden.')
C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-control-ffn-original-e55'
SUITE = R / 'moa-dev/experiments/candidates/model-prefix-suite-e55'
PARENTS = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55'
CHANGED = {'src/device/shared/mlp.rs', 'src/device/shared/residual.rs', 'src/ops.rs'}


def info(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def write_exact(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, ('refusing to overwrite changed frozen file', str(path))
    else:
        path.write_bytes(data)


def split_function(text, declaration, following):
    start = text.index(declaration)
    end = text.index(following, start)
    return text[:start], text[start:end], text[end:]


def main():
    parents, manifests = {}, {}
    for variant in ('original', 'e55'):
        manifests[variant] = json.loads((SUITE / f'{variant}-source-manifest.json').read_text())['files']
        base = PARENTS / variant / 'crate'
        assert len(manifests[variant]) == 47
        assert {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()} == set(manifests[variant])
        parents[variant] = {}
        for name, expected in manifests[variant].items():
            data = (base / name).read_bytes()
            assert info(data) == expected, (variant, name)
            parents[variant][name] = data
    original, e55 = parents['original'], parents['e55']
    assert original['src/bin/probe_model_prefix_suite.rs'] == e55['src/bin/probe_model_prefix_suite.rs']
    assert original['Cargo.lock'] == e55['Cargo.lock']
    assert e55['src/bin/probe_model_prefix_suite.rs'] == (SUITE / 'probe_model_prefix_suite.rs').read_bytes()
    result = dict(e55)
    result['src/device/shared/mlp.rs'] = original['src/device/shared/mlp.rs']

    declaration = '#[device(chip = 1)]\npub fn decoder_feedforward('
    following = '#[device(chip = 1)]\npub fn final_norm_and_logits('
    before, e55_ffn, after = split_function(e55['src/ops.rs'].decode(), declaration, following)
    _, original_ffn, _ = split_function(original['src/ops.rs'].decode(), declaration, following)
    assert original_ffn.split(') {', 1)[0] == e55_ffn.split(') {', 1)[0]
    assert original_ffn.count('shared::residual::add(') == 1
    restored_ffn = original_ffn.replace('shared::residual::add(', 'shared::residual::add_original_for_feedforward(')
    result['src/ops.rs'] = (before + restored_ffn + after).encode()

    _, original_add, _ = split_function(original['src/device/shared/residual.rs'].decode(),
                                        'pub(crate) fn add(', 'pub(crate) fn scale_by_layer_gate(')
    dedicated_add = original_add.replace('pub(crate) fn add(', 'pub(crate) fn add_original_for_feedforward(', 1)
    result['src/device/shared/residual.rs'] = e55['src/device/shared/residual.rs'] + (
        '\n// Isolated control: preserve E55 shared add for full-O; this loop is original.\n' + dedicated_add
    ).encode()
    # Original FFN still calls the generic RMS and layer gate, both unchanged.
    for name, declaration, following in (
        ('src/device/shared/rmsnorm.rs', 'pub(crate) fn normalize<', 'pub(crate) fn '),
    ):
        old = original[name].decode()
        new = e55[name].decode()
        start_old, start_new = old.index(declaration), new.index(declaration)
        # The original generic normalize is the whole original file's function
        # suffix; E55 appends separate new functions after that exact text.
        assert old[start_old:].rstrip() in new[start_new:]
    old_residual = original['src/device/shared/residual.rs'].decode()
    e55_residual = e55['src/device/shared/residual.rs'].decode()
    _, old_gate, _ = split_function(old_residual, 'pub(crate) fn scale_by_layer_gate(', 'pub(crate) fn add_vision<')
    _, new_gate, _ = split_function(e55_residual, 'pub(crate) fn scale_by_layer_gate(', 'pub(crate) fn add_vision<')
    assert old_gate == new_gate
    assert {name for name in result if result[name] != e55[name]} == CHANGED
    assert len(result) == 47
    crate = A / 'crate'
    for name, data in result.items():
        write_exact(crate / name, data)
    assert {str(p.relative_to(crate)) for p in crate.rglob('*') if p.is_file()} == set(result)
    for name in sorted(CHANGED):
        write_exact(C / Path(name).name, result[name])
    patch = ''.join(''.join(difflib.unified_diff(
        e55[name].decode().splitlines(keepends=True), result[name].decode().splitlines(keepends=True),
        fromfile='e55/' + name, tofile='ffn-original-control/' + name)) for name in sorted(CHANGED))
    write_exact(C / 'source.patch', patch.encode())
    manifest = {'schema': 'MOA_SINGLE_KERNEL_SOURCE_V1', 'base': 'e55',
                'target': 'decoder_feedforward', 'original_input_count': 46, 'added_probe_count': 1,
                'files': {name: info(data) for name, data in sorted(result.items())}}
    write_exact(C / 'source-manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
    audit = {
        'schema': 'MOA_FFN_ORIGINAL_CONTROL_SOURCE_AUDIT_V1',
        'parent_manifests': {v: info((SUITE / f'{v}-source-manifest.json').read_bytes()) for v in manifests},
        'parent_inputs_verified_each': 47, 'control_inputs': 47,
        'modified_from_e55': sorted(CHANGED),
        'mlp_file_byte_identical_original': True,
        'other_public_ops_byte_identical_e55': True,
        'decoder_public_signature_unchanged': True,
        'decoder_body_original_except_dedicated_add_call_name': True,
        'original_residual_add_body_preserved_under_dedicated_name': True,
        'existing_e55_residual_file_is_unchanged_prefix': True,
        'generic_rms_and_layer_gate_original_code_preserved': True,
        'probe_source': info(result['src/bin/probe_model_prefix_suite.rs']),
        'host_loader_tokenizer_runtime_and_official_harness_unchanged_e55': True,
        'image_identity_requires_postbuild_registry_gate': True,
        'npu_executed': False,
    }
    write_exact(C / 'source-audit.json', (json.dumps(audit, indent=2) + '\n').encode())
    print(json.dumps({'status': 'SOURCE_PREPARED', 'inputs': 47, 'changed': sorted(CHANGED)}))


if __name__ == '__main__':
    main()
