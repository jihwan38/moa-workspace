#!/usr/bin/env python3
"""Prepare exactly 47 E55 suite inputs, restoring only shared residual::add."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-control-full-o-original-e55'
SUITE_C = R / 'moa-dev/experiments/candidates/model-prefix-suite-e55'
SUITE_A = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55'
RESIDUAL = 'src/device/shared/residual.rs'
PROBE = 'src/bin/probe_model_prefix_suite.rs'


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def fingerprint(path):
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def add_range(text):
    marker = 'pub(crate) fn add('
    end_marker = '\npub(crate) fn scale_by_layer_gate('
    check(text.count(marker) == 1 and text.count(end_marker) == 1, 'ambiguous_residual_functions')
    begin = text.index(marker)
    end = text.index(end_marker, begin)
    check(text[begin:end].rstrip().endswith('}'), 'bad_add_function_boundary')
    return begin, end


def main():
    bases = {}
    for variant in ('original', 'e55'):
        manifest = json.loads((SUITE_C / f'{variant}-source-manifest.json').read_text())['files']
        compiled = json.loads((SUITE_C / f'{variant}-compiled-source-manifest.json').read_text())['files']
        check(len(manifest) == 47 and manifest == compiled, 'frozen_manifest_mismatch')
        base = SUITE_A / variant / 'crate'
        actual = {str(p.relative_to(base)): fingerprint(p) for p in base.rglob('*') if p.is_file()}
        check(actual == manifest, f'{variant}_frozen_source_mismatch')
        bases[variant] = (base, manifest)
    e55, expected = bases['e55']
    original, _ = bases['original']
    check(expected[PROBE]['sha256'] == '07cfaf4646da18a78e03bb386632ac455ccf545d98b858385bded281c3104da4', 'unexpected_host_source')
    check(expected[RESIDUAL]['sha256'] == 'deb29e529440170c97bdd65788ab2d11907e240dba8e7a5c7a2eb68e641eac3e', 'unexpected_e55_residual')
    old = (e55 / RESIDUAL).read_text()
    upstream = (original / RESIDUAL).read_text()
    b, e = add_range(old)
    ob, oe = add_range(upstream)
    check(old[:b] == upstream[:ob], 'imports_differ')
    restored = old[:b] + upstream[ob:oe] + old[e:]
    check(restored != old, 'no_source_change')
    rb, re = add_range(restored)
    check(restored[rb:re] == upstream[ob:oe] and restored[re:] == old[e:], 'replacement_scope_failure')
    # Generic add has exactly one current source call site, in full_attention_output.
    call = 'shared::residual::add('
    hits = []
    for path in (e55 / 'src').rglob('*.rs'):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if call in line:
                hits.append({'file': str(path.relative_to(e55)), 'line': number})
    check(hits == [{'file': 'src/ops.rs', 'line': 207}], 'generic_add_call_scope_changed')
    ops = (e55 / 'src/ops.rs').read_text()
    body = ops[ops.index('pub fn full_attention_output('):ops.index('pub fn decoder_feedforward(')]
    check(body.count(call) == 1, 'full_o_caller_mismatch')

    crate = A / 'crate'
    for name in expected:
        dest = crate / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(e55 / name, dest)
    (crate / RESIDUAL).write_text(restored)
    (C / 'residual.rs').write_text(restored)
    (C / 'source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True), restored.splitlines(True),
        fromfile='e55/' + RESIDUAL, tofile='control/' + RESIDUAL)))
    actual = {str(p.relative_to(crate)): fingerprint(p) for p in crate.rglob('*') if p.is_file()}
    check(set(actual) == set(expected) and len(actual) == 47, 'unexpected_or_missing_source_inputs')
    changed = [name for name in expected if actual[name] != expected[name]]
    check(changed == [RESIDUAL], 'more_than_residual_changed')
    (C / 'source-manifest.json').write_text(json.dumps({
        'variant':'e55_with_original_full_o_residual_add', 'files':actual,
        'source_input_count':47, 'unchanged_e55_inputs':46, 'changed_inputs':changed,
        'identical_suite_host':True, 'generic_add_call_sites':hits,
        'original_add_function_sha256':hashlib.sha256(upstream[ob:oe].encode()).hexdigest(),
        'suffix_after_add_byte_identical_e55':True,
        'e55_source_manifest_sha256':fingerprint(SUITE_C / 'e55-source-manifest.json')['sha256'],
        'original_source_manifest_sha256':fingerprint(SUITE_C / 'original-source-manifest.json')['sha256'],
    }, indent=2) + '\n')
    print(json.dumps({'status':'prepared', 'input_count':47, 'changed_inputs':changed,
        'residual':actual[RESIDUAL], 'host':actual[PROBE]}))


if __name__ == '__main__':
    main()
