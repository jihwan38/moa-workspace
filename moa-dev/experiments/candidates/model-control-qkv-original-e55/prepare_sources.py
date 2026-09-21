#!/usr/bin/env python3
"""Reconstruct one QKV-only original control from fully verified frozen suite inputs."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import shutil

if not __debug__:
    raise RuntimeError('Source integrity requires assertions; Python -O is forbidden.')
C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-control-qkv-original-e55'
S = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55'
P = R / 'moa-dev/experiments/candidates/model-prefix-suite-e55'
PROBE = 'src/bin/probe_model_prefix_suite.rs'
EXPECTED_PROBE_SHA = '07cfaf4646da18a78e03bb386632ac455ccf545d98b858385bded281c3104da4'
CHANGED = {'src/ops.rs', 'src/device/sliding/projection.rs',
           'src/device/sliding/rmsnorm.rs', 'src/device/sliding/rope.rs'}

def fp(path):
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}

def function_bounds(source, name):
    match = re.search(r'\bfn ' + re.escape(name) + r'\b', source)
    assert match, name
    body = source.index('{', match.start())
    end, depth = body + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return match.start(), body, end

parents = {}
for variant in ('original', 'e55'):
    base = S / variant / 'crate'
    manifest = json.loads((P / f'{variant}-source-manifest.json').read_text())['files']
    assert len(manifest) == 47
    assert {str(f.relative_to(base)) for f in base.rglob('*') if f.is_file()} == set(manifest)
    assert all(fp(base / name) == info for name, info in manifest.items())
    assert manifest[PROBE]['sha256'] == EXPECTED_PROBE_SHA
    assert manifest['Cargo.lock']['sha256'] == 'aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09'
    parents[variant] = {'base': str(base.relative_to(R)), 'files': manifest}
old = S / 'original/crate'
e55 = S / 'e55/crate'
sources = {name: (e55 / name).read_bytes() for name in parents['e55']['files']}

# Restore only the complete public QKV body, leaving its declaration and every
# other byte in ops.rs unchanged from E55.
name = 'sliding_project_qkv'
original_ops = (old / 'src/ops.rs').read_text()
e55_ops = sources['src/ops.rs'].decode()
o0, ob, oe = function_bounds(original_ops, name)
e0, eb, ee = function_bounds(e55_ops, name)
assert original_ops[o0:ob] == e55_ops[e0:eb]
control_ops = e55_ops[:eb] + original_ops[ob:oe] + e55_ops[ee:]
sources['src/ops.rs'] = control_ops.encode()
_, cb, ce = function_bounds(control_ops, name)
assert control_ops[:cb] == e55_ops[:eb] and control_ops[ce:] == e55_ops[ee:]
assert control_ops[cb:ce] == original_ops[ob:oe]
public_ops = re.findall(r'pub fn (\w+)\(', e55_ops)
assert len(public_ops) == 10
for op in public_ops:
    a0, ab, ae = function_bounds(e55_ops, op)
    b0, bb, be = function_bounds(control_ops, op)
    assert e55_ops[a0:ab] == control_ops[b0:bb]
    if op != name:
        assert e55_ops[a0:ae] == control_ops[b0:be], op

# QKV helper prefix comes from original; E55 O and all its helpers remain exact.
projection = 'src/device/sliding/projection.rs'
original_projection = (old / projection).read_text()
e55_projection = sources[projection].decode()
marker = 'pub(crate) fn project_output('
assert original_projection.count(marker) == e55_projection.count(marker) == 1
prefix = original_projection[:original_projection.index(marker)]
axis_import = 'use crate::axes::{Ds, Gs, H, Ns, Ps, Qs};'
assert prefix.count(axis_import) == 1
# The retained O suffix uses Dummy2; this is an import only, not a QKV mapping.
prefix = prefix.replace(axis_import, 'use crate::axes::{Ds, Dummy2, Gs, H, Ns, Ps, Qs};')
control_projection = prefix + e55_projection[e55_projection.index(marker):]
sources[projection] = control_projection.encode()
assert control_projection[control_projection.index(marker):] == e55_projection[e55_projection.index(marker):]
for name in ('project_query', 'project_one_kv_matrix', 'project_key_value'):
    a0, _, ae = function_bounds(original_projection, name)
    b0, _, be = function_bounds(control_projection, name)
    assert original_projection[a0:ae] == control_projection[b0:be], name
for name in ('src/device/sliding/rmsnorm.rs', 'src/device/sliding/rope.rs'):
    sources[name] = (old / name).read_bytes()

actual_changed = {name for name, value in sources.items() if value != (e55 / name).read_bytes()}
assert actual_changed == CHANGED
assert sources[PROBE] == (P / 'probe_model_prefix_suite.rs').read_bytes()
crate = A / 'crate'
for name, value in sources.items():
    dest = crate / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(value)
    if name in CHANGED:
        review = C / name
        review.parent.mkdir(parents=True, exist_ok=True)
        review.write_bytes(value)
assert {str(f.relative_to(crate)) for f in crate.rglob('*') if f.is_file()} == set(sources)
manifest = {name: fp(crate / name) for name in sorted(sources)}
(C / 'source-manifest.json').write_text(json.dumps({
    'schema': 'MOA_MODEL_CONTROL_QKV_ORIGINAL_SOURCE_V1', 'files': manifest,
    'existing_crate_inputs': 46, 'unchanged_added_suite_host_inputs': 1,
    'changed_from_e55': sorted(CHANGED), 'unchanged_from_e55_count': 43,
    'target': 'furiosa_opt_gemma4::ops::sliding_project_qkv',
}, indent=2) + '\n')
(C / 'parent-source-manifests.json').write_text(json.dumps(parents, indent=2) + '\n')
patch = ''.join(''.join(difflib.unified_diff((e55/name).read_text().splitlines(True),
                 sources[name].decode().splitlines(True), fromfile='a/'+name, tofile='b/'+name))
                for name in sorted(CHANGED))
(C / 'control.patch').write_text(patch)
(C / 'source-scope-verification.json').write_text(json.dumps({
    'status': 'SOURCE_SCOPE_PASS', 'actual_input_count': len(manifest),
    'changed_source_count': len(CHANGED), 'changed_sources': sorted(CHANGED),
    'all_ten_public_signatures_unchanged': True, 'other_nine_ops_functions_byte_identical_e55': True,
    'ops_outside_qkv_body_byte_identical_e55': True,
    'qkv_public_body_byte_identical_original': True,
    'three_projection_qkv_helper_functions_byte_identical_original': True,
    'head_rmsnorm_and_rope_complete_files_byte_identical_original': True,
    'projection_output_suffix_byte_identical_e55': True,
    'all_shared_helpers_byte_identical_e55': True,
    'suite_probe_sha256': EXPECTED_PROBE_SHA, 'npu_executed': False,
}, indent=2) + '\n')
print(json.dumps({'status': 'prepared', 'crate': str(crate), 'inputs': len(manifest), 'changed': sorted(CHANGED)}))
