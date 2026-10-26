#!/usr/bin/env python3
"""Read-only preservation and assembly audit; does not compile or run a device."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def function(source, name):
    start = source.index('async fn ' + name + '(')
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def main():
    manifest = json.loads((EXPERIMENT / 'donor-hashes.json').read_text())
    for name, expected in manifest['preserved'].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    name = manifest['quantizer_file']
    donor = subprocess.check_output(['git', 'show', manifest['quantizer_donor_commit'] + ':' + name], cwd=ROOT)
    assert (ROOT / name).read_bytes() == donor
    assert sha(donor) == manifest['quantizer_sha256']
    assert sha((ROOT / 'Cargo.lock').read_bytes()) == 'a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e'
    old = (ROOT / 'src/bin/test_int4_complete_ffn.rs').read_text()
    new = (ROOT / 'src/bin/test_int4_grouped_complete_ffn.rs').read_text()
    for name in ['production_chain', 'candidate_prefix']:
        assert function(old, name) == function(new, name), name
    rest = function(new, 'candidate_rest')
    clocked = function(new, 'candidate_clocked')
    calls = re.findall(r'(?:launch\([^;]+?\)\.await\?|raw::project_\w+\([^;]+?\)\.await\?)', rest)
    assert len(calls) == 6
    assert all(call in clocked for call in calls)
    assert 'candidate_prefix(d,p,x,s).await?' in clocked
    assert all(x not in clocked for x in ['read(', 'upload(', 'snapshot(', 'assert'])
    preserved = subprocess.check_output(['git', 'diff', '--name-only', manifest['base'], '--', 'src'], cwd=ROOT).decode().splitlines()
    new_names = ['src/ops_int4_grouped_activation_store.rs', 'src/bin/test_int4_grouped_complete_ffn.rs', 'src/bin/support/int4_grouped_complete_fixture.rs']
    assert set(preserved) <= {'src/lib.rs', *new_names}, preserved
    return {
        'scope': 'source preservation only; no runtime claim',
        'base': manifest['base'], 'preserved_files': len(manifest['preserved']),
        'quantizer_exact_donor_sha256': sha(donor),
        'current_production_call_exact': True, 'original_prefix_call_exact': True,
        'six_rest_calls_present_unchanged_in_clocked_path': True,
        'all_inherited_src_preserved_except_additive_lib_export': True,
        'new_source_sha256': {p: sha((ROOT / p).read_bytes()) for p in new_names},
        'initial_expected_checks': {'bf16_comparisons': 196608000, 'i64_f64_samples': 6144,
                                    'group_scales': 40960, 'relative_offsets': 40960, 'packed_bytes': 19660800},
    }


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
