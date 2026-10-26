#!/usr/bin/env python3
"""Replay frozen-source identity and hostcheck binding; no compiler/device calls."""
from pathlib import Path
import hashlib
import json
import subprocess

TREE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = 'ef53a3b3'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(TREE), *args])


def function(text, name):
    start = text.index('async fn ' + name + '(')
    opening = text.index('{', start)
    depth = 0
    for end in range(opening, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            return text[start:end + 1]
    raise AssertionError(name)


def main():
    proof = json.loads((HERE / 'source-proof.json').read_text())
    names = git('ls-tree', '-r', '--name-only', BASE).decode().splitlines()
    old = [p for p in names if p.startswith('src/') or p in ('Cargo.toml', 'rust-toolchain.toml')]
    assert len(old) == proof['unchanged_old_files'] == 423
    for path in old:
        assert (TREE / path).read_bytes() == git('show', BASE + ':' + path), path
    actual = {str(p.relative_to(TREE)) for p in (TREE / 'src').rglob('*') if p.is_file()}
    assert actual - {p for p in old if p.startswith('src/')} == set(proof['new_src'])
    before = (TREE / 'src/bin/test_int4_grouped_complete_ffn.rs').read_text()
    after = (TREE / 'src/bin/test_int4_weight_mse.rs').read_text()
    for name, expected in proof['unchanged_function_sha256'].items():
        a, b = function(before, name), function(after, name)
        assert a == b and sha(a.encode()) == expected, name
    plan = json.loads((HERE / 'plan.json').read_text())
    for path, expected in plan['dependencies'].items():
        assert sha((TREE / path).read_bytes()) == expected, path
    check = HERE / 'a003-int4-weight-mse-hostcheck'
    run = json.loads((check / 'run.json').read_text())
    hashes = json.loads((check / 'source-hashes.json').read_text())
    assert run['exit_code'] == 0 and run['result'] == 'command_succeeded'
    assert len(hashes) == 427 and run['fresh_package']['contents_changed'] is False
    for path, expected in hashes.items():
        assert sha((TREE / path).read_bytes()) == expected, path
    return {'base': BASE, 'old_source_config_unchanged': len(old),
            'new_source': proof['new_src'], 'copied_functions_identical': 6,
            'hostcheck_source': run['commit'], 'hostcheck_inputs': len(hashes),
            'hostcheck_pass': True, 'dependencies_unchanged': True,
            'future_registry_expected': 19,
            'scope': 'Seven-call numerical diagnostic. No threegraph timing or model quality claim.'}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
