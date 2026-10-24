#!/usr/bin/env python3
"""Freeze a successfully built server, source and public test harness together."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path(os.environ.get('MOA_ROUND2_ROOT', Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser()
parser.add_argument('--build-run', required=True)
parser.add_argument('--name', required=True)
args = parser.parse_args()
run = root / 'runs' / args.build_run
record = json.loads((run / 'run.json').read_text())
assert record['result'] == 'command_succeeded', 'build did not succeed'
tree = Path(record['worktree'])
commit = record['commit']
hashes = json.loads((run / 'source-hashes.json').read_text())
for relative, expected in hashes.items():
    assert hashlib.sha256((tree / relative).read_bytes()).hexdigest() == expected, f'changed since build: {relative}'
assert not subprocess.check_output(['git', 'diff', commit, '--', 'src', 'Cargo.toml', 'rust-toolchain.toml'], cwd=tree), 'source differs from recorded commit'
candidate = root / 'candidates' / args.name
candidate.mkdir(parents=True, exist_ok=False)
source = candidate / 'source.tar'
subprocess.run(['git', 'archive', commit, '-o', str(source)], cwd=tree, check=True)
subprocess.run(['tar', '-xf', str(source), '-C', str(candidate)], check=True)
binary = candidate / 'target/release/server'
binary.parent.mkdir(parents=True)
shutil.copy2(tree / 'target/release/server', binary)
shutil.copy2(tree / 'Cargo.lock', candidate / 'Cargo.lock')
manifest = dict(source_commit=commit, build_run=args.build_run,
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                bytes=binary.stat().st_size, source_hashes=hashes)
(candidate / 'candidate.json').write_text(json.dumps(manifest, indent=2) + '\n')
(run / 'candidate.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(candidate=str(candidate), commit=commit, sha256=manifest['binary_sha256']), indent=2))
