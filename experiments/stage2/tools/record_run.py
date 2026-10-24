#!/usr/bin/env python3
"""Run one local/container command with immutable source and result metadata."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ.get('MOA_ROUND2_ROOT', Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--worktree', default='server')
parser.add_argument('--purpose', required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
if not command:
    parser.error('a command is required')
tree = (root / args.worktree).resolve()
if tree.parent != root or not (tree / '.git').exists():
    parser.error('worktree must be a direct Git worktree under the round2 directory')
run = root / 'runs' / args.name
run.mkdir(exist_ok=False)
def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def git(*arguments):
    return subprocess.check_output(['git', '-C', str(tree), *arguments],
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS='0')).decode()
files = sorted((tree / 'src').rglob('*'))
files += [tree / name for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml']]
hashes = {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in files if p.is_file()}
(run / 'source-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
(run / 'source.diff').write_text(git('diff', '--binary', 'HEAD'))
metadata = {'name': args.name, 'purpose': args.purpose, 'started_at': utc(),
            'worktree': str(tree), 'commit': git('rev-parse', 'HEAD').strip(),
            'status': git('status', '--porcelain=v1'), 'command': command,
            'result': 'running', 'exit_code': None}
for argument in command:
    if argument.startswith('/round2/candidates/'):
        candidate = root / argument.removeprefix('/round2/')
        manifest = candidate / 'candidate.json'
        if manifest.is_file():
            metadata['tested_candidate'] = json.loads(manifest.read_text())
            (run / 'tested-candidate.json').write_text(manifest.read_text())
record = run / 'run.json'
record.write_text(json.dumps(metadata, indent=2) + '\n')
with (run / 'output.log').open('w') as log:
    process = subprocess.Popen(command, cwd=tree, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    metadata['pid'] = process.pid
    record.write_text(json.dumps(metadata, indent=2) + '\n')
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(line, end='', flush=True)
    code = process.wait()
metadata.update(finished_at=utc(), exit_code=code,
                result='command_succeeded' if code == 0 else 'command_failed')
record.write_text(json.dumps(metadata, indent=2) + '\n')
print(f'RUN_RECORD {record}', flush=True)
sys.exit(code)
