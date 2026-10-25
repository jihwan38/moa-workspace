#!/usr/bin/env python3
"""Run one local/container command with immutable source and result metadata."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(os.environ.get('MOA_ROUND2_ROOT', Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--worktree', default='server')
parser.add_argument('--purpose', required=True)
parser.add_argument('--fresh-package', action='store_true',
                    help='Refresh package source mtimes before a build in an exclusively owned shared target')
parser.add_argument('--capture-artifact', action='append', default=[], metavar='RELATIVE_PATH',
                    help='Freeze a successful build output immediately, after verifying unchanged inputs')
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
if args.fresh_package:
    # Cargo dep-info may contain relative paths. Switching a shared target back
    # to an older worktree can otherwise accept another tree's newer library.
    # Keep contents/hashes unchanged; make every package input newer than the
    # previously completed build. The caller must own the target exclusively.
    refreshed = set(files)
    refreshed.update(p for p in tree.glob('*.rs') if p.is_file())
    for path in sorted(refreshed):
        if path.is_file():
            path.touch()
    metadata['fresh_package'] = {'refreshed_files': len([p for p in refreshed if p.is_file()]),
                                 'contents_changed': False,
                                 'target_exclusivity': 'caller must coordinate'}
# Upload the exact frozen executable even if another build replaces target/release
# while the Arena client is preparing its request. Keep the launch working directory.
if 'furiosa-arena' in command:
    arena_index = command.index('furiosa-arena')
    if command[arena_index + 1:arena_index + 2] == ['submit']:
        executable_index = arena_index + 2
        requested = Path(command[executable_index])
        container_cwd = Path('/round2') / args.worktree
        if '--workdir' in command:
            container_cwd = Path(command[command.index('--workdir') + 1])
        container_executable = requested if requested.is_absolute() else container_cwd / requested
        try:
            local_executable = root / container_executable.relative_to('/round2')
        except ValueError:
            local_executable = None
        if local_executable is not None and local_executable.is_file():
            artifact_dir = run / 'submitted'
            artifact_dir.mkdir()
            frozen = artifact_dir / requested.name
            shutil.copy2(local_executable, frozen)
            metadata['requested_command'] = command.copy()
            metadata['submitted_artifact'] = {
                'source_path': str(local_executable),
                'path': str(frozen),
                'sha256': hashlib.sha256(frozen.read_bytes()).hexdigest(),
                'size': frozen.stat().st_size,
            }
            command[executable_index] = str(Path('/round2') / frozen.relative_to(root))
for argument in command:
    if argument.startswith(('/round2/candidates/', '/round2/evaluations/')):
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
if code == 0 and args.capture_artifact:
    try:
        for relative, expected in hashes.items():
            actual = hashlib.sha256((tree / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'build input changed: {relative}')
        captured = {}
        for relative in args.capture_artifact:
            source = tree / relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError(f'artifact must be relative to worktree: {relative}')
            destination = run / 'built' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            captured[relative] = {
                'source_path': str(source), 'path': str(destination),
                'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                'size': destination.stat().st_size,
            }
        metadata['built_artifacts'] = captured
        metadata['build_inputs_unchanged'] = True
    except (OSError, ValueError) as error:
        code = 1
        metadata.update(result='artifact_verification_failed', exit_code=code,
                        artifact_error=str(error))
record.write_text(json.dumps(metadata, indent=2) + '\n')
print(f'RUN_RECORD {record}', flush=True)
sys.exit(code)
