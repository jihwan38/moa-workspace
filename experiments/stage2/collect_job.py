#!/usr/bin/env python3
"""Collect an Arena job without treating command completion as model validation."""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('job', type=int)
parser.add_argument('run')
args = parser.parse_args()
root = Path(os.environ.get('MOA_ROUND2_ROOT', Path(__file__).resolve().parents[1]))
run = root / 'runs' / args.run
if not run.is_dir():
    parser.error('run directory must already exist')
base = ['docker', 'exec', 'moa2026-round2', 'furiosa-arena']
status = json.loads(subprocess.check_output([*base, 'status', str(args.job)]))
(run / 'arena-status.json').write_text(json.dumps(status, indent=2) + '\n')
log = subprocess.check_output([*base, 'logs', str(args.job)]).decode()
(run / 'arena.log').write_text(log)
records = []
for line in log.splitlines():
    prefix, _, payload = line.partition(' ')
    if prefix.endswith(('_RESULT', '_SUMMARY')):
        try:
            records.append({'type': prefix, 'data': json.loads(payload)})
        except json.JSONDecodeError:
            pass
(run / 'arena-records.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps({'job': args.job, 'status': status['status'],
                  'duration_sec': status.get('duration_sec'),
                  'exit_code': status.get('exit_code'), 'records': records}, indent=2))
