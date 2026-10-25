#!/usr/bin/env python3
"""Root-only frozen TRAIN32 experiments; retain accuracy failures."""
import argparse
import datetime
import fcntl
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--candidate', choices=['q020', 'q021'], default='q020')
args = parser.parse_args()
is_q021 = args.candidate == 'q021'
PLAN = ROOT / 'external-quality/experiments/external-gsm32' / ('q021-dispatch-plan.json' if is_q021 else 'dispatch-plan.json')
STATE = ROOT / 'state' / ('external-gsm32-q021-status.json' if is_q021 else 'external-gsm32-status.json')
assert not STATE.exists(), 'Inspect prior state; do not restart an active queue'
lock = (ROOT / 'state/automatic-arena-queue.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
source = json.loads(PLAN.read_text())
assert source['stop_on_failure'] is False
assert len(source['jobs']) == (2 if is_q021 else 4)
# Compare each shard on both frozen candidates before advancing to the next.
pending = []
for number, index in enumerate([0, 1] if is_q021 else [0, 2, 1, 3], start=1326 if is_q021 else 1308):
    item = dict(source['jobs'][index])
    evaluation = Path(item['evaluation']).name
    item['name'] = f'r{number}-{evaluation}'
    dest = ROOT / 'evaluations' / evaluation
    harness = json.loads((dest / 'harness.json').read_text())
    assert item['hard_timeout_seconds'] == harness['hard_timeout_seconds'] == 70
    assert hashlib.sha256((dest / 'target/release/server').read_bytes()).hexdigest() == item['server_sha256']
    assert harness['request_set_sha256'] == item['request_set_sha256']
    assert harness['total'] == harness['concurrency'] == 16
    pending.append(item)
active, finished = {}, []


def save():
    STATE.write_text(json.dumps({
        'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_plan': str(PLAN), 'pending': [x['name'] for x in pending],
        'active': active, 'finished': finished,
        'accuracy_failure_does_not_abort_comparison': True,
    }, indent=2) + '\n')


save()
while pending or active:
    while pending and len(active) < 2:
        item = pending.pop(0)
        command = list(item['argv'])
        container = command.index('moa2026-round2')
        command[container + 1:container + 1] = ['env', 'RNGD_JOB_NAME=beef-' + item['name']]
        recorded = ['python3', str(ROOT / 'tools/record_run.py'), '--name', item['name'],
                    '--worktree', 'external-quality', '--purpose',
                    'Preregistered distinct GSM TRAIN32 comparison; fixed seed/shards, '
                    'full answers retained, job completion separate from accuracy, no official claim',
                    '--', *command]
        run = subprocess.run(recorded, cwd=ROOT, capture_output=True, text=True)
        ids = set(re.findall(r'submitted job (\d+)', run.stdout))
        assert run.returncode == 0 and len(ids) == 1, (run.stdout[-1500:], run.stderr[-1500:])
        job = ids.pop()
        active[job] = item
        print(json.dumps({'submitted': int(job), 'run': item['name']}), flush=True)
        save()
    for job, item in list(active.items()):
        status = json.loads(subprocess.check_output(
            ['docker', 'exec', 'moa2026-round2', 'furiosa-arena', 'status', job], text=True))
        if status['status'] not in ('SUCCEEDED', 'FAILED', 'TIMED_OUT', 'CANCELLED'):
            continue
        subprocess.run(['python3', str(ROOT / 'tools/collect_job.py'), job, item['name']],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        folder = ROOT / 'runs' / item['name']
        tested = json.loads((folder / 'tested-candidate.json').read_text())
        assert tested['binary_sha256'] == item['server_sha256']
        records = json.loads((folder / 'arena-records.json').read_text())
        results = [x['data'] for x in records if x['type'] == 'STAGE2_RESULT']
        summaries = [x['data'] for x in records if x['type'] == 'STAGE2_SUMMARY']
        row = {'job': int(job), 'run': item['name'], 'status': status['status'],
               'duration_sec': status.get('duration_sec'), 'summaries': summaries,
               'observed_results': len(results), 'observed_accuracy_passed': sum(x['passed'] for x in results),
               'all_16_result_records': len(results) == 16,
               'accuracy_denominator_complete': bool(summaries) and summaries[0].get('accuracy_denominator_complete', False),
               'server_sha256': item['server_sha256'], 'request_set_sha256': item['request_set_sha256']}
        finished.append(row)
        del active[job]
        print(json.dumps(row), flush=True)
        save()
    if active:
        time.sleep(3)
assert len(finished) == len(source['jobs'])
print('All fixed experiments collected; inspect accuracy separately from job completion.', flush=True)
