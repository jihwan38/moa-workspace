#!/usr/bin/env python3
"""Root-owned public accuracy queue, maximum two Arena jobs, hard cap 70s."""
import datetime
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
plan_path = Path(sys.argv[1]).resolve()
state_path = root / 'state' / (plan_path.stem + '-status.json')
assert not state_path.exists(), 'Inspect prior state; never restart an active queue blindly'
lock = (root/'state/automatic-arena-queue.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
pending = json.loads(plan_path.read_text())
active, finished, failed = {}, [], False

def save():
    state_path.write_text(json.dumps({
        'plan': str(plan_path), 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'active': active, 'pending': [x['name'] for x in pending],
        'finished': finished, 'failed': failed,
    }, indent=2) + '\n')

save()
while pending or active:
    while pending and len(active) < 2 and not failed:
        item = pending.pop(0)
        assert item['test'] in ('gsm8k', 'ifeval', 'mmlu', 'vqav2', 'fleurs_en', 'fleurs_ko')
        assert 1 <= item['concurrency'] <= 64
        location = '/round2/candidates/' + item['candidate']
        if 'evaluation' in item:
            assert Path(item['evaluation']).name == item['evaluation']
            location = '/round2/evaluations/' + item['evaluation']
            harness = json.loads((root / 'evaluations' / item['evaluation'] / 'harness.json').read_text())
            assert harness['base_candidate'] == item['candidate']
        command = ['docker', 'exec', '--workdir', location,
                   'moa2026-round2', 'env', 'RNGD_SUBMIT_TIMEOUT=70',
                   'RNGD_JOB_NAME=beef-' + item['name'], 'bash',
                   'scripts/stage2/accuracy_test.sh', '--test', item['test'],
                   '--concurrency', str(item['concurrency']), '--no-build', '--no-wait']
        result = subprocess.run([sys.executable, str(root/'tools/record_run.py'),
                                 '--name', item['name'], '--worktree', item['worktree'],
                                 '--purpose', item['purpose'], '--', *command],
                                cwd=root, text=True, capture_output=True)
        ids = set(re.findall(r'submitted job (\d+)', result.stdout))
        if result.returncode or len(ids) != 1:
            failed = True
            print(json.dumps({'launch_failed': item['name'], 'stdout': result.stdout[-1500:],
                              'stderr': result.stderr[-1500:]}), flush=True)
            save()
            break
        job = int(ids.pop())
        active[str(job)] = item['name']
        print(json.dumps({'submitted': job, 'run': item['name']}), flush=True)
        save()
    for job, name in list(active.items()):
        status = json.loads(subprocess.check_output(
            ['docker', 'exec', 'moa2026-round2', 'furiosa-arena', 'status', job], text=True))
        if status['status'] not in ('SUCCEEDED', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
            continue
        subprocess.run([sys.executable, str(root/'tools/collect_job.py'), job, name],
                       cwd=root, stdout=subprocess.DEVNULL, check=True)
        records = json.loads((root/'runs'/name/'arena-records.json').read_text())
        summaries = [r['data'] for r in records if r['type'] == 'STAGE2_SUMMARY']
        passed = status['status'] == 'SUCCEEDED' and bool(summaries) and all(
            d['failed'] == 0 and d['passed'] == d['total'] and d['total'] > 0 for d in summaries)
        failed |= not passed
        row = {'job': int(job), 'run': name, 'status': status['status'],
               'duration_sec': status.get('duration_sec'), 'quality_passed': passed,
               'summaries': summaries}
        finished.append(row)
        del active[job]
        print(json.dumps(row), flush=True)
        save()
    if failed and not active:
        break
    if active:
        time.sleep(3)
sys.exit(1 if failed else 0)
