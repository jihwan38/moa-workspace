#!/usr/bin/env python3
"""Bind q020 public quality and matched HTTP measurements to frozen artifacts."""
import datetime
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = 'q020-b16-f4-mse-parallel'
CONTROL = 'c081-physical32-clean'


def read_run(name, candidate):
    run = ROOT / 'runs' / name
    manifest = json.loads((ROOT / 'candidates' / candidate / 'candidate.json').read_text())
    tested = json.loads((run / 'tested-candidate.json').read_text())
    assert tested['binary_sha256'] == manifest['binary_sha256']
    status = json.loads((run / 'arena-status.json').read_text())
    assert status['status'] == 'SUCCEEDED', (name, status)
    records = json.loads((run / 'arena-records.json').read_text())
    log = (run / 'arena.log').read_text()
    jobs = set(re.findall(r'submitted job (\d+)', (run / 'output.log').read_text()))
    assert len(jobs) == 1
    ready = re.findall(r'^Server ready after ([\d.]+)s$', log, re.M)
    assert len(ready) == 1
    evidence = {
        'run': name, 'job': int(jobs.pop()), 'candidate': candidate,
        'source_commit': manifest['source_commit'], 'binary_sha256': manifest['binary_sha256'],
        'status': status['status'], 'duration_sec': status['duration_sec'],
        'server_ready_seconds': float(ready[0]),
        'log_sha256': hashlib.sha256((run / 'arena.log').read_bytes()).hexdigest(),
    }
    return records, log, evidence


quality = []
for item in json.loads((ROOT / 'state/q020-quality-plan.json').read_text()):
    records, log, evidence = read_run(item['name'], CANDIDATE)
    results = [x['data'] for x in records if x['type'] == 'STAGE2_RESULT']
    summaries = [x['data'] for x in records if x['type'] == 'STAGE2_SUMMARY']
    assert results and all(x['passed'] is True for x in results)
    assert len(summaries) == 1 and summaries[0]['total'] == len(results)
    assert summaries[0]['failed'] == 0 and summaries[0]['passed'] == len(results)
    assert 'quantizer=MseDyadic projections=144' in log
    snapshots = [{int(w): int(n) for w, n in re.findall(r'\bd(\d+)=(\d+):', line)}
                 for line in log.splitlines() if line.startswith('SCHED_TOTAL')]
    assert snapshots
    for previous, current in zip(snapshots, snapshots[1:]):
        assert all(current.get(w, 0) >= n for w, n in previous.items())
    evidence.update(test=item['test'], passed=len(results), summary=summaries[0],
                    identity_marker_retained_in_log_tail='verified_owning_buffers=true' in log,
                    b16_decode_calls=sum(n for w, n in snapshots[-1].items() if 2 <= w <= 16))
    quality.append(evidence)
assert len(quality) == 6 and sum(x['passed'] for x in quality) == 124

performance = []
for name, candidate in [('r1301-q020-c16-warm-r96-submit', CANDIDATE),
                        ('r1302-c081-c16-warm-r96-submit', CONTROL)]:
    records, log, evidence = read_run(name, candidate)
    assert 'verified_owning_buffers=true' in log
    results = [x['data'] for x in records if x['type'] == 'STAGE2_PERFORMANCE_RESULT']
    warmup = [x['data'] for x in records if x['type'] == 'STAGE2_PERFORMANCE_WARMUP_RESULT']
    summaries = [x['data'] for x in records if x['type'] == 'STAGE2_PERFORMANCE_SUMMARY']
    assert len(summaries) == 1
    summary = summaries[0]
    assert len(results) == summary['total'] == 96 and len(warmup) == summary['concurrency'] == 16
    assert summary['failed'] == 0 and all(x['passed'] for x in results + warmup)
    assert len({x['request_index'] for x in results}) == 96
    completed = re.findall(r'^SCHED_REQ id=(\d+)', log, re.M)
    assert len(completed) == len(set(completed)) == 112
    assert summary['metrics']['ttft_ms']['p90'] <= 4000
    assert summary['metrics']['estimated_tpot_ms']['p90'] <= 100
    evidence.update(summary=summary, measured_requests=96, warmup_requests=16,
                    unique_worker_completions=112,
                    individual_ttft_over4s=sum(x['ttft_ms'] > 4000 for x in results))
    performance.append(evidence)

current, control = [x['summary']['metrics']['generated_tokens_per_second'] for x in performance]
result = {
    'candidate': CANDIDATE, 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'quality': quality, 'performance': performance,
    'matched_relative_tps_gain': current / control - 1,
    'scope': 'q019 approximate MSE F4 B16 FFN with byte-identical parallel startup encoding. '
             'All six public suites completed, 124/124. One longer matched C16 96-request HTTP pair. '
             'Not official 240-request performance or private quality; original-model equivalence is not claimed.',
    'retained_prior_timeout': 'q019 r1278 GSM planned34 completed17 before70s; never count it as a complete pass.',
}
destination = ROOT / 'r64-server/evidence/q020-http-validation.json'
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'quality': '124/124', 'q020_tps': current, 'c081_tps': control,
                  'matched_gain': current / control - 1}))
