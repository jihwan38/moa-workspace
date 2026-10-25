#!/usr/bin/env python3
"""Validate frozen q021 public quality and completed HTTP evidence."""
import datetime
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = 'q021-b16-f4-mse-lm-fp8'


def read(name, candidate=CANDIDATE):
    folder = ROOT / 'runs' / name
    manifest = json.loads((ROOT / 'candidates' / candidate / 'candidate.json').read_text())
    tested = json.loads((folder / 'tested-candidate.json').read_text())
    assert tested['binary_sha256'] == manifest['binary_sha256']
    status = json.loads((folder / 'arena-status.json').read_text())
    assert status['status'] == 'SUCCEEDED'
    records = json.loads((folder / 'arena-records.json').read_text())
    log = (folder / 'arena.log').read_text()
    jobs = set(re.findall(r'submitted job (\d+)', (folder / 'output.log').read_text()))
    assert len(jobs) == 1
    ready = re.findall(r'^Server ready after ([\d.]+)s$', log, re.M)
    assert len(ready) == 1
    return records, log, {
        'run': name, 'job': int(jobs.pop()), 'source_commit': manifest['source_commit'],
        'binary_sha256': manifest['binary_sha256'], 'status': status['status'],
        'duration_sec': status['duration_sec'], 'server_ready_seconds': float(ready[0]),
        'log_sha256': hashlib.sha256((folder / 'arena.log').read_bytes()).hexdigest(),
    }


quality = []
for item in json.loads((ROOT / 'state/q021-quality-plan.json').read_text()):
    records, log, evidence = read(item['name'])
    results = [x['data'] for x in records if x['type'] == 'STAGE2_RESULT']
    summaries = [x['data'] for x in records if x['type'] == 'STAGE2_SUMMARY']
    assert len(summaries) == 1 and results
    assert summaries[0]['total'] == summaries[0]['passed'] == len(results)
    assert summaries[0]['failed'] == 0 and all(x['passed'] is True for x in results)
    assert 'quantizer=MseDyadic projections=144' in log
    evidence.update(test=item['test'], passed=len(results), summary=summaries[0])
    quality.append(evidence)
assert len(quality) == 6 and sum(x['passed'] for x in quality) == 124

external = []
for name in ['r1326-q021-b16-f4-mse-lm-fp8-external-gsm-train32-s1',
             'r1327-q021-b16-f4-mse-lm-fp8-external-gsm-train32-s2']:
    records, _, evidence = read(name)
    results = [x['data'] for x in records if x['type'] == 'STAGE2_RESULT']
    summaries = [x['data'] for x in records if x['type'] == 'STAGE2_SUMMARY']
    assert len(summaries) == 1 and len(results) == 16
    summary = summaries[0]
    assert summary['accuracy_denominator_complete'] and summary['completed'] == 16
    assert summary['accuracy_scored'] == summary['accuracy_passed'] == 16
    assert all(x['passed'] is True for x in results)
    evidence.update(summary=summary)
    external.append(evidence)


def performance(name, concurrency, count, candidate=CANDIDATE):
    records, log, evidence = read(name, candidate)
    measured = [x['data'] for x in records if x['type'] == 'STAGE2_PERFORMANCE_RESULT']
    warm = [x['data'] for x in records if x['type'] == 'STAGE2_PERFORMANCE_WARMUP_RESULT']
    summaries = [x['data'] for x in records if x['type'] == 'STAGE2_PERFORMANCE_SUMMARY']
    assert len(summaries) == 1
    summary = summaries[0]
    assert len(measured) == summary['total'] == summary['passed'] == count
    assert len(warm) == summary['concurrency'] == concurrency
    assert summary['failed'] == 0 and all(x['passed'] for x in measured + warm)
    assert len({x['request_index'] for x in measured}) == count
    completed = re.findall(r'^SCHED_REQ id=(\d+)', log, re.M)
    assert len(completed) == len(set(completed)) == count + concurrency
    assert 'verified_owning_buffers=true' in log
    assert summary['metrics']['ttft_ms']['p90'] <= 4000
    assert summary['metrics']['estimated_tpot_ms']['p90'] <= 100
    shape = [(x['request_index'], x['id'], x['usage']['prompt_tokens'], x['usage']['completion_tokens'])
             for x in sorted(measured, key=lambda x: x['request_index'])]
    evidence.update(summary=summary, warmup_requests=concurrency, measured_requests=count,
                    unique_worker_completions=len(completed),
                    request_shape_sha256=hashlib.sha256(json.dumps(shape).encode()).hexdigest(),
                    individual_ttft_over4s=sum(x['ttft_ms'] > 4000 for x in measured))
    return evidence


c16 = performance('r1328-q021-c16-warm-r96-submit', 16, 96)
c15 = performance('r1329-q021-c15-warm-r90-submit', 15, 90)
control = performance('r1301-q020-c16-warm-r96-submit', 16, 96, 'q020-b16-f4-mse-parallel')
assert c16['request_shape_sha256'] == control['request_shape_sha256']
harnesses = [json.loads((ROOT / 'evaluations' / f'{candidate}-c16-warm-r96/harness.json').read_text())
             for candidate in ['q020', 'q021']]
assert harnesses[0]['accuracy_helper_sha256'] == harnesses[1]['accuracy_helper_sha256']
gain = (c16['summary']['metrics']['generated_tokens_per_second'] /
        control['summary']['metrics']['generated_tokens_per_second'] - 1)
c15_control = performance('r1332-q020-c15-warm-r90-submit', 15, 90, 'q020-b16-f4-mse-parallel')
c15_repeat = performance('r1333-q021-c15-warm-r90-repeat-submit', 15, 90)
assert c15_control['request_shape_sha256'] == c15_repeat['request_shape_sha256'] == c15['request_shape_sha256']
c15_gain = c15_repeat['summary']['metrics']['generated_tokens_per_second'] / c15_control['summary']['metrics']['generated_tokens_per_second'] - 1
result = {
    'candidate': CANDIDATE, 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'quality': quality, 'external_gsm_train32': external,
    'performance': [c16, c15], 'matched_q020_control': control,
    'matched_c16_relative_tps_gain': gain,
    'matched_c15_control': c15_control, 'matched_c15_q021_repeat': c15_repeat,
    'matched_c15_relative_tps_gain': c15_gain,
    'selection': 'q021 quality passes, but no material HTTP speed gain established: C16 flat and matched C15+0.36%, below observed run variation. Keep q020 as the simpler measured candidate; retain q021 as a validated alternate.',
    'scope': 'Approximate MSE F4 B16 FFN and FP8 B16 LM; byte-identical parallel CPU startup. '
             'Six organizer examples repeated over124 requests, plus32 distinct preregistered public TRAIN questions. '
             'One matched-size C16 comparison and a matched C15 pair plus an earlier standalone C15 run. No official240/225-request performance, '
             'private quality, global accuracy improvement or original-model equivalence claim.',
}
destination = ROOT / 'r64-server/evidence/q021-http-validation.json'
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'public': '124/124', 'external_train': '32/32', 'C16_tps': c16['summary']['metrics']['generated_tokens_per_second'],
                  'C15_tps': c15['summary']['metrics']['generated_tokens_per_second'], 'matched_C16_gain': gain, 'matched_C15_gain': c15_gain}))
