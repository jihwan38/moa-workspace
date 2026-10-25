#!/usr/bin/env python3
"""Read-only completed HTTP/lifecycle audit; excludes prompt and output text."""
from pathlib import Path
import collections
import hashlib
import json
import re

ROOT = Path('/home/jihwan38/moa-worktrees/round2/runs')
HERE = Path(__file__).parent
RUNS = {'control': 'r1148-c077-overlap-r96-submit', 'candidate': 'r1150-c078-overlap-r96-submit'}


def fields(line):
    return dict(re.findall(r'(\w+)=([^ ]*)', line))


def tuples(value):
    return [list(map(int, s.split(':'))) for s in value.split(',') if s]


def http_records(lines, prefix, expected):
    records = []
    for line in lines:
        if not line.startswith(prefix):
            continue
        x = json.loads(line[len(prefix):])
        ds = x['delta_offsets_ms']
        assert x['passed'] and x['finish_reason'] == 'length'
        assert len(ds) == x['usage']['completion_tokens']
        assert all(b >= a for a, b in zip(ds, ds[1:]))
        assert ds[-1] <= x['latency_ms'] + .01
        assert abs(ds[0] - x['ttft_ms']) < 1.0
        gaps = [b-a for a, b in zip(ds, ds[1:])]
        records.append({
            'request_index': x['request_index'], 'suite': x['suite'], 'usage': x['usage'],
            'delta_count': len(ds), 'ttft_ms': x['ttft_ms'], 'latency_ms': x['latency_ms'],
            'dispatch_offset_ms': x['dispatch_offset_ms'], 'completion_offset_ms': x['completion_offset_ms'],
            'max_gap_ms': max(gaps), 'gaps_over150': sum(g > 150 for g in gaps),
            'delta_offsets_ms': ds,
        })
    assert len(records) == expected
    assert sorted(x['request_index'] for x in records) == list(range(1, expected + 1))
    return records


def audit(name):
    path = ROOT / name / 'arena.log'
    lines = path.read_text().splitlines()
    manifest = json.loads((path.parent / 'tested-candidate.json').read_text())
    req = []
    for lineno, line in enumerate(lines, 1):
        if line.startswith('SCHED_REQ '):
            req.append({'line': lineno, **fields(line)})
    assert len(req) == 112
    assert sorted(int(x['id']) for x in req) == list(range(1, 113))
    measured = http_records(lines, 'STAGE2_PERFORMANCE_RESULT ', 96)
    warm = http_records(lines, 'STAGE2_PERFORMANCE_WARMUP_RESULT ', 16)
    assert all(x['usage']['completion_tokens'] == (128 if x['request_index'] == 1 else 64) for x in measured)
    assert all(x['usage']['completion_tokens'] == 64 for x in warm)
    summaries = [json.loads(l.split(' ', 1)[1]) for l in lines if l.startswith('STAGE2_PERFORMANCE_SUMMARY ')]
    assert len(summaries) == 1 and summaries[0]['passed'] == 96 and summaries[0]['failed'] == 0
    # A unique tuple is the only asserted HTTP↔Worker mapping. Arrival IDs are not HTTP indices.
    long_workers = [x for x in req if int(x['n']) == 128]
    assert len(long_workers) == 1 and int(long_workers[0]['p']) == 256
    long_http = next(x for x in measured if x['request_index'] == 1)
    assert long_http['usage']['prompt_tokens'] == 256
    snapshots, seen, current = [], set(), None
    for lineno, line in enumerate(lines, 1):
        if line.startswith('SCHED_COHORT ') and snapshots:
            f = fields(line)
            if 'cohort_summary' not in snapshots[-1]:
                snapshots[-1]['cohort_summary'] = f
                assert int(f['members']) == len(snapshots[-1]['release_members'])
                assert f['reason'] == snapshots[-1]['release']['action']
            continue
        if not line.startswith('SCHED_RECOVERY '):
            continue
        f = fields(line); members = tuples(f['members']); passthrough = tuples(f['passthrough'])
        ids = [m[0] for m in members]; passids = {m[0] for m in passthrough}
        assert len(ids) == len(set(ids)) and len(ids) <= 16 and not set(ids) & passids
        assert all(len(x) == 4 for x in members) and all(len(x) == 3 for x in passthrough)
        assert all(x[2] >= 0 and x[3] in [0,1] for x in members)
        if f['action'] == 'start':
            assert current is None and not set(ids) & seen
            assert all(x[3] == 0 and x[2] > 0 for x in members)
            assert abs(float(f['oldest_ms']) + float(f['left_ms']) - 3750.0) < .003
            seen.update(ids)
            current = {'start_line': lineno, 'start': f, 'start_members': members,
                       'start_passthrough': passthrough, 'joins': [], 'current_ids': ids,
                       'receipt_relative_ms': -float(f['oldest_ms']),
                       'deadline_relative_ms': float(f['left_ms']), 'last_elapsed_ms': 0.0,
                       'last_members': members}
        elif f['action'] == 'join':
            assert current is not None
            previous = [int(x) for x in f['previous'].split(',')]
            added = tuples(f['added']); addedids = [x[0] for x in added]
            assert previous == current['current_ids'] and ids == previous + addedids
            assert len(addedids) == len(set(addedids)) and not set(addedids) & seen
            assert all(next(m for m in members if m[0] == x[0]) == x + [0] for x in added)
            elapsed = float(f['elapsed_ms']); oldleft = float(f['old_left_ms']); left = float(f['left_ms'])
            assert elapsed >= current['last_elapsed_ms'] and left > 0 and left <= oldleft + .001
            assert abs(elapsed + oldleft - current['deadline_relative_ms']) < .005
            assert abs(elapsed - float(f['old_oldest_ms']) - current['receipt_relative_ms']) < .005
            assert float(f['oldest_ms']) >= float(f['old_oldest_ms']) - .001
            newdeadline = elapsed + left
            assert newdeadline <= current['deadline_relative_ms'] + .005
            assert abs(float(f['oldest_ms']) + left - 3750.0) < .003
            # All earlier members remain in the same slots with nonincreasing prompt work.
            for old in current['last_members']:
                new = next(m for m in members if m[0] == old[0])
                assert new[1] == old[1] and new[2] <= old[2] and new[3] >= old[3]
            # In this run every added slot was freed by an actual old passthrough completion.
            replacements = []
            for new in added:
                old = next(x for x in current['start_passthrough'] if x[1] == new[1])
                completion = next(x for x in req if int(x['id']) == old[0])
                assert current['start_line'] < completion['line'] < lineno
                replacements.append({'added': new, 'prior_owner': old,
                                     'prior_completion_line': completion['line']})
            current['joins'].append({'line': lineno, 'fields': f, 'members': members,
                                     'passthrough': passthrough, 'replacements': replacements})
            seen.update(addedids)
            current.update(current_ids=ids, last_members=members, last_elapsed_ms=elapsed,
                           deadline_relative_ms=newdeadline,
                           receipt_relative_ms=elapsed-float(f['oldest_ms']))
        else:
            assert current is not None and ids == current['current_ids']
            for old in current['last_members']:
                new = next(m for m in members if m[0] == old[0])
                assert new[1] == old[1] and new[2] <= old[2] and new[3] >= old[3]
            if f['action'] == 'ready':
                assert all(x[3] == 1 for x in members)
            elif f['action'] == 'deadline':
                assert float(f['oldest_ms']) >= 3750 and float(f['left_ms']) == 0
            else:
                raise AssertionError('Unexercised release reason needs explicit interpretation: ' + f['action'])
            elapsed = current['receipt_relative_ms'] + float(f['oldest_ms'])
            assert elapsed >= current['last_elapsed_ms'] - .005
            oldids = {x[0] for x in current['start_passthrough']}
            completed = [int(x['id']) for x in req if current['start_line'] < x['line'] < lineno and int(x['id']) in oldids]
            # No held member completes before its snapshot release; normal length-only requests stay allocated.
            assert not [x for x in req if current['start_line'] < x['line'] < lineno and int(x['id']) in ids]
            current.update(release_line=lineno, release=f, release_members=members,
                           release_passthrough=passthrough, elapsed_ms=elapsed,
                           completed_passthrough=completed,
                           remaining_at_release=sum(x[2] for x in members),
                           deadline_overshoot_ms=max(0., float(f['oldest_ms'])-3750.))
            snapshots.append(current); current = None
    assert current is None
    for snap in snapshots:
        assert abs(float(snap['cohort_summary']['hold_ms']) - snap['elapsed_ms']) < .1
    long_id = int(long_workers[0]['id'])
    intervals = [s for s in snapshots if long_id in {x[0] for x in s['start_passthrough']}
                 and long_id in {x[0] for x in s['release_passthrough']}]
    return {
        'run': name, 'log_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_commit': manifest['source_commit'], 'binary_sha256': manifest['binary_sha256'],
        'worker_completions': len(req), 'unique_worker_completions': True,
        'http_measured': len(measured), 'http_warm': len(warm), 'all_pass': True,
        'all_length_finishes': True, 'delta_count_matches_usage': True,
        'metrics': summaries[0]['metrics'],
        'max_http_gap_ms': max(x['max_gap_ms'] for x in measured),
        'requests_with_gap_over150': sum(x['max_gap_ms'] > 150 for x in measured),
        'requests_with_gap_over200': sum(x['max_gap_ms'] > 200 for x in measured),
        'snapshots': snapshots, 'snapshot_count': len(snapshots), 'unique_members': len(seen),
        'join_count': sum(len(s['joins']) for s in snapshots),
        'joined_members': sum(len(tuples(j['fields']['added'])) for s in snapshots for j in s['joins']),
        'long_request': {'http': long_http, 'worker': long_workers[0],
                         'passthrough_intervals_ms': [s['elapsed_ms'] for s in intervals]},
        'http_sanitized': measured, 'worker_sanitized': req,
    }


out = {k: audit(v) for k,v in RUNS.items()}
a = json.loads((ROOT/RUNS['control']/'tested-candidate.json').read_text())
b = json.loads((ROOT/RUNS['candidate']/'tested-candidate.json').read_text())
changed = sorted(s for s in a['source_hashes'] | b['source_hashes']
                 if a['source_hashes'].get(s) != b['source_hashes'].get(s))
assert changed == ['src/host/recovery_late_admission_tests.rs','src/host/startup_cohort.rs','src/host/worker.rs']
out['source_changed'] = changed
out['limits'] = [
    'HTTP request indices differ from Worker receipt IDs. Only unique prompt256/output128 is mapped.',
    'Recovery records expose Decode eligibility, not visible-versus-invisible phase; completion lines prove real passthrough progress.',
    'No per-row token execution trace exists. No extra held decode is source/CPU-tested; exact output counts alone do not prove execution count.',
    'All requests finish by length. Actual cancellations, EOS, fatal errors and EOF are not exercised here; CPU gates cover them.',
    'Receipt deadlines are unchanged, but atomic model work can overshoot; 150ms is a soft gap bound.',
    'No absolute admission timestamps or cross-clock client/server synchronization is assumed.',
]
(HERE/'r1150-lifecycle.json').write_text(json.dumps(out,indent=2)+'\n')
for label, r in out.items():
    if label in RUNS:
        print(label,{k:r[k] for k in ['http_measured','worker_completions','snapshot_count','join_count','joined_members','unique_members','max_http_gap_ms','requests_with_gap_over150']})
