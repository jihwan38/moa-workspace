#!/usr/bin/env python3
"""Verify inputs and reproduce the complete fixed-E4 compiler cost screen."""
import collections
import hashlib
import json
from pathlib import Path

tree = Path(__file__).resolve().parents[2]
root = tree.parent
out = {'functional_source': '4bf039b2', 'graphs': {},
       'scope': 'Complete live16 append+attention+O compiler estimate; no hardware or quality claim'}
for name, run in [('candidate', 'r1272-b16-fixed-e4-resource-decode'),
                  ('control', 'r1273-b16-fixed-e4-control-resource')]:
    record_dir = root / 'runs' / run
    record = json.loads((record_dir / 'run.json').read_text())
    assert record['result'] == 'command_succeeded' and record['build_inputs_unchanged']
    hashes = json.loads((record_dir / 'source-hashes.json').read_text())
    for relative, expected in hashes.items():
        assert hashlib.sha256((tree / relative).read_bytes()).hexdigest() == expected, relative
    path = Path(__file__).with_name(name + '-schedule.json')
    data = json.loads(path.read_text())
    end, sums = 0, collections.Counter()
    for key, events in data['timeline']['operators'].items():
        for event in events:
            life = event['lifetime']
            end = max(end, life['end'])
            sums[data['description']['operators'][key]['kind']] += life['end'] - life['begin']
    stack = {json.dumps(entry['allocation'], sort_keys=True)
             for entries in data['timeline']['tensors'].values() for entry in entries
             if 'DramStack' in entry['allocation']['buffer_type']}
    out['graphs'][name] = {
        'estimated_cycles': end, 'overlapping_operator_duration_sums': dict(sums),
        'dram_stack_allocations': [json.loads(value) for value in sorted(stack)],
        'source_hashes_verified': len(hashes),
        'schedule_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'run': run}
candidate, control = (out['graphs'][name] for name in ('candidate', 'control'))
out['relative_reduction'] = 1 - candidate['estimated_cycles'] / control['estimated_cycles']
out['same_index_bounce_allocation'] = candidate['dram_stack_allocations'] == control['dram_stack_allocations']
assert out['same_index_bounce_allocation']
assert [entry['size'] for entry in candidate['dram_stack_allocations']] == [32]
out['speed_gate_10pct'] = out['relative_reduction'] >= .1
out['decision'] = 'PASS resource gate; actual intended-code micro required' if out['speed_gate_10pct'] else 'HOLD resource gate'
Path(__file__).with_name('result.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({key: value for key, value in out.items() if key != 'graphs'}))
