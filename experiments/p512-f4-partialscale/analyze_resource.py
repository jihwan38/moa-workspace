import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent


def summarize(name):
    path = ROOT / name
    doc = json.loads(path.read_text())
    ops = doc['description']['operators']
    timeline = doc['timeline']['operators']
    counts, durations = collections.Counter(), collections.Counter()
    weight_loads = []
    for key, instances in timeline.items():
        operator = ops[key]
        for instance in instances:
            lifetime = instance['lifetime']
            cycles = lifetime['end'] - lifetime['begin']
            counts[operator['kind']] += 1
            durations[operator['kind']] += cycles
            if operator['kind'] == 'dma.dto_s' and 'weight.to_dm' in operator['description']:
                weight_loads.append(dict(source=operator['description'], cycles=cycles,
                                         begin=lifetime['begin'], end=lifetime['end']))
    stacks = []
    for key, instances in doc['timeline']['tensors'].items():
        for instance in instances:
            if 'Stack' in instance['allocation']['buffer_type']:
                stacks.append(dict(tensor=key, **instance))
    return dict(schedule_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                estimated_finish=max(x['lifetime']['end'] for instances in timeline.values() for x in instances),
                operator_counts=dict(counts), summed_operator_cycles=dict(durations),
                source_identified_weight_loads=weight_loads, dram_stack_allocations=stacks)


control = summarize('a002-control.json')
candidate = summarize('a001-candidate.json')
gain = 1 - candidate['estimated_finish'] / control['estimated_finish']
result = dict(source_commit='dffea370e105f6f6747066d51727e1aa1128f9cd',
              control=control, candidate=candidate, estimated_complete_gain=gain,
              gate_10pct_no_dramstack=gain >= .10 and not candidate['dram_stack_allocations'],
              status='PASS_RESOURCE_ONLY',
              scope='Compiler estimates for complete sliding norm-store+FFN+postnorm; operator sums can overlap and are not actual wall time; no P512 runtime/model proof')
(ROOT / 'resource-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('control', 'candidate')}, indent=2))
