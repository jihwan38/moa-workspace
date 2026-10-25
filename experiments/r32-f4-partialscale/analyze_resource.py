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
pre = summarize('a003-pre.json')
post = summarize('a004-post.json')
fixed = pre['estimated_finish'] + post['estimated_finish']
a, b = control['estimated_finish'] + fixed, candidate['estimated_finish'] + fixed
gain = 1 - b / a
stacks = candidate['dram_stack_allocations'] + pre['dram_stack_allocations'] + post['dram_stack_allocations']
result = dict(source_commit='1a4a57a3', control=control, candidate=candidate,
              pre_norm=pre, post_norm_residual_gate=post,
              complete_estimated_control=a, complete_estimated_candidate=b,
              complete_fraction_lower=gain, core_fraction_lower=1-candidate['estimated_finish']/control['estimated_finish'],
              gate_10pct_no_dramstack=gain>=.10 and not stacks,
              status='PASS_RESOURCE_ONLY' if gain>=.10 and not stacks else 'HOLD_RESOURCE',
              scope='Sum of existing three sequential launch schedules: pre-norm + complete four-stage core + post-norm/residual/gate. No fused-wrapper or host overhead measurement. No runtime/model proof.')
(ROOT/'resource-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('control','candidate','pre_norm','post_norm_residual_gate')},indent=2))
for name,x in [('control',control),('candidate',candidate),('pre',pre),('post',post)]:
 print(name,x['estimated_finish'],x['operator_counts'],x['summed_operator_cycles'],'DramStack',len(x['dram_stack_allocations']))
