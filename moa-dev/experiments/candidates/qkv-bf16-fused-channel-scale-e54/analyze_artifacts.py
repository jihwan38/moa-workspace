#!/usr/bin/env python3
"""Check lowered fusion and compare resource timelines with measured E54 image."""
from pathlib import Path
import collections
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ART = ROOT/'moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54'
BASE = ROOT/'moa-dev/experiments/artifacts/qkv-bf16-f32-scale-e53'

def timeline(i):
    return {'operator':i['history'].get('resourcelir',i['index']), 'resource':i['tpe'],
            **i['lifetime'], 'duration':i['lifetime']['end']-i['lifetime']['begin']}

def inspect(path, fused):
    graph=json.loads((path/'candidate-graph.json').read_text())
    schedule=json.loads((path/'candidate.json').read_text())
    producers={str(t):o for o in graph['operators'].values() for t in o['outputs']}
    consumers=collections.defaultdict(list)
    for o in graph['operators'].values():
        for t in o['inputs']: consumers[str(t)].append(o)
    tensors=graph['tensors']
    # Graph addresses are per-slice; public schedule plots 512 physical slices.
    def schedule_tensor(tid):
        t=tensors[str(tid)]
        address=int(t['optionals']['address'])*512
        matches=[s for s in schedule['tensors'] if s['size']==t['size'] and s['buffer_type']==t['buffer_type'] and s['address']==address]
        assert len(matches)==1, (tid,matches)
        return int(matches[0]['index'])
    result={}
    for dot in graph['operators'].values():
        d=dot['description']
        if 'operation_unit:' not in d or 'lookup_table:' not in d: continue
        weights=[t for t in dot['inputs'] if tensors[str(t)]['element_type']=='Float8e4m3']
        assert len(weights)==1
        weight_id=weights[0]
        weight_dma=producers[str(weight_id)]
        weight_io=tensors[str(weight_dma['inputs'][0])]['optionals']['address']
        label={'IO(1)+0':'Q','IO(2)+0':'K','IO(3)+0':'V'}[weight_io]
        schedule_weight=schedule_tensor(weight_id)
        sched_dot=next(i for i in schedule['instructions'] if schedule_weight in i['input_tensors'] and i['tpe']=='Main')
        sched_dma=next(i for i in schedule['instructions'] if schedule_weight in i['output_tensors'] and i['tpe']=='DmaLoad')
        switch=next(o for o in consumers[str(dot['outputs'][0])] if 'InterTranspose' in o['description'])
        sched_switch=next(i for i in schedule['instructions'] if set(i['input_tensors']) & set(sched_dot['output_tensors']) and i['tpe']=='Main')
        entry={'weight_hbm':weight_io, 'graph_dot':dot['name'], 'graph_switch':switch['name'],
               'weight_dma':timeline(sched_dma),'dot':timeline(sched_dot),'row_switch':timeline(sched_switch),
               'dot_output_dtype':tensors[str(dot['outputs'][0])]['element_type'],
               'switch_output_dtype':tensors[str(switch['outputs'][0])]['element_type']}
        if fused:
            assert d.index('operation_unit:')<d.index('FpMul0:')<d.index('cast_compaction:')<d.index('transpose_engine:')
            assert 'commit_in_size: 8B (=4EA)' in d
            assert 'data_type: Bfloat16' in d
            assert entry['dot_output_dtype']==entry['switch_output_dtype']=='Bfloat16'
            vrfs=[t for t in dot['inputs'] if tensors[str(t)]['buffer_type']=='Vrf']
            assert len(vrfs)==1
            vrf=vrfs[0];sub=producers[str(vrf)]
            assert sub['context']=='Sub' and 'InterTranspose' in sub['description']
            scale_dm=sub['inputs'][0];scale_dma=producers[str(scale_dm)]
            scale_io=tensors[str(scale_dma['inputs'][0])]['optionals']['address']
            assert scale_io=={'Q':'IO(4)+0','K':'IO(5)+0','V':'IO(6)+0'}[label]
            sched_vrf=schedule_tensor(vrf)
            sched_sub=next(i for i in schedule['instructions'] if sched_vrf in i['output_tensors'])
            sched_scale=next(i for i in schedule['instructions'] if set(i['output_tensors']) & set(sched_sub['input_tensors']) and i['tpe']=='DmaLoad')
            assert tensors[str(vrf)]['size']==256*512
            entry.update({'scale_hbm':scale_io, 'graph_scale_sub':sub['name'], 'scale_sub':timeline(sched_sub),
                          'scale_dma':timeline(sched_scale), 'vrf_tensor':vrf,'vrf_physical_bytes_per_slice':256,
                          'vrf_shape':tensors[str(vrf)]['shape'],
                          'first_bf16_cast_after_complete_dot_and_scale':True,'commit_bytes':8,'commit_rows':4})
        result[label]=entry
    assert set(result)=={'Q','K','V'}
    durations=collections.Counter()
    for i in schedule['instructions']:
        durations[i['tpe']]+=i['lifetime']['end']-i['lifetime']['begin']
    summary={'makespan':max(i['lifetime']['end'] for i in schedule['instructions']),
             'instruction_count':len(schedule['instructions']),'duration_sum_by_resource':dict(durations),
             'projection_paths':result}
    return summary

baseline=inspect(BASE,False)
candidate=inspect(ART,True)
assert baseline['makespan']==52282 and candidate['makespan']==51293
output={'baseline':baseline,'candidate':candidate,
        'delta_cycles':candidate['makespan']-baseline['makespan'],
        'delta_percent':100*(candidate['makespan']/baseline['makespan']-1),
        'definition':'Whole makespan=max lifetime.end; resource duration sums are not elapsed runtime.',
        'identity_method':'Q/K/V labels follow graph HBM IO(1/2/3), not source call/schedule order; graph memory address*512 matched unique same-size schedule tensor.',
        'scope':'Public lowered graph and static schedule only; no NPU execution, bitexact proof, or measured speedup.'}
(HERE/'static-comparison.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'baseline':baseline['makespan'],'candidate':candidate['makespan'],
                  'delta_cycles':output['delta_cycles'],'delta_percent':output['delta_percent']},indent=2))
