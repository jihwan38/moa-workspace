#!/usr/bin/env python3
"""Audit one shared graph TRF, its schedule reservation, and table critical path."""
from pathlib import Path
import collections
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def interval(i):
    return {'id':i['index'],'resource':i['tpe'],**i['lifetime'],
            'duration':i['lifetime']['end']-i['lifetime']['begin']}

def analyze(name):
    p=ROOT/'moa-dev/experiments/artifacts'/name
    g=json.loads((p/'candidate-graph.json').read_text())
    s=json.loads((p/'candidate.json').read_text())
    gt=g['tensors'];st={int(t['index']):t for t in s['tensors']}
    gp={str(t):o for o in g['operators'].values() for t in o['outputs']}
    sp={t:i for i in s['instructions'] for t in i['output_tensors']}
    def graph_root(tid):
        chain=[]
        while str(tid) in gp:
            o=gp[str(tid)]
            chain.append(o['name'])
            assert len(o['inputs'])==1,(tid,o['name'])
            tid=o['inputs'][0]
        return tid,chain
    paths={};all_trf=set()
    for dot in g['operators'].values():
        d=dot['description']
        if 'operation_unit:' not in d or 'lookup_table:' not in d:continue
        wid=next(t for t in dot['inputs'] if gt[str(t)]['element_type']=='Float8e4m3')
        root,weight_chain=graph_root(wid)
        io=gt[str(root)]['optionals']['address']
        label={'IO(1)+0':'Q','IO(2)+0':'K','IO(3)+0':'V'}[io]
        w=gt[str(wid)];address=int(w['optionals']['address'])*512
        matching_ids={tid for tid,t in st.items() if t['buffer_type']=='Sram' and t['size']==w['size'] and t['address']==address}
        sd=next(i for i in s['instructions'] if '.contract_outer' in i['description'] and matching_ids.intersection(i['input_tensors']))
        tf=next(t for t in dot['inputs'] if gt[str(t)]['buffer_type']=='Trf');all_trf.add(tf)
        sv=next(t for t in dot['inputs'] if gt[str(t)]['buffer_type']=='Vrf')
        scale_root,scale_chain=graph_root(sv)
        assert gt[str(scale_root)]['optionals']['address']=={'Q':'IO(4)+0','K':'IO(5)+0','V':'IO(6)+0'}[label]
        assert d.index('operation_unit:')<d.index('FpMul0:')<d.index('cast_compaction:')
        assert gt[str(dot['outputs'][0])]['element_type']=='Bfloat16'
        assert 'commit_in_size: 8B (=4EA)' in d
        lut=next(t for t in sd['input_tensors'] if st.get(t,{}).get('size')==8388608)
        lut_sub=sp[lut];lut_dma=sp[lut_sub['input_tensors'][0]]
        # Find the actual weight HBM DMA before zero-cost reshape aliases.
        sw=next(t for t in sd['input_tensors'] if t in matching_ids)
        dw=sp[sw]
        while dw['tpe']!='DmaLoad':
            assert len(dw['input_tensors'])==1
            dw=sp[dw['input_tensors'][0]]
        paths[label]={'graph_dot':dot['name'],'graph_weight_chain':weight_chain,
          'graph_scale_chain':scale_chain,'weight_hbm':io,
          'graph_activation_trf':tf,'weight_dma':interval(dw),'dot':interval(sd),
          'lookup_dma':interval(lut_dma),'lookup_sub':interval(lut_sub),
          'lookup_schedule_tensor':lut,'lookup_dm_address':st[lut]['address'],
          'lookup_dm_lifetime':st[lut]['lifetime'],
          'scale_then_first_bf16_cast':True,'bf16_commit_bytes':8}
    assert set(paths)=={'Q','K','V'}
    trfs=[t for t in s['tensors'] if t['buffer_type']=='Trf']
    loads=[i for i in s['instructions'] if '.to_trf()' in i['description'] and i['tpe']=='Sub']
    durations=collections.Counter()
    for i in s['instructions']:durations[i['tpe']]+=i['lifetime']['end']-i['lifetime']['begin']
    return {'makespan':max(i['lifetime']['end'] for i in s['instructions']),
      'instruction_count':len(s['instructions']),'duration_sum_by_resource':dict(durations),
      'graph_activation_trf_ids':sorted(all_trf),'schedule_trfs':trfs,
      'trf_loads':[interval(i) for i in loads],
      'trf_reservation_bytes_per_slice':sum(t['size'] for t in trfs)//512,
      'paths':paths}

baseline=analyze('qkv-bf16-fused-channel-scale-e54')
candidate=analyze('qkv-bf16-shared-trf-e55')
assert baseline['makespan']==51293 and candidate['makespan']==52562
assert len(candidate['graph_activation_trf_ids'])==len(candidate['schedule_trfs'])==len(candidate['trf_loads'])==1
assert candidate['duration_sum_by_resource']['Sub']==baseline['duration_sum_by_resource']['Sub']-1223
for resource in baseline['duration_sum_by_resource']:
    if resource!='Sub':assert candidate['duration_sum_by_resource'][resource]==baseline['duration_sum_by_resource'][resource]
assert candidate['trf_reservation_bytes_per_slice']==32768
assert baseline['trf_reservation_bytes_per_slice']==65536
assert candidate['paths']['Q']['lookup_dm_address']==candidate['paths']['K']['lookup_dm_address']
assert candidate['paths']['Q']['lookup_dm_lifetime']['end']<=candidate['paths']['K']['lookup_dm_lifetime']['begin']
assert candidate['paths']['K']['dot']['begin']-baseline['paths']['K']['dot']['begin']==1269
assert candidate['paths']['V']['dot']['begin']-baseline['paths']['V']['dot']['begin']==1269
report={'decision':'REJECT_STATIC_REGRESSION_NO_NPU','baseline':baseline,'candidate':candidate,
 'delta_cycles':1269,'delta_percent':100*(52562/51293-1),
 'critical_observation':'Candidate K weight DMA ends31553 then lookup DMA838+Sub1293+79 gap precedes dot33763. Baseline table ready6163, K weight ends32494 and dot begins immediately. Net1269 propagates through V/cache tail.',
 'causal_limit':'Q/K lookup DM address reuse and changed preparation are observed; why the allocator/scheduler selected them is not established.',
 'scope':'Static schedule/graph/source only; no NPU or all-input equivalence proof.'}
(HERE/'static-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'decision':report['decision'],'baseline':51293,'candidate':52562,
 'delta_percent':report['delta_percent'],'trf_count':1,'reservation_bytes_per_slice':32768},indent=2))
