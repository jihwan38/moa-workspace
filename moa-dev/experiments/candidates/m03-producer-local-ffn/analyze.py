#!/usr/bin/env python3
"""Audit logical coordinates and summarize emitted complete-FFN costs, not stalls."""
import hashlib,json
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
FAMILY='m03-producer-local-ffn'
BUILD='m03-native-local-v1'
def digest(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def coordinates():
    rows=[];down=[];scales=[]
    for c in range(2):
        for s in range(256):
            for t in range(30):
                local=30*s+t;l=7680*c+local
                r,u=divmod(local,120)
                assert r<64 and 4*r<256 and 7680*c+120*r+u==l
                rows.append(l)
            r,k=divmod(s,4)
            for t in range(60):
                h=60*r+t;l=7680*c+1920*k
                down.append(((h*15360+l)//2,(h*15360+l+1920)//2))
                scales.append((h*960+l//16,h*960+(l+1920)//16))
    assert sorted(rows)==list(range(15360))
    for spans,total in [(down,3840*15360//2),(scales,3840*15360//16)]:
        spans.sort();assert spans[0][0]==0 and spans[-1][1]==total
        assert all(a[1]==b[0] for a,b in zip(spans,spans[1:]))
    # Per-slice streams commit F32 pairs or >=four BF16 values: exact8B units.
    assert 30*4%8==120*2%8==60*4%8==0
    assert 6*240*4==12*120*4==5760
    return {'upgate_rows_per_projection':len(rows),'down_row_segments':len(down),
            'weight_and_scale_coverage':'every logical HBM element exactly once',
            'local_geglu_ownership':'same cluster for each of15360 activation values',
            'down_leader_logical_bytes':2*8*4*1920*2,
            'e104_down_leader_logical_bytes':2*8*8*1920*2,
            'down_replicated_logical_bytes':2*256*1920*2,
            'cluster_output_partial_logical_bytes':2*3840*4,
            'scope':'Source coordinate/alignment accounting, not arithmetic or actual bus traffic.'}
def analyze(path):
    s=json.loads(path.read_text());ins=s['instructions'];t={str(n['index']):n for n in s['tensors']}
    main=[n for n in ins if n['tpe']=='Main'];dma=[n for n in ins if n['tpe'].startswith('Dma')]
    expand=[n for n in main if 'fetch_table_lookup::<f8e' in n['description']]
    dots=[n for n in main if 'contract_outer' in n['description']]
    weights=[n for n in dma if n['tpe']=='DmaLoad' and 'src/device/shared/mlp.rs:' in n['description'] and any(x in n['description'] for x in ['weight_packed.to_dm','weight.to_dm'])]
    assert len(expand)==len(dots)==len(weights)==3
    am=[n for n in main if '0x7fff_ffff' in n['description']]
    rms=[n for n in main if 'src/device/shared/rmsnorm.rs:95:' in n['description']]
    assert len(am)==len(rms)==1
    def details(nodes):
        return [{'index':n['index'],'type':n['tpe'],**n['lifetime'],
                 'description':n['description'],
                 'outputs':[t[str(i)] for i in n['output_tensors']]} for n in nodes]
    transfers=[n for n in dma if n['tpe']=='DmaStos' and any(x in n['description'] for x in ['up.to_dm','gate.to_dm','down.to_dm'])]
    ring=[n for n in main if 'CustomBroadcast { ring_size: 32 }' in n['description']]
    expanded=[t[str(i)] for n in expand for i in n['output_tensors']]
    events=sorted([(v['lifetime']['begin'],v['size']) for v in expanded]+[(v['lifetime']['end'],-v['size']) for v in expanded])
    now=peak=0
    for _,delta in events:now+=delta;peak=max(peak,now)
    return {'schedule':digest(path.read_bytes()),'makespan':max(n['lifetime']['end'] for n in ins),
            'instruction_count':len(ins),'dma_count':len(dma),'weight_loads':details(weights),
            'fp8_expansion':details(expand),'dots':details(dots),'output_layout_transfers':details(transfers),
            'ring32_streams':details(ring),'pre_rms_to_amax_gap':am[0]['lifetime']['begin']-rms[0]['lifetime']['end'],
            'fp8_logical_write_bytes':3*3840*15360,'fp8_peak_live_allocated_bytes':peak,
            'scope':'Compiler allocation/intervals only, not hardware traffic/stalls. Overlapping intervals cannot be added as savings.'}
def result():
    control=HERE.parent/'e104-weight-path/evidence/profile-v1/schedule.json'
    candidate=HERE/'evidence'/BUILD/'schedule.json'
    a,b=analyze(control),analyze(candidate)
    return {'control':a,'candidate':b,'coordinates':coordinates(),'static_change_percent':100*(b['makespan']/a['makespan']-1)}
if __name__=='__main__':
    a=ROOT/'moa-dev/experiments/artifacts'/FAMILY/BUILD/'schedule.json';b=HERE/'evidence'/BUILD/'schedule.json'
    if b.exists():assert a.read_bytes()==b.read_bytes()
    else:shutil.copyfile(a,b)
    value=result();(HERE/'static-analysis.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:{n:v[n] for n in ['makespan','dma_count','pre_rms_to_amax_gap','fp8_peak_live_allocated_bytes']} for k,v in value.items() if k in ['control','candidate']},indent=2))
