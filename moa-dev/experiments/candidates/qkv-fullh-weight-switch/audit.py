#!/usr/bin/env python3
"""Independent E144 full-H Switch source/descriptor audit. No compile or NPU calls."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / 'moa-dev/experiments/candidates/qkv-native-consumer-local'
HELPER = ROOT / 'moa-dev/experiments/candidates/ffn-native-down-weighted-residual/verify_graph.py'
spec = importlib.util.spec_from_file_location('e144_graph_helper', HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
M = 4194304

def record(p):
    b = p.read_bytes()
    return {'path': str(p.relative_to(ROOT)), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}

def digest(p):
    return {k:v for k,v in record(p).items() if k != 'path'}

def load(p):
    return json.loads(p.read_text())

def sl(a, offset):
    return a['pe']*64 + a['dmn']*32 + a['slice'] + offset//M

def packets(op):
    for n, pair in sorted(h.dma_descriptors(op['description']).items()):
        a,b = pair['src'],pair['dst']; sa,da=a['address'],b['address']
        aa=list(h.intervals(sa.get('in_chip',sa.get('in_slice')),a['dims']))
        bb=list(h.intervals(da.get('in_chip',da.get('in_slice')),b['dims']))
        assert len(aa)==len(bb)
        for x,y in zip(aa,bb):
            yield n,sa,da,x,y

def signature(g,op):
    desc=re.sub(r'#O\d+', '#O', op['description'])
    tensors=[(g.tensor(t)['size'],g.tensor(t)['element_type'],g.tensor(t)['shape']) for t in op['inputs']+op['outputs']]
    return json.dumps([op['kind'],op['context'],desc,tensors],sort_keys=True)

def audit(label):
    assert label == 'e144-fullh-v2', 'This audit freezes the corrected low-two-bit owner, not v1.'
    bpath=HERE/f'build-{label}-repro.json'; b=load(bpath)
    assert b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container']
    assert b['base_commit']=='b3fe0ab4be2a5ece6382ae5745112722200f1b66'
    variant=HERE/b['variant']; ev=HERE/'evidence'/label
    art=ROOT/'moa-dev/experiments/artifacts'/HERE.name/label
    bound=[bpath,variant/'projection.rs',variant/'ops.rs',variant/'recipe.json',ev/'graph.json',ev/'schedule.json',OLD/'native-v2/projection.rs',OLD/'native-v2/ops.rs',OLD/'evidence/e138-native-v2/graph.json',HELPER]
    for name in ('graph','schedule'):
        assert digest(ev/f'{name}.json')==b[name]
        assert (art/f'{name}.json').read_bytes()==(ev/f'{name}.json').read_bytes()
    assert digest(art/'exact-kernel.bin')==b['exact_image']
    before=load(ROOT/'moa-dev/experiments/candidates/ffn-direct-full30-partials/measured-build-e142-adoption.json')
    assert len(b['source_inputs'])==46 and b['source_inputs'].keys()==before['source_inputs'].keys()
    assert {x for x in b['source_inputs'] if b['source_inputs'][x]!=before['source_inputs'][x]}=={'src/device/sliding/projection.rs','src/ops.rs'}
    for name,d in b['source_inputs'].items(): assert digest(art/'crate'/name)==d
    for name,file in [('src/device/sliding/projection.rs','projection.rs'),('src/ops.rs','ops.rs')]:
        assert digest(variant/file)==b['source_inputs'][name]
        assert (variant/file).read_bytes()==(ev/name).read_bytes()
    new=(variant/'projection.rs').read_text(); old=(OLD/'native-v2/projection.rs').read_text(); restored=new
    for axis,R in [('QkvQueryRow',8),('QkvKvRow',4)]:
        changed=f'let weight: DmTensor<f8e4m3, Chip, QkvCluster, m![QkvGroup, {axis} % 4], m![{axis} / 4, H]> = weight.to_dm(&mut device.tdma);'
        original=f'let weight: DmTensor<f8e4m3, Chip, QkvCluster, QkvColumns, m![{axis}, H % 960]> = weight.to_dm(&mut device.tdma);'
        assert restored.count(changed)==1
        restored=restored.replace(changed,original)
        changed=f'.begin(weight.view()).fetch::<m![H / 960, {axis} / 4], m![H % 960]>()\n        .switch::<QkvColumns, m![{axis}]>(SwitchConfig::InterTranspose {{ slice1: 4, slice0: 1, time0: {R} }})'
        original=f'.begin(weight.view()).fetch::<m![{axis}, H / 32 % 30], m![H % 32]>()'
        assert restored.count(changed)==1
        restored=restored.replace(changed,original)
    assert restored==old, 'Unexpected source arithmetic/head/suffix changes'
    assert (variant/'ops.rs').read_bytes()==(OLD/'native-v2/ops.rs').read_bytes()
    g=h.Graph(load(ev/'graph.json')); og=h.Graph(load(OLD/'evidence/e138-native-v2/graph.json'))
    o={int(x['name'].split('#O')[-1]):x for x in g.ops.values()}
    oo={int(x['name'].split('#O')[-1]):x for x in og.ops.values()}
    weight=[]; native=[]; changed_nodes=set(); total_row_packets=0
    for name,axis,R,wn,nn in [('Q','QkvQueryRow',8,78,79),('K','QkvKvRow',4,91,92),('V','QkvKvRow',4,104,105)]:
        rows=4*R; w=o[wn]; n=o[nn]; changed_nodes|={wn,nn}
        assert w['kind']=='Dma.DtoS' and g.tensor(w['inputs'][0])['optionals']['address']=='IO(%d)+0' % {'Q':1,'K':2,'V':3}[name]
        spans=[]; dest=set()
        for _,sa,da,(lo,hi),(dl,dh) in packets(w):
            assert sa['chip']==da['chip']==0
            c=da['cluster']; s=sl(da,dl); group,owner=divmod(s,4); local=dl%M; r,u=divmod(local,3840)
            assert 0<=s<256 and 0<=r<R and 0<=u<3840 and u%256==0
            assert hi-lo==dh-dl==256
            assert lo==(c*64*rows+group*rows+4*r+owner)*3840+u
            spans.append((lo,hi)); dest.add((c,s,local))
        coverage=h.full_coverage(spans,128*rows*3840,name+' original HBM')
        assert len(dest)==512*R*15 and g.tensor(w['outputs'][0])['size']==128*rows*3840
        weight.append({'projection':name,'operator':w['name'],'packet_bytes':256,'source_coverage':coverage,'per_slice_bytes':R*3840,'source_coordinate':f'{64*rows}*c+{rows}*g+4*r+b','descriptor':h.dma_descriptors(w['description'])})
        assert n['kind']=='TuExec' and n['context']=='Main' and n['inputs'][0]==w['outputs'][0]
        d=n['description']; assert f'InterTranspose {{ dim0_volume: 1, dim1_volume: 4, dim0_chunk_volume: {R} }}' in d
        assert d[d.index('operation_unit:'):].replace(n['name'],'NATIVE')==oo[nn]['description'][oo[nn]['description'].index('operation_unit:'):].replace(oo[nn]['name'],'NATIVE')
        assert g.tensor(n['outputs'][0])['shape']==og.tensor(oo[nn]['outputs'][0])['shape']
        assert [g.tensor(t)['shape'] for t in n['inputs'][1:]]==[og.tensor(t)['shape'] for t in oo[nn]['inputs'][1:]]
        assert 'lookup_table:' not in d and 'cast_compaction:' not in d
        assert h.offsets(h.sequence(d,'commit_projection'))==list(range(rows)) and 'commit_in_size: 8B (=2EA)' in d
        fetch=h.offsets(h.sequence(d,'fetch_sequencer'))
        expect=[3840*r+960*q+u for q in range(4) for r in range(R) for u in range(960)]
        assert fetch==expect, (name,'Actual pre-switch DM reader order differs')
        seen=set(); destinations=set()
        for c in range(2):
            for group in range(64):
                for owner in range(4):
                    for q in range(4):
                        for r in range(R):
                            row=c*64*rows+group*rows+4*r+owner
                            src=(c,4*group+owner,q*R+r)
                            dst=(c,4*group+q,4*r+owner)
                            assert row==c*64*rows+group*rows+dst[2]
                            assert (3840*r+960*q)+959<R*3840
                            seen.add((row,q));destinations.add(dst)
        assert len(seen)==len(destinations)==128*rows*4
        total_row_packets+=len(seen)
        native.append({'projection':name,'operator':n['name'],'switch':{'slice1':4,'slice0':1,'time0':R},'row_quarter_packets_bijection':len(seen),'packet_bytes':960,'u_affine_interval':[0,960],'reader_elements_checked':len(fetch),'output_rows_per_group':rows,'post_switch_operation_unit_and_commit_identical_to_e138_v2':True})
    assert total_row_packets==32768
    # All other emitted operators, tensor shapes, scale/head/cache paths are exact matches.
    untouched=Counter(signature(g,x) for num,x in o.items() if num not in changed_nodes)
    inherited=Counter(signature(og,x) for num,x in oo.items() if num not in changed_nodes)
    assert untouched==inherited, 'Unexpected changed operators outside three weight ingress/native readers'
    schedule=load(ev/'schedule.json'); ts={int(t['index']):t for t in schedule['tensors']}; regs=[]
    for ins in schedule['instructions']:
        if ins['tpe']!='Main' or '.contract_outer::' not in ins['description']:continue
        rr=[ts[t] for t in ins['input_tensors'] if ts[t]['buffer_type'] in ('Trf','Vrf')]
        assert sorted(t['buffer_type'] for t in rr)==['Trf','Vrf']
        for t in rr:assert t['lifetime']['begin']<=ins['lifetime']['begin']<=ins['lifetime']['end']<=t['lifetime']['end']
        assert 'InterTranspose' in ins['description']
        regs.append({'instruction':ins['index'],'lifetime':ins['lifetime'],'registers':rr})
    assert len(regs)==3 and len({tuple(t['index'] for t in row['registers']) for row in regs})==1
    return {'passed':True,'candidate':'E144','label':label,'scope':'Source, actual emitted descriptors/reader/Switch/numerical order/output shape and scheduled register lifetime. NPU correctness/performance separate.','inputdigests':{str(p.relative_to(ROOT)):record(p) for p in bound},'source_composition':{'e142_inputs':44,'e138_ops':1,'e138_projection_with_only_weight_ingress_reader_changes':1,'old_mlp_not_reintroduced':True},'row_quarter_bijection':total_row_packets,'weight_ingress':weight,'native_switch':native,'unchanged_emitted_operators':sum(untouched.values()),'scheduled_shared_register_lifetimes':regs,'cost_scope':{'original_weight_hbm_bytes':31457280,'original_weight_dm_write_bytes':31457280,'original_weight_dm_read_bytes':31457280,'new_full_weight_dm_transpose_bytes':0,'new_weight_trf_bytes':0,'new_work':'Three fused Main Switch routes and their setup/collect/backpressure; local DM read order changed. No latency claim.'},'limits':['Exact lowering and scheduled lifetime are not proof of hidden SFR read-index/bank timing or NPU arithmetic.','Graph exposes native input register shapes and subsequent VE order; hidden DPE SFR read-index is not decoded from the image here.','Finite source precision validation must be rebound to this exact image; no inherited NPU PASS.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('label',nargs='?',default='e144-fullh-v2');p.add_argument('--check',action='store_true');a=p.parse_args()
    result=json.loads(json.dumps(audit(a.label)));path=HERE/f'audit-{a.label}.json'
    if a.check:assert load(path)==result,'Audit record differs'
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',path)
if __name__=='__main__':main()
