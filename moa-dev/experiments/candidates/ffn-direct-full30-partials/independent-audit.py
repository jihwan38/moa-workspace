#!/usr/bin/env python3
"""Independent frozen E142 full30 F32partial source/descriptor/register gate.
No hardware arithmetic or performance is inferred from this graph check.
"""
import argparse,hashlib,importlib.util,itertools,json
from pathlib import Path

def helper_bounds(s):
    a=s.index('fn project_rows_with_scale_contraction(');b=s.index('pub(crate) fn project_up_and_gate(',a)
    return a,b

def analyze(root,candidate,label):
    p=root/'moa-dev/experiments/candidates'/candidate
    old=root/'moa-dev/experiments/candidates/ffn-independent-direct-copy-repair'
    spec=importlib.util.spec_from_file_location('e141',old/'independent-audit.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    h=prior.h;require=h.require;h.ROOT=root;h.HERE=p;h.ARTIFACTS=root/'moa-dev/experiments/artifacts'/candidate
    build,paths,recs=h.provenance(label)
    source=paths['source'].read_text();control=(old/'direct-six-row-copy-v1/mlp.rs').read_text()
    a,b=helper_bounds(source);c,d=helper_bounds(control)
    require(source[:a]+source[b:]==control[:c]+control[d:],'Only helper may change from E141')
    expected='6ebb74ca9e651e4b09e161206829e6ca6de3fd05d5a06e67b97627f6b2cd455c'
    require(hashlib.sha256(source.encode()).hexdigest()==expected,'Prepared frozen full30 source differs; any supported repair requires a separately frozen checker')
    g=h.Graph(json.loads(paths['graph'].read_text()));schedule=json.loads(paths['schedule'].read_text())
    prefix=prior.prefix_and_bitmap(g,schedule,paths['exact_image']);ingress=h.hbm_ingress(g)
    owned={}
    for item in ingress:
        mapping=g.mappings(item['output'])
        if mapping['Partitioning']!={'Symbol':{'symbol':'FfnSlice','size':256}}:continue
        require(mapping['Cluster']=={'Symbol':{'symbol':'Dummy2','size':2}},'Cluster ownership')
        rb=1920 if item['type']=='Float4e2m1' else 240
        for pair in item['descriptors']:
            src,dst=pair['source'],pair['destination'];sa,da=src['address'],dst['address']
            rr=list(h.intervals(sa['in_chip'],src['dims']));ww=list(h.intervals(da['in_slice'],dst['dims']));require(len(rr)==len(ww),'Ingress packet match')
            for (lo,hi),(dl,dh) in zip(rr,ww):
                sl=da['pe']*64+da['dmn']*32+da['slice']+dl//4194304;off=dl%4194304
                require(hi-lo==dh-dl and 0<=sl<256 and off+dh-dl<=30*rb,'Compact source/destination extent')
                require(lo==(da['cluster']*7680+sl*30)*rb+off,'Compact original row ownership')
        owned[item['output']]=item
    require(len(owned)==4,'Four independent up/gate public ingress matrices')
    producers=[];partials={}
    for o in g.ops.values():
        d=o['description']
        if o['kind']!='TuExec' or o['context']!='Main' or 'lookup_table:' not in d or 'FfnRow_1=30' not in d:continue
        require('LReduce(+) [axes=H]' not in d,'Full30 native must retain block axis')
        require(len(o['inputs'])==4,'Full30 packed/TRF/inputscale/table binding')
        weight,trf,xs,tab=o['inputs'];out=o['outputs'][0]
        require(weight in owned and g.tensor(weight)['element_type']=='Float4e2m1','Direct compact full30 FP4')
        require(g.tensor(trf)['buffer_type']=='Trf' and g.tensor(trf)['size']==512*3*3840,'Original 3term TRF')
        require(g.tensor(xs)['buffer_type']=='Vrf' and g.tensor(xs)['size']==512*32,'Input scale32B')
        require(g.producer(tab,'StoTab')[1]['outputs']==[tab],'Lookup dependency')
        require(h.offsets(h.sequence(d,'fetch_sequencer'))==list(range(30*3840)),'Full30 exact weight read')
        require('FpMul0:' in d and 'FpMul1:' not in d and 'cast_compaction:' not in d,'Native -> F32xscale only')
        require(g.tensor(out)['element_type']=='Float32' and g.tensor(out)['size']==512*28800,'F32partial [30,240] extent')
        require(h.offsets(h.sequence(d,'commit_projection'))==list(range(7200)) and 'commit_in_size: 16B (=4EA)' in d,'Full partial 1800x16B')
        partials[out]=[];producers.append({'operator':o['name'],'partial':out,'weight':weight,'trf':trf,'input_scale_vrf':xs,'table':tab,'partial_bytes_per_slice':28800})
    require(len(producers)==2,'Two direct native/lookup producers')
    outputs={};scales={};consumers=[]
    for o in g.ops.values():
        d=o['description']
        if o['kind']!='TuExec' or o['context']!='Main' or 'LReduce(+) [axes=H]' not in d or 'FfnRow_1=6' not in d:continue
        require(len(o['inputs'])==3,'Partial/blockscale/output binding')
        pv,vrf,dst=o['inputs'];v=g.view(pv);ov=g.view(dst)
        require(v['parent'] in partials,'Read current native F32partial')
        sp=g.producer(vrf,'TuExec')[1];sv=g.view(sp['inputs'][0])
        require(sp['context']=='Sub' and g.tensor(vrf)['buffer_type']=='Vrf' and g.tensor(vrf)['size']==5760*512,'Sixrow blockscaleVRF')
        for view,stride in [(v,240),(sv,240),(ov,1)]:
            require((view['stride_elements'],view['axis_extent'],view['window'])==(stride,30,6),'Same sixrow window')
        require(v['start']==sv['start']==ov['start'] and v['start'] in [0,6,12,18,24],'Partial/scale/output same rows')
        require(sv['parent'] in owned,'Original FP8 blockscale ingress')
        require(h.offsets(h.sequence(d,'fetch_sequencer'))==list(range(1440)),'Sixrow partial read5760B')
        require(h.offsets(h.sequence(sp['description'],'fetch_sequencer'))==list(range(1440)),'Sixrow scale read1440FP8')
        require(d.index('FpMul0:')<d.index('LReduce(+) [axes=H]') and 'FpMul1:' not in d and 'cast_compaction:' not in d,'F32 blockscale before Hreduce')
        require(h.offsets(h.sequence(d,'commit_projection'))==list(range(6)) and 'commit_in_size: 8B (=2EA)' in d,'Three F32rowpair writes')
        require(g.tensor(o['outputs'][0])['element_type']=='Float32','No early BF16')
        rootout=g.root_output(ov['parent']);outputs.setdefault(rootout,[]).append(v['start']);partials[v['parent']].append(v['start']);scales.setdefault(sv['parent'],[]).append(v['start'])
        consumers.append({'operator':o['name'],'partial_view':v,'scale_view':sv,'output_view':ov,'partial_byte_bounds':[v['start']*960,(v['start']+6)*960],'blockscale_vrf':vrf})
    require(len(consumers)==10,'Ten separate blockscale/reduction consumers')
    for group in [outputs,partials,scales]:require(len(group)==2 and all(sorted(v)==[0,6,12,18,24] for v in group.values()),'Five disjoint sixrow windows eachprojection')
    # Preserve E141 canonical output gather and first BF16 position.
    def root_output(tid):
        _,o=g.producer(tid)
        if o['kind'] in ['LowLevelReshape','ClusterSync']:return root_output(o['inputs'][0])
        return g.root_output(tid)
    gathers=[]
    for o in g.ops.values():
        if o['kind']!='Dma.StoS':continue
        try:r=root_output(o['inputs'][0])
        except (KeyError,ValueError):continue
        if r not in outputs:continue
        count=0
        for pair in h.dma_descriptors(o['description']).values():
            sa,da=pair['src']['address'],pair['dst']['address'];require(sa['cluster']==da['cluster'],'Gather cluster unchanged')
            rr=list(h.intervals(sa['in_slice'],pair['src']['dims']));ww=list(h.intervals(da['in_slice'],pair['dst']['dims']));require(len(rr)==len(ww),'Gather packets')
            for (lo,hi),(dl,dh) in zip(rr,ww):
                ss=sa['pe']*64+sa['dmn']*32+sa['slice']+lo//4194304;ds=da['pe']*64+da['dmn']*32+da['slice']+dl//4194304
                require(hi-lo==dh-dl==120 and lo%4194304==0 and ds==ss//4*4 and dl%4194304==(ss%4)*120,'Canonical F32row gather');count+=1
        require(count==512,'All512F32rowpackets')
        cast=[x for x in g.ops.values() if x['kind']=='TuExec' and o['outputs'][0] in x['inputs']];require(len(cast)==1 and 'output_element_type: bf16' in cast[0]['description'],'FirstBF16 aftergather');gathers.append(o['name'])
    require(len(gathers)==2,'Two canonical gathers')
    ts={int(t['index']):t for t in schedule['tensors']};vrfs=[t for t in ts.values() if t['buffer_type']=='Vrf']
    for t in vrfs:require(0<=t['address'] and t['address']+t['size']<=8192*512,'VRF capacity')
    for a,b in itertools.combinations(vrfs,2):
        live=max(a['lifetime']['begin'],b['lifetime']['begin'])<min(a['lifetime']['end'],b['lifetime']['end']);overlap=max(a['address'],b['address'])<min(a['address']+a['size'],b['address']+b['size']);require(not(live and overlap),'WholeVRF physical overlap')
    for i in schedule['instructions']:
        for tid in i['input_tensors']:
            t=ts.get(tid)
            if t is None or t['buffer_type'] not in ['Vrf','Trf']:continue
            require(t['lifetime']['begin']<=i['lifetime']['begin']<=i['lifetime']['end']<=t['lifetime']['end'],'Register consumer lifetime')
    require(sum(o['kind']=='StoTab' for o in g.ops.values())==3,'Two upgate +one inherited down table')
    return {'passed':True,'label':label,'inputdigests':recs,'verifiers':[{'path':str(Path(__file__).resolve().relative_to(root)),'bytes':len(Path(__file__).read_bytes()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},h.record(old/'independent-audit.py'),h.record(old.parent/'ffn-native-down-weighted-residual/verify_graph.py')],'prefix':prefix,'ingress':ingress,'native_producers':producers,'partial_consumers':consumers,'canonical_gathers':gathers,'f32_partial_bytes_each_direction':29491200,'whole_vrf_buffers_checked':len(vrfs),'scope':'Actual compiled bounds/order/register contracts only. Separate Mul0 blockscale has same semantic placement as E141 fused Mul1; hardware arithmetic equivalence requires the finite oracle and whole validation. No speed or generic precision claim.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('label');p.add_argument('--output',type=Path);p.add_argument('--check',action='store_true');q=p.parse_args()
    here=Path(__file__).resolve().parent;root=here.parents[3]
    if q.label!='e142-full30-v1':raise ValueError('Only frozen E142 full30-v1 is supported')
    out=q.output or here/'independent-audit.json'
    r=json.loads(json.dumps(analyze(root,here.name,q.label)))
    if q.check:assert json.loads(out.read_text())==r
    else:
        if out.exists() and json.loads(out.read_text()).get('passed'):raise ValueError('Successful gate is immutable; use --check')
        out.write_text(json.dumps(r,indent=2)+'\n')
    print('PASS actualgraph',q.label)
if __name__=='__main__':main()
