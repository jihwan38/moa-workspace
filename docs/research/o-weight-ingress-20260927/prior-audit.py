#!/usr/bin/env python3
"""Prior O evidence audit. --check is read-only; no build or NPU."""
import hashlib,itertools,json,re,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).with_suffix('.json')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(rel): return json.loads((ROOT/rel).read_text())
ids=['E2-O','E3-rows','E3-direct','E1-hier-O','E8-O','E9-O','E14-O','E24','E26','E29-O','E31','E34','E47','E53','E64','E111','E112','E113','E114','E115']
entries={x['id']:x for x in read('docs/experiments/catalog.json')['entries']}
reports={}
extra=['docs/experiments/c12-pipeline-screen-e104-after.json','docs/experiments/m01-remap-screen-e104-after.json','docs/experiments/m02-khalf-screen-e104-after.json']
paths=set(extra)
for eid in ids:
    paths.update([entries[eid]['report'],entries[eid]['reference']])
for rel in sorted(paths):
    j=read(rel); rows=[]
    for r in j['runs']:
        lp=ROOT/r['log']; raw=lp.read_text(); stage=r.get('batch_stage'); segment=raw
        if stage:
            segment=raw.split('STRUCTURAL_BEGIN '+stage+'\n',1)[1].split('STRUCTURAL_END '+stage,1)[0]
        match=re.search(r'==> sliding_attention_output\n(.*?)(?=\n==>|\Z)',segment,re.S)
        assert match, (rel,stage)
        m=re.search(r'median cycles=(\d+) \(of \d+ runs: (\[[^\]]+\])\)',match.group(1))
        assert m, rel
        rk=r['kernels']['sliding_attention_output']; cycles=json.loads(m.group(2))
        assert int(m.group(1))==rk['median_cycles'] and cycles==rk['cycles_by_run'],rel
        assert sha(lp)==r['log_sha256'],rel
        rows.append({'job_id':r['job']['id'],'batch_stage':stage,'log':r['log'],'log_sha256':r['log_sha256'],'log_hash_verified':True,'raw_stage_o_median_verified':True,'o_median':rk['median_cycles'],'o_cycles':cycles,'o_comparisons_passed':rk.get('output_comparisons_passed'),'o_correctness':rk['correctness'],'total_pass_lines_in_stage':segment.count('-> PASS'),'total_fail_lines_in_stage':segment.count('-> FAIL')})
    reports[rel]={'report_sha256':sha(ROOT/rel),'median':j['aggregate']['sliding_attention_output']['median_cycles'],'runs':rows,'provenance_source_binary_relationship':j.get('provenance',{}).get('source_binary_relationship')}
comparisons=[]
for eid in ids:
    e=entries[eid]; val=reports[e['report']]['median']; ctrl=reports[e['reference']]['median']; controls=[ctrl]
    if eid=='E111': controls.append(reports[extra[0]]['median'])
    if eid in ['E112','E113']: controls.append(reports[extra[1]]['median'])
    if eid in ['E114','E115']: controls.append(reports[extra[2]]['median'])
    comp=statistics.median(controls)
    comparisons.append({'id':eid,'method':e['method'],'report':e['report'],'reference_report':e['reference'],'median':val,'reference_medians':controls,'reference_comparison':comp,'change_percent':100*(val/comp-1),'comparison_scope':'bracketing controls in one allocation' if len(controls)==2 else 'historical report median, not randomized contemporary pair','note':e.get('note')})

# Enumerate emitted DMA descriptors of the already-frozen E24 row0/2/4 graphs.
# All final singleton dimensions represent contiguous packet byte count, not a loop.
descriptors=[]
for name,factor in [('o-row-control',1),('o-row-two',2),('o-row-four',4)]:
    rel=f'moa-dev/experiments/artifacts/output-row-interleave-schedules/{name}-graph.json'; g=read(rel)
    matches=[(nid,op) for nid,op in g['operators'].items() if op['kind']=='Dma.DtoS' and any(g['tensors'][str(t)]['size']==15728640 for t in op['inputs'])]
    assert len(matches)==1; nid,op=matches[0]; desc=op['description']; srcs={};dsts={}
    for line in desc.splitlines():
        m=re.match(r'(src|dst)(\d+): \[([^\]]+)\], (.*)',line)
        if not m: continue
        vec=[(s,int(n),int(st)) for s,n,st in re.findall(r'([^ ,=]+)=(\d+):(\d+)',m.group(3))]
        (srcs if m[1]=='src' else dsts)[int(m[2])]=(vec,m[4])
    assert len(srcs)==len(dsts)==8
    source_packets=set(); dest_packets=set(); totals=0; mapping_errors=0; max_local=0
    details=[]
    for i in sorted(srcs):
        sv,sbase=srcs[i];dv,dbase=dsts[i]
        assert [x[1] for x in sv]==[x[1] for x in dv]
        assert sv[-1][1:]==dv[-1][1:]==(1,256)
        soff=int(re.search(r'in_chip=(0x[0-9A-Fa-f]+)',sbase)[1],16)
        cm=re.search(r'cluster=(\d+), pe=(\d+), dmn=(\d+), slice=(\d+), in_slice=(0x[0-9A-Fa-f]+)',dbase)
        cl,pe,dmn,sl=[int(x) for x in cm.groups()[:4]];doff=int(cm[5],16)
        assert dmn==0 and sl==0
        for v in itertools.product(*(range(x[1]) for x in sv[:-1])):
            sa=soff+sum(a*b[2] for a,b in zip(v,sv)); da=doff+sum(a*b[2] for a,b in zip(v,dv))
            local_slice=pe*64+da//4194304; local_byte=da%4194304
            h,q=divmod(sa,4096); c=h//1920; hr=h%1920;k=q//512
            group=hr//(60*factor); rowlane=hr%factor; temporal=(hr//factor)%60
            expected_slice=(group*factor+rowlane)*8+k
            if (cl,local_slice,local_byte)!=(c,expected_slice,temporal*512+q%512):mapping_errors+=1
            assert sa%256==0 and local_byte%256==0
            source_packets.add(sa);dest_packets.add((cl,local_slice,local_byte));totals+=256;max_local=max(max_local,local_byte+256)
        details.append({'index':i,'src_dimensions':sv,'src_base':sbase,'dst_dimensions':dv,'dst_base':dbase})
    assert len(source_packets)==len(dest_packets)==61440 and totals==15728640 and mapping_errors==0
    assert source_packets==set(range(0,15728640,256)) and max_local==30720
    assert dest_packets=={(c,s,b) for c in range(2) for s in range(256) for b in range(0,30720,256)}
    descriptors.append({'variant':name,'graph':rel,'graph_sha256':sha(ROOT/rel),'graph_node':nid,'operator_name':op['name'],'inputs':op['inputs'],'outputs':op['outputs'],'row_interleave':factor,'descriptors':details,'packet_bytes':256,'distinct_source_packets':len(source_packets),'source_range_exclusive':[0,15728640],'distinct_destination_packets':len(dest_packets),'destination_slices':512,'destination_per_slice_range_exclusive':[0,30720],'mapping_errors':mapping_errors,'status':'PASS','coverage':'Every emitted 256B packet enumerated; all input bytes once, expected H/K producer mapping and all local DM packets once. Does not model transfer order, HBM bank/controller scheduling, physical transaction volume or NPU performance; consumer correctness remains logical source proof/limited NPU.'})

pair_rows=set(); pair_outputs=set()
for c,g,r,t,j in itertools.product(range(2),range(4),range(4),range(60),range(2)):
    h=1920*c+480*g+8*t+2*r+j
    assert h//480==4*c+g and h%480==8*t+2*r+j
    pair_rows.add(h)
for c,g,r,t in itertools.product(range(2),range(4),range(4),range(60)):
    # One contiguous source F32 row pair goes to one destination F32 row pair.
    source=(c,64*g+16*r,8*t)
    target=(4*c+g,32*t+8*r)
    pair_outputs.add(target)
assert pair_rows==set(range(3840))
assert pair_outputs=={(ep,b) for ep in range(8) for b in range(0,1920,8)}
proposal={'name':'pair-preserving four-way H ingress','classification':'unexecuted exact mapping; reopening an existing row-interleave family','h':'1920*c+480*g+8*t+2*r+j','slice':'64*g+16*r+q','ranges':{'c':2,'g':4,'r':4,'q':16,'t':60,'j':2},'valid_weight_bytes_per_slice':30720,'k_shards':16,'f32_result_bytes':15360,'logical_contiguous_gather_units':{'control':{'count':32,'bytes_each':480},'candidate':{'count':1920,'bytes_each':8},'unit_count_ratio':60},'logical_row_coverage':3840,'logical_pair_destination_coverage':1920,'scope_limit':'Python logical geometry only; no candidate source/compile/descriptor, support, bounds in emitted graph or NPU result. Eight-byte units are predicted source/destination contiguous units, not measured HBM/TDMA transactions.'}
result={'schema_version':1,'scope':'read-only existing O experiment/source/descriptor audit; no compile or Arena','head':'48bea3dea8d3b98b4f66a15981c30646d40ddfd0','baseline_current':read('baseline/current.json'),'comparisons':comparisons,'reports':reports,'row_interleave_descriptor_audit':descriptors,'proposed_pair4_logical_geometry':proposal,'source_provenance_checks':{'m01_verify_check':'PASS; 4 frozen sources, 3840 row reduction tags, 5 builds and stored Arena/package verification','m02_verify_check':'PASS; 4 frozen sources, 2 ownership maps, 6 builds, 4 Arena stages/60 comparisons'},'calculated_costs':{'weight_payload_bytes':3840*4096,'e104_tile_bytes':120*256,'old_8col_tile_bytes':60*512,'old_4col_sweep_tile_bytes':30*1024,'old_2col_sweep_and_m02_tile_bytes':15*2048,'m01_compact_partial_bytes':512*120*4,'m01_additional_valid_dm_edges_dma':983040,'m01_additional_valid_dm_edges_switch':491520,'m02_padded_partial_bytes':512*15*8,'m02_valid_partial_bytes':512*15*4,'m02_activation_replica_bytes':512*2048*2,'e104_activation_replica_bytes':512*256*2,'c09_static_percent':100*(27432/27239-1),'scale_fusion_static_percent':100*(27393/27239-1),'m02_v4_vs_v3_static_percent':100*(32382/37413-1),'m02_v4_vs_v3_npu_percent':100*(89203/78981-1)}}
if '--check' in sys.argv:
    assert json.loads(OUT.read_text()) == json.loads(json.dumps(result)), 'prior audit differs from preserved evidence'
    print('PASS: 34 reports, raw logs, historical descriptors and proposed row-pair coverage')
    raise SystemExit(0)
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'json':str(OUT),'reports':len(reports),'unique_jobs':len({r['job_id'] for j in reports.values() for r in j['runs']}),'verified_report_runs':sum(len(j['runs']) for j in reports.values()),'row_descriptor_packets_checked':sum(j['distinct_source_packets'] for j in descriptors),'comparisons':comparisons},ensure_ascii=False,indent=2))
