#!/usr/bin/env python3
"""Independent E149 actual descriptor/reader/Switch/output/register gate. Offline only."""
import argparse,hashlib,importlib.util,json,re,subprocess,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OLD=HERE.parent/'o-native-activation-resident';E147=HERE.parent/'qkv-scalar-sub-vrf'
spec=importlib.util.spec_from_file_location('e144_helpers',HERE.parent/'qkv-fullh-weight-switch/audit.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);h=a.h

def load(p):return json.loads(p.read_text())
def rec(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def digest(p):return {k:v for k,v in rec(p).items() if k!='path'}
def nodes(g):return {int(v['name'].split('#O')[-1]):v for v in g.ops.values()}
def audit(label):
 assert label=='e149-ingress-v1'
 bp=HERE/f'build-{label}-repro.json';b=load(bp);ev=HERE/'evidence'/label;art=ROOT/'moa-dev/experiments/artifacts'/HERE.name/label
 ref=E147/'measured-build-e147-adoption.json';base=load(ref);oldref=OLD/'measured-build-e137-adoption.json';older=load(oldref)
 assert b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container'] and b['base_commit']=='fb45202bba3c24537f74b4bf463276b76d823773'
 assert len(b['source_inputs'])==46 and b['source_inputs'].keys()==base['source_inputs'].keys()
 assert {k for k in b['source_inputs'] if b['source_inputs'][k]!=base['source_inputs'][k]}=={'src/device/sliding/projection.rs'}
 assert base['images']['sliding_attention_output']==older['images']['sliding_attention_output']
 for name,d in b['source_inputs'].items():assert digest(art/'crate'/name)==d
 src=HERE/b['variant']/'projection.rs';oldsrc=E147/'scalar-v1/projection.rs'
 assert digest(src)==b['source_inputs']['src/device/sliding/projection.rs'] and src.read_bytes()==(ev/'src/device/sliding/projection.rs').read_bytes()
 for n in ('graph','schedule'):
  assert digest(ev/f'{n}.json')==b[n] and (ev/f'{n}.json').read_bytes()==(art/f'{n}.json').read_bytes()
 assert digest(art/'exact-kernel.bin')==b['exact_image']
 old=oldsrc.read_text();new=src.read_text();changed='OutputCluster, m![H / 120 % 16, Qs / 1024, H % 120 = 60 % 4], m![H % 120 = 60 / 4, Qs % 1024]';original='OutputCluster, OutputColumns, m![H % 120 = 60, Qs % 256]'
 assert new.count(changed)==3
 restored=new.replace(changed,original);part='.fetch::<m![Qs / 256 % 4, H % 120 = 60 / 4], m![Qs % 256]>()\n        .switch::<OutputColumns, m![H % 120 = 60]>(SwitchConfig::InterTranspose { slice1: 4, slice0: 1, time0: 15 })'
 assert restored.count(part)==1 and restored.replace(part,'.fetch::<m![H % 120 = 60, Qs / 32 % 8], m![Qs % 32]>()')==old,'Unexpected arithmetic/source change'
 subprocess.run([sys.executable,str(HERE/'geometry.py'),'--check'],check=True,stdout=subprocess.DEVNULL)
 g=h.Graph(load(ev/'graph.json'));og=h.Graph(load(OLD/'evidence/e137-native-v3/graph.json'));o=nodes(g);oo=nodes(og)
 changednodes={70,73,78,83};nc=Counter(a.signature(g,x)for k,x in o.items()if k not in changednodes);oc=Counter(a.signature(og,x)for k,x in oo.items()if k not in changednodes);assert nc==oc,'Unexpected emitted change outside weight routing'
 allspans=[];weight=[];native=[]
 for wn,nn,offset in [(70,78,0),(73,83,60)]:
  w,n=o[wn],o[nn];v=g.view(w['inputs'][0]);ov=og.view(oo[wn]['inputs'][0]);assert all(v[k]==ov[k]for k in ('stride_elements','axis_extent','window','start'))
  assert (v['stride_elements'],v['axis_extent'],v['window'],v['start'])==(4096,120,60,offset)
  root=g.tensor(v['parent']);assert root['optionals']['address']==og.tensor(ov['parent'])['optionals']['address']=='IO(2)+0'
  dest=set();span=[]
  for _,sa,da,(lo,hi),(dl,dh)in a.packets(w):
   assert sa['chip']==da['chip']==0 and hi-lo==dh-dl==256
   c=da['cluster'];s=a.sl(da,dl);group,remainder=divmod(s,16);quarter,owner=divmod(remainder,4);local=dl%a.M;r,u=divmod(local,1024)
   assert 0<=s<256 and 0<=r<15 and 0<=u<1024 and u%256==0
   assert lo==(1920*c+120*group+4*r+owner)*4096+quarter*1024+u
   abslo=lo+offset*4096;span.append((abslo,abslo+256));assert (c,s,local)not in dest;dest.add((c,s,local))
  assert len(dest)==30720 and g.tensor(w['outputs'][0])['size']==7864320
  allspans.extend(span);weight.append({'operator':wn,'source_view':v,'descriptor':h.dma_descriptors(w['description']),'packets':len(span),'packet_bytes':256,'per_slice_bytes':15360,'logical_contiguous_region_bytes':1024,'source_rows':'1920*c+120*g+offset+4*r+b','descriptor_offsets_relative_to_indexed_hbm_view':True})
  d=n['description'];assert n['kind']=='TuExec' and n['context']=='Main' and n['inputs'][0]==w['outputs'][0]
  assert 'InterTranspose { dim0_volume: 1, dim1_volume: 4, dim0_chunk_volume: 15 }'in d
  assert h.offsets(h.sequence(d,'fetch_sequencer'))==[1024*r+256*q+u for q in range(4)for r in range(15)for u in range(256)]
  assert d[d.index('operation_unit:'):].replace(n['name'],'N')==oo[nn]['description'][oo[nn]['description'].index('operation_unit:'):].replace(oo[nn]['name'],'N')
  assert [g.tensor(t)['shape']for t in n['inputs'][1:]+n['outputs']]==[og.tensor(t)['shape']for t in oo[nn]['inputs'][1:]+oo[nn]['outputs']]
  dv=g.view(n['inputs'][3]);odv=og.view(oo[nn]['inputs'][3]);assert all(dv[k]==odv[k]for k in ('stride_elements','axis_extent','window','start')) and dv['start']==offset
  assert h.offsets(h.sequence(d,'commit_projection'))==list(range(60)) and 'commit_in_size: 8B (=2EA)'in d
  native.append({'operator':nn,'source_weight':wn,'source_reader_elements':15360,'switch':{'slice1':4,'slice0':1,'time0':15},'time_output':'row=4*r+b, canonical60rows','destination_view':dv,'post_switch_native_term_k16_reduce_and_f32_commit_exact_old':True})
 coverage=h.full_coverage(allspans,15728640,'Both stage original HBM weight')
 schedule=load(ev/'schedule.json');ts={int(t['index']):t for t in schedule['tensors']};regs=[]
 for ins in schedule['instructions']:
  if ins['tpe']=='Main' and '.contract_outer' in ins['description']:
   rr=[ts[t]for t in ins['input_tensors']if t in ts and ts[t]['buffer_type']in ('Trf','Vrf')];assert sorted(t['buffer_type']for t in rr)==['Trf','Vrf']
   for t in rr:assert t['lifetime']['begin']<=ins['lifetime']['begin']<=ins['lifetime']['end']<=t['lifetime']['end']
   assert 'InterTranspose' in ins['description'];regs.append({'instruction':ins['index'],'lifetime':ins['lifetime'],'registers':rr})
 assert len(regs)==2 and len({tuple(t['index']for t in r['registers'])for r in regs})==1
 vrfs=[t for t in ts.values()if t['buffer_type']=='Vrf']
 for i,x in enumerate(vrfs):
  assert 0<=x['address']<x['address']+x['size']<=8192*512
  for y in vrfs[i+1:]:
   time=max(x['lifetime']['begin'],y['lifetime']['begin'])<min(x['lifetime']['end'],y['lifetime']['end']);addr=max(x['address'],y['address'])<min(x['address']+x['size'],y['address']+y['size']);assert not(time and addr),(x['index'],y['index'])
 bound={'build_record':bp,'graph':ev/'graph.json','schedule':ev/'schedule.json','source':src,'recipe':HERE/b['variant']/'recipe.json','baseline_build':ref,'historical_same_o_build':oldref,'baseline_source':oldsrc,'historical_same_o_graph':OLD/'evidence/e137-native-v3/graph.json','geometry_verifier':HERE/'geometry.py','geometry':HERE/'geometry.json','helpers':HERE.parent/'qkv-fullh-weight-switch/audit.py','descriptor_helpers':a.HELPER,'verifier':Path(__file__)}
 return {'passed':True,'candidate':'E149','label':label,'inputdigests':{k:rec(p)for k,p in bound.items()},'source':{'compiled_inputs':46,'unchanged_other_inputs':45,'changed':['src/device/sliding/projection.rs'],'only_three_weight_dm_types_and_one_fetch_switch_changed':True,'quantizer_precision_qkv_native_stages_gather_scales_rms_residual_exact_e147':True},'unchanged_emitted_operators':sum(nc.values()),'weight_ingress':weight,'actual_all_weight_coverage':coverage,'logical_packet_bijection':61440,'native_switch':native,'shared_register_lifetimes':regs,'all_vrf_lifetimes_nonoverlapping_within_8kib':True,'cost':{'original_weight_hbm_bytes':15728640,'original_weight_dm_write_bytes':15728640,'original_weight_dm_read_bytes':15728640,'new_full_weight_transpose_dm_bytes':0,'added_main_switch_stream_bytes':15728640,'native_count':2,'latency_claim':False},'scope':'Exact source and emitted weight descriptor/reader/Switch/output/register contracts. Finite NPU arithmetic and performance require this image to be measured.','limits':['Logical1024B ingress regions are emitted as four256B packets, not a1024B DMA packet.','No hidden SFR read-index or physical bank-service timing is inferred.','Static lifetime and unchanged arithmetic do not inherit previous NPU correctness.']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('label',nargs='?',default='e149-ingress-v1');p.add_argument('--check',action='store_true');args=p.parse_args();r=json.loads(json.dumps(audit(args.label)));out=HERE/f'audit-{args.label}.json'
 if args.check:assert load(out)==r,'Audit record differs'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS',out)
