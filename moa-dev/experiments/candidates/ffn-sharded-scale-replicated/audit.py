#!/usr/bin/env python3
"""E154 replicated-Max source/RMS/shards/copies/register gate; no builds or NPU."""
import argparse,hashlib,importlib.util,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OLD=HERE.parent/'ffn-down-scale-ingress-switch'
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=imp('e151_helpers',OLD/'audit.py');a=p.a;h=p.h
load=p.load
PREV=HERE.parent/'ffn-prerms-sharded-consumer'
source=imp('e153_source',PREV/'source-review/source_review.py')
def record(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def digest(p):return {k:v for k,v in record(p).items()if k!='path'}
def ops(g):return {int(o['name'].split('#O')[-1]):o for o in g.ops.values()}
def sig(g,o):return re.sub(r'T\d+','T',a.signature(g,o))
def actual(g,sch,image,og=None):
 oo=ops(g)
 cast=next(o for o in g.ops.values()if o['kind']=='TuExec' and o['context']=='Main' and 'output_element_type: bf16' in o['description'] and g.mappings(o['outputs'][0]).get('InSlice')=={'Modulo':{'inner':{'Symbol':{'symbol':'H','size':3840}},'modulo':480}} and 've_pass:'not in o['description']) if og else None
 if og:
  od=ops(og)
  for i in range(10):assert a.signature(g,oo[i])==a.signature(og,od[i]),('RMS arithmetic',i)
  assert cast==oo[10] and cast['inputs']==oo[9]['outputs']
  assert h.offsets(h.sequence(cast['description'],'fetch_sequencer'))==list(range(480))
  assert h.offsets(h.sequence(cast['description'],'commit_projection'))==list(range(480)) and 'commit_in_size: 16B (=8EA)'in cast['description']
 # Locate the actual new quantizer via local absolute Max; raw and whole IDs differ.
 local=next(o for o in g.ops.values()if 'LogicAnd:'in o['description'] and '2147483647i'in o['description'] and 'LReduce(max_f) [axes=H]'in o['description'] and 'H_1=480'in o['description'])
 x=local['inputs'][0];xm=g.mappings(x)
 assert xm['Cluster']=={'Padding':{'inner':{'Broadcast':{'size':1}},'padding':2,'kind':'Top'}}
 assert xm['Partitioning']=={'Padding':{'inner':{'Stride':{'inner':{'Symbol':{'symbol':'H','size':3840}},'stride':480}},'padding':256,'kind':'Top'}}
 assert 'ClipMax:'in local['description'] and '1.92593e-34f'in local['description']
 glob=next(o for o in g.ops.values()if o['inputs']==local['outputs'] and 'GReduceMax'in o['description'])
 ratio=int(re.search(r'GReduceMaxF \[ratio=(\d+)\]',glob['description'])[1])
 assert '[[[H_480=8]+248]=256]'in glob['description']
 gm=g.mappings(glob['outputs'][0]);assert gm['Cluster']==xm['Cluster']
 assert gm['Partitioning']=={'Padding':{'inner':{'Symbol':{'symbol':'Dummy8','size':8}},'padding':256,'kind':'Top'}}
 resh=next(o for o in g.ops.values()if o['kind']=='LowLevelReshape' and o['inputs']==glob['outputs'])
 assert g.mappings(resh['outputs'][0])==g.mappings(local['outputs'][0])
 scale=next(o for o in g.ops.values()if o['inputs']==resh['outputs'] and '2139095040i'in o['description'])
 assert 'FpMul0:'in scale['description'] and '6.1035156e-5f'in scale['description']
 assert g.mappings(scale['outputs'][0])==g.mappings(local['outputs'][0])
 scalar_vrf=next(o for o in g.ops.values()if o['kind']=='StoVrf' and o['inputs']==scale['outputs'])
 assert g.mappings(scalar_vrf['outputs'][0])==g.mappings(scale['outputs'][0])
 assert not any(o['kind'].startswith('Dma') and o['inputs'][0]in scale['outputs'] for o in g.ops.values())
 qdest=set(range(8))
 prefix=p.p.prior.prefix_and_bitmap(g,sch,image)
 packed=prefix['copies'][0]['root'];late=[]
 for o in g.ops.values():
  if o['kind']!='Dma.StoS':continue
  try:rt=g.root_output(o['inputs'][0])
  except (ValueError,KeyError):continue
  if rt==packed:late.append(o)
 assert len(late)==1;late=late[0];seen=set();count=0
 for _,sa,da,(lo,hi),(dl,dh)in a.packets(late):
  ss=a.sl(sa,lo);ds=a.sl(da,dl);so=lo%a.M;do=dl%a.M
  assert sa['cluster']==0 and da['cluster']in(0,1) and 0<=ss<8 and ss==ds and so==do and so in(0,720) and hi-lo==dh-dl==720
  key=(da['cluster'],ds,do);assert key not in seen;seen.add(key);count+=1
 assert count==32 and len(seen)==32
 seed=next(o for o in g.ops.values()if o['kind']=='Dma.StoS' and g.tensor(o['outputs'][0])['size']==5898240 and g.tensor(o['outputs'][0])['element_type']=='Float8e5m2')
 dest=set()
 for _,sa,da,(lo,hi),(dl,dh)in a.packets(seed):
  ss=a.sl(sa,lo);ds=a.sl(da,dl);t,u=divmod(lo%a.M,480)
  assert sa['cluster']==da['cluster'] and 0<=ss<8 and 0<=t<3 and u==0 and hi-lo==dh-dl==480 and ds%32==0 and ds<256 and dl%a.M==t*3840+ss*480
  key=(da['cluster'],ds,t,ss);assert key not in dest;dest.add(key)
 assert len(dest)==2*8*3*8;seed_count=len(dest)
 anonymous=next(o for o in g.ops.values()if o['kind']=='LowLevelReshape' and o['inputs']==scale['outputs'])
 am=g.mappings(anonymous['outputs'][0]);assert am['Cluster']==xm['Cluster']
 assert am['Partitioning']=={'Padding':{'inner':{'Broadcast':{'size':8}},'padding':256,'kind':'Top'}}
 restore=next(o for o in g.ops.values()if o['kind']=='Dma.StoS' and o['inputs'][0]==anonymous['outputs'][0])
 dest=set();restore_sources=set()
 for _,sa,da,(lo,hi),(dl,dh)in a.packets(restore):
  ss=a.sl(sa,lo)
  assert sa['cluster']==0 and 0<=ss<8 and lo%a.M==0 and hi-lo==dh-dl==32 and dl%a.M==0
  key=(da['cluster'],a.sl(da,dl));assert key not in dest;dest.add(key);restore_sources.add(ss)
 assert dest=={(c,s)for c in range(2)for s in range(256)}
 # All registered inputs remain live through consumers, disjoint where lifetimes overlap.
 ts={int(t['index']):t for t in sch['tensors']};vrfs=[t for t in ts.values()if t['buffer_type']=='Vrf']
 for i,t in enumerate(vrfs):
  assert 0<=t['address']<t['address']+t['size']<=8192*512
  for u in vrfs[i+1:]:
   live=max(t['lifetime']['begin'],u['lifetime']['begin'])<min(t['lifetime']['end'],u['lifetime']['end'])
   overlap=max(t['address'],u['address'])<min(t['address']+t['size'],u['address']+u['size'])
   assert not(live and overlap),(t['index'],u['index'])
 for ins in sch['instructions']:
  for tid in ins['input_tensors']:
   t=ts.get(tid)
   if t and t['buffer_type']in('Vrf','Trf'):assert t['lifetime']['begin']<=ins['lifetime']['begin']<=ins['lifetime']['end']<=t['lifetime']['end']
 natives=[o for o in g.ops.values()if 'lookup_table:'in o['description'] and o['context']=='Main'];assert len(natives)==(4 if og else 2)
 if og:
  for n in natives:assert any(sig(g,n)==sig(og,z)for z in og.ops.values()),'Native descriptor/arithmetic changed'
  newsw=[o for o in g.ops.values()if o['context']=='Sub' and 'InterTranspose'in o['description']]
  oldsw=[o for o in og.ops.values()if o['context']=='Sub' and 'InterTranspose'in o['description']]
  assert len(newsw)==len(oldsw)==5
  for sw in newsw:assert any(sig(g,sw)==sig(og,z)for z in oldsw),'E151 scale supplier changed'
 return {'rms_first_ten_emitted_operations_identical_to_e151':bool(og),'local_cast':cast and cast['name'],'input_cluster':0,'valid_input_slices':list(range(8)),'h480_local_max':local['name'],'global_max':{'operator':glob['name'],'actual_ratio':ratio,'valid_source_shards':8,'output_cluster':0,'output_slices':list(range(8))},'scale_operator':scale['name'],'quant_scalar_dma_removed':True,'scalar_vrf':scalar_vrf['name'],'restore_source_slices':sorted(restore_sources),'scale_supply_valid_shards':sorted(qdest),'late_packed_dma':{'operator':late['name'],'packets':count,'packet_bytes':720,'destination_bytes':23040,'source_cluster':0,'destination_clusters':[0,1],'source_target_same_h_shard':True},'ring_seed_dma':{'operator':seed['name'],'packets':seed_count,'packet_bytes':480,'destination_bytes':184320},'scalar_restore_all512':True,'copy_bitmap':prefix,'all_vrf_lifetimes_nonoverlapping_within_8kib':True,'register_buffers':vrfs,'all_register_consumers_live':True,'native_count':len(natives),'native_and_e151_scale_arithmetic_unchanged':bool(og)}
def audit(label):
 assert label=='e154-replicated-v1';bp=HERE/f'build-{label}-repro.json';b=load(bp);ev=HERE/'evidence'/label;art=ROOT/'moa-dev/experiments/artifacts'/HERE.name/label;basepath=OLD/'measured-build-e151-adoption.json';base=load(basepath)
 assert b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container'] and b['base_commit']=='8adbbae1776aa8080a96348a9d9a64022a0cf6d0'
 assert len(b['source_inputs'])==46 and b['source_inputs'].keys()==base['source_inputs'].keys()
 changed=sorted(k for k in b['source_inputs']if b['source_inputs'][k]!=base['source_inputs'][k]);assert changed==['src/device/shared/mlp.rs','src/device/shared/rmsnorm.rs','src/ops.rs']
 for k,d in b['source_inputs'].items():assert digest(art/'crate'/k)==d
 for k,v in [('ops.rs','src/ops.rs'),('mlp.rs','src/device/shared/mlp.rs'),('rmsnorm.rs','src/device/shared/rmsnorm.rs')]:assert digest(HERE/'replicated-v1'/k)==b['source_inputs'][v] and (HERE/'replicated-v1'/k).read_bytes()==(ev/v).read_bytes()
 for k in ('graph','schedule'):assert digest(ev/f'{k}.json')==b[k] and (ev/f'{k}.json').read_bytes()==(art/f'{k}.json').read_bytes()
 assert digest(art/'exact-kernel.bin')==b['exact_image']
 import subprocess,sys
 subprocess.run([sys.executable,str(HERE/'replicated-v1/prepare.py'),'--source',str(PREV/'sharded-v1'),'--check'],check=True,stdout=subprocess.PIPE)
 sr=source.run();assert load(PREV/'source-review/source_review.json')==sr
 for n in ('ops.rs','rmsnorm.rs'):assert (HERE/'replicated-v1'/n).read_bytes()==(PREV/'sharded-v1'/n).read_bytes()
 recipe=load(HERE/'replicated-v1/recipe.json');assert recipe['actual_start_commit']=='b336c45c20bcf95815d8f66cfea4d9b8b8d99922' and recipe['comparison_recipe']=='baseline/e151.json'
 g=h.Graph(load(ev/'graph.json'));og=h.Graph(load(PREV/'evidence/e153-sharded-v1/graph.json'));sch=load(ev/'schedule.json')
 aa=actual(g,sch,art/'exact-kernel.bin',og)
 d=ops(g)[1];seen=set()
 for _,sa,da,(lo,hi),(dl,dh)in a.packets(d):
  ss=a.sl(sa,lo);ds=a.sl(da,dl);assert sa['cluster']==da['cluster']==0 and ss==0 and 0<=ds<8 and hi-lo==dh-dl==960 and lo%a.M==960*ds and dl%a.M==0;seen.add(ds)
 assert seen==set(range(8));aa['input_h480_typed_dma_packets']=8
 images={}
 for name,sha in b['source_plan']['expected_unchanged_images'].items():
  pp=art/'standard-kernels'/f'furiosa_opt_gemma4::ops::{name}.bin';dd=digest(pp);assert dd['sha256']==sha;images[name]=record(pp)
 assert set(images)=={'sliding_project_qkv','sliding_attention_output'}
 bound={'build_record':bp,'graph':ev/'graph.json','schedule':ev/'schedule.json','baseline_build':basepath,'e153_graph':PREV/'evidence/e153-sharded-v1/graph.json','recipe':HERE/'replicated-v1/recipe.json','source_review':PREV/'source-review/source_review.json','source_verifier':PREV/'source-review/source_review.py','source_reproduction':HERE/'replicated-v1/prepare.py','source_contract':HERE/'replicated-v1/source-contract.json','verifier':Path(__file__),'descriptor_helper':a.HELPER,'copy_helper':HERE.parent/'ffn-independent-direct-copy-repair/independent-audit.py'}
 for k in('ops.rs','rmsnorm.rs','mlp.rs'):bound[k]=HERE/'replicated-v1'/k
 return {'passed':True,'candidate':'E154','label':label,'inputdigests':{k:record(v)for k,v in bound.items()},'source':{'compiled_input_count':46,'changed_from_e151':changed,'unchanged_other_inputs':43,'source_math_review_passed':True,'finite_bf16_patterns':65280,'unchanged_qkv_o_images':images,'e153_source_composition_reproduced':True},'actual':aa,'scope':'Actual source/RMS operation identity, replicated scalar masks, emitted DMA/copy/bitmap/consumer and register-lifetime gate; finite source-Max proof. NPU arithmetic/liveness, complete FFN correctness and speed remain separate.'}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('label',nargs='?',default='e154-replicated-v1');ap.add_argument('--check',action='store_true');args=ap.parse_args();r=json.loads(json.dumps(audit(args.label)));out=HERE/f'audit-{args.label}.json'
 if args.check:assert load(out)==r,'Audit record differs'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS',out)
