#!/usr/bin/env python3
"""E146 independent source/actual packet/partial-output/register gate. Read only with --check."""
import argparse,hashlib,importlib.util,json,re
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
OLD=HERE.parent/'qkv-fullh-weight-switch'
spec=importlib.util.spec_from_file_location('e146_prior_audit',OLD/'audit.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
h=a.h;M=a.M

def load(p):return json.loads(p.read_text())
def rec(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def dig(p):return {k:v for k,v in rec(p).items() if k!='path'}
def function(s,name):
 start=s.index('fn '+name+'(');begin=s.index('{',start);depth=1;i=begin+1
 while depth:
  depth+=(s[i]=='{')-(s[i]=='}');i+=1
 return s[start:i]
def core(s,name,axis,half=None):
 f=function(s,name);begin=f.index('.collect::<m!['+axis)
 end=f.index('.commit_trim::<m!['+axis,begin);end=f.index('>()',end)+3
 q=f[begin:end]
 return q.replace(axis+' = '+str(half),axis) if half else q

def audit(label):
 assert label=='e146-qk-half-v4'
 bp=HERE/f'build-{label}-repro.json';b=load(bp);ev=HERE/'evidence'/label;v=HERE/b['variant'];art=ROOT/'moa-dev/experiments/artifacts'/HERE.name/label
 assert b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container']
 assert b['base_commit']=='3586c417e43f4ea3afc7df7c428d381b7dcbcc5e'
 oldb=load(OLD/'measured-build-e144-adoption.json')
 assert len(b['source_inputs'])==46 and b['source_inputs'].keys()==oldb['source_inputs'].keys()
 assert {k for k in b['source_inputs'] if b['source_inputs'][k]!=oldb['source_inputs'][k]}=={'src/device/sliding/projection.rs'}
 for name,d in b['source_inputs'].items():assert dig(art/'crate'/name)==d
 assert dig(v/'projection.rs')==b['source_inputs']['src/device/sliding/projection.rs']
 assert (v/'projection.rs').read_bytes()==(ev/'src/device/sliding/projection.rs').read_bytes()
 for key in ['graph','schedule']:
  assert dig(ev/f'{key}.json')==b[key] and (ev/f'{key}.json').read_bytes()==(art/f'{key}.json').read_bytes()
 assert dig(art/'exact-kernel.bin')==b['exact_image']
 src=(v/'projection.rs').read_text();old=(OLD/'fullh-v2/projection.rs').read_text()
 marker='// E137 uses E134 QKV unchanged; O changes activation representation and native reduction.'
 assert src[src.index(marker):]==old[old.index(marker):]
 prefix=src[:src.index('fn project_native_query(')].replace('project_native_key(device,','project_native_kv(device,').replace('project_native_value(device,','project_native_kv(device,')
 assert prefix==old[:old.index('fn project_native_query(')]
 assert 'staged: bool' not in src and '.memset(' not in src
 for axis,half,name,oldname in [('QkvQueryRow',16,'project_native_query_stage','project_native_query'),('QkvKvRow',8,'project_native_kv_stage','project_native_kv')]:assert core(src,name,axis,half)==core(old,oldname,axis)
 # The scale/first-BF16 suffix is identical for Q, K and V; V's complete old body is retained.
 suffix='    // Full-allocation relabel preserves'
 for nn,on in [('project_native_query','project_native_query'),('project_native_key','project_native_kv')]:
  nf,of=function(src,nn),function(old,on);assert nf[nf.index(suffix):]==of[of.index(suffix):]
 assert function(src,'project_native_value').replace('project_native_value(','project_native_kv(',1)==function(old,'project_native_kv')
 g=h.Graph(load(ev/'graph.json'));og=h.Graph(load(OLD/'evidence/e144-fullh-v2/graph.json'))
 o={int(x['name'].split('#O')[-1]):x for x in g.ops.values()};oo={int(x['name'].split('#O')[-1]):x for x in og.ops.values()}
 bound=[bp,v/'projection.rs',v/'recipe.json',ev/'graph.json',ev/'schedule.json',OLD/'fullh-v2/projection.rs',OLD/'evidence/e144-fullh-v2/graph.json',OLD/'measured-build-e144-adoption.json',OLD/'audit.py',a.HELPER]
 stage_specs=[('Q',32,16,0,80,88,89),('Q',32,16,16,83,93,94),('K',16,8,0,108,116,117),('K',16,8,8,111,121,122),('V',16,16,0,134,135,None)]
 coverage={x:[] for x in 'QKV'};seen={x:set() for x in 'QKV'};reports=[]
 for name,rows,half,start,wn,nn,iw in stage_specs:
  w,n=o[wn],o[nn];io={'Q':1,'K':2,'V':3}[name];R=half//4
  tid=w['inputs'][0];base=0
  if name!='V':
   view=h.Graph.view(g,tid);assert view['stride_elements']==3840 and view['axis_extent']==rows and view['window']==half and view['start']==start
   tid=view['parent'];base=start*3840
  assert g.tensor(tid)['optionals']['address']==f'IO({io})+0'
  dest=set();count=0
  for _,sa,da,(lo,hi),(dl,dh) in a.packets(w):
   c=da['cluster'];s=a.sl(da,dl);group,owner=divmod(s,4);local=dl%M;r,u=divmod(local,3840)
   assert sa['chip']==da['chip']==0 and 0<=c<2 and 0<=s<256 and 0<=r<R and u%256==0 and 0<=u<3840
   assert hi-lo==dh-dl==256
   assert lo+base==(c*64*rows+group*rows+start+4*r+owner)*3840+u
   coverage[name].append((lo+base,hi+base));dest.add((c,s,local));count+=1
  assert len(dest)==count==512*R*15 and g.tensor(w['outputs'][0])['size']==512*R*3840
  assert n['kind']=='TuExec' and n['context']=='Main' and n['inputs'][:3]==[w['outputs'][0],94,95]
  d=n['description'];assert f'InterTranspose {{ dim0_volume: 1, dim1_volume: 4, dim0_chunk_volume: {R} }}' in d
  assert h.offsets(h.sequence(d,'fetch_sequencer'))==[3840*r+960*q+u for q in range(4) for r in range(R) for u in range(960)]
  assert h.offsets(h.sequence(d,'commit_projection'))==list(range(half)) and 'commit_in_size: 8B (=2EA)' in d
  assert d.index('FpMul0:')<d.index('ReduceAccTree: LReduce(+) [axes=QkvTerm]')<d.index('GReduceAdd: GReduceAddF [ratio=4]')
  assert g.mappings(n['outputs'][0])['Partitioning']==og.mappings(oo[{'Q':79,'K':92,'V':105}[name]]['outputs'][0])['Partitioning']
  if iw is not None:
   destview=h.Graph.view(g,n['inputs'][3]);assert destview['stride_elements']==1 and destview['axis_extent']==rows and destview['window']==half and destview['start']==start
   write=o[iw];assert write['kind']=='IndexWrite' and write['inputs'][0]==destview['parent'] and write['inputs'][2]==n['outputs'][0] and g.scalar(write['inputs'][1])==start
   assert 'window_size: '+str(half) in write['description']
  for c in range(2):
   for group in range(64):
    for owner in range(4):
     for q in range(4):
      for r in range(R):
       row=c*64*rows+group*rows+start+4*r+owner
       dst=(c,4*group+q,start+4*r+owner)
       assert row==c*64*rows+group*rows+dst[2] and 3840*r+960*q+959<R*3840
       assert (row,q) not in seen[name];seen[name].add((row,q))
  reports.append({'projection':name,'rows':[start,start+half],'weight_operator':w['name'],'native_operator':n['name'],'source_base_bytes_from_index_access':base,'actual_256B_packets':count,'source_destination_affine_mapping_checked':True,'switch_time0':R,'row_quarter_packets':128*half*4,'output_byte_interval_per_live_slice':[start*4,(start+half)*4],'output_pair_bytes':8,'index_write':o[iw]['name'] if iw else None})
 cov={name:h.full_coverage(spans,{'Q':4096,'K':2048,'V':2048}[name]*3840,name) for name,spans in coverage.items()}
 assert sum(len(x) for x in seen.values())==32768
 # Final stage writes join the same complete raw allocation, then unchanged canonical DMA once.
 for name,ids,root,final,reshape,gather,oldgather in [('Q',[89,94],103,112,95,98,83),('K',[117,122],131,140,123,126,96)]:
  assert g.root_output(final)==root and o[reshape]['inputs']==[final]
  assert o[gather]['inputs'][0]==o[reshape]['outputs'][0]
  assert a.signature(g,o[gather])==a.signature(og,oo[oldgather])
 assert a.signature(g,o[135])==a.signature(og,oo[105])
 excluded={78,79,91,92}
 inherited=Counter(a.signature(og,x) for n,x in oo.items() if n not in excluded)
 now=Counter(a.signature(g,x) for x in o.values())
 assert not (inherited-now),'An inherited arithmetic, head/cache or gather operator differs'
 sch=load(ev/'schedule.json');ts={int(t['index']):t for t in sch['tensors']};regs=[]
 for ins in sch['instructions']:
  if ins['tpe']!='Main' or '.contract_outer::' not in ins['description']:continue
  rr=[ts[t] for t in ins['input_tensors'] if t in ts and ts[t]['buffer_type'] in ('Trf','Vrf')]
  assert sorted(t['buffer_type'] for t in rr)==['Trf','Vrf']
  for t in rr:assert t['lifetime']['begin']<=ins['lifetime']['begin']<=ins['lifetime']['end']<=t['lifetime']['end']
  regs.append({'instruction':ins['index'],'lifetime':ins['lifetime'],'registers':rr})
 assert len(regs)==5 and len({tuple(t['index'] for t in r['registers']) for r in regs})==1
 return {'passed':True,'candidate':'E146','label':label,'inputdigests':{str(p.relative_to(ROOT)):rec(p) for p in bound},'source_composition':{'e144_source_inputs':45,'projection_only_change':True,'quantizer_and_four_term_precision_unchanged':True,'scale_first_bf16_head_cache_and_o_suffix_unchanged':True,'no_new_memset_or_sub_assembly':True},'weight_coverage':cov,'stage_mapping':reports,'row_quarter_bijection':32768,'native_register_lifetimes':regs,'inherited_emitted_operator_signatures':sum(inherited.values()),'cost_scope':{'original_weight_hbm_bytes':31457280,'original_weight_dm_read_write_each_bytes':31457280,'native_switch_tasks_before':3,'native_switch_tasks_after':5,'extra_full_raw_copy_bytes':0,'direct_stage_writes_total_valid_bytes':24576,'new_index_write_and_join_dependencies':True},'limits':['Graph and scheduled lifetime do not establish NPU correctness, timing or actual overlap.','Hidden DPE SFR read-index is not independently decoded; source native shapes and shared register operands are checked.','Two IndexWrite stages serialize their output dependency and canonical gather waits for both; no free overlap or gain claimed.','The v1-v3 zero-size scheduler failures are not reclassified as a proven private allocator root cause.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('label',nargs='?',default='e146-qk-half-v4');p.add_argument('--check',action='store_true');args=p.parse_args()
 r=json.loads(json.dumps(audit(args.label)));out=HERE/f'audit-{args.label}.json'
 if args.check:assert load(out)==r,'Audit record differs'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS',out)
if __name__=='__main__':main()
