#!/usr/bin/env python3
"""E147 independent emitted-graph gate; no compile, Arena, or production edits."""
import argparse,hashlib,importlib.util,json,re,subprocess,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
OLD=HERE.parent/'qkv-fullh-weight-switch'
spec=importlib.util.spec_from_file_location('e144_audit_helpers',OLD/'audit.py'); a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);h=a.h

def load(p):return json.loads(p.read_text())
def record(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def digest(p):return {k:v for k,v in record(p).items() if k!='path'}
def nodes(g):return {int(x['name'].split('#O')[-1]):x for x in g.ops.values()}
def audit(label):
 assert label=='e147-scalar-v1'
 bp=HERE/f'build-{label}-repro.json';b=load(bp);ev=HERE/'evidence'/label;art=ROOT/'moa-dev/experiments/artifacts'/HERE.name/label
 assert b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container']
 assert b['base_commit']=='3586c417e43f4ea3afc7df7c428d381b7dcbcc5e'
 ref=OLD/'measured-build-e144-adoption.json';oldbuild=load(ref)
 assert len(b['source_inputs'])==46 and b['source_inputs'].keys()==oldbuild['source_inputs'].keys()
 assert {k for k in b['source_inputs'] if b['source_inputs'][k]!=oldbuild['source_inputs'][k]}=={'src/device/sliding/projection.rs'}
 for name,d in b['source_inputs'].items():assert digest(art/'crate'/name)==d
 src=HERE/b['variant']/'projection.rs';assert digest(src)==b['source_inputs']['src/device/sliding/projection.rs']
 assert src.read_bytes()==(ev/'src/device/sliding/projection.rs').read_bytes()
 for name in ('graph','schedule'):
  assert digest(ev/f'{name}.json')==b[name]
  assert (art/f'{name}.json').read_bytes()==(ev/f'{name}.json').read_bytes()
 assert digest(art/'exact-kernel.bin')==b['exact_image']
 oldsrc=OLD/'fullh-v2/projection.rs';new=src.read_text();old=oldsrc.read_text()
 assert new[new.index('fn project_native_query'):]==old[old.index('fn project_native_query'):]
 begin='    let mut packed: DmTensor'
 assert new[new.index(begin):new.index('    // Preserve compact Term4 source fetch')]==old[old.index(begin):old.index('    // Input DMA gives each')]
 start='pub(crate) fn project_qkv_native';end='fn project_native_query'
 ncall=new[new.index(start):new.index(end)];ocall=old[old.index(start):old.index(end)]
 removed='    let coefficients: VrfTensor<f32, Chip, QkvCluster, QkvColumns, m![QkvTerm # 8]> = device.sub\n        .begin(coefficients.view()).fetch::<m![1], m![QkvTerm]>()\n        .collect::<m![1], m![QkvTerm # 8]>().to_vrf();\n'
 assert ocall.count(removed)==1 and ocall.replace(removed,'')==ncall
 subprocess.run([sys.executable,str(HERE/'numerical/scale-proof.py'),'--check'],check=True,stdout=subprocess.DEVNULL)
 assert load(HERE/'numerical/scale-proof.json')==load(HERE/'numerical/original-scale-fusion-audit.json')
 g=h.Graph(load(ev/'graph.json'));og=h.Graph(load(OLD/'evidence/e144-fullh-v2/graph.json'));o=nodes(g);oo=nodes(og)
 # All other operation descriptions, logical tensor shapes/types/sizes, and multiplicities are identical.
 nchanged={12,14,15,70,71};ochanged={12,13,14,15,16,74,76}
 nc=Counter(a.signature(g,x) for k,x in o.items() if k not in nchanged)
 oc=Counter(a.signature(og,x) for k,x in oo.items() if k not in ochanged)
 # New bitmap DtoS has an identical existing ring4 config operation fingerprint elsewhere.
 assert nc-oc==Counter({a.signature(g,o[13]):1}) and not oc-nc
 d=o[12]['description'];tokens=['LogicAnd:','2139095040i','FpMul0:','7.8125e-3f','LReduce(max_f)','ClipMax:','1.1754944e-38f','GReduceMaxF [ratio=4]','commit_in_size: 32B']
 assert [d.index(t) for t in tokens]==sorted(d.index(t) for t in tokens)
 assert o[12]['inputs']==[30] and o[12]['outputs']==[31] and h.offsets(h.sequence(d,'fetch_sequencer'))==list(range(960))
 leader=g.mappings(31)['Partitioning'];assert leader['Pair']['right']=={'Padding':{'inner':{'Broadcast':{'size':1}},'padding':4,'kind':'Top'}}
 switches=[]
 for number,valid,alias in [(14,8,15),(70,4,71)]:
  op=o[number];desc=op['description'];srcid=op['inputs'][0];tid=op['outputs'][0];dst=g.tensor(tid);rename=o[alias]
  assert op['context']=='Sub' and 'to Vrf' in desc and 'ring_size: 4' in desc
  assert [int(v) for v in re.findall(r'SliceBitmap \[(\d+)\]',desc)]==[4*(s//4) for s in range(256)]
  assert op['inputs'][1:]==[34] and g.mappings(srcid)['Partitioning']==leader
  assert h.offsets(h.sequence(desc,'fetch_sequencer'))==list(range(valid))
  assert g.tensor(srcid)['size']==512*valid*4 and dst['buffer_type']=='Vrf' and dst['size']==512*32
  assert g.mappings(tid)['Partitioning']['Pair']['right']=={'Broadcast':{'size':4}}
  assert rename['inputs']==[tid];renamed=g.tensor(rename['outputs'][0]);assert renamed['optionals']['address']==dst['optionals']['address'] and renamed['size']==dst['size']
  assert g.mappings(rename['outputs'][0])['InSlice']==g.mappings(tid)['InSlice']
  assert g.mappings(rename['outputs'][0])['Partitioning']['Pair']['right']=={'Stride':{'inner':{'Symbol':{'symbol':'H','size':3840}},'stride':960}}
  switches.append({'operator':number,'source_tensor':srcid,'source_valid_f32_indices':list(range(valid)),'source_valid_bytes_per_slice':valid*4,'output_vrf_bytes_per_slice':32,'bitmap_entries':256,'mapping':'source=4*(destination//4), same cluster','renamed_output':rename['outputs'][0]})
 assert h.sequence(o[70]['description'],'fetch_sequencer')==h.sequence(oo[76]['description'],'fetch_sequencer')
 assert 'fetch_size: 8' in o[70]['description'] and 'fetch_size: 8' in oo[76]['description']
 assert '[QkvTerm_1=4]+4]=8' in o[70]['description'] and '[QkvTerm_1=4]+4]=8' in oo[76]['description']
 assert a.signature(g,o[69])==a.signature(og,oo[70]) and 'commit_in_size: 8B' in o[69]['description']
 cfg=list(a.packets(o[13]));assert len(cfg)==512 and all(hi-lo==dh-dl==32 for _,_,_,(lo,hi),(dl,dh) in cfg)
 weight=[]
 for name,wn,oldwn,nn,oldnn in [('Q',77,78,78,79),('K',90,91,91,92),('V',103,104,104,105)]:
  assert a.signature(g,o[wn])==a.signature(og,oo[oldwn]) and a.signature(g,o[nn])==a.signature(og,oo[oldnn])
  ps=list(a.packets(o[wn]));assert all(hi-lo==dh-dl==256 for _,_,_,(lo,hi),(dl,dh) in ps)
  spans=[(lo,hi) for _,_,_,(lo,hi),_ in ps];size={'Q':15728640,'K':7864320,'V':7864320}[name]
  h.full_coverage(spans,size,name+' HBM')
  weight.append({'projection':name,'dma':wn,'native':nn,'hbm_bytes':size,'packets_256B':len(ps),'fullh_switch_native_and_output_shape_exact_e144':True})
 schedule=load(ev/'schedule.json');ts={int(t['index']):t for t in schedule['tensors']};regs=[]
 for ins in schedule['instructions']:
  if ins['tpe']!='Main' or '.contract_outer::' not in ins['description']:continue
  rr=[ts[t] for t in ins['input_tensors'] if ts[t]['buffer_type'] in ('Trf','Vrf')]
  assert sorted(t['buffer_type'] for t in rr)==['Trf','Vrf']
  for t in rr:assert t['lifetime']['begin']<=ins['lifetime']['begin']<=ins['lifetime']['end']<=t['lifetime']['end']
  regs.append({'instruction':ins['index'],'lifetime':ins['lifetime'],'registers':rr})
 assert len(regs)==3 and len({tuple(t['index'] for t in x['registers']) for x in regs})==1
 vrfs=[t for t in ts.values() if t['buffer_type']=='Vrf']
 for t in vrfs:assert 0<=t['address'] and t['address']+t['size']<=8192*512
 for i,x in enumerate(vrfs):
  for y in vrfs[i+1:]:
   time=max(x['lifetime']['begin'],y['lifetime']['begin'])<min(x['lifetime']['end'],y['lifetime']['end'])
   addr=max(x['address'],y['address'])<min(x['address']+x['size'],y['address']+y['size'])
   assert not (time and addr),(x['index'],y['index'],'live VRF overlap')
 # Alias entries split producer/consumer lifetimes at an identical physical register region.
 for left,right in [(96,97),(121,126)]:
  x,y=ts[left],ts[right];assert (x['address'],x['size'])==(y['address'],y['size']) and x['lifetime']['end']==y['lifetime']['begin']
 scaleusers=[]
 for ins in schedule['instructions']:
  if 97 in ins['input_tensors']:
   assert ts[97]['lifetime']['begin']<=ins['lifetime']['begin']<=ins['lifetime']['end']<=ts[97]['lifetime']['end'];scaleusers.append(ins['index'])
 assert len(scaleusers)==4
 bound=[bp,src,HERE/b['variant']/'recipe.json',ev/'graph.json',ev/'schedule.json',ref,oldsrc,OLD/'evidence/e144-fullh-v2/graph.json',OLD/'audit.py',a.HELPER,HERE/'audit.py']+sorted((HERE/'numerical').glob('*'))
 return {'passed':True,'candidate':'E147','label':label,'scope':'Actual scalar pipeline, bitmap, valid sequencer/masks, register allocation, and unchanged E144 numerical consumers. NPU correctness/performance are separate.','inputdigests':{str(p.relative_to(ROOT)):record(p) for p in bound if p.is_file()},'source':{'inputs':46,'changed':['src/device/sliding/projection.rs'],'remaining_45_exact_e144':True,'quant_four_terms_and_coeff_constants_unchanged':True,'native_head_output_suffix_byte_identical':True},'scale_pipeline':{'operator':12,'source_values_per_slice':960,'global_reduction_ratio':4,'output_only_q0':True,'finite_bf16_patterns_cpu_checked':65280,'nonfinite_not_claimed':True},'broadcasts':switches,'coefficient_old_valid_reader_and_padding_identical':True,'bitmap_shared_config':{'operator':13,'bytes':16384,'packets':512,'two_consumers':[14,70]},'unchanged_emitted_operation_signatures':sum(oc.values()),'weight_ingress_and_native':weight,'scheduled_shared_register_lifetimes':regs,'scale_register_consumer_instructions':scaleusers,'all_vrf_lifetimes_disjoint_in_8192_bytes_per_slice':True,'cost':{'source_mul_values':491520,'removed_main_starts':2,'removed_scalar_consumer_dm_writes_bytes':24576,'removed_next_scalar_consumer_dm_reads_bytes':24576,'added_shared_bitmap_hbm_bytes':16384,'two_sub_broadcasts_replace_existing_two_sub_loads':True,'common_scale_head_dma_remains':True,'no_latency_claim':True},'limits':['Printed TU fetch_size=8 is shared with E144 compact-coefficient fetch; valid sequencer is exactly 0..3 and collect pads to8. Hidden physical bus-read granularity is not decoded or inferred.','Source/graph and finite-BF16 CPU equivalence do not replace exact-image NPU correctness.','Static schedule lifetimes establish allocation and dependency checks, not observed hardware latency.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('label',nargs='?',default='e147-scalar-v1');p.add_argument('--check',action='store_true');args=p.parse_args();result=json.loads(json.dumps(audit(args.label)));out=HERE/f'audit-{args.label}.json'
 if args.check:assert load(out)==result,'Audit record differs'
 else:out.write_text(json.dumps(result,indent=2)+'\n')
 print('PASS',out)
if __name__=='__main__':main()
