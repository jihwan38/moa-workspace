#!/usr/bin/env python3
"""Independent E148 emitted O graph/VRF/source gate; no compilation or NPU calls."""
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
 assert label=='e148-direct-v1'
 bp=HERE/f'build-{label}-repro.json';b=load(bp);ev=HERE/'evidence'/label;art=ROOT/'moa-dev/experiments/artifacts'/HERE.name/label
 ref=E147/'measured-build-e147-adoption.json';baseline=load(ref);oldref=OLD/'measured-build-e137-adoption.json';older=load(oldref)
 assert b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container'] and b['base_commit']=='fb45202bba3c24537f74b4bf463276b76d823773'
 assert len(b['source_inputs'])==46 and b['source_inputs'].keys()==baseline['source_inputs'].keys()
 assert {k for k in b['source_inputs'] if b['source_inputs'][k]!=baseline['source_inputs'][k]}=={'src/device/sliding/projection.rs'}
 assert baseline['images']['sliding_attention_output']==older['images']['sliding_attention_output']
 for name,d in b['source_inputs'].items():assert digest(art/'crate'/name)==d
 src=HERE/b['variant']/'projection.rs';oldsrc=E147/'scalar-v1/projection.rs'
 assert digest(src)==b['source_inputs']['src/device/sliding/projection.rs'] and src.read_bytes()==(ev/'src/device/sliding/projection.rs').read_bytes()
 for n in ('graph','schedule'):
  assert digest(ev/f'{n}.json')==b[n] and (ev/f'{n}.json').read_bytes()==(art/f'{n}.json').read_bytes()
 assert digest(art/'exact-kernel.bin')==b['exact_image']
 new=src.read_text();old=oldsrc.read_text();mark='pub(crate) fn prepare_output_terms('
 assert new.split(mark)[0]==old.split(mark)[0], 'QKV prefix changed'
 removed='    let coefficients: VrfTensor<f32, Chip, OutputCluster, OutputColumns, m![OutputTerm # 8]> = device\n        .sub.begin(coefficients.view()).fetch::<m![1], m![OutputTerm]>()\n        .collect::<m![1], m![OutputTerm # 8]>().to_vrf();\n'
 ns=new[new.index('pub(crate) fn project_output('):];os=old[old.index('pub(crate) fn project_output('):]
 assert os.count(removed)==1 and os.replace(removed,'')==ns,'Native/stages/epilogue source changed'
 qstart='    let mut packed: DmTensor<f8e4m3, Chip, m![2], OutputQuantSlices';qend='    // Constant normalized coefficients:'
 nq=new[new.index(qstart):new.index(qend,new.index(qstart))];oq=old[old.index(qstart):old.index(qend,old.index(qstart))]
 nh='let sum_vrf: VrfTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.main'
 oh='let sum: DmTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.main'
 replacement='        .commit_trim::<m![Qs % 8]>().commit();\n    let sum_vrf: VrfTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.sub\n        .begin(sum.view()).fetch::<m![Qs / 8 % 32], m![Qs % 8]>()\n        .collect::<m![Qs / 8 % 32], m![Qs % 8]>().to_vrf();'
 assert nq.count(nh)==2 and nq.count('        .to_vrf(&mut device.sub);')==2
 assert nq.replace(nh,oh).replace('        .to_vrf(&mut device.sub);',replacement)==oq,'Residual arithmetic changed'
 cs='    // Constant normalized coefficients:';assert new[new.index(cs,new.index(qstart)):new.index('    // Compact16B coefficients')]==old[old.index(cs,old.index(qstart)):old.index('    // Input DMA gives each (c,g) the identical4096')]
 subprocess.run([sys.executable,str(HERE/'numerical/scale-proof.py'),'--check'],check=True,stdout=subprocess.DEVNULL)
 g=h.Graph(load(ev/'graph.json'));og=h.Graph(load(OLD/'evidence/e137-native-v3/graph.json'));o=nodes(g);oo=nodes(og)
 changed={2,3,4,5,22,30,58,59};oldchanged={2,3,4,5,6,23,24,32,33,64,66}
 nc=Counter(a.signature(g,x) for k,x in o.items() if k not in changed);oc=Counter(a.signature(og,x) for k,x in oo.items() if k not in oldchanged)
 assert nc==oc,'Unexpected emitted changes outside scalar transport and two cumulative sums'
 d=o[2]['description'];tokens=['LogicAnd:','2139095040i','FpMul0:','7.8125e-3f','LReduce(max_f)','ClipMax:','1.1754944e-38f','GReduceMaxF [ratio=16]','commit_in_size: 32B']
 assert [d.index(t) for t in tokens]==sorted(d.index(t) for t in tokens)
 assert h.offsets(h.sequence(d,'fetch_sequencer'))==list(range(256))
 leader=g.mappings(7)['Partitioning'];assert leader['Pair']['right']=={'Padding':{'inner':{'Broadcast':{'size':1}},'padding':16,'kind':'Top'}}
 broadcasts=[]
 for num,valid,alias in [(4,8,5),(58,4,59)]:
  op=o[num];desc=op['description'];tid=op['outputs'][0];sid=op['inputs'][0]
  assert op['context']=='Sub' and 'to Vrf' in desc and 'ring_size: 16' in desc and op['inputs'][1:]==[10]
  assert [int(v) for v in re.findall(r'SliceBitmap \[(\d+)\]',desc)]==[16*(s//16) for s in range(256)]
  assert h.offsets(h.sequence(desc,'fetch_sequencer'))==list(range(valid)) and g.mappings(sid)['Partitioning']==leader
  assert g.tensor(sid)['size']==512*valid*4 and g.tensor(tid)['size']==512*32 and g.tensor(tid)['buffer_type']=='Vrf'
  assert g.mappings(tid)['Partitioning']['Pair']['right']=={'Broadcast':{'size':16}}
  renamed=o[alias]['outputs'][0];assert o[alias]['inputs']==[tid]
  assert g.tensor(renamed)['optionals']['address']==g.tensor(tid)['optionals']['address'] and g.tensor(renamed)['size']==g.tensor(tid)['size']
  assert g.mappings(renamed)['InSlice']==g.mappings(tid)['InSlice']
  assert g.mappings(renamed)['Partitioning']['Pair']['right']=={'Stride':{'inner':{'Symbol':{'symbol':'Qs','size':4096}},'stride':256}}
  broadcasts.append({'operator':num,'source_valid_f32_indices':list(range(valid)),'valid_source_bytes_per_slice':valid*4,'destination_vrf_bytes_per_slice':32,'bitmap_entries':256,'mapping':'same cluster, source=16*(destination//16)','rename_operator':alias})
 assert h.sequence(o[58]['description'],'fetch_sequencer')==h.sequence(oo[66]['description'],'fetch_sequencer')
 assert 'fetch_size: 8' in o[58]['description'] and 'fetch_size: 8' in oo[66]['description']
 assert '[OutputTerm_1=4]+4]=8' in o[58]['description'] and '[OutputTerm_1=4]+4]=8' in oo[66]['description']
 cfg=list(a.packets(o[3]));assert len(cfg)==512 and all(hi-lo==dh-dl==32 for _,_,_,(lo,hi),(dl,dh) in cfg)
 sums=[]
 for num,oldnum,oldload in [(22,23,24),(30,32,33)]:
  n,z=o[num],oo[oldnum];nd=n['description'];od=z['description']
  assert nd.split('WriteVrf:')[0].rstrip()==od.split('commit_unit:')[0].rstrip()
  assert n['context']=='Main' and 'WriteVrf: WriteVrf' in nd and 'to Vrf' in nd and 'commit_unit:' not in nd
  assert g.tensor(n['outputs'][0])['buffer_type']=='Vrf' and g.tensor(n['outputs'][0])['size']==512*256*4
  assert g.tensor(n['outputs'][0])['shape']==og.tensor(oo[oldload]['outputs'][0])['shape']
  assert [g.tensor(t)['shape'] for t in n['inputs']]==[og.tensor(t)['shape'] for t in z['inputs']]
  assert h.offsets(h.sequence(nd,'fetch_sequencer'))==list(range(256))
  sums.append({'operator':num,'old_sum_operator':oldnum,'old_reload_operator':oldload,'f32_ve_arithmetic_and_readers_exact':True,'direct_vrf_bytes_per_slice':1024,'eliminated_sram_write_bytes':524288,'eliminated_sub_sram_read_bytes':524288})
 # Unchanged descriptor matching also covers both 60-row HBM windows, disjoint native commits and canonical gather.
 oldnative=[oo[78],oo[83]]
 natives=[x for x in o.values() if a.signature(g,x)==a.signature(og,oo[78])]
 assert len(natives)==2
 natives.sort(key=lambda x:g.view(x['inputs'][3])['start'])
 oldnative.sort(key=lambda x:og.view(x['inputs'][3])['start'])
 newnative=[]
 for n,z,start in zip(natives,oldnative,[0,60]):
  nv,ov=g.view(n['inputs'][3]),og.view(z['inputs'][3])
  for key in ('stride_elements','axis_extent','window','start'):assert nv[key]==ov[key]
  assert nv['start']==start and nv['window']==60 and nv['axis_extent']==120
  assert h.offsets(h.sequence(n['description'],'commit_projection'))==list(range(60)) and 'commit_in_size: 8B' in n['description']
  newnative.append({'operator':n['name'],'destination_view':nv})
 oldweights=[oo[70],oo[73]];weight=[]
 weights=[x for x in o.values() if a.signature(g,x)==a.signature(og,oo[70])];assert len(weights)==2
 weights.sort(key=lambda x:g.view(x['inputs'][0])['start']);oldweights.sort(key=lambda x:og.view(x['inputs'][0])['start'])
 for w,z,start in zip(weights,oldweights,[0,60]):
  nv,ov=g.view(w['inputs'][0]),og.view(z['inputs'][0])
  for key in ('stride_elements','axis_extent','window','start'):assert nv[key]==ov[key]
  assert nv['start']==start and nv['window']==60 and nv['axis_extent']==120
  ps=list(a.packets(w));assert all(hi-lo==dh-dl==256 for _,_,_,(lo,hi),(dl,dh) in ps)
  weight.append({'operator':w['name'],'old_operator':z['name'],'source_view':nv,'packets_256B':len(ps),'logical_bytes':sum(hi-lo for _,_,_,(lo,hi),_ in ps),'exact_source_destination_descriptor_match':True})
 assert sum(x['logical_bytes'] for x in weight)==15728640
 schedule=load(ev/'schedule.json');ts={int(t['index']):t for t in schedule['tensors']};regs=[];direct=[]
 for ins in schedule['instructions']:
  if '.to_vrf(&mut device.sub)' in ins['description']:
   assert ins['tpe']=='Main' and set(ins['contexts'])=={'MainContext','SubContext','VectorEngine'}
   rr=[ts[t] for t in ins['input_tensors'] if t in ts and ts[t]['buffer_type']=='Vrf'];out=[ts[t] for t in ins['output_tensors']];assert len(rr)==len(out)==1 and out[0]['buffer_type']=='Vrf' and out[0]['size']==524288
   assert rr[0]['lifetime']['begin']<=ins['lifetime']['begin']<=ins['lifetime']['end']<=rr[0]['lifetime']['end'];assert out[0]['lifetime']['begin']<=ins['lifetime']['begin']<=ins['lifetime']['end']<=out[0]['lifetime']['end']
   direct.append({'instruction':ins['index'],'contexts':ins['contexts'],'lifetime':ins['lifetime'],'input_vrf':rr[0],'output_vrf':out[0]})
  if ins['tpe']=='Main' and '.contract_outer' in ins['description']:
   rr=[ts[t] for t in ins['input_tensors'] if t in ts and ts[t]['buffer_type'] in ('Trf','Vrf')];assert sorted(t['buffer_type'] for t in rr)==['Trf','Vrf']
   for t in rr:assert t['lifetime']['begin']<=ins['lifetime']['begin']<=ins['lifetime']['end']<=t['lifetime']['end']
   regs.append({'instruction':ins['index'],'lifetime':ins['lifetime'],'registers':rr})
 assert len(direct)==2 and len(regs)==2 and len({tuple(t['index'] for t in row['registers']) for row in regs})==1
 vrfs=[t for t in ts.values() if t['buffer_type']=='Vrf']
 for i,x in enumerate(vrfs):
  assert 0<=x['address']<x['address']+x['size']<=8192*512
  for y in vrfs[i+1:]:
   time=max(x['lifetime']['begin'],y['lifetime']['begin'])<min(x['lifetime']['end'],y['lifetime']['end']);addr=max(x['address'],y['address'])<min(x['address']+x['size'],y['address']+y['size']);assert not(time and addr),(x['index'],y['index'])
 bound=[bp,src,HERE/b['variant']/'recipe.json',ev/'graph.json',ev/'schedule.json',ref,oldref,oldsrc,OLD/'evidence/e137-native-v3/graph.json',HERE.parent/'qkv-fullh-weight-switch/audit.py',a.HELPER,HERE/'audit.py']+sorted((HERE/'numerical').glob('*'))
 return {'passed':True,'candidate':'E148','label':label,'inputdigests':{str(p.relative_to(ROOT)):rec(p) for p in bound if p.is_file()},'source':{'compiled_inputs':46,'unchanged_other_inputs':45,'changed':['src/device/sliding/projection.rs'],'qkv_prefix_exact_e147':True,'native_two_stages_gather_scale_epilogue_exact_except_obsolete_coefficient_load':True,'four_residual_arithmetic_and_constants_byte_identical':True},'scale':{'operator':2,'finite_bf16_cpu_patterns':65280,'source_values_per_slice':256,'global_max_ratio':16,'leader_q0_only':True},'scalar_broadcasts':broadcasts,'coefficient_valid_reader_and_padding_equal_old_o':True,'shared_bitmap':{'operator':3,'hbm_bytes':16384,'consumers':[4,58]},'direct_cumulative_sums':sums,'unchanged_emitted_signatures':sum(nc.values()),'weight_ingress':weight,'unchanged_native_operators':newnative,'direct_vrf_scheduled_resources':direct,'native_shared_register_lifetimes':regs,'all_vrf_lifetimes_nonoverlapping_within_8kib':True,'cost':{'removed_two_cumulative_sum_sram_writes_bytes':1048576,'removed_two_sub_sum_reads_bytes':1048576,'removed_scalar_consumer_dm_write_and_sub_read_bytes_each':24576,'added_shared_bitmap_bytes':16384,'new_scale_mul_input_values':131072,'two_direct_sum_operations_use_Main_Sub_VectorEngine':True,'no_performance_inference':True},'scope':'Source, actual descriptors/valid readers and scheduled register/engine contracts. Exact-image NPU correctness/performance are separate.','limits':['Printed fetch_size=8 also occurs in the old compact coefficient path; exposed valid sequencer is0..3, padded VRF32B. Hidden physical bus-read granularity is not inferred.','Finite-BF16 scalar equality does not establish full native arithmetic, nonfinite or end-to-end model correctness.','Removing DM traffic adds direct Main/Sub VRF occupancy; static schedule is not measured overlap or latency.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('label',nargs='?',default='e148-direct-v1');p.add_argument('--check',action='store_true');args=p.parse_args();r=json.loads(json.dumps(audit(args.label)));out=HERE/f'audit-{args.label}.json'
 if args.check:assert load(out)==r,'Audit record differs'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS',out)
if __name__=='__main__':main()
