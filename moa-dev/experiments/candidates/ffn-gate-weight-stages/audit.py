#!/usr/bin/env python3
"""E152 actual gate-half ingress/native/partial/register audit; offline only."""
import argparse,importlib.util,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OLD=HERE.parent/'ffn-down-weight-stages'
s=importlib.util.spec_from_file_location('e150_helpers',OLD/'audit.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p);a=p.a;h=p.h
load=p.load;rec=p.rec;digest=p.digest
def audit(label):
 assert label=='e152-staged-v1';bp=HERE/f'build-{label}-repro.json';b=load(bp);ev=HERE/'evidence'/label;art=ROOT/'moa-dev/experiments/artifacts'/HERE.name/label;basepath=OLD/'measured-build-e150-adoption.json';base=load(basepath)
 assert b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container'] and b['base_commit']=='ad1ecb3d3868c7e0ca2e4d1428bbd66bf1111821'
 assert len(b['source_inputs'])==46 and b['source_inputs'].keys()==base['source_inputs'].keys() and {k for k in b['source_inputs']if b['source_inputs'][k]!=base['source_inputs'][k]}=={'src/device/shared/mlp.rs'}
 for name,v in b['source_inputs'].items():assert digest(art/'crate'/name)==v
 src=HERE/b['variant']/'mlp.rs';assert digest(src)==b['source_inputs']['src/device/shared/mlp.rs'] and src.read_bytes()==(ev/'src/device/shared/mlp.rs').read_bytes()
 for name in ('graph','schedule'):assert digest(ev/f'{name}.json')==b[name] and (ev/f'{name}.json').read_bytes()==(art/f'{name}.json').read_bytes()
 assert digest(art/'exact-kernel.bin')==b['exact_image']
 oldsrc=OLD/'staged-v1/mlp.rs';old=oldsrc.read_text();new=src.read_text();s=new
 start=s.index('// Prepared source only: compact gate half');end=s.index('pub(crate) fn project_up_and_gate',start);s=s[:start]+s[end:]
 split='''    let gate_weight_first: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow = 15, H]> = gate_weight_packed
        .tile::<m![FfnRow], 15, m![Dummy2, FfnSlice, FfnRow = 15 # 30, H]>(0)
        .to_dm(&mut device.tdma);
    let gate_weight_second: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow = 15, H]> = gate_weight_packed
        .tile::<m![FfnRow], 15, m![Dummy2, FfnSlice, FfnRow = 15 # 30, H]>(15)
        .to_dm(&mut device.tdma);'''
 original='''    let gate_weight_packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]> =
        gate_weight_packed.to_dm(&mut device.tdma);'''
 assert s.count(split)==1;s=s.replace(split,original).replace('project_gate_rows_staged(device, &x_trf, &x_scale_vrf, &gate_weight_first, &gate_weight_second, &gate_weight_scale, &mut gate);','project_rows_with_scale_contraction(device, &x_trf, &x_scale_vrf, &gate_weight_packed, &gate_weight_scale, &mut gate);');assert s==old
 g=h.Graph(load(ev/'graph.json'));og=h.Graph(load(OLD/'evidence/e150-staged-v1/graph.json'));sc=load(ev/'schedule.json');osc=load(OLD/'evidence/e150-staged-v1/schedule.json')
 pred=lambda o:o['context']=='Main'and'lookup_table:'in o['description']and'FfnRow'in o['description']
 alln=[o for o in g.ops.values()if pred(o)];oldn=[o for o in og.ops.values()if pred(o)];assert len(alln)==3 and len(oldn)==2
 ns=[o for o in alln if 'FfnRow_1=15'in o['description']];ns.sort(key=lambda o:g.view(o['inputs'][-1])['start']);assert len(ns)==2
 on=[o for o in oldn if og.tensor(og.producer(o['inputs'][0],'Dma.DtoS')[1]['inputs'][0])['optionals']['address']=='IO(3)+0'][0]
 native=[];ingress=[];spans=[];roots=set();trfs=set();coeffs=set()
 for n,off in zip(ns,[0,15]):
  weight,trf,coeff,table,dst=n['inputs'];w=g.producer(weight,'Dma.DtoS')[1];v=g.view(w['inputs'][0]);dv=g.view(dst)
  assert (v['stride_elements'],v['axis_extent'],v['window'],v['start'])==(3840,30,15,off) and g.tensor(v['parent'])['optionals']['address']=='IO(3)+0'
  assert (dv['stride_elements'],dv['axis_extent'],dv['window'],dv['start'])==(240,30,15,off)
  assert g.tensor(weight)['size']==512*28800 and g.tensor(weight)['element_type']=='Float4e2m1';seen=set()
  for _,sa,da,(lo,hi),(dl,dh) in a.packets(w):
   assert sa['chip']==da['chip']==0 and hi-lo==dh-dl==960
   c=da['cluster'];sl=a.sl(da,dl);local=dl%a.M;r,u=divmod(local,1920)
   assert 0<=sl<256 and 0<=r<15 and u in (0,960) and local+960<=28800
   assert lo==(c*7680+sl*30+r)*1920+u
   absolute=lo+off*1920;spans.append((absolute,absolute+960));assert(c,sl,local)not in seen;seen.add((c,sl,local))
  assert len(seen)==15360
  d=n['description'];od=on['description'];assert h.offsets(h.sequence(d,'fetch_sequencer'))==list(range(15*3840))
  assert d[d.index('operation_unit:'):d.index('commit_unit:')].replace('FfnRow_1=15','FfnRow_1=30')==od[od.index('operation_unit:'):od.index('commit_unit:')]
  assert h.offsets(h.sequence(d,'commit_projection'))==list(range(3600)) and 'commit_in_size: 16B (=4EA)'in d
  root=g.root_output(dv['parent']);roots.add(root);trfs.add(trf);coeffs.add(coeff);assert g.tensor(root)['size']==512*28800
  assert g.tensor(trf)['buffer_type']=='Trf'and g.tensor(trf)['size']==512*3*3840 and g.tensor(coeff)['buffer_type']=='Vrf'and g.tensor(coeff)['size']==512*32
  ingress.append({'operator':w['name'],'source_view':v,'packet_bytes':960,'packets':15360,'bytes':14745600,'per_slice_bytes':28800,'descriptor':h.dma_descriptors(w['description'])})
  native.append({'operator':n['name'],'weight':weight,'trf':trf,'coefficient_vrf':coeff,'table':table,'partial_view':dv,'partial_bytes':[off*960,(off+15)*960],'commit_packets_per_slice':900,'commit_bytes':16})
 assert len(roots)==len(trfs)==len(coeffs)==1;root=next(iter(roots));coverage=h.full_coverage(spans,29491200,'Gate two compact halves original FP4')
 assert len({n['inputs'][1]for n in alln})==1,'One emitted activation TRF must remain shared by up and both gate halves'
 up=[n for n in alln if n not in ns][0];assert any(a.signature(g,up)==a.signature(og,o)for o in oldn)
 def reduce(o):return o['context']=='Main'and'LReduce(+) [axes=H]'in o['description']and'FfnRow_1=6'in o['description']
 oldcons=[o for o in og.ops.values()if reduce(o)];cons=[o for o in g.ops.values()if reduce(o)];assert len(cons)==len(oldcons)==10
 gatecons=[];joins=set()
 for o in cons:
  assert any(a.signature(g,o)==a.signature(og,z)for z in oldcons)
  pv=g.view(o['inputs'][0]);
  if pv['parent']==up['outputs'][0]:continue
  assert g.root_output(pv['parent'])==root
  dv=g.view(o['inputs'][-1]);sp=g.producer(o['inputs'][1],'TuExec')[1];sv=g.view(sp['inputs'][0]);assert pv['start']==dv['start']==sv['start']
  assert(pv['stride_elements'],pv['axis_extent'],pv['window'])==(240,30,6) and pv['start']in (0,6,12,18,24)
  assert h.offsets(h.sequence(o['description'],'fetch_sequencer'))==list(range(1440)) and g.tensor(o['inputs'][1])['size']==512*5760
  joins.add(pv['parent']);gatecons.append({'operator':o['name'],'partial_view':pv,'scale_view':sv,'output_view':dv})
 assert len(gatecons)==5 and len(joins)==1;t=next(iter(joins));chain=[]
 while t!=root:
  _,op=g.producer(t,'IndexWrite');chain.append(op['inputs'][-1]);t=op['inputs'][0]
 assert set(chain)=={n['outputs'][0]for n in ns}
 # Unchanged adopted two-half weighted down arithmetic and original scale path.
 dn=lambda o:o['context']=='Main'and'lookup_table:'in o['description']and'LReduce(+) [axes=FfnTerm]'in o['description']
 dns=[o for o in g.ops.values()if dn(o)];odns=[o for o in og.ops.values()if dn(o)];assert len(dns)==len(odns)==2 and all(any(re.sub(r'T\d+','T',a.signature(g,o))==re.sub(r'T\d+','T',a.signature(og,z))for z in odns)for o in dns)
 ts={int(t['index']):t for t in sc['tensors']};vrfs=[t for t in ts.values()if t['buffer_type']=='Vrf'];shared=[]
 for i,x in enumerate(vrfs):
  assert 0<=x['address']<x['address']+x['size']<=8192*512
  for y in vrfs[i+1:]:
   assert not(max(x['lifetime']['begin'],y['lifetime']['begin'])<min(x['lifetime']['end'],y['lifetime']['end'])and max(x['address'],y['address'])<min(x['address']+x['size'],y['address']+y['size']))
 for ins in sc['instructions']:
  for tid in ins['input_tensors']:
   t=ts.get(tid)
   if t and t['buffer_type']in('Vrf','Trf'):assert t['lifetime']['begin']<=ins['lifetime']['begin']<=ins['lifetime']['end']<=t['lifetime']['end']
  if ins['tpe']=='Main'and '.contract_outer' in ins['description']and'FfnRow'in ins['description']:
   rr=[ts[t]for t in ins['input_tensors']if t in ts and ts[t]['buffer_type']in('Trf','Vrf')];shared.append({'instruction':ins['index'],'lifetime':ins['lifetime'],'registers':rr})
 assert len(shared)==3 and len({t['index']for i in shared for t in i['registers']if t['buffer_type']=='Trf'})==1
 prefix=p.prior.prefix_and_bitmap(g,sc,art/'exact-kernel.bin')
 static=[]
 for ins in sc['instructions']:
  if (ins['tpe'].startswith('Dma')and'gate_weight_packed' in ins['description'])or ins['index']=='141':static.append({'index':ins['index'],'tpe':ins['tpe'],'lifetime':ins['lifetime'],'source':ins['description']})
 bound={'build_record':bp,'graph':ev/'graph.json','schedule':ev/'schedule.json','source':src,'recipe':HERE/b['variant']/'recipe.json','baseline_build':basepath,'original_source':oldsrc,'original_graph':OLD/'evidence/e150-staged-v1/graph.json','helpers':OLD/'audit.py','descriptor_helpers':a.HELPER,'verifier':Path(__file__)}
 return {'passed':True,'candidate':'E152','label':label,'inputdigests':{k:rec(v)for k,v in bound.items()},'source':{'compiled_inputs':46,'changed':['src/device/shared/mlp.rs'],'other45_unchanged':True,'up_and_all_quantizers_and_e150_down_unchanged':True},'gate_ingress':ingress,'gate_weight_coverage':coverage,'gate_native_halves':native,'gate_six_row_consumers':gatecons,'both_half_write_join':True,'cross_half_consumer_rows':[12,17],'shared_up_gate_activation_trf':shared,'whole_vrf_buffers_checked':len(vrfs),'all_vrf_lifetimes_nonoverlapping_within_8kib':True,'original_down_natives_unchanged':True,'up_copy_bitmap':prefix,'static_ingress_and_input_gather':static,'lookup_tables_before_after':[sum(o['kind']=='StoTab'for o in og.ops.values()),sum(o['kind']=='StoTab'for o in g.ops.values())],'scope':'Actual compiled source, packet coverage, 15-row native/F32partial bounds, unchanged numerical consumers and whole register lifetimes. Static input-gather interleaving is not NPU timing or a performance promise.'}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('label',nargs='?',default='e152-staged-v1');ap.add_argument('--check',action='store_true');args=ap.parse_args();r=json.loads(json.dumps(audit(args.label)));out=HERE/f'audit-{args.label}.json'
 if args.check:assert load(out)==r,'Audit record differs'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS',out)
