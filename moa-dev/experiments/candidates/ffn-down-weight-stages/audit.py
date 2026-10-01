#!/usr/bin/env python3
"""Independent E150 exact whole graph bounds/partial/register gate; offline only."""
import argparse,hashlib,importlib.util,json,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];E147=HERE.parent/'qkv-scalar-sub-vrf';E142=HERE.parent/'ffn-direct-full30-partials'
def imp(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=imp('e144_helpers',HERE.parent/'qkv-fullh-weight-switch/audit.py');h=a.h
prior=imp('e141_prefix',HERE.parent/'ffn-independent-direct-copy-repair/independent-audit.py')
def load(p):return json.loads(p.read_text())
def rec(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def digest(p):return {k:v for k,v in rec(p).items()if k!='path'}
def audit(label):
 assert label=='e150-staged-v1'
 bp=HERE/f'build-{label}-repro.json';b=load(bp);ev=HERE/'evidence'/label;art=ROOT/'moa-dev/experiments/artifacts'/HERE.name/label
 ref=E147/'measured-build-e147-adoption.json';base=load(ref);oldref=E142/'measured-build-e142-adoption.json';older=load(oldref)
 assert b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container'] and b['base_commit']=='fb45202bba3c24537f74b4bf463276b76d823773'
 assert len(b['source_inputs'])==46 and b['source_inputs'].keys()==base['source_inputs'].keys()
 assert {k for k in b['source_inputs']if b['source_inputs'][k]!=base['source_inputs'][k]}=={'src/device/shared/mlp.rs'}
 assert base['images']['decoder_feedforward']==older['images']['decoder_feedforward']
 for name,d in b['source_inputs'].items():assert digest(art/'crate'/name)==d
 src=HERE/b['variant']/'mlp.rs';assert digest(src)==b['source_inputs']['src/device/shared/mlp.rs'] and src.read_bytes()==(ev/'src/device/shared/mlp.rs').read_bytes()
 for n in ('graph','schedule'):
  assert digest(ev/f'{n}.json')==b[n] and (ev/f'{n}.json').read_bytes()==(art/f'{n}.json').read_bytes()
 assert digest(art/'exact-kernel.bin')==b['exact_image']
 subprocess.run([sys.executable,str(HERE/'geometry.py'),'--check'],check=True,stdout=subprocess.DEVNULL)
 g=h.Graph(load(ev/'graph.json'));og=h.Graph(load(E142/'evidence/e142-full30-v1/graph.json'));schedule=load(ev/'schedule.json')
 prefix=prior.prefix_and_bitmap(g,schedule,art/'exact-kernel.bin')
 def isdown(o):return o['kind']=='TuExec' and o['context']=='Main' and 'lookup_table:'in o['description'] and 'LReduce(+) [axes=FfnTerm]'in o['description']
 ns=[o for o in g.ops.values()if isdown(o)];oldnative=[o for o in og.ops.values()if isdown(o)];assert len(ns)==2 and len(oldnative)==1;oldnative=oldnative[0]
 ns.sort(key=lambda n:g.view(n['inputs'][-1])['start']);native=[];ingress=[];allspans=[];roots=set();downregs=set()
 for n,off in zip(ns,[0,30]):
  weight,trf,coeff,table,dst=n['inputs'];w=g.producer(weight,'Dma.DtoS')[1];v=g.view(w['inputs'][0]);dv=g.view(dst)
  assert (v['stride_elements'],v['axis_extent'],v['window'],v['start'])==(15360,60,30,off)
  assert g.tensor(v['parent'])['optionals']['address']=='IO(4)+0'
  assert (dv['stride_elements'],dv['axis_extent'],dv['window'],dv['start'])==(120,60,30,off)
  root=g.root_output(dv['parent']);roots.add(root);assert g.tensor(root)['size']==512*28800 and g.tensor(root)['element_type']=='Float32'
  assert g.tensor(weight)['size']==512*28800 and g.tensor(weight)['element_type']=='Float4e2m1'
  seen=set();spans=[]
  for _,sa,da,(lo,hi),(dl,dh)in a.packets(w):
   assert sa['chip']==da['chip']==0 and hi-lo==dh-dl==960
   c=da['cluster'];sl=a.sl(da,dl);group,q=divmod(sl,4);local=dl%a.M;r,u=divmod(local,960)
   assert 0<=sl<256 and 0<=r<30 and u==0 and local+960<=28800
   assert lo==(60*group+r)*7680+c*3840+q*960
   abslo=lo+off*7680;spans.append((abslo,abslo+960));assert(c,sl,local)not in seen;seen.add((c,sl,local))
  assert len(seen)==15360;allspans+=spans;ingress.append({'operator':w['name'],'source_view':v,'descriptor':h.dma_descriptors(w['description']),'packet_bytes':960,'packet_count':15360,'bytes':14745600,'slice_bytes':28800,'slice_128B_units':225,'source_row_pitch_bytes':7680})
  d=n['description'];od=oldnative['description'];assert h.offsets(h.sequence(d,'fetch_sequencer'))==list(range(30*1920))
  assert d[d.index('operation_unit:'):d.index('commit_unit:')].replace('H_1=30','H_1=60')==od[od.index('operation_unit:'):od.index('commit_unit:')]
  assert h.offsets(h.sequence(d,'commit_projection'))==list(range(3600)) and 'commit_in_size: 16B (=4EA)'in d
  assert g.tensor(trf)['buffer_type']=='Trf' and g.tensor(trf)['size']==512*3*1920
  assert g.tensor(coeff)['buffer_type']=='Vrf' and g.tensor(coeff)['size']==512*96
  assert g.tensor(trf)['shape']==og.tensor(oldnative['inputs'][1])['shape'] and g.tensor(coeff)['shape']==og.tensor(oldnative['inputs'][2])['shape']
  assert g.producer(table,'StoTab')[1]['outputs']==[table];downregs.add((trf,coeff))
  native.append({'operator':n['name'],'weight':weight,'trf':trf,'coefficient_vrf':coeff,'lookup_table':table,'output_view':dv,'partial_root':root,'partial_byte_bounds':[off*480,(off+30)*480],'commit_packets_per_slice':900,'commit_bytes':16,'weighted_term_ve_order_identical_to_original':True})
 assert len(roots)==len(downregs)==1;partial_root=next(iter(roots));coverage=h.full_coverage(allspans,29491200,'Both staged original down FP4 ingress')
 consumers=[];oldcons=[o for o in og.ops.values()if o['kind']=='TuExec' and o['context']=='Main' and 'LReduce(+) [axes=TileL]'in o['description'] and 'H_1=12'in o['description']];assert len(oldcons)==5
 for o in g.ops.values():
  d=o['description']
  if o['kind']!='TuExec' or o['context']!='Main' or 'LReduce(+) [axes=TileL]'not in d or 'H_1=12'not in d:continue
  pv,vrf,dst=o['inputs'];v=g.view(pv);dv=g.view(dst);sp=g.producer(vrf,'TuExec')[1];sv=g.view(sp['inputs'][0])
  assert g.root_output(v['parent'])==partial_root
  for view,stride in [(v,120),(sv,120),(dv,1)]:assert(view['stride_elements'],view['axis_extent'],view['window'])==(stride,60,12)
  assert v['start']==sv['start']==dv['start'] and v['start']in (0,12,24,36,48)
  assert h.offsets(h.sequence(d,'fetch_sequencer'))==list(range(1440)) and h.offsets(h.sequence(sp['description'],'fetch_sequencer'))==list(range(1440))
  assert sp['context']=='Sub' and g.tensor(vrf)['buffer_type']=='Vrf' and g.tensor(vrf)['size']==512*5760
  assert h.offsets(h.sequence(d,'commit_projection'))==list(range(12)) and 'commit_in_size: 8B (=2EA)'in d
  assert any(a.signature(g,o)==a.signature(og,z)for z in oldcons),'Changed original partial/scale/reduction arithmetic'
  consumers.append({'operator':o['name'],'partial_view':v,'scale_view':sv,'output_view':dv,'partial_byte_bounds':[v['start']*480,(v['start']+12)*480],'read_after_both_stages':v['parent']})
 assert len(consumers)==5 and sorted(x['partial_view']['start']for x in consumers)==[0,12,24,36,48]
 parent=consumers[0]['partial_view']['parent'];assert len({x['partial_view']['parent']for x in consumers})==1
 chain=[];t=parent
 while t!=partial_root:
  _,op=g.producer(t,'IndexWrite');chain.append(op['inputs'][-1]);t=op['inputs'][0]
 assert set(chain)=={n['outputs'][0]for n in ns},'Consumers must see both native writes'
 ts={int(t['index']):t for t in schedule['tensors']};vrfs=[t for t in ts.values()if t['buffer_type']=='Vrf'];regs=[]
 for i,x in enumerate(vrfs):
  assert 0<=x['address']<x['address']+x['size']<=8192*512
  for y in vrfs[i+1:]:
   live=max(x['lifetime']['begin'],y['lifetime']['begin'])<min(x['lifetime']['end'],y['lifetime']['end']);overlap=max(x['address'],y['address'])<min(x['address']+x['size'],y['address']+y['size']);assert not(live and overlap),(x['index'],y['index'])
 for i in schedule['instructions']:
  for tid in i['input_tensors']:
   t=ts.get(tid)
   if t is None or t['buffer_type']not in ('Vrf','Trf'):continue
   assert t['lifetime']['begin']<=i['lifetime']['begin']<=i['lifetime']['end']<=t['lifetime']['end']
  if i['tpe']=='Main' and '.contract_outer' in i['description'] and 'H % 60 = 30' in i['description']:
   rr=[ts[t]for t in i['input_tensors']if t in ts and ts[t]['buffer_type']in ('Trf','Vrf')];assert sorted(x['buffer_type']for x in rr)==['Trf','Vrf'];regs.append({'instruction':i['index'],'lifetime':i['lifetime'],'registers':rr})
 assert len(regs)==2 and len({tuple(t['index']for t in r['registers'])for r in regs})==1
 bound={'build_record':bp,'graph':ev/'graph.json','schedule':ev/'schedule.json','source':src,'recipe':HERE/b['variant']/'recipe.json','baseline_build':ref,'original_ffn_build':oldref,'original_ffn_graph':E142/'evidence/e142-full30-v1/graph.json','geometry_verifier':HERE/'geometry.py','geometry':HERE/'geometry.json','helpers':HERE.parent/'qkv-fullh-weight-switch/audit.py','descriptor_helpers':a.HELPER,'prefix_helpers':HERE.parent/'ffn-independent-direct-copy-repair/independent-audit.py','verifier':Path(__file__)}
 return {'passed':True,'candidate':'E150','label':label,'inputdigests':{k:rec(p)for k,p in bound.items()},'source':{'compiled_inputs':46,'unchanged_other_inputs':45,'changed':['src/device/shared/mlp.rs'],'quantizer_upgate_original_down_consumers_and_finish_unchanged':True},'up_prefix_copy_and_bitmap':prefix,'staged_weight_ingress':ingress,'all_weight_coverage':coverage,'native_half_producers':native,'partial_consumers':consumers,'two_native_write_join_checked':True,'cross_half_consumer_rows_inclusive':[24,35],'shared_down_register_lifetimes':regs,'whole_vrf_buffers_checked':len(vrfs),'all_vrf_lifetimes_nonoverlapping_within_8kib':True,'lookup_tables_before_after':[sum(o['kind']=='StoTab'for o in og.ops.values()),sum(o['kind']=='StoTab'for o in g.ops.values())],'cost':{'packed_weight_bytes':29491200,'partial_write_read_bytes_each':14745600,'new_partial_assembly_bytes':0,'down_native_before_after':[1,2]},'scope':'Actual compiled source/bounds/term order/partial join/register contracts; separate finite whole and down-boundary NPU checks are required. No schedule-order or latency claim.','limits':['28800B packed half has128B phase; odd225x128B count is not a demonstrated bank/runtime hazard.','Splitting source transfer does not force up gather to run between the two DMA tasks.','Native setup/table and full-partial join are added work; overlapping static envelopes cannot be summed as savings.']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('label',nargs='?',default='e150-staged-v1');p.add_argument('--check',action='store_true');args=p.parse_args();r=json.loads(json.dumps(audit(args.label)));out=HERE/f'audit-{args.label}.json'
 if args.check:assert load(out)==r,'Audit record differs'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS',out)
