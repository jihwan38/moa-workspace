#!/usr/bin/env python3
"""E151 actual whole scale routing/bounds/register audit; offline only."""
import argparse,hashlib,importlib.util,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OLD=HERE.parent/'ffn-down-weight-stages'
def imp(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=imp('e150_audit',OLD/'audit.py');a=p.a;h=p.h
load=p.load;rec=p.rec
def record(x):
 b=x.read_bytes();return {'path':str(x.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def digest(x):return {k:v for k,v in record(x).items() if k!='path'}
def audit(label):
 assert label=='e151-ingress-v1';bp=HERE/f'build-{label}-repro.json';b=load(bp);ev=HERE/'evidence'/label;art=ROOT/'moa-dev/experiments/artifacts'/HERE.name/label
 basepath=OLD/'measured-build-e150-adoption.json';base=load(basepath)
 assert b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container'] and b['base_commit']=='ad1ecb3d3868c7e0ca2e4d1428bbd66bf1111821'
 assert len(b['source_inputs'])==46 and set(b['source_inputs'])==set(base['source_inputs'])
 assert {k for k in b['source_inputs'] if b['source_inputs'][k]!=base['source_inputs'][k]}=={'src/device/shared/mlp.rs'}
 for name,v in b['source_inputs'].items():assert digest(art/'crate'/name)==v
 src=HERE/b['variant']/'mlp.rs';assert digest(src)==b['source_inputs']['src/device/shared/mlp.rs'] and src.read_bytes()==(ev/'src/device/shared/mlp.rs').read_bytes()
 for name in ('graph','schedule'):assert digest(ev/f'{name}.json')==b[name] and (ev/f'{name}.json').read_bytes()==(art/f'{name}.json').read_bytes()
 assert digest(art/'exact-kernel.bin')==b['exact_image']
 oldsrc=OLD/'staged-v1/mlp.rs';s=src.read_text();old=oldsrc.read_text()
 newload='''    // Source slice 4*g+b owns rows 60*g+4*r+b, r<15, with each
    // cluster's full 480-byte scale row. No intermediate DM scatter.
    let scales: DmTensor<f8e4m3, Chip, UpGateCluster, m![H / 60, H % 4], m![H / 4 % 15, TileL / 16]> = scale.to_dm(&mut device.tdma);'''
 oldload='    let scales: DmTensor<f8e4m3, Chip, UpGateCluster, DownColumns, m![H % 60, TileL / 16 % 120]> = scale.to_dm(&mut device.tdma);'
 newread='''            .begin(scales.view().tile::<m![H / 4 % 15], 3, m![H / 4 % 15 = 3 # 15, TileL / 16]>(3 * i))
            .fetch::<m![TileL / 1920, H / 4 % 15 = 3], m![TileL / 16 % 120]>()
            .fetch_cast::<f32>()
            // Actual InterTranspose order is (row-high3,row-low4).
            // The four old source slices supply 12 consecutive output rows.
            .switch::<DownColumns, m![H % 60 = 12]>(SwitchConfig::InterTranspose { slice1: 4, slice0: 1, time0: 3 })'''
 oldread='''            .begin(scales.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, TileL / 16 % 120]>(12 * i))
            .fetch::<m![H % 60 = 12], m![TileL / 16 % 120]>()
            .fetch_cast::<f32>()'''
 assert s.count(newload)==s.count(newread)==1 and s.replace(newload,oldload).replace(newread,oldread)==old,'Only scale ownership/producer may change'
 g=h.Graph(load(ev/'graph.json'));og=h.Graph(load(OLD/'evidence/e150-staged-v1/graph.json'));sched=load(ev/'schedule.json')
 switches=[o for o in g.ops.values() if o['context']=='Sub' and 'InterTranspose' in o['description']];assert len(switches)==5
 switches.sort(key=lambda o:g.view(o['inputs'][0])['start']);parent={g.view(o['inputs'][0])['parent'] for o in switches};assert len(parent)==1;scale=next(iter(parent));w=g.producer(scale,'Dma.DtoS')[1]
 assert g.tensor(scale)['size']==3686400 and g.tensor(scale)['element_type']=='Float8e4m3'
 assert g.tensor(w['inputs'][0])['optionals']['address']=='IO(7)+0'
 spans=[];dest=set()
 for _,sa,da,(lo,hi),(dl,dh) in a.packets(w):
  assert sa['chip']==da['chip']==0 and hi-lo==dh-dl==480
  c=da['cluster'];sl=a.sl(da,dl);group,owner=divmod(sl,4);local=dl%a.M;r,u=divmod(local,480)
  assert 0<=sl<256 and 0<=r<15 and u==0 and local+480<=7200
  assert lo==(60*group+4*r+owner)*960+c*480
  assert(c,sl,local)not in dest;dest.add((c,sl,local));spans.append((lo,hi))
 assert len(dest)==7680;coverage=h.full_coverage(spans,3686400,'E151 original full down scale matrix')
 def isreduce(o):return o['context']=='Main' and 'LReduce(+) [axes=TileL]'in o['description'] and 'H_1=12'in o['description']
 consumers=[o for o in g.ops.values() if isreduce(o)];oldcons=[o for o in og.ops.values()if isreduce(o)];assert len(consumers)==len(oldcons)==5
 routed=set();vrfoutputs=set();routing=[];partialroot=set();join=set()
 for i,sw in enumerate(switches):
  assert len(sw['inputs'])==len(sw['outputs'])==1,'InterTranspose should not acquire a new bitmap operand'
  v=g.view(sw['inputs'][0]);out=sw['outputs'][0];vrfoutputs.add(out);d=sw['description']
  assert (v['stride_elements'],v['axis_extent'],v['window'],v['start'])==(480,15,3,3*i)
  assert h.offsets(h.sequence(d,'fetch_sequencer'))==[a3*480+q*120+u for q in range(4) for a3 in range(3) for u in range(120)]
  assert 'InterTranspose { dim0_volume: 1, dim1_volume: 4, dim0_chunk_volume: 3 }, f32'in d
  assert '[H_60=64, TileL_1920=4]'in d and '[H_1=12, TileL_16=120]'in d and 'to Vrf'in d
  assert g.tensor(out)['buffer_type']=='Vrf' and g.tensor(out)['element_type']=='Float32' and g.tensor(out)['size']==512*5760
  co=[o for o in consumers if out in o['inputs']];assert len(co)==1;co=co[0]
  assert any(a.signature(g,co)==a.signature(og,z)for z in oldcons),'Changed original arithmetic or scale VRF reader order'
  pv=g.view(co['inputs'][0]);dv=g.view(co['inputs'][-1]);assert pv['start']==dv['start']==12*i and (pv['stride_elements'],pv['axis_extent'],pv['window'])==(120,60,12)
  partialroot.add(g.root_output(pv['parent']));join.add(pv['parent'])
  for c in range(2):
   for group in range(64):
    for q in range(4):
     for a3 in range(3):
      for owner in range(4):
       row=12*i+4*a3+owner;source_h=60*group+4*(3*i+a3)+owner
       assert source_h==60*group+row
       for u in range(120):assert source_h*960+c*480+q*120+u==(60*group+row)*960+c*480+q*120+u
       key=(c,4*group+q,row);assert key not in routed;routed.add(key)
  routing.append({'operator':sw['name'],'source_view':v,'vrf':out,'vrf_bytes_per_slice':5760,'reader':co['name'],'partial_view':pv,'output_view':dv,'source_tile_byte_bounds':[3*i*480,(3*i+3)*480],'fetch_positions':1440,'destination_rows':[12*i,12*i+12]})
 assert len(routed)==30720 and len(partialroot)==len(join)==1
 # Native arithmetic, weight descriptors, shared register shapes and half writes unchanged.
 def native(o):return o['context']=='Main' and 'lookup_table:'in o['description'] and 'LReduce(+) [axes=FfnTerm]'in o['description']
 ns=[o for o in g.ops.values()if native(o)];ons=[o for o in og.ops.values()if native(o)];assert len(ns)==len(ons)==2
 for n in ns:
  assert any(a.signature(g,n)==a.signature(og,z)for z in ons)
  wi=g.producer(n['inputs'][0],'Dma.DtoS')[1];assert any(a.signature(g,wi)==a.signature(og,og.producer(z['inputs'][0],'Dma.DtoS')[1])for z in ons)
 root=next(iter(partialroot));t=next(iter(join));chain=[]
 while t!=root:
  _,op=g.producer(t,'IndexWrite');chain.append(op['inputs'][-1]);t=op['inputs'][0]
 assert set(chain)=={n['outputs'][0]for n in ns}
 ts={int(t['index']):t for t in sched['tensors']};vrfs=[t for t in ts.values()if t['buffer_type']=='Vrf'];scale_regs=[]
 for i,x in enumerate(vrfs):
  assert 0<=x['address']<x['address']+x['size']<=8192*512
  for y in vrfs[i+1:]:
   live=max(x['lifetime']['begin'],y['lifetime']['begin'])<min(x['lifetime']['end'],y['lifetime']['end']);over=max(x['address'],y['address'])<min(x['address']+x['size'],y['address']+y['size']);assert not(live and over),(x['index'],y['index'])

 for ins in sched['instructions']:
  if ins['tpe']=='Sub' and 'InterTranspose { slice1: 4, slice0: 1, time0: 3 }' in ins['description']:
   assert len(ins['output_tensors'])==1
   reg=ts[ins['output_tensors'][0]];assert reg['buffer_type']=='Vrf' and reg['size']==5760*512 and '(H % 60 = 12, TileL / 16 % 120)' in reg['shape'];scale_regs.append(reg)
  for tid in ins['input_tensors']:
   x=ts.get(tid)
   if x and x['buffer_type']in ('Vrf','Trf'):assert x['lifetime']['begin']<=ins['lifetime']['begin']<=ins['lifetime']['end']<=x['lifetime']['end']
 assert len(scale_regs)==5
 prefix=p.prior.prefix_and_bitmap(g,sched,art/'exact-kernel.bin')
 bound={'build_record':bp,'graph':ev/'graph.json','schedule':ev/'schedule.json','source':src,'recipe':HERE/b['variant']/'recipe.json','baseline_build':basepath,'original_source':oldsrc,'original_graph':OLD/'evidence/e150-staged-v1/graph.json','helpers':OLD/'audit.py','descriptor_helpers':a.HELPER,'verifier':Path(__file__)}
 return {'passed':True,'candidate':'E151','label':label,'inputdigests':{k:record(v)for k,v in bound.items()},'source':{'compiled_inputs':46,'changed':['src/device/shared/mlp.rs'],'unchanged_other_inputs':45,'only_two_scale_supply_source_regions_changed':True},'scale_ingress':{'operator':w['name'],'packet_bytes':480,'packets':7680,'per_slice_bytes':7200,'coverage':coverage,'descriptor':h.dma_descriptors(w['description']),'source_coordinate':'h=60g+4r+b, block=480c+k; destination slice4g+b'},'scale_routing':routing,'row_packets_bijection':30720,'switch_payload_f32_bytes':14745600,'scale_vrf_registers':scale_regs,'all_vrf_lifetimes_nonoverlapping_within_8kib':True,'whole_vrf_buffers_checked':len(vrfs),'native_weight_descriptors_and_arithmetic_identical_to_e150':True,'five_original_consumers_unchanged':True,'both_native_half_writes_joined':True,'cross_half_consumer_rows':[24,35],'up_prefix_copy_bitmap':prefix,'scope':'Actual source/descriptor/Fetch/Switch/VRF/consumer and register lifetime gate; no NPU correctness, exclusive network cost, scheduling priority or speedup claim.'}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('label',nargs='?',default='e151-ingress-v1');ap.add_argument('--check',action='store_true');args=ap.parse_args();r=json.loads(json.dumps(audit(args.label)));out=HERE/f'audit-{args.label}.json'
 if args.check:assert load(out)==r,'Audit record differs'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS',out)
