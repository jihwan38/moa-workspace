#!/usr/bin/env python3
"""Offline source/image/name-only and source-supported marker gate for E153 readiness."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[4]
def load(p): return json.loads(p.read_text())
def rec(p):
 b=p.read_bytes(); return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def verify(r): assert rec(ROOT/r['path'])==r,r['path']
def digest(r): return {k:r[k] for k in ('bytes','sha256')}
spec=importlib.util.spec_from_file_location('profile_abi',HERE/'image_profile.py'); abi=importlib.util.module_from_spec(spec); spec.loader.exec_module(abi)
def kind(n):
 if n['tpe'].startswith('Dma'): return 'DMA'
 if n['tpe']=='Main': return 'Renegade::TuExec'
 if n['tpe']=='Sub' and n['contexts']!=['Noop']:
  d=n['description']
  if '.to_trf()' in d:return 'Renegade::StoTrf'
  if not d.strip():return 'Renegade::StoTab'
  if '.to_vrf()' in d and 'fetch_cast' not in d and 'vector_' not in d and '.switch' not in d and '.collect::<m![1], m![OutputTerm # 8]>()' not in d:return 'Renegade::StoVrf'
  return 'Renegade::TuExec'
def patch_check(raw,host,probe):
 labelled,meta=abi.inspect(raw,True);_,after=abi.inspect(labelled)
 for k in ('version','chips','pes','header_bytes','tasks','weight','stacks','slots','inputs','outputs','profile_depth'):assert meta[k]==after[k],k
 allowed={i for row in meta['spans'] for s in row for i in range(s['name_offset'],s['name_offset']+s['name_bytes'])}
 assert len(raw)==len(labelled) and all(a==b or i in allowed for i,(a,b) in enumerate(zip(raw,labelled)))
 assert host.count(raw)==1 and probe==host.replace(raw,labelled) and probe.count(labelled)==1
 return meta,abi.digest(labelled)
def mapping(schedule,mp,meta,count):
 ops=[n for n in schedule['instructions'] if kind(n)]; assert len(meta['spans'])==1
 spans=sorted([s for s in meta['spans'][0] if s['name'] not in ('Task','Core','Cluster')],key=lambda s:s['begin_marker'])
 assert len(ops)==len(spans)==count
 assert [n['schedule']for n in mp['operations']]==ops and [n['marker']for n in mp['operations']]==spans
 assert [kind(n)for n in ops]==[s['name']for s in spans]
 return {n['index']:n for n in ops}
def review():
 p=load(HERE/'package-profile-v1.json');assert p['status']=='ready'
 c=HERE.parent;b=load(ROOT/p['origin']['build']['path']);parent=load(ROOT/p['origin']['package']['path']);control=load(c.parent/'ffn-down-scale-ingress-switch/measured-build-e151-adoption.json')
 bound=[HERE/'package-profile-v1.json',HERE/'image_profile.py',ROOT/p['origin']['build']['path'],ROOT/p['origin']['package']['path'],ROOT/p['origin']['graph_gate']['record']['path'],c.parent/'ffn-down-scale-ingress-switch/measured-build-e151-adoption.json']
 for x in [p['origin']['build'],p['origin']['package'],p['origin']['graph_gate']['record']]:verify(x)
 roles={
 'candidate':{'pre_rms_f32':135,'local_bf16':140,'local_amax':145,'global_amax':150,'up_pow2_scale':155,'quant_scale_dma':168,'quant_scale_vrf':172,'hi':178,'hi_vrf':182,'lo':191,'copy_hi':193,'copy_lo':202,'sum_terms':207,'sum_vrf':214,'third':218,'copy_third':228,'late_packed_dma':232,'ring_seed':262,'ring32':279,'up_shared_trf':300,'restore_scale_dma':186,'gate_weight':71,'gate_native':309,'up_native':394,'up_gather':512,'geglu':526,'down_input_gather':534,'down_weight0':410,'down_weight1':659,'down_ring':658,'down_trf':665,'down_native0':669,'down_native1':677,'down_scale_hbm':678,'down_scale_vrf_last':743,'down_reduce_last':751,'output_normalize':811,'residual_add':816,'layer_mul':821,'final_hbm':824},
 'control':{'pre_rms_f32':132,'canonical_bf16':137,'early_cluster_dma':153,'quant_reshard':165,'canonical_amax':164,'up_pow2_scale':171,'quant_scale_dma':174,'quant_scale_vrf':178,'hi':185,'hi_vrf':189,'lo':194,'copy_hi':197,'copy_lo':218,'sum_terms':206,'sum_vrf':210,'third':215,'copy_third':237,'ring_seed':251,'ring32':275,'up_shared_trf':291,'restore_scale_dma':180,'gate_weight':45,'gate_native':295,'up_native':411,'up_gather':498,'geglu':513,'down_input_gather':520,'down_weight0':430,'down_weight1':647,'down_ring':646,'down_trf':653,'down_native0':657,'down_native1':665,'down_scale_hbm':666,'down_scale_vrf_last':731,'down_reduce_last':739,'output_normalize':799,'residual_add':804,'layer_mul':809,'final_hbm':812}}
 summaries={}
 for side,original in [('control',control),('candidate',b)]:
  r=p['sides'][side]
  assert len(r['source_inputs'])==46 and r['source_inputs']==original['source_inputs']
  assert r['original_host_source']==original['source_inputs']['src/bin/test_kernels.rs']
  assert digest(r['diagnostic_host_source'])==r['diagnostic_inputs']['src/bin/test_kernels.rs']
  assert {k:v for k,v in r['diagnostic_inputs'].items()if k!='src/bin/test_kernels.rs'}=={k:v for k,v in original['source_inputs'].items()if k!='src/bin/test_kernels.rs'}
  for x in [r['unlabelled_host'],r['diagnostic_probe'],r['schedule'],r['graph'],r['diagnostic_host_source'],*r['images'].values()]:verify(x);bound.append(ROOT/x['path'])
  raw=(ROOT/r['images']['decoder_feedforward']['path']).read_bytes();expected=original['exact_image'] if side=='candidate' else original['images']['decoder_feedforward'];assert abi.digest(raw)==digest(expected)
  host=(ROOT/r['unlabelled_host']['path']).read_bytes();probe=(ROOT/r['diagnostic_probe']['path']).read_bytes();meta,labelled=patch_check(raw,host,probe)
  for name in ('sliding_project_qkv','sliding_attention_output'):
   im=(ROOT/r['images'][name]['path']).read_bytes();assert host.count(im)==probe.count(im)==1 and digest(r['images'][name])==digest(control['images'][name])
  mp=HERE/f'evidence/profile-v1/{side}-marker-map.json';md=HERE/f'evidence/profile-v1/{side}-ffn.image-profile.json';assert load(md)==json.loads(json.dumps(meta));bound.extend([mp,md])
  n=mapping(load(ROOT/r['schedule']['path']),load(mp),meta,r['operations'])
  for role,idx in roles[side].items():assert str(idx)in n,(side,role,idx)
  node=lambda role:n[str(roles[side][role])]
  for producer,consumer in [('quant_scale_dma','quant_scale_vrf'),('quant_scale_vrf','hi'),('hi','hi_vrf'),('sum_terms','sum_vrf'),('sum_vrf','third'),('ring_seed','ring32'),('gate_weight','gate_native'),('up_shared_trf','gate_native'),('up_shared_trf','up_native'),('down_weight0','down_native0'),('down_weight1','down_native1'),('down_trf','down_native0'),('down_trf','down_native1')]:
   assert set(node(producer)['output_tensors'])&set(node(consumer)['input_tensors']),(side,producer,consumer)
  assert 'gate_weight_packed.to_dm'in node('gate_weight')['description'] and 'Sub'==node('quant_scale_vrf')['tpe']
  assert 'CustomBroadcast { ring_size: 32 }'in node('ring32')['description'] and '.to_trf()'in node('up_shared_trf')['description']
  if side=='candidate':
   assert 'vector_inter_slice_reduce::<Slice, m![1]>(InterSliceReduceOpF32::Max)'in node('global_amax')['description']
   assert 'packed.to_dm'in node('late_packed_dma')['description'] and '.begin(normalized.view())'in node('local_bf16')['description'] and '.switch'not in node('local_bf16')['description']
  else:assert 'SwitchConfig::Broadcast1' in node('canonical_bf16')['description']
  for suffix in ['graph','schedule']:
   expected=original[suffix];assert digest(r[suffix])==digest(expected)
  src=HERE/f'evidence/profile-v1/{side}/production-mlp.rs';assert digest(rec(src))==original['source_inputs']['src/device/shared/mlp.rs'];bound.append(src)
  summaries[side]={'source_inputs':46,'operations':r['operations'],'original_image':abi.digest(raw),'labelled_image':labelled,'roles':roles[side]}
 return {'passed':True,'candidate':'E153-readiness','verifier':rec(Path(__file__)),'inputdigests':{str(x.relative_to(ROOT)):rec(x)for x in bound},'sides':summaries,'names_only_binary_difference':True,'instructions_addresses_io_constants_unchanged':True,'unchanged_qkv_o_images':True,'scope':'Paired actual E151/E153 FFN source/image/name-only and ordered-marker/producer-consumer audit. Marker spans include waiting and overlap; not exclusive service time or predicted savings.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=json.loads(json.dumps(review()));out=HERE/'independent-review.json'
 if args.check:assert load(out)==r,'Independent profile review changed'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS E153 readiness: actual E151/E153 FFN names-only and ordered markers')
