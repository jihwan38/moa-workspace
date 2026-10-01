#!/usr/bin/env python3
"""Offline source/image/name-only and source-supported marker gate for E148 readiness."""
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
 cb=load(ROOT/p['control']['measured_build']['path']); b=load(ROOT/p['origin']['build']['path']); parent=load(ROOT/p['origin']['package']['path'])
 bound=[HERE/'package-profile-v1.json',HERE/'image_profile.py',HERE/'landmarks.json',ROOT/p['control']['measured_build']['path'],ROOT/p['origin']['build']['path'],ROOT/p['origin']['package']['path']]; out={}; indices={}
 for side,count,src,images in [('control',45,cb['source_inputs'],cb['images']),('candidate',40,b['source_inputs'],parent['image_files'])]:
  r=p['sides'][side];assert r['source_inputs']==src and len(src)==46
  assert {k:v for k,v in r['diagnostic_inputs'].items() if k!='src/bin/test_kernels.rs'}=={k:v for k,v in src.items() if k!='src/bin/test_kernels.rs'}
  for x in [r['unlabelled_host'],r['diagnostic_probe'],r['schedule'],r['graph'],*r['images'].values()]:verify(x)
  raw=(ROOT/r['images']['sliding_attention_output']['path']).read_bytes();assert abi.digest(raw)==digest(images['sliding_attention_output'])
  host=(ROOT/r['unlabelled_host']['path']).read_bytes();probe=(ROOT/r['diagnostic_probe']['path']).read_bytes();meta,labelled=patch_check(raw,host,probe)
  for name in ('sliding_project_qkv','decoder_feedforward'):
   im=(ROOT/r['images'][name]['path']).read_bytes();assert host.count(im)==probe.count(im)==1 and abi.digest(im)==digest(images[name])
  mp=HERE/f'evidence/profile-v1/{side}-marker-map.json';md=HERE/f'evidence/profile-v1/{side}-o.image-profile.json'; assert load(md)==json.loads(json.dumps(meta))
  indices[side]=mapping(load(ROOT/r['schedule']['path']),load(mp),meta,count)
  out[side]={'compiled_inputs':46,'original_image':abi.digest(raw),'labelled_image':labelled,'marker_operations':count,'names_only_binary_difference':True,'task_bytes_addresses_io_constants_preserved':True,'other_two_images_unchanged':True}
  bound.extend([mp,md,ROOT/r['schedule']['path'],ROOT/r['graph']['path'],ROOT/r['images']['sliding_attention_output']['path'],ROOT/r['unlabelled_host']['path'],ROOT/r['diagnostic_probe']['path']])
 f=p['ffn'];fb=load(ROOT/f['build']['path']);
 for k in ('build','schedule','graph','source','image','probe'):verify(f[k]);bound.append(ROOT/f[k]['path'])
 assert digest(f['source'])==cb['source_inputs']['src/device/shared/mlp.rs']==fb['source_inputs']['src/device/shared/mlp.rs']
 assert digest(f['image'])==digest(cb['images']['decoder_feedforward'])==digest(fb['exact_image'])
 host=(ROOT/p['sides']['control']['unlabelled_host']['path']).read_bytes();raw=(ROOT/f['image']['path']).read_bytes();probe=(ROOT/f['probe']['path']).read_bytes();meta,labelled=patch_check(raw,host,probe)
 for name in ('sliding_project_qkv','sliding_attention_output'):
  im=(ROOT/p['sides']['control']['images'][name]['path']).read_bytes();assert host.count(im)==probe.count(im)==1
 mp=HERE/'evidence/profile-v1/ffn-marker-map.json';md=HERE/'evidence/profile-v1/ffn.image-profile.json';assert load(md)==json.loads(json.dumps(meta))
 indices['ffn']=mapping(load(ROOT/f['schedule']['path']),load(mp),meta,138);bound.extend([mp,md])
 out['ffn']={'source_and_image_equal_current_e147_and_e142':True,'original_image':abi.digest(raw),'labelled_image':labelled,'marker_operations':138,'names_only_binary_difference':True,'task_bytes_addresses_io_constants_preserved':True,'other_two_images_unchanged':True}
 lm=load(HERE/'landmarks.json'); assert all(v in indices[side] for side,roles in lm.items()for v in roles.values())
 c=indices['candidate'];assert all(c[i]['contexts']==['MainContext','SubContext','VectorEngine'] and '.to_vrf(&mut device.sub)' in c[i]['description'] for i in ['93','106'])
 assert all(c[i]['tpe']=='Sub' and 'CustomBroadcast { ring_size: 16 }' in c[i]['description'] for i in ['55','146'])
 assert 'vector_inter_slice_reduce' in c['45']['description'] and '0x7f80_0000' in c['45']['description']
 f=indices['ffn'];shared=f['274']['output_tensors'][0];assert shared in f['283']['input_tensors'] and shared in f['374']['input_tensors']
 return {'passed':True,'scope':'Independent pre-timing name-only image/source/marker review. Marker intervals include waiting and instrumentation; no physical contention or speedup attribution follows from this gate.','verifier':rec(Path(__file__)),'inputdigests':{str(x.relative_to(ROOT)):rec(x)for x in bound},'sides':out,'source_roles':{'candidate_scale':'45','candidate_direct_main_sub_sum':['93','106'],'candidate_sub_scalar_broadcast':['55','146'],'candidate_native':['152','162'],'ffn_shared_activation_trf':'274','ffn_native_gate_then_up':['283','374']},'ffn_cse_scope':'Both up/gate native consumers read the same emitted activation TRF; source-level two reload calls are not two emitted loads.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=json.loads(json.dumps(review()));out=HERE/'independent-review.json'
 if args.check:assert load(out)==r,'Audit record differs'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS E148 readiness: exact names-only images and 45/40/138 ordered markers')
