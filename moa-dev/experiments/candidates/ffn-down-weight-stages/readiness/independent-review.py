#!/usr/bin/env python3
"""Offline source/image/name-only and source-supported marker gate for E150 readiness."""
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
 p=load(HERE/'package-profile-v1.json'); assert p['status']=='ready'
 b=load(ROOT/p['origin']['build']['path']); parent=load(ROOT/p['origin']['package']['path']); r=p['sides']['candidate']
 assert len(r['source_inputs'])==46 and r['source_inputs']==b['source_inputs']
 assert {k:v for k,v in r['diagnostic_inputs'].items() if k!='src/bin/test_kernels.rs'}=={k:v for k,v in b['source_inputs'].items() if k!='src/bin/test_kernels.rs'}
 bound=[HERE/'package-profile-v1.json',HERE/'image_profile.py',ROOT/p['origin']['build']['path'],ROOT/p['origin']['package']['path'],ROOT/p['origin']['graph_gate']['record']['path']]
 for x in [p['origin']['build'],p['origin']['package'],p['origin']['graph_gate']['record'],r['unlabelled_host'],r['diagnostic_probe'],r['schedule'],r['graph'],r['diagnostic_host_source'],*r['images'].values()]: verify(x);bound.append(ROOT/x['path'])
 for name,v in r['images'].items(): assert digest(v)==digest(parent['image_files'][name])
 raw=(ROOT/r['images']['decoder_feedforward']['path']).read_bytes();assert abi.digest(raw)==digest(b['exact_image'])
 host=(ROOT/r['unlabelled_host']['path']).read_bytes();probe=(ROOT/r['diagnostic_probe']['path']).read_bytes();meta,labelled=patch_check(raw,host,probe)
 for name in ('sliding_project_qkv','sliding_attention_output'):
  im=(ROOT/r['images'][name]['path']).read_bytes();assert host.count(im)==probe.count(im)==1 and digest(r['images'][name])==digest(p['control']['images'][name])
 mp=HERE/'evidence/profile-v1/candidate-marker-map.json';md=HERE/'evidence/profile-v1/candidate-ffn.image-profile.json';assert load(md)==json.loads(json.dumps(meta));bound.extend([mp,md])
 n=mapping(load(ROOT/r['schedule']['path']),load(mp),meta,142)
 for f in ('graph.json','schedule.json'):
  assert digest(rec(HERE/'evidence/profile-v1/candidate'/f))==digest(rec(HERE.parent/'evidence/e150-staged-v1'/f))
 source=HERE/'evidence/profile-v1/candidate/production-mlp.rs';assert digest(rec(source))==b['source_inputs']['src/device/shared/mlp.rs'];bound.append(source)
 for wi,di,ti in [('430','657','516'),('647','665','532')]:
  assert n[wi]['tpe']=='DmaLoad' and n[di]['tpe']=='Main' and n[ti]['tpe']=='Sub' and not n[ti]['description'].strip()
  assert n[wi]['output_tensors'][0] in n[di]['input_tensors'] and n[ti]['output_tensors'][0] in n[di]['input_tensors']
  assert 'fetch_table_lookup::<f8e5m2>' in n[di]['description']
 for shared in ['653','630']:
  assert all(n[shared]['output_tensors'][0] in n[di]['input_tensors'] for di in ['657','665'])
 assert '.tile::<m![H % 60], 30' in n['430']['description'] and '(0)' in n['430']['description']
 assert '.tile::<m![H % 60], 30' in n['647']['description'] and '(30)' in n['647']['description']
 assert 'up.to_dm' in n['498']['description'] and 'device\n.main' in n['513']['description']
 roles={'down_weight_first':'430','down_weight_second':'647','down_native_first':'657','down_native_second':'665','down_table_first':'516','down_table_second':'532','down_shared_activation_trf':'653','down_term_coeff_vrf':'630','up_canonical_gather':'498','geglu':'513','down_scale_hbm':'666'}
 return {'passed':True,'candidate':'E150','scope':'Independent name-only E150 FFN image, actual source binding and 142 ordered marker review; one candidate trace with E147 whole controls. No current E147 trace or causal service-time attribution is implied.','verifier':rec(Path(__file__)),'inputdigests':{str(x.relative_to(ROOT)):rec(x)for x in bound},'source_inputs':46,'original_image':abi.digest(raw),'labelled_image':labelled,'names_only_binary_difference':True,'task_bytes_addresses_io_constants_preserved':True,'unchanged_qkv_o_images':True,'marker_operations':142,'source_roles':roles,'shared_register_scope':'Both down half-native consumers reference the same emitted activation TRF and coefficient VRF.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=json.loads(json.dumps(review()));out=HERE/'independent-review.json'
 if args.check: assert load(out)==r,'Audit record differs'
 else: out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS E150 readiness: exact names-only FFN and 142 ordered markers')
