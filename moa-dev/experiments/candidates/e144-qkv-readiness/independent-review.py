#!/usr/bin/env python3
"""Offline independent D-E145 source, image-name edit, marker and visibility gate."""
import argparse,hashlib,importlib.util,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def load(p):return json.loads(p.read_text())
def rec(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def verify_rec(r):assert rec(ROOT/r['path'])==r
spec=importlib.util.spec_from_file_location('profile_abi',HERE/'image_profile.py');abi=importlib.util.module_from_spec(spec);spec.loader.exec_module(abi)
def kind(n):
 if n['tpe'].startswith('Dma'):return 'DMA'
 if n['tpe']=='Main':return 'Renegade::TuExec'
 if n['tpe']=='Sub' and n['contexts']!=['Noop']:
  d=n['description']
  if '.to_trf()' in d:return 'Renegade::StoTrf'
  if not d.strip():return 'Renegade::StoTab'
  if '.collect::<m![1], m![QkvTerm # 8]>()' in d:return 'Renegade::TuExec'
  if '.to_vrf()' in d and 'fetch_cast' not in d and 'vector_' not in d:return 'Renegade::StoVrf'
  return 'Renegade::TuExec'
def review():
 p=load(HERE/'package-profile-v1.json');assert p['status']=='ready'
 out={};bound=[HERE/'package-profile-v1.json',HERE/'image_profile.py']
 for side,count,oldbuild in [('control',76,'ffn-direct-full30-partials/measured-build-e142-adoption.json'),('candidate',92,'qkv-fullh-weight-switch/measured-build-e144-adoption.json')]:
  r=p['sides'][side];build=load(HERE.parent/oldbuild);assert r['source_inputs']==build['source_inputs'] and len(r['source_inputs'])==46
  assert {k:v for k,v in r['diagnostic_inputs'].items() if k!='src/bin/test_kernels.rs'}=={k:v for k,v in r['source_inputs'].items() if k!='src/bin/test_kernels.rs'}
  for x in [r['unlabelled_host'],r['diagnostic_probe'],r['schedule'],r['graph'],*r['images'].values()]:verify_rec(x)
  raw=(ROOT/r['images']['sliding_project_qkv']['path']).read_bytes();assert abi.digest(raw)=={k:build['images']['sliding_project_qkv'][k] for k in ['bytes','sha256']}
  labelled,meta=abi.inspect(raw,True);_,again=abi.inspect(labelled)
  for key in ('version','chips','pes','header_bytes','tasks','weight','stacks','slots','inputs','outputs','profile_depth'):assert meta[key]==again[key]
  allowed={i for row in meta['spans'] for s in row for i in range(s['name_offset'],s['name_offset']+s['name_bytes'])}
  assert len(raw)==len(labelled) and all(a==b or i in allowed for i,(a,b) in enumerate(zip(raw,labelled)))
  host=(ROOT/r['unlabelled_host']['path']).read_bytes();probe=(ROOT/r['diagnostic_probe']['path']).read_bytes()
  assert host.count(raw)==1 and probe==host.replace(raw,labelled) and probe.count(labelled)==1
  for name in ('sliding_attention_output','decoder_feedforward'):
   im=(ROOT/r['images'][name]['path']).read_bytes();assert host.count(im)==probe.count(im)==1
   assert abi.digest(im)=={k:build['images'][name][k] for k in ['bytes','sha256']}
  mp=HERE/f'evidence/profile-v1/{side}-marker-map.json';m=load(mp);schedule=load(ROOT/r['schedule']['path']);ops=[i for i in schedule['instructions'] if kind(i)]
  spans=[s for s in meta['spans'][0] if s['name'] in {'DMA','Renegade::TuExec','Renegade::StoTrf','Renegade::StoVrf','Renegade::StoTab'}]
  spans.sort(key=lambda s:s['begin_marker'])
  assert len(ops)==len(spans)==count and [x['schedule'] for x in m['operations']]==ops and [x['marker'] for x in m['operations']]==spans
  assert [s['name'] for s in spans]==[kind(i) for i in ops]
  absent=['97','107','331','374'] if side=='control' else ['269','281','422','500']
  assert m['cluster_absent']=={'0':[],'1':absent}
  out[side]={'source_inputs':46,'original_image':abi.digest(raw),'labelled_image':abi.digest(labelled),'whole_task_bytes_addresses_io_constants_preserved':True,'ordered_markers':count,'cluster_absent':m['cluster_absent'],'host_original_and_names_only_patch':True}
  bound.extend([HERE.parent/oldbuild,mp,ROOT/r['schedule']['path'],ROOT/r['graph']['path']])
 g=load(HERE/'evidence/profile-v1/candidate/graph.json');s=load(HERE/'evidence/profile-v1/candidate/schedule.json');ss={x['index']:x for x in s['instructions']};ops={int(o['name'].split('#O')[-1]):o for o in g['operators'].values()}
 visibility=[]
 for idx,n,role,io in [('269',134,'cos',12),('281',138,'sin',13),('422',196,'V cache',15),('500',189,'K cache',14)]:
  o=ops[n];d=o['description'];assert re.findall(r'--- \((.*?)\) ---',d)==['chip0,cluster0,pe0']
  assert any(g['tensors'][str(t)]['optionals'].get('address')==f'IO({io})+0' for t in o['inputs'])
  desc=ss[idx]['description'];assert ('.dma_gather_scaled' in desc if io in (12,13) else '.dma_scatter' in desc)
  assert ('cos.' in desc if io==12 else 'sin.' in desc if io==13 else 'v.' in desc if io==15 else 'k.' in desc)
  visibility.append({'schedule':idx,'graph_operator':o['name'],'role':role,'public_io':io,'cluster0_only_actual_descriptor':d})
 return {'passed':True,'scope':'Pre-NPU image/source/marker/cluster visibility only; current E144 timing remains unmeasured by this gate.','verifier':rec(Path(__file__)),'inputdigests':{str(x.relative_to(ROOT)):rec(x) for x in bound},'sides':out,'candidate_cluster_visibility':visibility,'control_graph_scope':'Preserved control graph.json is binary IR. Control schedule is tied by byte-identical original E142 QKV image; no decoded control descriptor claim.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();r=json.loads(json.dumps(review()));f=HERE/'independent-review.json'
 if a.check:assert load(f)==r
 else:f.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS D-E145 original images, names-only edits, 76/92 ordered markers and actual E144 cluster0-only descriptors')
