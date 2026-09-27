#!/usr/bin/env python3
"""Small exact-source and independent emitted-copy gate; no compiler or NPU calls."""
import argparse,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LABEL='e140-control-copy-fix'
BASE='334a7059a82252a310436c6d7e02f8ba577e7f51'
def read(p):return json.loads(p.read_text())
def digest(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def record(p):return {'path':str(p.relative_to(ROOT)),**digest(p.read_bytes())}
def expected_source():
 s=subprocess.check_output(['git','show',BASE+':furiosa-opt-gemma4-12B/src/device/shared/mlp.rs'],cwd=ROOT,text=True)
 for name,i in [('hi',0),('lo',1),('third',2)]:
  old=f'''    {name}.view().to_dm_view(&mut device.tdma,
        packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{{!}} 3, H % 480]>({i}));'''
  new=f'''    // Exact480B copy; padded destination must not enlarge the source DMA extent.
    device.sub.begin({name}.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .collect::<m![H / 32 % 15], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit_view(packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{{!}} 3, H % 480]>({i}));'''
  assert s.count(old)==1
  s=s.replace(old,new)
 return s.encode()
def output():
 bp=HERE/f'build-{LABEL}-repro.json';b=read(bp)
 assert b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container']
 assert b['base_commit']==BASE and b['source_plan']['performance_comparison']['id']=='E137'
 src=HERE/'control-copy-fix/mlp.rs';assert src.read_bytes()==expected_source()
 art=ROOT/'moa-dev/experiments/artifacts'/HERE.name/LABEL
 assert b['changed_sources']==['src/device/shared/mlp.rs']
 expected=dict(b['source_plan']['source_origin_inputs']);expected['src/device/shared/mlp.rs']=digest(src.read_bytes())
 assert expected==b['source_inputs']
 for name,rec in b['source_inputs'].items():assert digest((art/'crate'/name).read_bytes())==rec,name
 for name in ('graph','schedule'):
  assert digest((art/(name+'.json')).read_bytes())==b[name]
  assert (art/(name+'.json')).read_bytes()==(HERE/'evidence'/LABEL/(name+'.json')).read_bytes()
 assert digest((art/'exact-kernel.bin').read_bytes())==b['exact_image']
 ip=HERE/'independent-control-copy-review.json';review=read(ip)
 assert review['passed'] is True
 for key,rec in review['inputdigests'].items():assert record(ROOT/rec['path'])==rec,key
 assert any(rec==record(src) for rec in review['inputdigests'].values())
 assert any(rec['sha256']==b['graph']['sha256'] for rec in review['inputdigests'].values())
 assert any(rec['sha256']==b['schedule']['sha256'] for rec in review['inputdigests'].values())
 return {'passed':True,'candidate':'E140','variant':'control-copy-fix','label':LABEL,
 'inputdigests':{'build_record':record(bp),'graph':record(art/'graph.json'),'schedule':record(art/'schedule.json'),
 'source':record(src),'independent_review':record(ip),'exact_image':record(art/'exact-kernel.bin')},
 'verifiers':[record(Path(__file__).resolve())],
 'source_change':'Exactly three up equal-scale term copies; all other E137 source bytes unchanged, including weighted down.',
 'emitted_copy_review':review,
 'scope':'Source/copy-bound correctness gate. No sole-cause or hardware performance conclusion; full original standard host still required.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
 result=output();path=HERE/f'graph-verification-{LABEL}.json'
 if a.check:assert read(path)==result
 else:
  with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('E140 control-only copy gate PASS')
