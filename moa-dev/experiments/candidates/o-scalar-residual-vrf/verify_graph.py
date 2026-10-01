#!/usr/bin/env python3
"""Bind the independently reproduced audit to the existing host-package gate."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def read(p):return json.loads(p.read_text())
def record(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def verify(label):
 subprocess.run([sys.executable,str(HERE/'audit.py'),label,'--check'],check=True)
 path=HERE/f'audit-{label}.json';a=read(path);assert a['passed'] and a['label']==label
 for r in a['inputdigests'].values():assert record(ROOT/r['path'])==r
 bp=HERE/f'build-{label}-repro.json';b=read(bp);assert b['exact_compile_exit_code']==0
 art=ROOT/'moa-dev/experiments/artifacts'/HERE.name/label
 for name in ('graph','schedule'):
  assert {k:record(art/(name+'.json'))[k] for k in ('bytes','sha256')}==b[name]
 image=record(art/'exact-kernel.bin');assert {k:image[k] for k in ('bytes','sha256')}==b['exact_image']
 return {'passed':True,'label':label,'independent_audit':record(path),'inputdigests':{'build_record':record(bp),'graph':record(art/'graph.json'),'schedule':record(art/'schedule.json'),'exact_image':image},'verifiers':[record(HERE/'audit.py'),record(Path(__file__))],'scope':'Actual O fused scale, ring16 bitmap/valid-mask, direct scalar/cumulative-sum VRF and two unchanged native consumers/register-lifetime gate. Finite-BF16 CPU scale identity and NPU numerical correctness have separate scope.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--check',action='store_true');a=p.parse_args();v=verify(a.label);out=HERE/f'graph-verification-{a.label}.json'
 if a.check:assert read(out)==v
 else:
  with out.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
 print('Host graph gate PASS')
