#!/usr/bin/env python3
"""Read-only four-image binding to actual323173 and fresh resources; no objcopy."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=HERE.parent/'int4-grouped-activation-quant'
sys.path.insert(0,str(PRIOR))
import registry_audit
import analyze_metadata_store as provenance

def sha(p):return provenance.sha(p)
def load(p):return json.loads(p.read_text())
def main():
 d=HERE/'device-build';r=load(d/'run.json');h=load(d/'source-hashes.json')
 assert r['commit'].startswith('0488a75a') and r['exit_code']==0 and r['result']=='command_succeeded'
 assert r['fresh_package']['contents_changed'] is False and r['build_inputs_unchanged'] is True
 assert r['status']=='' and (d/'source.diff').read_bytes()==b''
 assert len(h)==421
 provenance.git_sources(r['commit'],h)
 for name,digest in h.items():assert sha(ROOT/name)==digest,name
 a=r['built_artifacts']['target/release/test_int4_grouped_lut_replicas'];path=Path(a['path'])
 assert sha(path)==a['sha256'] and path.stat().st_size==a['size']
 registry=registry_audit.read_artifact(path);assert registry['record_count']==4
 oldrun=load(PRIOR/'actual-323173/run.json');oldstatus=load(PRIOR/'actual-323173/arena-status.json')
 assert oldstatus['id']==323173 and oldstatus['status']=='SUCCEEDED' and oldstatus['exit_code']==0
 oldartifact=oldrun['submitted_artifact']
 assert oldartifact['sha256']=='b71c05511a3217c8d6a3758360996203232df88d6bd5dfb6ae3d429edd37bfb3'
 for p in ['path','source_path']:assert sha(Path(oldartifact[p]))==oldartifact['sha256']
 oldreg=registry_audit.read_artifact(Path(oldartifact['path']));assert oldreg['record_count']==2
 old={x['path']:x for x in oldreg['records']}
 pairs=[]
 for item in registry['records']:
  name=item['path'];assert item['key']==[]
  assert name in [f'furiosa_opt_gemma4::{module}::quantize_pack_{width}'for module in ['ops_int4_grouped_activation_store','ops_int4_grouped_lut_replicas']for width in ['input','hidden']]
  fresh=HERE/'kernel-out-a004/furiosa-opt-gemma4'/f'{name}.bin'
  assert sha(fresh)==item['sha256'] and fresh.stat().st_size==item['bytes']
  pair={'path':name,'sha256':item['sha256'],'bytes':item['bytes'],'fresh_build_equal':True}
  if name in old:
   assert (old[name]['sha256'],old[name]['bytes'])==(item['sha256'],item['bytes'])
   pair['baseline_actual323173_equal']=True
  else:
   resource=HERE/'resource-images'/f'{name}.bin'
   assert sha(resource)==item['sha256'] and resource.stat().st_size==item['bytes']
   pair['candidate_frozen_resource_equal']=True
  pairs.append(pair)
 assert sum(x.get('baseline_actual323173_equal',False)for x in pairs)==2
 assert sum(x.get('candidate_frozen_resource_equal',False)for x in pairs)==2
 out={'source_commit':r['commit'],'functional_source':'78054937','captured_artifact':a,
 'source_inputs':len(h),'original_dependencies':{k:h[k]for k in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']},
 'run':r,'record_files':{p.name:sha(p)for p in sorted(d.iterdir())},'registry':registry,'four_image_bindings':pairs,
 'prior_actual':{'job':323173,'artifact':oldartifact,'status':oldstatus,'record_hashes':{n:sha(PRIOR/'actual-323173'/n)for n in ['run.json','arena-status.json','arena.log','source-hashes.json']}},
 'parser_sha256':sha(PRIOR/'registry_audit.py'),
 'decision':'FRESH_BUILD_CAPTURE_AND_4_IMAGE_BINDINGS_PASS; runtime values/address-cause/speed still unexecuted; root owns Arena.',
 'limits':['Matched source probe has5cases×2widths, independent full codes/scales/new offsets and original old-offset control, ABBA and immutable table/base/input/shifts after reuse.','New offset ABI is512*m+32*(p%16); mantissa follows data permutation but replica follows destination row.','No added wholeFFN/model precision claim, no speed target or causal assertion from static schedule.']}
 print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
