#!/usr/bin/env python3
"""Read-only verification of E132 source, compiled graph, package and Arena evidence."""
import json, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE))
import package,screen,verify_graph,analyze_profile

def verify():
 label='e132-k4-v2'
 built=package.read(HERE/f'build-{label}-repro.json')
 assert package.read(HERE/'build-e132-k4-v1-repro.json')['failed_exit_code']!=0
 assert built['source_plan']['source_base_commit']=='cc82b9ccff22795165faffd3ba9a68cc557da001'
 assert 'CARGO_TARGET_DIR=/tmp/moa-m03-validation-probe/target' in built['compile_command']
 assert json.loads(json.dumps(verify_graph.audit(label)))==package.read(HERE/f'graph-verification-{label}.json')
 package.checked_package(label);screen.checked_screen('e132-o-k4-screen')
 result,batch=screen.verified_batch(HERE/'e132-o-k4-screen-job-109654-results.json')
 a=result['whole_o_comparison'];assert a['candidate_change_percent']>0
 profile=HERE/'e132-o-k4-profile-job-109683-results.json'
 assert json.loads(json.dumps(analyze_profile.analyze(profile)))==package.read(HERE/'profile-analysis.json')
 assert json.loads(json.dumps(verify_graph.audit('e132-k4-v3')))==package.read(HERE/'graph-verification-e132-k4-v3.json')
 art=package.ARTIFACTS
 assert (art/'e132-k4-v2/exact-kernel.bin').read_bytes()==(art/'e132-k4-v3/exact-kernel.bin').read_bytes()
 a=(HERE/'two-stage-v2/projection.rs').read_text();z=(HERE/'two-stage-v3/projection.rs').read_text()
 begin=a.index('    let x_trf:',a.index('pub(crate) fn project_output('));end=a.index('    let weight: HbmTensorView',begin);block=a[begin:end]
 assert a.count(block)==z.count(block)==1 and a.replace(block,'')==z.replace(block,'')
 gate=package.read(HERE/'v3-issue-order-gate.json')
 assert gate['same_image'] is True and gate['changed_order'] is False
 for v in (2,3):
  label3=f'e132-k4-v{v}';image=art/label3/'exact-kernel.bin'
  assert gate[f'v{v}']['image']==package.digest(image.read_bytes())
  assert gate[f'v{v}']['schedule']==package.file_record(art/label3/'schedule.json')
  assert gate[f'v{v}']['weight0_position']==4 and gate[f'v{v}']['TRF_position']==3
 s2=package.read(art/'e132-k4-v2/schedule.json');s3=package.read(art/'e132-k4-v3/schedule.json')
 assert [{k:v for k,v in n.items() if k!='description'} for n in s2['instructions']]==[{k:v for k,v in n.items() if k!='description'} for n in s3['instructions']]
 assert not (HERE/'validation/probe-build.json').exists(),'Negative candidate adoption probe should remain unrun'
 out={'valid':True,'source_origin':'E128 cc82b9c','runtime_changed':False,'official_submission':False,'source_plan':package.file_record(HERE/'source-plan-e132-k4-v2.json'),'whole_screen':package.file_record(HERE/'e132-o-k4-screen-job-109654-results.json'),'profile':package.file_record(profile),'graph_gate':package.file_record(HERE/f'graph-verification-{label}.json'),'profile_analysis':package.file_record(HERE/'profile-analysis.json'),'verifier':package.file_record(Path(__file__)),'checks':['Source/control/3image/fixture/runner bindings','V1 frozen compile failure and V2 independent-row repair','Exact whole O descriptor coverage and F32scale-before-BF16','Whole allocation actualupload/rawlog/45PASS','Frozen image O26marker mapping and samecluster trace/all57PASS','Prepared adoption oracle unrun after negative screen','V3 source-only reorder; exactimage byte-identical and all49 instruction metadata exceptsource descriptions identical; noNPU'],'scope':'Revalidates retained evidence; no new compilation or Arena execution.'}
 return out
if __name__=='__main__':
 v=verify();p=HERE/'verification.json'
 if '--check' in sys.argv:assert package.read(p)==v
 else:package.write(p,v)
 print('E132 source/graph/package/whole/profile evidence PASS; no adoption')
