#!/usr/bin/env python3
"""Gate/stage E137, inspect its official record, or explicitly submit it once.

check/preflight never contact the official server. preflight uses the existing
loopback-only payload capture tool. submit/status are separate explicit actions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

CAND='moa-dev/experiments/candidates/o-native-activation-resident'
RECIPE='baseline/e137.json'
REMOTE='/tmp/moa-official-e137-20260927'
PREFLIGHT='docs/experiments/e137-official-payload-preflight-20260927.json'
RECORD='docs/experiments/e137-official-submit-20260927.json'
SERVER='https://micro2026-api.duckdns.org:7777'
CONTAINER='moa2026-dev'
EXPECTED_KERNELS={'sliding_project_qkv','sliding_attention_output','decoder_feedforward'}


def require(value,message):
    if not value:raise RuntimeError(message)
def digest(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(root,path):return json.loads((root/path).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()
def file_record(root,path):return {'path':path,**digest((root/path).read_bytes())}
def run(root,command,**kwargs):return subprocess.run(command,cwd=root,check=True,**kwargs)
def json_write(path,value,exclusive=False):
    with path.open('x' if exclusive else 'w') as out:json.dump(value,out,indent=2);out.write('\n')
def check_record(root,record):
    path=(root/record['path']).resolve();path.relative_to(root)
    require(digest(path.read_bytes())=={k:record[k] for k in ('bytes','sha256')},'Changed evidence: '+record['path'])
def archive_digests(raw):
    result={}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for item in tar:
            if item.isdir():continue
            require(item.isfile() and item.name not in result,'Unexpected/duplicate archive entry: '+item.name)
            result[item.name]=digest(tar.extractfile(item).read())
    return result


def verify(root,check_payload=True):
    current=read(root,'baseline/current.json')
    require(current.get('recipe')==RECIPE,'E137 is not the adopted baseline: '+str(current.get('recipe')))
    recipe=read(root,RECIPE)
    require(recipe['id']=='E137' and recipe['status']=='adopted','E137 recipe is not adopted')
    require(recipe['git']['tag']=='baseline-e137','Wrong source tag')
    require(recipe['compile_inputs']==CAND+'/compiled-source-manifest.json','Wrong E137 manifest path')
    require(recipe['performance_report']=='docs/experiments/e137-adoption-performance.json','Wrong performance report')
    commit=subprocess.check_output(['git','rev-parse','baseline-e137^{commit}'],cwd=root,text=True).strip()
    require(commit==recipe['git']['commit'],'Tag/recipe commit mismatch')
    manifest=read(root,recipe['compile_inputs'])
    measured=read(root,CAND+'/measured-build-e137-adoption.json')
    require(len(manifest)==46 and measured['source_inputs']==manifest,'All46 measured compile inputs must match')
    require(measured['variant']=='native-v3' and measured['label']=='e137-native-v3','Wrong measured variant')
    require(measured['exact_compile_exit_code']==measured['host_build_exit_code']==0,'Build did not complete')
    require(measured['source_inputs_verified_in_container'] and not measured['compile_only'],'Missing full build provenance')
    gate_path=CAND+'/validation/final-verification.json'
    gate=read(root,gate_path)
    require(gate['valid'] and gate['adoption_ready'] and gate['candidate']=='E137','Final validation is not adoption-ready')
    check_record(root,gate['verifier'])
    require(gate['verifier']['path']==CAND+'/validation_verify.py','Wrong final verifier')
    for record in gate['inputdigests'].values():check_record(root,record)
    require(gate['performance_allocations']==3 and gate['standard_input_cases']==9,'Three allocations/nine standard inputs required')
    require(all(gate['projection_oracle'].get(k)==v for k,v in {'cases':9,'outputs':34560,'exact_bitwise':True,'signed_zero_equivalent':True,'actual_production_quantizer':True,'before_rms':True}.items()),'Wrong finite pre-RMS oracle scope')
    require(all(gate['full_o_stress'].get(k)==v for k,v in {'cases':1,'outputs':3840,'exact_bitwise':True,'actual_public_image':True,'original_reference_and_tolerance_unchanged':True,'performance_excluded':True}.items()),'Wrong actual whole-O stress scope')
    require(gate['unchanged_images']=={'qkv':True,'ffn':True},'Unchanged-image gate failed')
    # Recompute existing gates; no new adoption wrapper or extra NPU execution.
    run(root,[sys.executable,str(root/CAND/'validation_verify.py'),'--check'])
    run(root,[sys.executable,str(root/CAND/'validation/oracle_v3.py'),'--check'])
    run(root,[sys.executable,str(root/CAND/'validation/full_stress_v3.py'),'--check'])
    run(root,[sys.executable,str(root/CAND/'package.py'),'e137-native-v3','--check'])
    run(root,[sys.executable,str(root/'moa-dev/experiment_index.py'),'--check','--check-live'])
    report=read(root,recipe['performance_report'])
    require(report['valid'] and report['independent_arena_jobs']==3,'Invalid performance report')
    require(report['all_paired_o_improved'] and report['median_paired_delta_percent']<0,'O performance was not consistently improved')
    require(len(report['paired_observations'])==3 and len({v['job_id'] for v in report['paired_observations']})==3,'Three distinct performance jobs required')
    require(all(v['delta_percent']<0 for v in report['paired_observations']),'Non-positive paired performance gate')
    extra=read(root,'docs/experiments/e137-extra-correctness.json')
    require(extra['valid'] and extra['output_comparisons_passed']==30 and extra['actual_upload_matches'],'Invalid six-input report')
    oracle=read(root,CAND+'/validation/oracle-results-v3.json')
    stress=read(root,CAND+'/validation/stress-results-v3.json')
    require(oracle['valid'] and oracle['case_count']==9 and oracle['output_elements']==34560 and oracle['exact_bitwise'],'Pre-RMS finite oracle failed')
    require(stress['valid'] and stress['case_count']==1 and stress['output_elements']==3840,'Actual whole-O case7 failed')
    require(len(stress['cases'])==1 and stress['cases'][0]['case']=='7' and int(stress['cases'][0]['exact_f32'])==3840,'Whole-O case7 exact-output evidence differs')
    require(oracle['actual_upload_matches'] and stress['actual_upload_matches'],'Correctness upload evidence missing')
    require(oracle['production_source_full_file_byte_identical'],'Oracle did not call the full production helper')
    require(stress['production_source_inputs_match'],'Whole-O stress source differs')
    require(stress['image']['matches_whole_exact'],'Stress used a different device image')
    # Bind the root adoption audit to frozen compile, standard package and source.
    audit=read(root,recipe['source_build_audit'])
    compiled_path=CAND+'/build-e137-native-v3-repro.json'
    package_path=CAND+'/package-e137-native-v3.json'
    measured_path=CAND+'/measured-build-e137-adoption.json'
    compiled,package=read(root,compiled_path),read(root,package_path)
    require(audit['candidate']=='E137' and audit['adopted_source_commit']==commit,'Adoption source/audit mismatch')
    require(audit['measured_build']==measured_path and audit['measured_build_digest']==file_record(root,measured_path),'Measured build audit mismatch')
    require(audit['original_compile_record']==file_record(root,compiled_path),'Wrong original compile record')
    require(audit['original_package_record']==file_record(root,package_path),'Wrong original standard package')
    require(audit['source_binding']==file_record(root,CAND+'/source-build-binding.json'),'Source binding audit mismatch')
    binding=read(root,CAND+'/source-build-binding.json')
    require(binding['valid'],'Invalid source/image binding')
    require(compiled['source_inputs']==measured['source_inputs']==manifest,'Manifest/compile/measured source disagreement')
    require(compiled['variant']=='native-v3' and compiled['label']=='e137-native-v3','Wrong original source variant')
    require(package['status']=='ready' and package['compile_record']==file_record(root,compiled_path),'Standard package is not the exact build')
    require(measured['images']==audit['images']==package['images'],'Image binding disagreement')
    require(measured['package_files']==package['package_files'],'Measured package differs')
    require(measured['original_compile_record']==audit['original_compile_record'] and measured['original_package_record']==audit['original_package_record'],'Measured provenance differs')
    require(audit['compile_inputs']==46,'Audit must bind all46 compile inputs')
    prior_recipe=read(root,'baseline/e134.json');prior_audit=read(root,prior_recipe['source_build_audit'])
    prior_build=read(root,prior_audit['measured_build'])
    require(set(manifest)==set(prior_build['source_inputs']),'Compile inventory differs from E134')
    changed=sorted(n for n in manifest if manifest[n]!=prior_build['source_inputs'][n])
    require(changed==['src/device/sliding/projection.rs'],'Unexpected changed compile inputs')
    source_expected=manifest['src/device/sliding/projection.rs']
    require({k:oracle['production_source'][k] for k in ('bytes','sha256')}==source_expected,'Oracle bound another projection')
    require({k:stress['image'][k] for k in ('bytes','sha256')}==compiled['exact_image'],'Whole-O stress image differs from measured O')
    require({k:measured['images']['sliding_attention_output'][k] for k in ('bytes','sha256')}==compiled['exact_image'],'Measured O image differs from exact compile')
    for kernel in ('sliding_project_qkv','decoder_feedforward'):
        require(measured['images'][kernel]==prior_build['images'][kernel],'E134 image changed: '+kernel)
    package_dir=root/measured['package_path']
    require({f.name:digest(f.read_bytes()) for f in package_dir.iterdir() if f.is_file()}==measured['package_files'],'Actual measured host/fixture/runner changed')
    host=(package_dir/'test_runtime').read_bytes()
    require(set(measured['standard_image_files'])==EXPECTED_KERNELS,'Wrong standard image inventory')
    for kernel,image_record in measured['standard_image_files'].items():
        check_record(root,image_record)
        image=(root/image_record['path']).read_bytes()
        require(digest(image)=={k:measured['images'][kernel][k] for k in ('bytes','sha256')},'Actual image digest differs: '+kernel)
        require(host.count(image)==measured['images'][kernel]['host_occurrences']==1,'Wrong embedded image count: '+kernel)
    crate=root/'furiosa-opt-gemma4-12B'
    actual_src={p.relative_to(crate).as_posix() for p in (crate/'src').rglob('*') if p.is_file()}
    require(actual_src=={p for p in manifest if p.startswith('src/')},'Live source inventory differs')
    files={}
    for name,expected in manifest.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe manifest path')
        source=recipe['tracked_lock'] if name=='Cargo.lock' else 'furiosa-opt-gemma4-12B/'+name
        data=subprocess.check_output(['git','show',f'{commit}:{source}'],cwd=root)
        require(digest(data)==expected,'Tag mismatch: '+name)
        require(not (crate/name).is_symlink() and digest((crate/name).read_bytes())==expected,'Live mismatch: '+name)
        files[name]=data
    allowed={name:info for name,info in manifest.items() if name=='src/ops.rs' or name.startswith('src/device/')}
    require(len(allowed)==23,'Submission payload must contain exactly23 allowed paths')
    summary={'source_checkpoint':'E137','source_commit':commit,'compile_input_count':46,
             'source_build_audit':recipe['source_build_audit'],'source_build_audit_digest':file_record(root,recipe['source_build_audit']),'source_files':allowed,'source_count':23,
             'compile_inputs':file_record(root,recipe['compile_inputs']),
             'validation':file_record(root,CAND+'/validation/final-verification.json'),
             'validation_verifier':file_record(root,CAND+'/validation_verify.py'),
             'projection_oracle':file_record(root,CAND+'/validation/oracle-results-v3.json'),
             'full_o_stress':file_record(root,CAND+'/validation/stress-results-v3.json'),
             'numerical_scope':'Nine selected standard inputs; nine pre-RMS cases/34560 exact BF16 outputs and one actual whole-O case/3840 outputs. Not arbitrary-input or full-model correctness.',
             'measured_build':file_record(root,CAND+'/measured-build-e137-adoption.json')}
    if check_payload:
        preflight=read(root,PREFLIGHT)
        require(preflight['status']=='LOCAL_CLI_PAYLOAD_CHECK_PASS' and preflight['source']==REMOTE,'Wrong payload preflight')
        require(preflight['source_files']==allowed and preflight['source_count']==23,'Preflight sources differ')
        require(preflight['three_requests_byte_identical'] and preflight['source_unchanged_after_checks'],'Unstable CLI payload')
        require(preflight['synthetic_token_only'] and not preflight['saved_config_used_or_written'] and not preflight['official_submission_performed'],'Preflight scope changed')
        require(len(preflight['cases'])==4 and all(c['passed'] for c in preflight['cases']),'Preflight cases incomplete')
        # The staged crate has all46 measured inputs, not only the23 uploaded files.
        raw=subprocess.check_output(['docker','exec','--workdir',REMOTE,CONTAINER,'tar','-cf','-',*sorted(manifest)],cwd=root)
        require(archive_digests(raw)==manifest,'Staged all46 input bytes differ')
        cli_raw=subprocess.check_output(['docker','exec',CONTAINER,'cat','/home/moa/.cargo/bin/moa-submitter'],cwd=root)
        require(digest(cli_raw)==preflight['cli_binary'],'Installed submitter changed since loopback check')
        summary['payload_preflight']=file_record(root,PREFLIGHT)
    return summary,files


def preflight(root):
    require(not (root/PREFLIGHT).exists(),'Preflight already recorded; check it instead of overwriting')
    require(not (root/RECORD).exists(),'A submission attempt exists; do not restage its source')
    summary,files=verify(root,check_payload=False)
    # Fail instead of modifying an earlier source snapshot or capture directory.
    for path in (REMOTE,REMOTE+'-payload-check'):
        require(subprocess.run(['docker','exec',CONTAINER,'test','-e',path],cwd=root).returncode==1,'Docker path already exists: '+path)
    run(root,['docker','exec',CONTAINER,'mkdir',REMOTE])
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for name,data in sorted(files.items()):
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644;tar.addfile(item,io.BytesIO(data))
    run(root,['docker','exec','-i',CONTAINER,'tar','-xf','-','-C',REMOTE],input=buf.getvalue())
    helper=root/'moa-dev/experiments/candidates/official-submit-preflight-e55/preflight.py'
    remote_helper='/tmp/moa-official-e137-payload-preflight.py'
    run(root,['docker','cp',str(helper),CONTAINER+':'+remote_helper])
    run(root,['docker','exec',CONTAINER,'python',remote_helper,'--source',REMOTE,'--output',REMOTE+'-payload-check'])
    raw=subprocess.check_output(['docker','exec',CONTAINER,'cat',REMOTE+'-payload-check/report.json'],cwd=root)
    data=json.loads(raw)
    require(data['source_files']==summary['source_files'],'Captured CLI bytes differ from E137')
    require(data['status']=='LOCAL_CLI_PAYLOAD_CHECK_PASS' and not data['official_submission_performed'],'Loopback preflight failed')
    evidence=root/'moa-dev/experiments/artifacts/official-submit-preflight-e137-20260927'
    require(not evidence.exists(),'Local preflight artifacts already exist')
    evidence.parent.mkdir(parents=True,exist_ok=True)
    run(root,['docker','cp',CONTAINER+':'+REMOTE+'-payload-check',str(evidence)])
    json_write(root/PREFLIGHT,data,exclusive=True)
    print(json.dumps({'status':data['status'],'source_count':23,'compile_input_count':46,'official_submission_performed':False},indent=2))


def submit(root):
    path=root/RECORD
    require(not path.exists(),'A submission was already attempted; inspect its record, never automatically retry')
    summary,_=verify(root)
    command=[str(root/'moa-dev/moa'),'moa-submitter','--server',SERVER,'submit','--source',REMOTE]
    data={**summary,'started_at_utc':stamp(),'command':command,'accepted':False,'official_submission_attempted':True}
    json_write(path,data,exclusive=True)
    try:
        result=subprocess.run(command,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        data.update(exit_code=result.returncode,response=result.stdout,finished_at_utc=stamp())
        matches=re.findall(r'^Submission:\s+([0-9a-f]{8})\s*$',result.stdout,re.M)
        if result.returncode==0 and len(matches)==1:data.update(accepted=True,submission_id=matches[0])
    except subprocess.TimeoutExpired as exc:
        response=exc.stdout or ''
        if isinstance(response,bytes):response=response.decode(errors='replace')
        data.update(timed_out=True,response=response,finished_at_utc=stamp(),error='Unknown server acceptance; inspect record/status before any manual decision')
    finally:json_write(path,data)
    print(data.get('response',''))
    require(data['accepted'],'Submission response requires inspection; no automatic retry')


def status(root):
    path=root/RECORD;data=read(root,RECORD)
    require(data['accepted'] and re.fullmatch(r'[0-9a-f]{8}',data['submission_id']),'No accepted submission ID')
    observation={'checked_at_utc':stamp()}
    for action in ('status','log'):
        command=[str(root/'moa-dev/moa'),'moa-submitter','--server',SERVER,action,data['submission_id']]
        result=subprocess.run(command,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45)
        observation[action]={'command':command,'exit_code':result.returncode,'output':result.stdout}
        require(result.returncode==0,result.stdout)
    status_text,log=observation['status']['output'],observation['log']['output']
    statuses=re.findall(r'^Status:\s+(\w+)',status_text,re.M)
    require(len(statuses)==1,'Missing/duplicate official status')
    data.update(current_status=statuses[0],server_contract_ok='contract OK' in log)
    if data['current_status']=='completed':
        scores=re.findall(r'^Score:\s+([\d.]+)\s*$',status_text,re.M)
        rows=re.findall(r'^\s+(ops::\w+)\s+(\d+)\s*$',status_text,re.M)
        require(len(scores)==1 and len(rows)==3 and {k.removeprefix('ops::') for k,_ in rows}==EXPECTED_KERNELS,'Invalid official score/cycle report')
        sys.path.insert(0,str(root/'moa-dev'))
        from measure import parse_log
        checked=parse_log(log)
        require(all(checked[k.removeprefix('ops::')]['median_cycles']==int(v) for k,v in rows),'Status/log cycle disagreement')
        require(sum(v['output_comparisons_passed'] for v in checked.values())==15,'Missing official comparisons')
        data.update(official_score=float(scores[0]),official_cycles={k:int(v) for k,v in rows},official_rank=None,official_basic_comparisons_passed=15)
    local=root/f"moa-dev/logs/e137-official-{data['submission_id']}-observation-{len(data.get('observations',[]))+1}.log"
    local.parent.mkdir(parents=True,exist_ok=True);local.write_text(log)
    observation['log'].update(local_log=str(local.relative_to(root)),**digest(log.encode()))
    data.setdefault('observations',[]).append(observation);json_write(path,data)
    print(status_text)
    if data['current_status'] in {'failed','rejected'}:print(log[-5000:])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','preflight','submit','status'])
    default=Path(__file__).resolve().parents[1]
    if not (default/'baseline/current.json').is_file():default=Path.cwd()
    parser.add_argument('--root',type=Path,default=default)
    args=parser.parse_args();root=args.root.resolve()
    require((root/'baseline/current.json').is_file(),'Use --root with the intended moa workspace')
    if args.action=='check':
        summary,_=verify(root);print(json.dumps({'valid':True,**summary},indent=2))
    else:{'preflight':preflight,'submit':submit,'status':status}[args.action](root)

if __name__=='__main__':main()
