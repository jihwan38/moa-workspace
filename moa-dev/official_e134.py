#!/usr/bin/env python3
"""Gate/stage E134, inspect its official record, or explicitly submit it once.

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

CAND='moa-dev/experiments/candidates/ffn-native-down-weighted-residual'
RECIPE='baseline/e134.json'
REMOTE='/tmp/moa-official-e134-20260927'
PREFLIGHT='docs/experiments/e134-official-payload-preflight-20260927.json'
RECORD='docs/experiments/e134-official-submit-20260927.json'
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
    require(current.get('recipe')==RECIPE,'E134 is not the adopted baseline: '+str(current.get('recipe')))
    recipe=read(root,RECIPE)
    require(recipe['id']=='E134' and recipe['status']=='adopted','E134 recipe is not adopted')
    require(recipe['git']['tag']=='baseline-e134','Wrong source tag')
    require(recipe['compile_inputs']==CAND+'/compiled-source-manifest.json','Wrong E134 manifest path')
    require(recipe['performance_report']=='docs/experiments/e134-adoption-performance.json','Wrong performance report')
    commit=subprocess.check_output(['git','rev-parse','baseline-e134^{commit}'],cwd=root,text=True).strip()
    require(commit==recipe['git']['commit'],'Tag/recipe commit mismatch')
    manifest=read(root,recipe['compile_inputs'])
    measured=read(root,CAND+'/measured-build-e134-adoption.json')
    require(len(manifest)==46 and measured['source_inputs']==manifest,'All46 measured compile inputs must match')
    require(measured['variant']=='weighted-v3' and measured['label']=='e134-weighted-v3','Wrong measured variant')
    require(measured['exact_compile_exit_code']==measured['host_build_exit_code']==0,'Build did not complete')
    require(measured['source_inputs_verified_in_container'] and not measured['compile_only'],'Missing full build provenance')
    gate=read(root,CAND+'/validation/final-verification.json')
    require(gate['valid'] and gate['adoption_ready'] and gate['candidate']=='E134','Final validation is not adoption-ready')
    check_record(root,gate['verifier'])
    for record in gate['inputdigests'].values():check_record(root,record)
    require(gate['performance']['independent_allocations']==3 and gate['performance']['all_paired_ffn_improved'],'Performance repeat gate failed')
    require(gate['standard_correctness']['distinct_inputs']==9,'Nine standard/extra inputs required')
    require(gate['down_oracle']['cases']==9 and gate['down_oracle']['exact_mapping_outputs']==34560,'Nine exact raw-down cases required')
    require(gate['full_ffn_stress']['e128_exact_outputs']==gate['full_ffn_stress']['e134_exact_outputs']==3840,'Constructed full-FFN repair check missing')
    require(gate['full_ffn_stress']['e133_tolerance_failures']==1920,'Frozen diagnostic lineage changed')
    # Recompute all original graph/package/upload/performance/extra/oracle/stress gates.
    run(root,[sys.executable,str(root/CAND/'verify_adoption.py'),'--check','--check-live'])
    adoption=read(root,CAND+'/validation/adoption-verification.json')
    require(adoption['valid'],'Live E134 adoption verification failed')
    run(root,[sys.executable,str(root/'moa-dev/experiment_index.py'),'--check','--check-live'])
    report=read(root,recipe['performance_report'])
    require(report['valid'] and report['independent_arena_jobs']==3,'Invalid performance report')
    for name in ('extra','probe'):
        require(read(root,f'docs/experiments/e134-{name}-correctness.json')['valid'],'Invalid '+name+' report')
    binding=read(root,CAND+'/source-build-binding.json')
    require(binding['valid'] and binding['source_count']==46 and binding['qkv_o_images_identical_to_e128'],'Invalid source/image binding')
    require(binding['changed_compile_files']==['src/device/shared/mlp.rs'],'Unexpected source changes')
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
    summary={'source_checkpoint':'E134','source_commit':commit,'compile_input_count':46,
             'source_build_audit':recipe['source_build_audit'],'source_files':allowed,'source_count':23,
             'compile_inputs':file_record(root,recipe['compile_inputs']),
             'validation':file_record(root,CAND+'/validation/final-verification.json'),
             'adoption_verification':file_record(root,CAND+'/validation/adoption-verification.json'),
             'adoption_verifier':file_record(root,CAND+'/verify_adoption.py'),
             'stage1_scope_decision':file_record(root,CAND+'/stage1-scope-decision.json'),
             'adoption_decision':file_record(root,CAND+'/adoption-decision.json'),
             'measured_build':file_record(root,CAND+'/measured-build-e134-adoption.json')}
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
    remote_helper='/tmp/moa-official-e134-payload-preflight.py'
    run(root,['docker','cp',str(helper),CONTAINER+':'+remote_helper])
    run(root,['docker','exec',CONTAINER,'python',remote_helper,'--source',REMOTE,'--output',REMOTE+'-payload-check'])
    raw=subprocess.check_output(['docker','exec',CONTAINER,'cat',REMOTE+'-payload-check/report.json'],cwd=root)
    data=json.loads(raw)
    require(data['source_files']==summary['source_files'],'Captured CLI bytes differ from E134')
    require(data['status']=='LOCAL_CLI_PAYLOAD_CHECK_PASS' and not data['official_submission_performed'],'Loopback preflight failed')
    evidence=root/'moa-dev/experiments/artifacts/official-submit-preflight-e134-20260927'
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
    local=root/f"moa-dev/logs/e134-official-{data['submission_id']}-observation-{len(data.get('observations',[]))+1}.log"
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
