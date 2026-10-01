#!/usr/bin/env python3
"""Gate/stage E151, inspect its official record, or explicitly submit it once.

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

CAND='moa-dev/experiments/candidates/ffn-down-scale-ingress-switch'
RECIPE='baseline/e151.json'
REMOTE='/tmp/moa-official-e151-20261001'
PREFLIGHT='docs/experiments/e151-official-payload-preflight-20261001.json'
RECORD='docs/experiments/e151-official-submit-20261001.json'
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
    require(read(root,'baseline/current.json')=={'recipe':RECIPE},'E151 is not adopted')
    recipe=read(root,RECIPE);require(recipe['id']=='E151' and recipe['status']=='adopted','Wrong recipe')
    require(recipe['git']['tag']=='baseline-e151','Wrong source tag')
    commit=subprocess.check_output(['git','rev-parse','baseline-e151^{commit}'],cwd=root,text=True).strip()
    require(commit==recipe['git']['commit'],'Tag mismatch')
    require(recipe['compile_inputs']==CAND+'/compiled-source-manifest.json','Wrong source manifest')
    manifest=read(root,recipe['compile_inputs'])
    audit=read(root,recipe['source_build_audit'])
    require(audit['candidate']=='E151' and audit['adopted_source_commit']==commit,'Wrong adoption audit')
    measured_path=CAND+'/measured-build-e151-adoption.json'
    require(audit['measured_build']==measured_path and audit['measured_build_digest']==file_record(root,measured_path),'Measured build mismatch')
    measured=read(root,measured_path)
    require(measured['variant']=='ingress-v1' and measured['label']=='e151-ingress-v1','Wrong measured variant')
    require(measured['exact_compile_exit_code']==measured['host_build_exit_code']==0 and not measured['compile_only'],'Build incomplete')
    require(measured['source_inputs_verified_in_container'] and measured['source_inputs']==manifest and len(manifest)==46,'Measured source binding failed')
    for k,n in [('original_compile_record','build-e151-ingress-v1-repro.json'),('original_package_record','package-e151-ingress-v1.json'),('source_binding','source-build-binding.json'),('validation','adoption/independent-final-review.json')]:
        require(audit[k]==file_record(root,CAND+'/'+n),'Changed audit evidence '+k)
    compiled=read(root,CAND+'/build-e151-ingress-v1-repro.json');package=read(root,CAND+'/package-e151-ingress-v1.json')
    require(compiled['source_inputs']==manifest and package['status']=='ready','Wrong compiled source/package')
    require(package['compile_record']==file_record(root,CAND+'/build-e151-ingress-v1-repro.json'),'Package compile binding failed')
    require(audit['images']==measured['images']==package['images'] and measured['package_files']==package['package_files'],'Image/package mismatch')
    gate_path=CAND+'/adoption/independent-final-review.json';gate=read(root,gate_path)
    require(gate['valid'] and gate['adoption_ready'] and gate['candidate']=='E151','Validation not ready')
    require(gate['verifier']==file_record(root,CAND+'/adoption/independent-final-review.py'),'Verifier mismatch')
    for record in gate['inputdigests'].values():check_record(root,record)
    require(gate['label']=='e151-ingress-v1','Wrong validation label')
    source=gate['source']; performance=gate['performance']; correctness=gate['correctness']
    require(source['compiled_input_count']==46 and source['source_manifest']==file_record(root,CAND+'/compiled-source-manifest.json'),'Wrong final source binding')
    require(source['image']==compiled['exact_image'] and source['production_mlp']==manifest['src/device/shared/mlp.rs'],'Final production image/MLP differs')
    require(source['changed_from_e150']==['src/device/shared/mlp.rs'] and source['e150_qkv_o_preserved'],'Unexpected source composition')
    require(source['whole_structural_gate']['passed'],'Whole structural gate failed')
    require(performance['independent_allocations']==3 and performance['all_paired_improved'] and performance['median_paired_delta_percent']<0,'No repeated complete-FFN gain')
    observations=performance['observations']
    require(len(observations)==3 and len({row['job'] for row in observations})==3 and all(row['paired_delta_percent']<0 for row in observations),'Wrong allocation count or mixed gain')
    require(correctness['standard_distinct_inputs']==9 and correctness['performance_repeated_inputs']==3 and correctness['extra_distinct_inputs']==6,'Incomplete standard inputs')
    raw=correctness['raw_down']; full=correctness['full_ffn']
    require(raw['cases']==9 and raw['values']==34560 and raw['bitwise_bf16'] and raw['signed_zero_equivalent'],'Raw down boundary failed')
    require(raw['production_mlp_prefix_preserved'] and raw['original_oracle_and_host_unchanged'] and raw['diagnostic_image_distinct_from_whole'],'Wrong raw diagnostic scope')
    require(full['cases']==1 and full['values']==3840 and full['bitwise_bf16'] and full['actual_production_image'] and full['original_reference_and_host_unchanged'],'Whole FFN regression/image/reference failed')
    for check in gate['checks']:
        check_record(root,check['verifier']);require(check['exit_code']==0,'Validation checker failed')
    run(root,[sys.executable,str(root/CAND/'adoption/independent-final-review.py'),'--check'])
    run(root,[sys.executable,str(root/'moa-dev/experiment_index.py'),'--check','--check-live'])
    prior_recipe=read(root,'baseline/e150.json');prior_audit=read(root,prior_recipe['source_build_audit']);prior=read(root,prior_audit['measured_build'])
    require(set(manifest)==set(prior['source_inputs']),'Source inventory differs from E150')
    require(sorted(k for k in manifest if manifest[k]!=prior['source_inputs'][k])==['src/device/shared/mlp.rs'],'Unexpected source changes')
    for k in ('sliding_project_qkv','sliding_attention_output'):
        require(measured['images'][k]==prior['images'][k],'Unchanged image differs '+k)
        require(source['unchanged_qkv_o_images'][k]=={name:measured['images'][k][name] for name in ('bytes','sha256')},'Unchanged image gate differs '+k)
    package_dir=root/measured['package_path'];require({f.name:digest(f.read_bytes()) for f in package_dir.iterdir() if f.is_file()}==measured['package_files'],'Actual measured package changed')
    host=(package_dir/'test_runtime').read_bytes();require(set(measured['standard_image_files'])==EXPECTED_KERNELS,'Wrong image inventory')
    for k,record in measured['standard_image_files'].items():
        check_record(root,record);raw=(root/record['path']).read_bytes()
        require(digest(raw)=={n:measured['images'][k][n] for n in ('bytes','sha256')} and host.count(raw)==1,'Actual image changed '+k)
    crate=root/'furiosa-opt-gemma4-12B';require({p.relative_to(crate).as_posix() for p in (crate/'src').rglob('*') if p.is_file()}=={p for p in manifest if p.startswith('src/')},'Source inventory changed')
    files={}
    for name,expected in manifest.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe path')
        source=recipe['tracked_lock'] if name=='Cargo.lock' else 'furiosa-opt-gemma4-12B/'+name
        raw=subprocess.check_output(['git','show',f'{commit}:{source}'],cwd=root)
        require(digest(raw)==expected and not (crate/name).is_symlink() and digest((crate/name).read_bytes())==expected,'Live/tag/source differs '+name);files[name]=raw
    allowed={n:m for n,m in manifest.items() if n=='src/ops.rs' or n.startswith('src/device/')};require(len(allowed)==23,'Wrong payload paths')
    summary={'source_checkpoint':'E151','source_commit':commit,'source_files':allowed,'source_count':23,'compile_input_count':46,'source_build_audit':recipe['source_build_audit'],'source_build_audit_digest':file_record(root,recipe['source_build_audit']),'compile_inputs':file_record(root,recipe['compile_inputs']),'validation':file_record(root,gate_path),'validation_verifier':gate['verifier'],'measured_build':file_record(root,measured_path),'numerical_scope':'Nine standard inputs,34560 exact BF16 outputs (signed-zero equivalent) from nine original down mapping cases, and3840 exact outputs of the original whole-FFN construction. Full-row480-value scale ingress under row-low2 ownership and five Sub Switch4 streams into original twelve-row VRFs; original E150 weighted native/partial arithmetic unchanged. Raw diagnostic preserves production MLP as an identical prefix plus original diagnostic append; whole uses actual production image. Finite correctness, no arbitrary-input or end-to-end model claim.'}
    if check_payload:
        pre=read(root,PREFLIGHT);require(pre['status']=='LOCAL_CLI_PAYLOAD_CHECK_PASS' and pre['source']==REMOTE,'Wrong payload preflight')
        require(pre['source_files']==allowed and pre['source_count']==23 and pre['three_requests_byte_identical'] and pre['source_unchanged_after_checks'],'Payload differs')
        require(pre['synthetic_token_only'] and not pre['saved_config_used_or_written'] and not pre['official_submission_performed'],'Preflight scope changed')
        require(len(pre['cases'])==4 and all(c['passed'] for c in pre['cases']),'Incomplete preflight')
        raw=subprocess.check_output(['docker','exec','--workdir',REMOTE,CONTAINER,'tar','-cf','-',*sorted(manifest)],cwd=root)
        require(archive_digests(raw)==manifest,'Staged inputs differ')
        binary=subprocess.check_output(['docker','exec',CONTAINER,'cat','/home/moa/.cargo/bin/moa-submitter'],cwd=root)
        require(digest(binary)==pre['cli_binary'],'Submitter changed');summary['payload_preflight']=file_record(root,PREFLIGHT)
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
    remote_helper='/tmp/moa-official-e151-payload-preflight.py'
    run(root,['docker','cp',str(helper),CONTAINER+':'+remote_helper])
    run(root,['docker','exec',CONTAINER,'python',remote_helper,'--source',REMOTE,'--output',REMOTE+'-payload-check'])
    raw=subprocess.check_output(['docker','exec',CONTAINER,'cat',REMOTE+'-payload-check/report.json'],cwd=root)
    data=json.loads(raw)
    require(data['source_files']==summary['source_files'],'Captured CLI bytes differ from E151')
    require(data['status']=='LOCAL_CLI_PAYLOAD_CHECK_PASS' and not data['official_submission_performed'],'Loopback preflight failed')
    evidence=root/'moa-dev/experiments/artifacts/official-submit-preflight-e151-20261001'
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
    local=root/f"moa-dev/logs/e151-official-{data['submission_id']}-observation-{len(data.get('observations',[]))+1}.log"
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
