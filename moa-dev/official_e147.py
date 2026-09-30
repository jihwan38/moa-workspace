#!/usr/bin/env python3
"""Gate/stage E147, inspect its official record, or explicitly submit it once.

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

CAND='moa-dev/experiments/candidates/qkv-scalar-sub-vrf'
RECIPE='baseline/e147.json'
REMOTE='/tmp/moa-official-e147-20261001'
PREFLIGHT='docs/experiments/e147-official-payload-preflight-20261001.json'
RECORD='docs/experiments/e147-official-submit-20261001.json'
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


def verify_validation_fields(root,gate,compiled,manifest):
    """Check E147's frozen final schema; the independent verifier rechecks raw evidence below."""
    require(gate['label']=='e147-scalar-v1','Wrong final-gate variant')
    source=gate['source'];performance=gate['performance'];correctness=gate['correctness']
    require(source['compiled_input_count']==46 and source['source_manifest']==file_record(root,CAND+'/compiled-source-manifest.json'),'Final source manifest differs')
    require(source['image']==compiled['exact_image'] and source['production_projection']==manifest['src/device/sliding/projection.rs'],'Final production image/source differs')
    require(source['changed_from_e144']==['src/device/sliding/projection.rs'] and source['e144_o_ffn_preserved'],'Wrong source composition')
    require(source['whole_structural_gate']['passed'],'Whole structural gate failed')
    structure=source['whole_structural_gate']
    require(structure['native_count']==3 and structure['one_main_scale'] and structure['direct_scalar_sub_vrf_broadcasts']==2 and structure['shared_bitmap_hbm_bytes']==16384,'Wrong scalar supply structure')
    proof=correctness['finite_bf16_scalar_cpu_proof']
    require(proof['patterns']==65280 and proof['bitwise_f32_scale_equivalent'] and proof['nonfinite_not_claimed'],'Wrong finite scalar equivalence scope')
    require(proof['proof']==file_record(root,CAND+'/numerical/scale-proof.json'),'Scale proof changed')
    require(performance['independent_allocations']==3 and performance['all_paired_improved'] and performance['median_paired_delta_percent']<0,'No repeated whole-QKV gain')
    observations=performance['observations']
    require(len(observations)==3 and len({r['job'] for r in observations})==3 and all(r['paired_delta_percent']<0 for r in observations),'Wrong independent allocation count')
    require(correctness['standard_distinct_inputs']==9 and correctness['performance_repeated_inputs']==3 and correctness['extra_distinct_inputs']==6,'Incomplete standard-input validation')
    raw=correctness['raw_before_head'];full=correctness['full_qkv']
    require(raw['cases']==9 and raw['values']==73728 and raw['bitwise_bf16'] and raw['signed_zero_equivalent'],'Raw Q/K/V boundary failed')
    require(raw['actual_input_sha256_checks']==63 and raw['original_oracle_and_host_unchanged'] and raw['diagnostic_image_distinct_from_whole'],'Wrong raw-oracle provenance/scope')
    require(full['cases']==2 and full['values']==16384 and full['untouched_cache_values']==8380416,'Whole-QKV boundary/cache incomplete')
    require(full['actual_input_sha256_checks']==38 and full['actual_production_image'] and full['original_tolerances']==[.04,.01],'Whole-QKV reference/image changed')
    required_checks={
        CAND+'/numerical/scale-proof.py': ['--check'],
        CAND+'/audit.py': ['e147-scalar-v1','--check'],
        CAND+'/verify_graph.py': ['e147-scalar-v1','--check'],
        CAND+'/validation_performance.py': ['--check'],
        CAND+'/validation_extra.py': ['--check'],
        CAND+'/provenance.py': ['--check'],
        CAND+'/boundary-suite/raw_oracle.py': ['--check'],
        CAND+'/boundary-suite/check_full.py': [],
    }
    checks=gate['checks'];require(len(checks)==len(required_checks),'Wrong validation checker count')
    require({c['verifier']['path']:c['arguments'] for c in checks}==required_checks,'Validation checker/arguments changed')
    for check in checks:
        check_record(root,check['verifier']);require(check['exit_code']==0,'Validation checker failed')
    result=read(root,CAND+'/boundary-suite/raw-results-v1.json')
    require(result['valid'] and result['actual_upload_matches'] and result['case_count']==9 and result['output_elements']==73728,'Raw result not verified')
    require(result['exact_bitwise'] and result['signed_zero_equivalent'] and result['production_source_full_file_byte_identical'],'Raw source/precision scope changed')
    require({k:result['production_source'][k] for k in ('bytes','sha256')}==manifest['src/device/sliding/projection.rs'],'Raw diagnostic uses another projection')
    require(len(result['cases'])==27 and sum(int(c['checked']) for c in result['cases'])==73728 and all(int(c['failures'])==int(c['nonfinite'])==0 for c in result['cases']),'Raw mismatch/nonfinite/incomplete outputs')
    raw_build=read(root,CAND+'/boundary-suite/raw-build-v1.json')
    require(raw_build['original_source_inputs']==manifest and raw_build['production_whole_exact_image']==compiled['exact_image'],'Raw build not bound to measured production')
    require(raw_build['host_diagnostic_image_matches_exact'] and raw_build['image']==result['diagnostic_image'],'Raw exact/host/result images differ')


def verify(root,check_payload=True):
    require(read(root,'baseline/current.json')=={'recipe':RECIPE},'E147 is not adopted')
    recipe=read(root,RECIPE);require(recipe['id']=='E147' and recipe['status']=='adopted','Wrong recipe')
    require(recipe['git']['tag']=='baseline-e147','Wrong source tag')
    commit=subprocess.check_output(['git','rev-parse','baseline-e147^{commit}'],cwd=root,text=True).strip()
    require(commit==recipe['git']['commit'],'Tag mismatch')
    require(recipe['compile_inputs']==CAND+'/compiled-source-manifest.json','Wrong source manifest')
    manifest=read(root,recipe['compile_inputs'])
    audit=read(root,recipe['source_build_audit'])
    require(audit['candidate']=='E147' and audit['adopted_source_commit']==commit,'Wrong adoption audit')
    measured_path=CAND+'/measured-build-e147-adoption.json'
    require(audit['measured_build']==measured_path and audit['measured_build_digest']==file_record(root,measured_path),'Measured build mismatch')
    measured=read(root,measured_path)
    require(measured['variant']=='scalar-v1' and measured['label']=='e147-scalar-v1','Wrong measured variant')
    require(measured['exact_compile_exit_code']==measured['host_build_exit_code']==0 and not measured['compile_only'],'Build incomplete')
    require(measured['source_inputs_verified_in_container'] and measured['source_inputs']==manifest and len(manifest)==46,'Measured source binding failed')
    for k,n in [('original_compile_record','build-e147-scalar-v1-repro.json'),('original_package_record','package-e147-scalar-v1.json'),('source_binding','source-build-binding.json'),('validation','adoption/independent-final-review.json')]:
        require(audit[k]==file_record(root,CAND+'/'+n),'Changed audit evidence '+k)
    compiled=read(root,CAND+'/build-e147-scalar-v1-repro.json');package=read(root,CAND+'/package-e147-scalar-v1.json')
    require(compiled['source_inputs']==manifest and package['status']=='ready','Wrong compiled source/package')
    require(package['compile_record']==file_record(root,CAND+'/build-e147-scalar-v1-repro.json'),'Package compile binding failed')
    require(audit['images']==measured['images']==package['images'] and measured['package_files']==package['package_files'],'Image/package mismatch')
    gate_path=CAND+'/adoption/independent-final-review.json';gate=read(root,gate_path)
    require(gate['valid'] and gate['adoption_ready'] and gate['candidate']=='E147','Validation not ready')
    require(gate['verifier']==file_record(root,CAND+'/adoption/independent-final-review.py'),'Verifier mismatch')
    for record in gate['inputdigests'].values():check_record(root,record)
    verify_validation_fields(root,gate,compiled,manifest)
    run(root,[sys.executable,str(root/CAND/'adoption/independent-final-review.py'),'--check'])
    run(root,[sys.executable,str(root/'moa-dev/experiment_index.py'),'--check','--check-live'])
    prior_recipe=read(root,'baseline/e144.json');prior_audit=read(root,prior_recipe['source_build_audit']);prior=read(root,prior_audit['measured_build'])
    require(set(manifest)==set(prior['source_inputs']),'Source inventory differs from E144')
    require(sorted(k for k in manifest if manifest[k]!=prior['source_inputs'][k])==['src/device/sliding/projection.rs'],'Unexpected source changes')
    for k in ('sliding_attention_output','decoder_feedforward'):
        require(measured['images'][k]==prior['images'][k],'Unchanged E144 image differs '+k)
        require(gate['source']['unchanged_o_ffn_images'][k]=={n:measured['images'][k][n] for n in ('bytes','sha256')},'Unchanged image gate differs '+k)
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
    summary={'source_checkpoint':'E147','source_commit':commit,'source_files':allowed,'source_count':23,'compile_input_count':46,'source_build_audit':recipe['source_build_audit'],'source_build_audit_digest':file_record(root,recipe['source_build_audit']),'compile_inputs':file_record(root,recipe['compile_inputs']),'validation':file_record(root,gate_path),'validation_verifier':gate['verifier'],'measured_build':file_record(root,measured_path),'numerical_scope':'Nine standard inputs;73728 exact BF16 outputs (signed zero equivalent) of nine original raw Q/K/V cases;16384 whole-QKV outputs within original atol0.04/rtol0.01 and8380416 untouched cache values across two cancellation cases. Actual one-Main finite-BF16 scale and Sub direct-VRF scalar supply with unchanged full-H native projections; finite correctness, no arbitrary-input or end-to-end model claim.'}
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
    remote_helper='/tmp/moa-official-e147-payload-preflight.py'
    run(root,['docker','cp',str(helper),CONTAINER+':'+remote_helper])
    run(root,['docker','exec',CONTAINER,'python',remote_helper,'--source',REMOTE,'--output',REMOTE+'-payload-check'])
    raw=subprocess.check_output(['docker','exec',CONTAINER,'cat',REMOTE+'-payload-check/report.json'],cwd=root)
    data=json.loads(raw)
    require(data['source_files']==summary['source_files'],'Captured CLI bytes differ from E147')
    require(data['status']=='LOCAL_CLI_PAYLOAD_CHECK_PASS' and not data['official_submission_performed'],'Loopback preflight failed')
    evidence=root/'moa-dev/experiments/artifacts/official-submit-preflight-e147-20261001'
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
    local=root/f"moa-dev/logs/e147-official-{data['submission_id']}-observation-{len(data.get('observations',[]))+1}.log"
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
