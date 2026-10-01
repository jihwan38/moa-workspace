#!/usr/bin/env python3
"""One immutable E47 case7 against the unchanged actual E149 whole-O image."""
import argparse,ast,io,json,re,shutil,subprocess,sys,tarfile
from pathlib import Path
from datetime import datetime,timezone
import oracle as common
from oracle import ROOT,HERE,VALID,TARGET,digest,file,load,write,sourcefiles,run,cli
SRC=VALID/'stress-src';LABEL='e149-full-o-case7-v1';ART=ROOT/'moa-dev/experiments/artifacts'/HERE.name/LABEL;REMOTE='/tmp/moa-e149-validation/'+LABEL
BUILD=VALID/'stress-build-v1.json';RESULTS=VALID/'stress-results-v1.json';PROJECTION=common.PROJECTION
LEGACY=ROOT/'moa-dev/experiments/candidates/output-boundary-probe-e47'
FIXTURE=ROOT/'moa-dev/experiments/artifacts/output-boundary-probe-e47-package/boundary.safetensors'
SCRIPTS=ROOT/'furiosa-opt-gemma4-12B/scripts';OLD_SCRIPTS=ROOT/'moa-dev/experiments/artifacts/e47-output-sixteen-columns-extra/crate/scripts'
REFERENCE_FILES=['generate_references.py','reference/gemma4.py','fixture_prng.py']
def functions(path):
    s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
def reference_audit():
    manifest=load(LEGACY/'manifest.json');assert digest(FIXTURE.read_bytes())==manifest['package']['boundary.safetensors']
    old=load(LEGACY/'cpu-references.json');a=functions(SCRIPTS/'generate_references.py');b=functions(OLD_SCRIPTS/'generate_references.py')
    funcs={}
    for n in ['scaled_projection','gen_sliding_attention_output','rms_norm']:
        assert a[n]==b[n];funcs[n]=digest(a[n].encode())
    for n in ['reference/gemma4.py','fixture_prng.py']:assert digest((SCRIPTS/n).read_bytes())['sha256']==old['official_sources'][n]
    return {'legacy_manifest':file(LEGACY/'manifest.json'),'legacy_cpu_truth':file(LEGACY/'cpu-references.json'),'fixture':file(FIXTURE),'original_functions_unchanged':funcs,'current_reference_files':{n:file(SCRIPTS/n) for n in REFERENCE_FILES}}
def build(label):
    common.verify_preparation()
    sys.path.insert(0,str(HERE));from package import checked_compile
    base_path,base,*_=checked_compile(label);from package import checked_gate;checked_gate(label);artifact=ROOT/'moa-dev/experiments/artifacts'/HERE.name/label
    files=sourcefiles(artifact/'crate');assert {k:digest(v) for k,v in files.items()}==base['source_inputs']
    assert 'src/bin/probe_output.rs' not in files
    files['src/bin/probe_output.rs']=(SRC/'probe_output.rs').read_bytes()
    ART.mkdir(parents=True,exist_ok=False);crate=ART/'crate';crate.mkdir();evidence=VALID/'stress-evidence-v1';evidence.mkdir(exist_ok=False)
    for n,raw in files.items():p=crate/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    record={'label':LABEL,'build_label':label,'base_build':file(base_path),'valid':False,'diagnostic_only':True,'official_submission':False,'performance_excluded':True,'original_sources_unchanged':True,'source_inputs':{n:digest(v) for n,v in files.items()},'host_source':file(SRC/'probe_output.rs'),'cpu_checker':file(SRC/'cpu_reference.py'),'builder':file(Path(__file__)),'helper':file(Path(common.__file__)),'target_dir':TARGET,'reference':reference_audit(),'commands':[]}
    def recorded(args,log=None):record['commands'].append(args);run(args,log)
    try:
        recorded(['docker','exec','moa2026-dev','mkdir','-p',REMOTE])
        recorded(['docker','cp',str(crate)+'/.','moa2026-dev:'+REMOTE])
        raw=subprocess.check_output(['docker','exec','--workdir',REMOTE,'moa2026-dev','tar','-cf','-',*files])
        with tarfile.open(fileobj=io.BytesIO(raw)) as a:assert {m.name:digest(a.extractfile(m).read()) for m in a if m.isfile()}==record['source_inputs']
        refs=ART/'reference';refs.mkdir()
        for n in REFERENCE_FILES:p=refs/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SCRIPTS/n,p)
        recorded(['docker','cp',str(refs),'moa2026-dev:'+REMOTE+'/reference'])
        recorded(['docker','cp',str(FIXTURE),'moa2026-dev:'+REMOTE+'/boundary.safetensors'])
        recorded(['docker','cp',str(SRC/'cpu_reference.py'),'moa2026-dev:'+REMOTE+'/cpu_reference.py'])
        recorded(['docker','exec','moa2026-dev','python3',REMOTE+'/cpu_reference.py',REMOTE+'/reference',REMOTE+'/boundary.safetensors',REMOTE+'/cpu-reference.json'],evidence/'cpu-reference.log')
        recorded(['docker','cp','moa2026-dev:'+REMOTE+'/cpu-reference.json',str(VALID/'stress-cpu-reference-v1.json')])
        env=['docker','exec','--workdir',REMOTE,'-e','CARGO_HOME=/home/moa/.cargo','-e','CARGO_NET_OFFLINE=true','-e','CARGO_TARGET_DIR='+TARGET,'-e','FURIOSA_OPT_OUT_DIR='+REMOTE+'/kernels','moa2026-dev']
        print('Building one-case whole-O host from frozen production source',flush=True)
        recorded(env+['cargo','furiosa-opt','build','--release','--locked','--bin','probe_output'],evidence/'host-build.log')
        package=ART/'package';package.mkdir();recorded(['docker','cp','moa2026-dev:'+TARGET+'/release/probe_output',str(package/'probe_output')]);recorded(['docker','cp','moa2026-dev:'+REMOTE+'/kernels',str(ART/'kernels')])
        images=list((ART/'kernels').rglob('*.bin'));assert len(images)==1
        image=images[0].read_bytes();assert digest(image)==base['exact_image'] and (package/'probe_output').read_bytes().count(image)==1
        record['image']={'name':str(images[0].relative_to(ART/'kernels')),**digest(image),'host_occurrences':1,'matches_whole_exact':True}
        shutil.copyfile(FIXTURE,package/'boundary.safetensors');(package/'run.sh').write_text('#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nchmod +x probe_output\n./probe_output --fixture boundary.safetensors\n')
        record['package_files']={p.name:digest(p.read_bytes()) for p in package.iterdir()}
        recorded(['docker','cp',str(package),'moa2026-dev:'+REMOTE+'/package'])
        recorded(['docker','exec','--workdir',REMOTE+'/package','moa2026-dev','./probe_output','--fixture','boundary.safetensors','--host-check-only'],evidence/'host-check.log')
        log=(evidence/'host-check.log').read_text();assert log.count('INPUT_CHECK case=7 ')==5 and 'completed=1 planned=1 total_failed_values=0 status=PASS' in log
        record.update(valid=True,host_build_exit_code=0,cpu_reference=file(VALID/'stress-cpu-reference-v1.json'))
    except Exception as e:record['error']=repr(e);raise
    finally:record['evidence']={p.name:file(p) for p in evidence.iterdir() if p.is_file()};write(BUILD,record)
    print('Actual whole-O exact image and one-case host checks PASS')
def verify():
    common.verify_preparation()
    b=load(BUILD);assert b['valid'] and b['builder']==file(Path(__file__)) and b['helper']==file(Path(common.__file__))
    assert b['host_source']==file(SRC/'probe_output.rs') and b['cpu_checker']==file(SRC/'cpu_reference.py') and b['reference']==reference_audit()
    base_path=ROOT/b['base_build']['path'];assert file(base_path)==b['base_build'];base=load(base_path)
    files=sourcefiles(ART/'crate');assert {n:digest(v) for n,v in files.items()}==b['source_inputs']
    assert {n:v for n,v in b['source_inputs'].items() if n!='src/bin/probe_output.rs'}==base['source_inputs']
    assert files['src/bin/probe_output.rs']==(SRC/'probe_output.rs').read_bytes()
    assert {p.name:digest(p.read_bytes()) for p in (ART/'package').iterdir()}==b['package_files']
    image=(ART/'kernels'/b['image']['name']).read_bytes();assert digest(image)==base['exact_image']=={k:b['image'][k] for k in ['bytes','sha256']};assert (ART/'package/probe_output').read_bytes().count(image)==1
    for rec in [*b['evidence'].values(),b['cpu_reference']]:assert file(ROOT/rec['path'])==rec
    assert load(VALID/'stress-cpu-reference-v1.json')['passed'];return b
def parsed(log):
    results=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in log.splitlines() if s.startswith('O_RESULT ')]
    assert len(results)==1 and results[0]['case']=='7' and results[0]['checked']=='3840' and results[0]['failures']==results[0]['nonfinite']=='0' and results[0]['status']=='PASS'
    assert 'O_MISMATCH' not in log and 'panicked' not in log and 'status=FAIL' not in log
    assert log.count('O_SUITE_RESULT completed=1 planned=1 total_failed_values=0 status=PASS')==1
    inputs=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in log.splitlines() if s.startswith('INPUT_CHECK ')]
    cpu=load(VALID/'stress-cpu-reference-v1.json');assert len(inputs)==5
    assert {v['tensor'] for v in inputs}==set(cpu['input_checksums'])
    for v in inputs:assert v['case']=='7' and v['status']=='PASS' and int(v['checksum'],16)==cpu['input_checksums'][v['tensor']]
    return results

def submit():
    b=verify();sys.path.insert(0,str(HERE));from package import checked_gate
    checked_gate(b['build_label'])
    for n,r in b['package_files'].items():assert subprocess.check_output(['docker','exec','moa2026-dev','sha256sum',REMOTE+'/package/'+n],text=True).split()[0]==r['sha256']
    p=VALID/'stress-submission-v1.json';write(p,{'attempted_at_utc':datetime.now(timezone.utc).isoformat(),'build':file(BUILD)})
    response=cli('submit',*[REMOTE+'/package/'+n for n in b['package_files']],'--entrypoint','run.sh','--timeout',70,'--name','beef-e149-whole-o-case7-v1');r=load(p);r['response']=response;ids=re.findall(r'submitted job (\d+)',response)
    if len(ids)==1:r['job_id']=int(ids[0])
    p.write_text(json.dumps(r,indent=2)+'\n');assert len(ids)==1,response;print(response,end='')
def collect():
    b=verify();job=load(VALID/'stress-submission-v1.json')['job_id'];status=json.loads(cli('status',job));assert status['status'] in {'SUCCEEDED','COMPLETED','FAILED','CANCELLED'}
    log=cli('logs',job);logpath=VALID/f'stress-evidence-v1/arena-{job}.log';logpath.write_text(log)
    r={'valid':False,'job':status,'build':file(BUILD),'raw_log':file(logpath),'observed_result_lines':[s for s in log.splitlines() if s.startswith('O_RESULT ')],'scope':'One original whole O/RMS/residual case; public atol=.05 rtol=.01. No performance or arbitrary-input claim.'}
    try:
        archive=ART/f'job-{job}.tar';remote=REMOTE+f'/job-{job}.tar';cli('download',job,'--out',remote);run(['docker','cp','moa2026-dev:'+remote,str(archive)])
        with tarfile.open(archive) as a:
            members=[m for m in a if m.isfile()];assert len(members)==len({m.name for m in members});uploaded={m.name:digest(a.extractfile(m).read()) for m in members}
        r.update(archive=file(archive),package_files=uploaded,actual_upload_matches={k:v for k,v in uploaded.items() if k!='job.yaml'}==b['package_files'])
        assert r['actual_upload_matches'] and status['exit_code']==0 and status['status'] in {'SUCCEEDED','COMPLETED'}
        r.update(valid=True,cases=parsed(log),case_count=1,output_elements=3840,image=b['image'],production_source_inputs_match=True,cpu_reference=b['cpu_reference'])
    except Exception as e:r['error']=repr(e)
    write(RESULTS,r);print(json.dumps(r,indent=2));assert r['valid']
def check():
    b=verify();r=load(RESULTS);assert r['valid'] and r['build']==file(BUILD)
    for key in ['raw_log','archive']:assert file(ROOT/r[key]['path'])==r[key]
    assert parsed((ROOT/r['raw_log']['path']).read_text())==r['cases']
    with tarfile.open(ROOT/r['archive']['path']) as a:
        members=[m for m in a if m.isfile()];assert len(members)==len({m.name for m in members});uploaded={m.name:digest(a.extractfile(m).read()) for m in members}
    assert uploaded==r['package_files'] and {k:v for k,v in uploaded.items() if k!='job.yaml'}==b['package_files']
    assert r['image']==b['image'] and r['cpu_reference']==b['cpu_reference'] and r['case_count']==1 and r['output_elements']==3840
    print('Actual whole O source/image/upload/CPU reference/case7 verified PASS')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','verify','submit','collect'],nargs='?');p.add_argument('--build-label');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:check()
    elif a.action=='build':build(a.build_label)
    elif a.action:globals()[a.action]()
    else:p.error('action or --check required')
