#!/usr/bin/env python3
"""Build/execute/check the frozen C09 pre-RMS correctness boundary diagnostic."""
import argparse, hashlib, io, json, re, shutil, subprocess, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]; VALID=HERE/'validation'; SRC=VALID/'probe-v1'
LABEL='c09-adoption-probe-v1'; ART=ROOT/'moa-dev/experiments/artifacts'/HERE.name/LABEL
REMOTE='/tmp/moa-c09-validation-probe/'+LABEL; TARGET='/tmp/moa-c09-validation-probe/target'
PROJECTION='src/device/sliding/projection.rs'; BUILD=VALID/'probe-build.json'; RESULTS=VALID/'probe-results.json'
sys.path.insert(0,str(ROOT/'moa-dev')); from measure import cli

def digest(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def file(p):return {'path':str(p.relative_to(ROOT)),**digest(p.read_bytes())}
def load(p):return json.loads(p.read_text())
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def sourcefiles(crate):
    return {str(p.relative_to(crate)):p.read_bytes() for p in crate.rglob('*') if p.is_file() and (p.is_relative_to(crate/'src') or p.name in {'Cargo.toml','Cargo.lock','rust-toolchain.toml'})}
def run(args,log=None):
    if log:
        with log.open('x') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
    else:subprocess.run(args,stdout=subprocess.DEVNULL,check=True)

def build():
    base=load(HERE/'build-c09-first-npu-v1-repro.json'); recipe=load(SRC/'recipe.json')
    original=ROOT/'moa-dev/experiments/artifacts'/HERE.name/base['label']/'crate'
    files=sourcefiles(original);assert {k:digest(v) for k,v in files.items()}==base['source_inputs']
    assert files[PROJECTION]==(HERE/'frozen-two-stage-v1/projection.rs').read_bytes()
    assert digest(files[PROJECTION])['sha256']==recipe['projection_sha256']
    original_files={k:digest(v) for k,v in files.items()}
    files['src/lib.rs']+=b'\npub mod probe;\n'
    files['src/probe.rs']=(SRC/'probe.rs').read_bytes();files['src/bin/probe_output.rs']=(SRC/'probe_output.rs').read_bytes()
    ART.mkdir(parents=True,exist_ok=False);crate=ART/'crate';crate.mkdir()
    for name,raw in files.items():p=crate/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    evidence=VALID/'probe-evidence';evidence.mkdir(exist_ok=False)
    record={'label':LABEL,'diagnostic_only':True,'performance_excluded':True,'official_submission':False,
            'base_build':file(HERE/'build-c09-first-npu-v1-repro.json'),'recipe':file(SRC/'recipe.json'),
            'original_source_inputs':original_files,'source_inputs':{k:digest(v) for k,v in files.items()},
            'production_projection':file(HERE/'frozen-two-stage-v1/projection.rs'),
            'production_projection_byte_identical':True,'all_original_sources_unchanged_except_lib_append':True,
            'wrapper_source':file(SRC/'probe.rs'),'host_source':file(SRC/'probe_output.rs'),'oracle_source':file(SRC/'generate.py'),
            'builder':file(Path(__file__)),'target_dir':TARGET,'commands':[],'valid':False}
    def recorded(args,log=None):record['commands'].append(args);run(args,log)
    try:
        recorded(['docker','exec','moa2026-dev','mkdir','-p',REMOTE])
        recorded(['docker','cp',str(crate)+'/.','moa2026-dev:'+REMOTE])
        raw=subprocess.check_output(['docker','exec','--workdir',REMOTE,'moa2026-dev','tar','-cf','-',*files])
        with tarfile.open(fileobj=io.BytesIO(raw)) as a:
            assert {m.name:digest(a.extractfile(m).read()) for m in a if m.isfile()}==record['source_inputs']
        record['container_source_verified']=True
        recorded(['docker','cp',str(SRC/'generate.py'),'moa2026-dev:'+REMOTE+'/generate.py'])
        recorded(['docker','exec','moa2026-dev','python3',REMOTE+'/generate.py',REMOTE+'/fixture'],evidence/'oracle-generation.log')
        recorded(['docker','cp','moa2026-dev:'+REMOTE+'/fixture',str(ART/'fixture')])
        shutil.copyfile(ART/'fixture/probe-oracle.json',VALID/'probe-oracle.json')
        env=['docker','exec','--workdir',REMOTE,'-e','CARGO_NET_OFFLINE=true','-e','CARGO_TARGET_DIR='+TARGET,'-e','FURIOSA_OPT_OUT_DIR='+REMOTE+'/kernels','moa2026-dev']
        print('Compiling exact pre-RMS C09 probe with separate target '+TARGET,flush=True)
        recorded(env+['cargo','furiosa-opt','compile','probe::output_before_rms','--exact','--dump-schedule','probe-schedule.json','--dump-graph','probe-graph.json'],evidence/'compile.log')
        record['exact_compile_exit_code']=0
        for name in ['probe-schedule.json','probe-graph.json']:
            recorded(['docker','cp','moa2026-dev:'+REMOTE+'/'+name,str(evidence/name)])
        print('Building pre-RMS host',flush=True)
        recorded(env+['cargo','furiosa-opt','build','--release','--locked','--bin','probe_output'],evidence/'host-build.log')
        record['host_build_exit_code']=0
        package=ART/'package';package.mkdir()
        recorded(['docker','cp','moa2026-dev:'+TARGET+'/release/probe_output',str(package/'probe_output')])
        shutil.copyfile(ART/'fixture/probe-fixture.bin',package/'probe-fixture.bin')
        (package/'run.sh').write_text('#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nchmod +x probe_output\n./probe_output --fixture probe-fixture.bin\n')
        recorded(['docker','cp','moa2026-dev:'+REMOTE+'/kernels',str(ART/'kernels')])
        host=(package/'probe_output').read_bytes();images=list((ART/'kernels').rglob('*.bin'));assert len(images)==1
        image=images[0].read_bytes();assert host.count(image)==1
        record['image']={'name':str(images[0].relative_to(ART/'kernels')),**digest(image),'host_occurrences':1}
        record['package_files']={p.name:digest(p.read_bytes()) for p in package.iterdir()}
        recorded(['docker','cp',str(package),'moa2026-dev:'+REMOTE+'/package'])
        recorded(['docker','exec','--workdir',REMOTE+'/package','moa2026-dev','./probe_output','--fixture','probe-fixture.bin','--host-check-only'],evidence/'host-check.log')
        hostcheck=(evidence/'host-check.log').read_text();assert hostcheck.count('PROBE_INPUT ')==24 and 'failed_values=0 host_only=true status=PASS' in hostcheck
        record['valid']=True
    except Exception as e:
        record['error']=repr(e);raise
    finally:
        record['evidence']={p.name:file(p) for p in evidence.iterdir() if p.is_file()}
        write(BUILD,record)
    print('Probe build and all24 input checksum checks PASS')

def finish_package():
    # Attempt1 compiled successfully; repair only recursive image discovery.
    b=load(BUILD);assert not b['valid'] and b['error']=='AssertionError()'
    assert b['exact_compile_exit_code']==b['host_build_exit_code']==0
    failed=VALID/'probe-build-attempt1.json';BUILD.rename(failed)
    b['packaging_repair']={'failed_attempt':file(failed),'reason':'SDK output is nested kernels/furiosa-opt-gemma4; glob changed to rglob. No source or compile changes.'}
    snapshot=SRC/'build-driver-at-compile.py'
    assert digest(snapshot.read_bytes())=={k:b['builder'][k] for k in ('bytes','sha256')}
    b['build_driver_snapshot']=file(snapshot)
    package=ART/'package';images=list((ART/'kernels').rglob('*.bin'));assert len(images)==1
    image=images[0].read_bytes();assert (package/'probe_output').read_bytes().count(image)==1
    b['image']={'name':str(images[0].relative_to(ART/'kernels')),**digest(image),'host_occurrences':1}
    b['package_files']={p.name:digest(p.read_bytes()) for p in package.iterdir()}
    run(['docker','cp',str(package),'moa2026-dev:'+REMOTE+'/package'])
    log=VALID/'probe-evidence/host-check.log'
    run(['docker','exec','--workdir',REMOTE+'/package','moa2026-dev','./probe_output','--fixture','probe-fixture.bin','--host-check-only'],log)
    raw=log.read_text();assert raw.count('PROBE_INPUT ')==24 and 'failed_values=0 host_only=true status=PASS' in raw
    b['evidence'][log.name]=file(log);b['valid']=True;b.pop('error');b['packaging_driver']=file(Path(__file__))
    write(BUILD,b);print('Packaging-only repair; immutable exact image and24 host checks PASS')

def verify_build():
    b=load(BUILD);assert b['valid'] and b['target_dir']==TARGET
    if 'build_driver_snapshot' in b:
        assert file(ROOT/b['build_driver_snapshot']['path'])==b['build_driver_snapshot']
        assert {k:b['build_driver_snapshot'][k] for k in ('bytes','sha256')}=={k:b['builder'][k] for k in ('bytes','sha256')}
        assert file(ROOT/b['packaging_repair']['failed_attempt']['path'])==b['packaging_repair']['failed_attempt']
    assert file(SRC/'recipe.json')==b['recipe'] and file(SRC/'probe.rs')==b['wrapper_source'] and file(SRC/'probe_output.rs')==b['host_source'] and file(SRC/'generate.py')==b['oracle_source']
    files=sourcefiles(ART/'crate');assert {k:digest(v) for k,v in files.items()}==b['source_inputs']
    assert files[PROJECTION]==(HERE/'frozen-two-stage-v1/projection.rs').read_bytes()
    assert file(HERE/'frozen-two-stage-v1/projection.rs')==b['production_projection']
    base=load(HERE/'build-c09-first-npu-v1-repro.json');assert file(HERE/'build-c09-first-npu-v1-repro.json')==b['base_build']
    assert b['original_source_inputs']==base['source_inputs']
    assert {k:v for k,v in b['source_inputs'].items() if k not in {'src/lib.rs','src/probe.rs','src/bin/probe_output.rs'}}=={k:v for k,v in base['source_inputs'].items() if k!='src/lib.rs'}
    original=ROOT/'moa-dev/experiments/artifacts'/HERE.name/base['label']/'crate'
    assert files['src/lib.rs']==(original/'src/lib.rs').read_bytes()+b'\npub mod probe;\n'
    assert files['src/probe.rs']==(SRC/'probe.rs').read_bytes() and files['src/bin/probe_output.rs']==(SRC/'probe_output.rs').read_bytes()
    assert {p.name:digest(p.read_bytes()) for p in (ART/'package').iterdir()}==b['package_files']
    image=(ART/'kernels'/b['image']['name']).read_bytes();assert digest(image)=={k:b['image'][k] for k in ('bytes','sha256')};assert (ART/'package/probe_output').read_bytes().count(image)==1
    for rec in b['evidence'].values():assert file(ROOT/rec['path'])==rec
    oracle=load(VALID/'probe-oracle.json');assert digest((ART/'package/probe-fixture.bin').read_bytes())==oracle['fixture']
    return b

def submit():
    b=verify_build()
    for name,rec in b['package_files'].items():assert subprocess.check_output(['docker','exec','moa2026-dev','sha256sum',REMOTE+'/package/'+name],text=True).split()[0]==rec['sha256']
    p=VALID/'probe-submission.json';write(p,{'attempted_at_utc':datetime.now(timezone.utc).isoformat(),'build':file(BUILD)})
    response=cli('submit',*[REMOTE+'/package/'+n for n in b['package_files']],'--entrypoint','run.sh','--timeout',70,'--name','beef-c09-pre-rms-boundary')
    r=load(p);r['response']=response;ids=re.findall(r'submitted job (\d+)',response)
    if len(ids)==1:r['job_id']=int(ids[0])
    p.write_text(json.dumps(r,indent=2)+'\n');assert len(ids)==1,response;print(response,end='')

def parsed(log):
    assert 'panicked' not in log and 'status=FAIL' not in log and 'PROBE_MISMATCH' not in log
    cases=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in log.splitlines() if line.startswith('PROBE_RESULT ')]
    assert len(cases)==8 and [int(c['case']) for c in cases]==list(range(8))
    oracle=load(VALID/'probe-oracle.json')
    for c,o in zip(cases,oracle['cases']):assert c['name']==o['name'] and c['checked']=='3840' and c['failures']==c['nonfinite']=='0' and c['status']=='PASS'
    inputs=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in log.splitlines() if line.startswith('PROBE_INPUT ')]
    assert len(inputs)==24
    assert [(int(i['case']),i['name']) for i in inputs]==[(c,n) for c in range(8) for n in ['x','scale','weight']]
    for i in inputs:assert int(i['checksum'],16)==oracle['cases'][int(i['case'])]['checksums'][i['name']] and i['status']=='PASS'
    assert log.count('PROBE_COMPLETE completed=8 planned=8 failed_values=0 host_only=false status=PASS')==1
    return cases

def collect():
    b=verify_build();job=load(VALID/'probe-submission.json')['job_id'];status=json.loads(cli('status',job))
    assert status['status'] in {'SUCCEEDED','COMPLETED','FAILED','CANCELLED'},status
    log=cli('logs',job);logpath=VALID/f'probe-evidence/arena-{job}.log';logpath.write_text(log)
    report={'valid':False,'job':status,'build':file(BUILD),'raw_log':file(logpath),'scope':'C09 helper + original F32 scale→BF16 before RMS; separate diagnostic image, no performance claim.','official_submission':False}
    try:
        remote=REMOTE+f'/job-{job}.tar';archive=ART/f'job-{job}.tar';cli('download',job,'--out',remote);run(['docker','cp','moa2026-dev:'+remote,str(archive)])
        with tarfile.open(archive) as a:
            members=[m for m in a if m.isfile()];assert len({m.name for m in members})==len(members)
            uploaded={m.name:digest(a.extractfile(m).read()) for m in members}
        assert {k:v for k,v in uploaded.items() if k!='job.yaml'}==b['package_files']
        assert status['status'] in {'SUCCEEDED','COMPLETED'} and status['exit_code']==0
        cases=parsed(log)
        report.update(valid=True,cases=cases,case_count=8,output_elements=30720,exact_bitwise=True,signed_zero_equivalent=True,production_source=b['production_projection'],production_projection_byte_identical=True,production_source_full_file_byte_identical=True,production_compile_source=file(ART/'crate'/PROJECTION),probe_source=b['wrapper_source'],host_source=b['host_source'],oracle_source=b['oracle_source'],oracle=file(VALID/'probe-oracle.json'),archive=file(archive),actual_upload_matches=True,package_files=uploaded,diagnostic_image=b['image'])
    except Exception as e:report['error']=repr(e)
    write(RESULTS,report);write(ROOT/'docs/experiments/e127-c09-probe-correctness.json',report)
    print(json.dumps(report,indent=2));assert report['valid']

def check():
    b=verify_build();r=load(RESULTS);assert r['valid']
    assert r==load(ROOT/'docs/experiments/e127-c09-probe-correctness.json')
    assert r['build']==file(BUILD) and r['oracle']==file(VALID/'probe-oracle.json')
    for key in ['raw_log','archive']:assert file(ROOT/r[key]['path'])==r[key]
    assert parsed((ROOT/r['raw_log']['path']).read_text())==r['cases']
    with tarfile.open(ROOT/r['archive']['path']) as a:
        members=[m for m in a if m.isfile()];assert len({m.name for m in members})==len(members)
        uploaded={m.name:digest(a.extractfile(m).read()) for m in members}
    assert uploaded==r['package_files'] and {k:v for k,v in uploaded.items() if k!='job.yaml'}==b['package_files']
    assert r['production_source']==b['production_projection'] and r['probe_source']==b['wrapper_source']
    assert r['production_source_full_file_byte_identical'] and r['production_compile_source']==file(ART/'crate'/PROJECTION)
    assert {k:r['production_compile_source'][k] for k in ('bytes','sha256')}=={k:r['production_source'][k] for k in ('bytes','sha256')}
    assert r['oracle_source']==b['oracle_source'] and r['host_source']==b['host_source'] and r['diagnostic_image']==b['image']
    assert r['case_count']==8 and r['output_elements']==30720 and r['exact_bitwise'] and r['signed_zero_equivalent']
    print('C09 pre-RMS probe source/package/oracle/all30720 values verified PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','finish_package','submit','collect','verify'],nargs='?');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:check()
    elif a.action=='verify':verify_build();print('Probe build verified PASS')
    elif a.action:globals()[a.action]()
    else:p.error('action or --check required')
