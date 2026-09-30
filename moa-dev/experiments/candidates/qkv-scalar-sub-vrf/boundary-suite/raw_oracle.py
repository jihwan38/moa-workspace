#!/usr/bin/env python3
"""Build/execute/check the frozen E147 pre-head-RMS correctness boundary diagnostic."""
import argparse, hashlib, io, json, re, shutil, subprocess, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path
VALID=Path(__file__).resolve().parent; HERE=VALID.parent; ROOT=HERE.parents[3]; SRC=VALID/'raw-src'
LABEL='e147-production-qkv-raw-v1'; ART=ROOT/'moa-dev/experiments/artifacts'/HERE.name/LABEL
REMOTE='/tmp/moa-e147-raw/'+LABEL; TARGET=None
PROJECTION='src/device/sliding/projection.rs'; BUILD=VALID/'raw-build-v1.json'; RESULTS=VALID/'raw-results-v1.json'
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

def build(build_label):
    assert TARGET and TARGET.startswith('/tmp/') and TARGET.endswith('/target'), 'explicit --target ownership required'
    assert build_label and build_label.replace('-', '').isalnum()
    from verify_preparation import check_preparation
    assert check_preparation(require_selected=True)['production_build_label'] == build_label
    sys.path.insert(0,str(HERE));from package import checked_gate
    checked_gate(build_label)
    base_path=HERE/f'build-{build_label}-repro.json';base=load(base_path);recipe=load(SRC/'recipe.json')
    assert base['exact_compile_exit_code']==0 and base['source_inputs_verified_in_container']
    production=ROOT/base['source_plan']['source_overrides'][PROJECTION]['path']
    original=ROOT/'moa-dev/experiments/artifacts'/HERE.name/base['label']/'crate'
    files=sourcefiles(original);assert {k:digest(v) for k,v in files.items()}==base['source_inputs']
    assert files[PROJECTION]==production.read_bytes()
    assert digest(files[PROJECTION])==base['source_inputs'][PROJECTION]
    original_files={k:digest(v) for k,v in files.items()}
    files['src/lib.rs']+=b'\npub mod probe;\n'
    files['src/probe.rs']=(SRC/'probe.rs').read_bytes();files['src/bin/probe_qkv_raw.rs']=(SRC/'probe_qkv_raw.rs').read_bytes()
    ART.mkdir(parents=True,exist_ok=False);crate=ART/'crate';crate.mkdir()
    for name,raw in files.items():p=crate/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    evidence=VALID/'raw-evidence-v1';evidence.mkdir(exist_ok=False)
    record={'label':LABEL,'diagnostic_only':True,'performance_excluded':True,'official_submission':False,
            'build_label':build_label,'production_selection':file(VALID/'production-selection.json'),'base_build':file(base_path),'recipe':file(SRC/'recipe.json'),
            'original_source_inputs':original_files,'source_inputs':{k:digest(v) for k,v in files.items()},
            'production_projection':file(production),
            'production_whole_exact_image':base['exact_image'],
            'production_projection_byte_identical':True,'all_original_sources_unchanged_except_lib_append':True,
            'wrapper_source':file(SRC/'probe.rs'),'host_source':file(SRC/'probe_qkv_raw.rs'),'oracle_source':file(SRC/'generate.py'),
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
        original_truth=ROOT/'moa-dev/experiments/candidates/qkv-native-consumer-local/boundary-suite/raw-truth-v1.json'
        assert (ART/'fixture/probe-oracle.json').read_bytes()==original_truth.read_bytes()
        shutil.copyfile(ART/'fixture/probe-oracle.json',VALID/'raw-truth-v1.json')
        env=['docker','exec','--workdir',REMOTE,'-e','CARGO_HOME=/home/moa/.cargo','-e','CARGO_NET_OFFLINE=true','-e','CARGO_TARGET_DIR='+TARGET,'-e','FURIOSA_OPT_OUT_DIR='+REMOTE+'/kernels','moa2026-dev']
        print('Compiling exact pre-head-RMS E147 probe with separate target '+TARGET,flush=True)
        recorded(env+['cargo','furiosa-opt','compile','probe::qkv_before_head_rms','--exact','--dump-schedule','probe-schedule.json','--dump-graph','probe-graph.json'],evidence/'compile.log')
        record['exact_compile_exit_code']=0
        for name in ['probe-schedule.json','probe-graph.json']:
            recorded(['docker','cp','moa2026-dev:'+REMOTE+'/'+name,str(evidence/name)])
        recorded(['docker','cp','moa2026-dev:'+REMOTE+'/kernels',str(ART/'exact-kernels')])
        exact_images=list((ART/'exact-kernels').rglob('*.bin'));assert len(exact_images)==1
        record['exact_diagnostic_image']=file(exact_images[0])
        print('Building pre-head-RMS host',flush=True)
        recorded(env+['cargo','furiosa-opt','build','--release','--locked','--bin','probe_qkv_raw'],evidence/'host-build.log')
        record['host_build_exit_code']=0
        package=ART/'package';package.mkdir()
        recorded(['docker','cp','moa2026-dev:'+TARGET+'/release/probe_qkv_raw',str(package/'probe_qkv_raw')])
        shutil.copyfile(ART/'fixture/probe-fixture.bin',package/'probe-fixture.bin')
        (package/'run.sh').write_text('#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nchmod +x probe_qkv_raw\n./probe_qkv_raw --fixture probe-fixture.bin\n')
        recorded(['docker','cp','moa2026-dev:'+REMOTE+'/kernels',str(ART/'kernels')])
        host=(package/'probe_qkv_raw').read_bytes();images=list((ART/'kernels').rglob('*.bin'));assert len(images)==1
        image=images[0].read_bytes();assert host.count(image)==1
        assert digest(image)=={k:record['exact_diagnostic_image'][k] for k in ('bytes','sha256')}
        record['host_diagnostic_image_matches_exact']=True
        record['image']={'name':str(images[0].relative_to(ART/'kernels')),**digest(image),'host_occurrences':1}
        record['package_files']={p.name:digest(p.read_bytes()) for p in package.iterdir()}
        recorded(['docker','cp',str(package),'moa2026-dev:'+REMOTE+'/package'])
        recorded(['docker','exec','--workdir',REMOTE+'/package','moa2026-dev','./probe_qkv_raw','--fixture','probe-fixture.bin','--host-check-only','--dump-generated',REMOTE+'/dump'],evidence/'host-check.log')
        recorded(['docker','cp','moa2026-dev:'+REMOTE+'/dump',str(ART/'dump')])
        truth=load(VALID/'raw-truth-v1.json');dump_matches={}
        for case in truth['cases']:
            for name,sha in case['input_sha256'].items():
                blob=ART/'dump'/f"case{case['case']}.{name}.bin"
                assert digest(blob.read_bytes())['sha256']==sha
                dump_matches[blob.name]=file(blob)
        assert len(dump_matches)==63
        record['generated_input_dumps']=dump_matches
        hostcheck=(evidence/'host-check.log').read_text();assert hostcheck.count('RAW_INPUT ')==63 and 'failed_values=0 host_only=true status=PASS' in hostcheck
        record['valid']=True
    except Exception as e:
        record['error']=repr(e);raise
    finally:
        record['evidence']={p.name:file(p) for p in evidence.iterdir() if p.is_file()}
        write(BUILD,record)
    print('Probe build and all63 input checksum/SHA256 checks PASS')

def verify_build():
    b=load(BUILD);assert file(VALID/'production-selection.json')==b['production_selection'];assert b['valid'] and b['target_dir'].endswith('/target')
    production=ROOT/b['production_projection']['path'];base_path=ROOT/b['base_build']['path']
    assert file(Path(__file__))==b['builder']
    assert file(SRC/'recipe.json')==b['recipe'] and file(SRC/'probe.rs')==b['wrapper_source'] and file(SRC/'probe_qkv_raw.rs')==b['host_source'] and file(SRC/'generate.py')==b['oracle_source']
    files=sourcefiles(ART/'crate');assert {k:digest(v) for k,v in files.items()}==b['source_inputs']
    assert files[PROJECTION]==production.read_bytes()
    assert file(production)==b['production_projection']
    base=load(base_path);assert file(base_path)==b['base_build']
    assert b['original_source_inputs']==base['source_inputs']
    assert b['production_whole_exact_image']==base['exact_image']
    assert b['host_diagnostic_image_matches_exact'] and file(ROOT/b['exact_diagnostic_image']['path'])==b['exact_diagnostic_image']
    assert {k:b['exact_diagnostic_image'][k] for k in ('bytes','sha256')}=={k:b['image'][k] for k in ('bytes','sha256')}
    assert {k:v for k,v in b['source_inputs'].items() if k not in {'src/lib.rs','src/probe.rs','src/bin/probe_qkv_raw.rs'}}=={k:v for k,v in base['source_inputs'].items() if k!='src/lib.rs'}
    original=ROOT/'moa-dev/experiments/artifacts'/HERE.name/base['label']/'crate'
    assert files['src/lib.rs']==(original/'src/lib.rs').read_bytes()+b'\npub mod probe;\n'
    assert files['src/probe.rs']==(SRC/'probe.rs').read_bytes() and files['src/bin/probe_qkv_raw.rs']==(SRC/'probe_qkv_raw.rs').read_bytes()
    assert {p.name:digest(p.read_bytes()) for p in (ART/'package').iterdir()}==b['package_files']
    image=(ART/'kernels'/b['image']['name']).read_bytes();assert digest(image)=={k:b['image'][k] for k in ('bytes','sha256')};assert (ART/'package/probe_qkv_raw').read_bytes().count(image)==1
    for rec in b['evidence'].values():assert file(ROOT/rec['path'])==rec
    for rec in b['generated_input_dumps'].values():assert file(ROOT/rec['path'])==rec
    oracle=load(VALID/'raw-truth-v1.json');assert digest((ART/'package/probe-fixture.bin').read_bytes())==oracle['fixture']
    return b

def submit():
    b=verify_build()
    sys.path.insert(0,str(HERE));from package import checked_gate
    checked_gate(b['build_label'])
    for name,rec in b['package_files'].items():assert subprocess.check_output(['docker','exec','moa2026-dev','sha256sum',REMOTE+'/package/'+name],text=True).split()[0]==rec['sha256']
    p=VALID/'raw-submission-v1.json';write(p,{'attempted_at_utc':datetime.now(timezone.utc).isoformat(),'build':file(BUILD)})
    response=cli('submit',*[REMOTE+'/package/'+n for n in b['package_files']],'--entrypoint','run.sh','--timeout',70,'--name','beef-e147-qkv-raw-v1')
    r=load(p);r['response']=response;ids=re.findall(r'submitted job (\d+)',response)
    if len(ids)==1:r['job_id']=int(ids[0])
    p.write_text(json.dumps(r,indent=2)+'\n');assert len(ids)==1,response;print(response,end='')

def parsed(log):
    assert 'panicked' not in log and 'status=FAIL' not in log and 'RAW_MISMATCH' not in log
    cases=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in log.splitlines() if line.startswith('RAW_RESULT ')]
    assert len(cases)==27
    assert [(int(c['case']),c['output']) for c in cases]==[(case,key) for case in range(9) for key in ['q','k','v']]
    oracle=load(VALID/'raw-truth-v1.json')
    for c in cases:
        o=oracle['cases'][int(c['case'])]
        assert c['name']==o['name'] and int(c['checked'])==o['outputs'][c['output']]['rows']
        assert c['failures']==c['nonfinite']=='0' and c['status']=='PASS'
    inputs=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in log.splitlines() if line.startswith('RAW_INPUT ')]
    names=['x','q_scale','k_scale','v_scale','q_weight','k_weight','v_weight']
    assert len(inputs)==63 and [(int(i['case']),i['name']) for i in inputs]==[(c,n) for c in range(9) for n in names]
    for i in inputs:assert int(i['checksum'],16)==oracle['cases'][int(i['case'])]['checksums'][i['name']] and i['status']=='PASS'
    assert log.count('RAW_COMPLETE completed=9 planned=9 failed_values=0 host_only=false status=PASS')==1
    return cases

def collect():
    b=verify_build();job=load(VALID/'raw-submission-v1.json')['job_id'];status=json.loads(cli('status',job))
    assert status['status'] in {'SUCCEEDED','COMPLETED','FAILED','CANCELLED'},status
    log=cli('logs',job);logpath=VALID/f'raw-evidence-v1/arena-{job}.log';logpath.write_text(log)
    report={'valid':False,'job':status,'build':file(BUILD),'raw_log':file(logpath),'scope':'E147 production project_qkv_native with actual post-preRMS BF16 input and original F32 channel-scale→BF16 boundary before head RMS; separate diagnostic image, no performance claim.','official_submission':False,'observed_result_lines':[s for s in log.splitlines() if s.startswith('RAW_RESULT ')]}
    try:
        remote=REMOTE+f'/job-{job}.tar';archive=ART/f'job-{job}.tar';cli('download',job,'--out',remote);run(['docker','cp','moa2026-dev:'+remote,str(archive)])
        with tarfile.open(archive) as a:
            members=[m for m in a if m.isfile()];assert len({m.name for m in members})==len(members)
            uploaded={m.name:digest(a.extractfile(m).read()) for m in members}
        report.update(archive=file(archive),package_files=uploaded,actual_upload_matches={k:v for k,v in uploaded.items() if k!='job.yaml'}==b['package_files'])
        assert report['actual_upload_matches']
        assert status['status'] in {'SUCCEEDED','COMPLETED'} and status['exit_code']==0
        cases=parsed(log)
        report.update(valid=True,cases=cases,case_count=9,output_elements=73728,exact_bitwise=True,signed_zero_equivalent=True,production_source=b['production_projection'],production_projection_byte_identical=True,production_source_full_file_byte_identical=True,production_compile_source=file(ART/'crate'/PROJECTION),probe_source=b['wrapper_source'],host_source=b['host_source'],oracle_source=b['oracle_source'],oracle=file(VALID/'raw-truth-v1.json'),archive=file(archive),actual_upload_matches=True,package_files=uploaded,diagnostic_image=b['image'])
    except Exception as e:report['error']=repr(e)
    write(RESULTS,report)
    print(json.dumps({k:report[k] for k in ['valid','job','scope']}));assert report['valid']

def check():
    b=verify_build();r=load(RESULTS);assert r['valid']
    assert r['build']==file(BUILD) and r['oracle']==file(VALID/'raw-truth-v1.json')
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
    assert r['case_count']==9 and r['output_elements']==73728 and r['exact_bitwise'] and r['signed_zero_equivalent']
    print('E147 pre-head-RMS probe source/package/oracle/all73728 values verified PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','submit','collect','verify'],nargs='?');p.add_argument('--check',action='store_true');p.add_argument('--build-label');p.add_argument('--target',default=TARGET);a=p.parse_args()
    if a.action=='build':TARGET=a.target
    if a.check:check()
    elif a.action=='verify':verify_build();print('Probe build verified PASS')
    elif a.action=='build':build(a.build_label)
    elif a.action:globals()[a.action]()
    else:p.error('action or --check required')
