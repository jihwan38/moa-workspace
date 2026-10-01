#!/usr/bin/env python3
"""Build/execute/verify the frozen E150 actual-down boundary diagnostic; never performance."""
import argparse, hashlib, io, json, re, shutil, subprocess, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path
SRC=Path(__file__).resolve().parent; HERE=SRC.parent; ROOT=HERE.parents[3]; VALID=SRC
LABEL='e150-down-oracle-v1'; ART=ROOT/'moa-dev/experiments/artifacts'/HERE.name/LABEL
REMOTE='/tmp/moa-e150-down-oracle/'+LABEL; TARGET='/tmp/moa-m03-validation-probe/target'
BUILD=VALID/'build.json'; RESULTS=VALID/'results.json'; EVID=VALID/'evidence'
MLP='src/device/shared/mlp.rs'
sys.path.insert(0,str(ROOT/'moa-dev')); from measure import cli

def digest(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def file(p):return {'path':str(p.relative_to(ROOT)),**digest(p.read_bytes())}
def load(p):return json.loads(p.read_text())
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def run(args,log=None):
    if log:
        with log.open('x') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
    else:subprocess.run(args,stdout=subprocess.DEVNULL,check=True)
def case_lines(log):
    return [s for s in log.splitlines() if s.startswith(('E134_INPUT ','E134_ORACLE '))]
def validate_oracle(log):
    assert 'panicked' not in log
    inputs=re.findall(r'^E134_INPUT case=(\d+) x=([0-9a-f]{16}) count=15360$',log,re.M)
    details=re.findall(r'^E134_ORACLE case=(\d+) rows=3840 width=15360 code_mask=([0-9a-f]+) packed=([0-9a-f]{16}) scales=([0-9a-f]{16}) expected=([0-9a-f]{16})$',log,re.M)
    assert [int(c) for c,_ in inputs]==list(range(9))
    assert [int(row[0]) for row in details]==list(range(9))
    prior=load(ROOT/'moa-dev/experiments/candidates/ffn-native-down-e5/down-oracle/build.json')['oracle']
    assert [list(v) for v in inputs[:8]]==prior['input_checksums']
    assert [list(v) for v in details[:8]]==prior['projection_checksums']
    result={'input_checksums':[list(v) for v in inputs],'projection_checksums':[list(v) for v in details],'case_count':9,'projection_count':9}
    assert result==load(ROOT/'moa-dev/experiments/candidates/ffn-native-down-weighted-residual/down-oracle/build.json')['oracle']
    return result


def verify_original_sources(recipe):
    for name,rec in recipe['original_sources'].items():
        assert file(ROOT/rec['path'])==rec
        assert digest((SRC/name).read_bytes())=={k:rec[k] for k in ('bytes','sha256')}

def build():
    recipe=load(SRC/'recipe.json');verify_original_sources(recipe);base_path=HERE/recipe['source_build'];base=load(base_path)
    sys.path.insert(0,str(HERE));import package as standard
    standard.checked_compile(base['label']);standard.checked_gate(base['label'])
    original=ROOT/'moa-dev/experiments/artifacts'/HERE.name/base['label']/'crate'
    raw=(original/MLP).read_bytes();assert digest(raw)==base['source_inputs'][MLP]
    assert raw==(HERE/recipe['production_source']).read_bytes() and digest(raw)['sha256']==recipe['production_source_sha256']
    append=(SRC/'probe_module.rs').read_bytes()
    host=(SRC/'probe_stream.rs').read_text()
    original_host=(original/'src/bin/test_kernels.rs').read_text()
    compare=original_host[original_host.index('fn compare('):original_host.index('// --- on-device')].strip()
    assert host.count(compare)==1
    assert 'const RTOL: f32 = 1e-2;' in original_host and 'const ATOL: f32 = 0.01;' in host
    assert re.search(r'name: "decoder_feedforward",\s+atol: 0.01,\s+rtol: RTOL,',original_host)
    files={'src/lib.rs':(SRC/'probe_lib.rs').read_bytes(),'src/stream_probe.rs':(SRC/'stream_probe.rs').read_bytes(),
           'src/device/mlp.rs':raw+append,'src/bin/probe_stream.rs':(SRC/'probe_stream.rs').read_bytes()}
    for name in ('src/axes.rs','src/device/layout.rs','Cargo.toml','Cargo.lock','rust-toolchain.toml'):
        files[name]=(original/name).read_bytes();assert digest(files[name])==base['source_inputs'][name]
    ART.mkdir(parents=True,exist_ok=False);crate=ART/'crate';crate.mkdir();EVID.mkdir(exist_ok=False)
    for name,content in files.items():
        for dest in [crate,EVID/'crate']:
            p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
    (EVID/'production-mlp.rs').write_bytes(raw);(EVID/'source-build-at-probe.json').write_bytes(base_path.read_bytes())
    shutil.copyfile(Path(__file__),EVID/'builder-at-compile.py')
    record={'label':LABEL,'diagnostic_only':True,'performance_excluded':True,'official_submission':False,'valid':False,
            'source_build_path':str(base_path.relative_to(ROOT)),'source_build_snapshot':file(EVID/'source-build-at-probe.json'),
            'production_build':file(base_path),'production_whole_exact_image':base['exact_image'],
            'production_whole_source_inputs':base['source_inputs'],
            'source_base_commit':base['base_commit'],'production_source':file(HERE/recipe['production_source']),
            'production_mlp_prefix':digest(raw),'production_source_full_file_byte_identical_prefix':True,
            'diagnostic_append':file(SRC/'probe_module.rs'),'source_inputs':{k:digest(v) for k,v in files.items()},
            'recipe':file(SRC/'recipe.json'),'builder':file(EVID/'builder-at-compile.py'),'target_dir':TARGET,'commands':[],
            'scope':'Actual candidate down preparation/project_down_tile/finish_down from synthetic original BF16 L input; nine required exact mapping/weighted-restoration cases, including the prior underflow case and the range-based CPU construction down boundary. Not public full-FFN acceptance; no proven post-GeGLU reachability.'}
    def recorded(args,log=None):record['commands'].append(args);run(args,log)
    try:
        recorded(['docker','exec','moa2026-dev','mkdir','-p',REMOTE]);recorded(['docker','cp',str(crate)+'/.','moa2026-dev:'+REMOTE])
        archive=subprocess.check_output(['docker','exec','--workdir',REMOTE,'moa2026-dev','tar','-cf','-',*files])
        with tarfile.open(fileobj=io.BytesIO(archive)) as a:
            assert {m.name:digest(a.extractfile(m).read()) for m in a if m.isfile()}==record['source_inputs']
        record['container_source_verified']=True
        env=['docker','exec','--workdir',REMOTE,'-e','CARGO_HOME=/home/moa/.cargo','-e','CARGO_NET_OFFLINE=true','-e','CARGO_TARGET_DIR='+TARGET,'-e','FURIOSA_OPT_OUT_DIR='+REMOTE+'/kernels','moa2026-dev']
        print('Exact diagnostic compile with isolated warm target '+TARGET,flush=True)
        recorded(env+['cargo','furiosa-opt','compile','stream_probe::down','--exact','--dump-schedule','schedule.json','--dump-graph','graph.json'],EVID/'compile.log')
        record['exact_compile_exit_code']=0
        recorded(['docker','cp','moa2026-dev:'+REMOTE+'/kernels',str(ART/'exact-kernels')])
        exact_images=list((ART/'exact-kernels').rglob('*.bin'));assert len(exact_images)==1
        record['exact_image']=file(exact_images[0])
        for n in ['schedule.json','graph.json']:recorded(['docker','cp','moa2026-dev:'+REMOTE+'/'+n,str(EVID/n)])
        print('Diagnostic host build',flush=True)
        recorded(env+['cargo','furiosa-opt','build','--release','--locked','--bin','probe_stream'],EVID/'host-build.log');record['host_build_exit_code']=0
        package=ART/'package';package.mkdir();recorded(['docker','cp','moa2026-dev:'+TARGET+'/release/probe_stream',str(package/'probe_stream')])
        (package/'run.sh').write_text('#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nchmod +x probe_stream\nexec ./probe_stream\n')
        recorded(['docker','cp','moa2026-dev:'+REMOTE+'/kernels',str(ART/'kernels')])
        images=list((ART/'kernels').rglob('*.bin'));assert len(images)==1
        image=images[0].read_bytes();assert (package/'probe_stream').read_bytes().count(image)==1
        assert image==(ROOT/record['exact_image']['path']).read_bytes()
        record['image']={'name':str(images[0].relative_to(ART/'kernels')),**digest(image),'host_occurrences':1}
        record['package_files']={p.name:digest(p.read_bytes()) for p in package.iterdir()}
        recorded(['docker','cp',str(package),'moa2026-dev:'+REMOTE+'/package'])
        recorded(['docker','exec','moa2026-dev',REMOTE+'/package/probe_stream','--oracle-only'],EVID/'oracle-only.log')
        log=(EVID/'oracle-only.log').read_text();record['oracle']=validate_oracle(log)
        assert log.count('E134_PROBE_COMPLETE oracle_only=true cases=9 checked=0 required_failures=0')==1
        record['oracle_exit_code']=0;record['valid']=True
    except Exception as e:record['error']=repr(e);raise
    finally:
        record['evidence']={str(p.relative_to(EVID)):file(p) for p in EVID.rglob('*') if p.is_file()}
        record['probe_sources']={p.name:file(p) for p in SRC.iterdir() if p.is_file() and p.name not in {'build.json','results.json','submission.json'}}
        write(BUILD,record)
    print('Exact source/image and CPU oracle PASS; NPU not submitted',flush=True)

def verify_build():
    b=load(BUILD);assert b['valid'] and b['target_dir']==TARGET
    assert file(SRC/'recipe.json')==b['recipe'];verify_original_sources(load(SRC/'recipe.json'))
    assert digest(Path(__file__).read_bytes())==digest((EVID/'builder-at-compile.py').read_bytes())
    assert file(ROOT/b['source_build_path'])==b['production_build']
    assert digest((ROOT/b['source_build_path']).read_bytes())==digest((EVID/'source-build-at-probe.json').read_bytes())
    assert b['exact_compile_exit_code']==b['host_build_exit_code']==b['oracle_exit_code']==0
    for rec in b['probe_sources'].values():assert file(ROOT/rec['path'])==rec
    for rec in b['evidence'].values():assert file(ROOT/rec['path'])==rec
    source=(ROOT/b['production_source']['path']).read_bytes();assert file(ROOT/b['production_source']['path'])==b['production_source']
    assert digest(source)==b['production_mlp_prefix'] and source==(EVID/'production-mlp.rs').read_bytes()
    assert (ART/'crate/src/device/mlp.rs').read_bytes()==source+(SRC/'probe_module.rs').read_bytes()
    for n,rec in b['source_inputs'].items():assert digest((ART/'crate'/n).read_bytes())==rec
    frozen_base=load(EVID/'source-build-at-probe.json');assert frozen_base['source_inputs'][MLP]==b['production_mlp_prefix']
    assert b['production_whole_source_inputs']==frozen_base['source_inputs'] and len(b['production_whole_source_inputs'])==46
    assert b['production_whole_exact_image']==frozen_base['exact_image']
    assert {p.name:digest(p.read_bytes()) for p in (ART/'package').iterdir()}==b['package_files']
    image=(ART/'kernels'/b['image']['name']).read_bytes();assert digest(image)=={k:b['image'][k] for k in ['bytes','sha256']};assert (ART/'package/probe_stream').read_bytes().count(image)==1
    assert file(ROOT/b['exact_image']['path'])==b['exact_image']
    assert image==(ROOT/b['exact_image']['path']).read_bytes()
    assert digest(image)!=b['production_whole_exact_image']
    assert validate_oracle((EVID/'oracle-only.log').read_text())==b['oracle']
    return b

def submit():
    b=verify_build()
    sys.path.insert(0,str(HERE));import package as standard
    standard.checked_gate(load(EVID/'source-build-at-probe.json')['label'])
    for n,v in b['package_files'].items():assert subprocess.check_output(['docker','exec','moa2026-dev','sha256sum',REMOTE+'/package/'+n],text=True).split()[0]==v['sha256']
    path=VALID/'submission.json';write(path,{'attempted_at_utc':datetime.now(timezone.utc).isoformat(),'build':file(BUILD)})
    response=cli('submit',*[REMOTE+'/package/'+n for n in b['package_files']],'--entrypoint','run.sh','--timeout',70,'--name','beef-e150-staged-down-oracle')
    r=load(path);r['response']=response;ids=re.findall(r'submitted job (\d+)',response)
    if len(ids)==1:r['job_id']=int(ids[0])
    path.write_text(json.dumps(r,indent=2)+'\n');assert len(ids)==1,response;print(response,end='')

def parse(log):
    oracle=validate_oracle(log);assert oracle==load(BUILD)['oracle']
    assert case_lines(log)==case_lines((EVID/'oracle-only.log').read_text())
    details=re.findall(r'^E134_CASE case=(\d+) role=mapping checked=3840 exact_mismatches=(\d+) tolerance_pass=(true|false) max_abs=(\S+) first_actual=(\S+) first_expected=(\S+)$',log,re.M)
    assert [int(v[0]) for v in details]==list(range(9))
    for case,mismatches,tol,*_ in details:assert mismatches=='0' and tol=='true',details
    assert log.count('E134_PROBE_COMPLETE oracle_only=false cases=9 checked=34560 required_failures=0')==1
    assert details[8][-1]=='0.16503906' or float(details[8][-1])==0.1650390625
    return {'details':[list(v) for v in details],'oracle':oracle,'complete':True,'case_count':9,'output_elements':34560,
            'exact_mapping_case_count':9,'exact_mapping_output_elements':34560,
            'prior_e133_underflow_case4_exact':True,'cpu_construction_down_boundary_case8_exact':True,
            'scope':'All nine raw-down mapping/weighted-restoration cases required exact BF16. This is not a complete FFN replay of the CPU construction or a general BF16 correctness proof.'}


def collect():
    b=verify_build();job=load(VALID/'submission.json')['job_id'];status=json.loads(cli('status',job))
    assert status['status'] in {'SUCCEEDED','COMPLETED','FAILED','CANCELLED','CANCELED'},status
    log=cli('logs',job);path=EVID/f'arena-{job}.log';path.write_text(log)
    r={'valid':False,'job':status,'build':file(BUILD),'raw_log':file(path),'scope':b['scope'],'official_submission':False,'performance_excluded':True}
    try:
        archive=ART/f'job-{job}.tar';remote=REMOTE+f'/job-{job}.tar';cli('download',job,'--out',remote);run(['docker','cp','moa2026-dev:'+remote,str(archive)])
        with tarfile.open(archive) as a:
            members=[m for m in a if m.isfile()];assert len({m.name for m in members})==len(members)
            uploaded={m.name:digest(a.extractfile(m).read()) for m in members}
        r.update(archive=file(archive),package_files=uploaded,actual_upload_matches={k:v for k,v in uploaded.items() if k!='job.yaml'}==b['package_files'])
        assert r['actual_upload_matches'] and status['status'] in {'SUCCEEDED','COMPLETED'} and status['exit_code']==0
        r.update(parse(log));r.update(valid=True,mapping_cases_exact_bitwise=True,signed_zero_equivalent=True,production_source=b['production_source'],production_source_full_file_byte_identical_prefix=True,production_compile_source=file(ART/'crate/src/device/mlp.rs'),probe_source=b['diagnostic_append'],diagnostic_image=b['image'])
    except Exception as e:r['error']=repr(e)
    write(RESULTS,r);print(json.dumps(r,indent=2));assert r['valid']

def check():
    b=verify_build();r=load(RESULTS);assert r['valid'] and r['build']==file(BUILD)
    for k in ['raw_log','archive']:assert file(ROOT/r[k]['path'])==r[k]
    parsed=parse((ROOT/r['raw_log']['path']).read_text())
    assert json.loads(json.dumps(parsed))=={k:r[k] for k in parsed}
    with tarfile.open(ROOT/r['archive']['path']) as a:
        members=[m for m in a if m.isfile()];assert len({m.name for m in members})==len(members)
        uploaded={m.name:digest(a.extractfile(m).read()) for m in members}
    assert uploaded==r['package_files'] and {k:v for k,v in uploaded.items() if k!='job.yaml'}==b['package_files']
    assert r['actual_upload_matches'] and r['production_source']==b['production_source'] and r['diagnostic_image']==b['image']
    assert r['production_compile_source']==file(ART/'crate/src/device/mlp.rs') and r['probe_source']==b['diagnostic_append']
    assert r['case_count']==9 and r['output_elements']==34560 and r['mapping_cases_exact_bitwise'] and r['signed_zero_equivalent']
    print('E150 source/oracle/upload and34560 exact outputs verified, including both known underflow regression boundaries')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',choices=['build','verify','submit','collect']);p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:check()
    elif a.action=='verify':verify_build();print('E150 probe build verified PASS')
    elif a.action:globals()[a.action]()
    else:p.error('action or --check required')
