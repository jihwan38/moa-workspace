#!/usr/bin/env python3
"""Replay one official-reference full FFN stress input on unchanged kernel images."""
import argparse, hashlib, io, json, re, subprocess, sys, tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
CAND=HERE.parent
ROOT=CAND.parents[3]
ART=ROOT/'moa-dev/experiments/artifacts/ffn-down-scale-ingress-switch/full-ffn-stress'
REMOTE='/tmp/moa-e151-fullstress-hosts'
TARGET='/tmp/moa-m03-validation-probe/target'
sys.path.insert(0,str(ROOT/'moa-dev'))
from measure import cli

def digest(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def file(path):return {'path':str(path.relative_to(ROOT)),**digest(path.read_bytes())}
def load(path):return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def call(cmd,log=None):
    if log:
        with log.open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    else:subprocess.run(cmd,stdout=subprocess.DEVNULL,check=True)

def build(side,build_record):
    assert re.fullmatch('[a-z0-9-]+',side)
    built=load(build_record);family=build_record.parent.name;label=built['label']
    assert built['exact_compile_exit_code']==0
    sys.path.insert(0,str(CAND));import package as standard
    standard.checked_compile(label);standard.checked_gate(label)
    assert len(built['source_inputs'])==46
    original=ROOT/'moa-dev/experiments/candidates/ffn-native-down-weighted-residual/full-ffn-stress'
    for n in ['full_ffn_stress.rs','expected-bf16.bin','reference.json','reference-build.json']:
        assert (HERE/n).read_bytes()==(original/n).read_bytes()
    origin=ROOT/'moa-dev/experiments/artifacts'/family/label/'crate'
    out=ART/side;out.mkdir(parents=True,exist_ok=False)
    crate=out/'crate';crate.mkdir();evidence=HERE/'evidence'/side;evidence.mkdir(parents=True,exist_ok=False)
    for name,rec in built['source_inputs'].items():
        raw=(origin/name).read_bytes();assert digest(raw)==rec
        p=crate/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    expected=(HERE/'expected-bf16.bin').read_bytes()
    assert digest(expected)=={'bytes':7680,'sha256':'d2930beb2df2274c5b316865d4563e4babe550c1ef930cf2bbcd02cca79f3a74'}
    (crate/'src/bin/expected.bf16.bin').write_bytes(expected)
    (crate/'src/bin/full_ffn_stress.rs').write_bytes((HERE/'full_ffn_stress.rs').read_bytes())
    record={'side':side,'production_build':file(build_record),'production_manifest':built['source_inputs'],
            'host':file(HERE/'full_ffn_stress.rs'),'reference':file(HERE/'reference.json'),
            'reference_build':file(HERE/'reference-build.json'),'expected':file(HERE/'expected-bf16.bin'),
            'builder':file(Path(__file__)),'diagnostic_only':True,'performance_excluded':True,
            'source_kernel_edits':False,'valid':False,'commands':[]}
    def run(cmd,log=None):record['commands'].append(cmd);call(cmd,log)
    remote=REMOTE+'/'+side
    try:
        run(['docker','exec','moa2026-dev','mkdir','-p',remote])
        run(['docker','cp',str(crate)+'/.','moa2026-dev:'+remote])
        names=list(built['source_inputs'])+['src/bin/full_ffn_stress.rs','src/bin/expected.bf16.bin']
        data=subprocess.check_output(['docker','exec','--workdir',remote,'moa2026-dev','tar','-cf','-',*names])
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            assert {m.name:digest(tar.extractfile(m).read()) for m in tar if m.isfile()}=={n:digest((crate/n).read_bytes()) for n in names}
        env=['docker','exec','--workdir',remote,'-e','CARGO_HOME=/home/moa/.cargo','-e','CARGO_NET_OFFLINE=true',
             '-e','CARGO_TARGET_DIR='+TARGET,'-e','FURIOSA_OPT_OUT_DIR='+remote+'/kernels','moa2026-dev']
        print('Build full stress host '+side,flush=True)
        run(env+['cargo','furiosa-opt','build','--release','--locked','--bin','full_ffn_stress'],evidence/'build.log')
        run(['docker','cp','moa2026-dev:'+remote+'/kernels',str(out/'kernels')])
        images=list((out/'kernels').rglob('furiosa_opt_gemma4::ops::decoder_feedforward.bin'));assert len(images)==1
        image=images[0]
        assert digest(image.read_bytes())==built['exact_image'], 'Production FFN image changed'
        host=out/'full_ffn_stress';run(['docker','cp','moa2026-dev:'+TARGET+'/release/full_ffn_stress',str(host)])
        assert host.read_bytes().count(image.read_bytes())==1
        run(['docker','exec','moa2026-dev',TARGET+'/release/full_ffn_stress','--prepare-only'],evidence/'prepare.log')
        assert (evidence/'prepare.log').read_text().count('FULL_STRESS_PREPARED')==1
        record.update(valid=True,image=file(image),executable=file(host),prepare_log=file(evidence/'prepare.log'),
                      kernel_image_byte_identical=True,build_log=file(evidence/'build.log'))
    except Exception as e:record['error']=repr(e);raise
    finally:write(HERE/f'build-{side}.json',record)
    print('PASS unchanged production image '+side,flush=True)

def verified(side):
    path=HERE/f'build-{side}.json';r=load(path);assert r['valid'] and r['kernel_image_byte_identical']
    for key in ['production_build','host','reference','reference_build','expected','builder','image','executable','prepare_log','build_log']:
        assert file(ROOT/r[key]['path'])==r[key]
    b=load(ROOT/r['production_build']['path']);assert b['source_inputs']==r['production_manifest'] and b['exact_image']==digest((ROOT/r['image']['path']).read_bytes())
    crate=ART/side/'crate'
    for n,rec in b['source_inputs'].items():assert digest((crate/n).read_bytes())==rec
    assert (crate/'src/bin/full_ffn_stress.rs').read_bytes()==(HERE/'full_ffn_stress.rs').read_bytes()
    assert (crate/'src/bin/expected.bf16.bin').read_bytes()==(HERE/'expected-bf16.bin').read_bytes()
    original=ROOT/'moa-dev/experiments/candidates/ffn-native-down-weighted-residual/full-ffn-stress'
    for n in ['full_ffn_stress.rs','expected-bf16.bin','reference.json','reference-build.json']:
        assert (HERE/n).read_bytes()==(original/n).read_bytes()
    assert (ROOT/r['executable']['path']).read_bytes().count((ROOT/r['image']['path']).read_bytes())==1
    call([sys.executable,str(original/'verify_reference.py'),'--check'])
    call([sys.executable,str(original/'verify_input_codegen.py'),'--check','--prepare-log',str(ROOT/r['prepare_log']['path'])])
    return r

def submit(label,sides):
    assert re.fullmatch('[a-z0-9-]+',label) and len(set(sides))==len(sides)
    folder=ART/label;folder.mkdir(parents=True,exist_ok=False)
    plans={side:verified(side) for side in sides}
    prep=[(ROOT/p['prepare_log']['path']).read_text() for p in plans.values()];assert len(set(prep))==1
    lines=['#!/bin/sh','set -eu','cd "$(dirname "$0")"']
    for side,rec in plans.items():
        name=side+'_full_ffn_stress';(folder/name).write_bytes((ROOT/rec['executable']['path']).read_bytes())
        lines += ['echo STRESS_STAGE_'+side,'chmod +x '+name,'./'+name]
    lines+=['echo FULL_STRESS_BATCH_COMPLETE'];(folder/'run.sh').write_text('\n'.join(lines)+'\n')
    remote=REMOTE+'/'+label;call(['docker','cp',str(folder),'moa2026-dev:'+remote])
    package={p.name:digest(p.read_bytes()) for p in folder.iterdir()}
    record={'label':label,'sides':sides,'builds':{s:file(HERE/f'build-{s}.json') for s in sides},'package_files':package,
            'input_preparation':prep[0],'diagnostic_only':True,'performance_excluded':True,'official_submission':False}
    path=HERE/(label+'-submission.json');write(path,record)
    response=cli('submit',*[remote+'/'+n for n in package],'--entrypoint','run.sh','--timeout',70,'--name','beef-'+label)
    ids=re.findall(r'submitted job (\d+)',response);record['response']=response
    if len(ids)==1:record['job_id']=int(ids[0])
    path.write_text(json.dumps(record,indent=2)+'\n');assert len(ids)==1,response;print(response,end='')

def collect(label):
    submission=HERE/(label+'-submission.json');submitted=load(submission);job=submitted['job_id'];status=json.loads(cli('status',job))
    assert status['status'] in ['SUCCEEDED','COMPLETED','FAILED','CANCELED','CANCELLED']
    log=cli('logs',job);logpath=HERE/'evidence'/f'arena-{job}.log';logpath.write_text(log)
    r={'submission':file(submission),'job':status,'log':file(logpath),'valid':False,'performance_excluded':True}
    try:
        assert status['status'] in ['SUCCEEDED','COMPLETED'] and status['exit_code']==0
        archive=ART/f'job-{job}.tar';remote=REMOTE+f'/job-{job}.tar';cli('download',job,'--out',remote)
        call(['docker','cp','moa2026-dev:'+remote,str(archive)])
        with tarfile.open(archive) as tar:
            members=[m for m in tar if m.isfile()];assert len(members)==len({m.name for m in members})
            uploaded={m.name:digest(tar.extractfile(m).read()) for m in members}
        assert {k:v for k,v in uploaded.items() if k!='job.yaml'}==submitted['package_files']
        stages={};names=re.findall(r'^STRESS_STAGE_(\S+)$',log,re.M);assert names==submitted['sides']
        for side in names:
            part=log.split('STRESS_STAGE_'+side+'\n',1)[1].split('STRESS_STAGE_',1)[0]
            assert part.count('FULL_STRESS_COMPLETE')==1
            inp=re.findall(r'^FULL_STRESS_INPUT .+$',part,re.M);assert len(inp)==1 and inp[0] in submitted['input_preparation']
            rows=re.findall(r'^FULL_STRESS_RESULT checked=3840 exact=(\d+) mismatches=(\d+) nonfinite=(\d+) max_abs=(\S+) first=(\S+) second=(\S+) expected=(\S+)$',part,re.M);assert len(rows)==1
            exact,bad,nan,delta,first,second,expected=rows[0]
            stages[side]={'checked':3840,'exact':int(exact),'mismatches':int(bad),'nonfinite':int(nan),'max_abs':float(delta),'first':float(first),'second':float(second),'expected':float(expected),'passed':int(bad)==0 and int(nan)==0}
            verified(side)
        assert log.count('FULL_STRESS_BATCH_COMPLETE')==1
        assert all(v['passed'] and v['exact']==3840 for v in stages.values())
        r.update(valid=True,archive=file(archive),uploaded_files=uploaded,actual_upload_matches=True,stages=stages,
                 scope='Actual whole public FFN images on one constructed in-component-range input, with unchanged 0.01+0.01*abs(expected) tolerance and actual official PyTorch reference. Not a PRNG seed or official grading run.')
    except Exception as e:r['error']=repr(e)
    write(HERE/(label+'-results.json'),r);print(json.dumps(r,indent=2));assert r['valid']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','submit','collect','verify']);p.add_argument('label');p.add_argument('rest',nargs='*');a=p.parse_args()
    if a.action=='build':assert len(a.rest)==1;build(a.label,(ROOT/a.rest[0]).resolve())
    elif a.action=='submit':submit(a.label,a.rest)
    elif a.action=='collect':assert not a.rest;collect(a.label)
    else:assert not a.rest;verified(a.label);print('PASS full stress source/image/reference '+a.label)
