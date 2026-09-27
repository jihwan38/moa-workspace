#!/usr/bin/env python3
"""One parent-authorized exact-only compile of unchanged weighted-v1 down helper."""
import hashlib,io,json,shutil,subprocess,tarfile
from pathlib import Path
SRC=Path(__file__).resolve().parent;HERE=SRC.parent;ROOT=HERE.parents[3]
LABEL='e134-weighted-v1-down-feasibility'
ART=ROOT/'moa-dev/experiments/artifacts'/HERE.name/LABEL
EVID=SRC/'feasibility';REMOTE='/tmp/moa-e134-down-feasibility';TARGET='/tmp/moa-m03-validation-probe/target'
def dg(b):return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def fr(p):return {'path':str(p.relative_to(ROOT)),**dg(p.read_bytes())}
def read(p):return json.loads(p.read_text())
def execute(cmd):subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
def main():
    assert not EVID.exists() and not ART.exists()
    recipe=read(SRC/'recipe.json');basepath=HERE/recipe['source_build'];base=read(basepath)
    origin=ROOT/'moa-dev/experiments/artifacts'/HERE.name/base['label']/'crate'
    actual={str(p.relative_to(origin)):dg(p.read_bytes()) for p in origin.rglob('*') if p.is_file() and (p.is_relative_to(origin/'src') or p.name in {'Cargo.toml','Cargo.lock','rust-toolchain.toml'})}
    assert actual==base['source_inputs']
    source=HERE/recipe['production_source'];raw=source.read_bytes()
    assert dg(raw)==base['source_inputs']['src/device/shared/mlp.rs'] and dg(raw)['sha256']==recipe['production_source_sha256']
    assert raw==(origin/'src/device/shared/mlp.rs').read_bytes()
    files={'src/lib.rs':(SRC/'probe_lib.rs').read_bytes(),'src/stream_probe.rs':(SRC/'stream_probe.rs').read_bytes(),'src/device/mlp.rs':raw+(SRC/'probe_module.rs').read_bytes(),'src/bin/probe_stream.rs':(SRC/'probe_stream.rs').read_bytes()}
    for n in ['src/axes.rs','src/device/layout.rs','Cargo.toml','Cargo.lock','rust-toolchain.toml']:
        files[n]=(origin/n).read_bytes();assert dg(files[n])==base['source_inputs'][n]
    EVID.mkdir();ART.mkdir(parents=True);crate=ART/'crate';crate.mkdir()
    for n,b in files.items():
        for folder in [crate,EVID/'crate']:
            q=folder/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
    record={'label':LABEL,'diagnostic_only':True,'host_built':False,'npu_submitted':False,'whole_graph_gate_required':True,'production_source':fr(source),'source_build':fr(basepath),'recipe':fr(SRC/'recipe.json'),'source_inputs':{n:dg(b) for n,b in files.items()},'target_dir':TARGET,'commands':[],'source_full_file_byte_identical_prefix':True,'scope':'Exact-only standalone feasibility; does not waive failed whole-kernel graph/LIR gate or establish arithmetic/performance.'}
    shutil.copyfile(__file__,EVID/'builder-at-compile.py')
    try:
        execute(['docker','exec','moa2026-dev','mkdir','-p',REMOTE]);execute(['docker','cp',str(crate)+'/.','moa2026-dev:'+REMOTE])
        data=subprocess.check_output(['docker','exec','--workdir',REMOTE,'moa2026-dev','tar','-cf','-',*files])
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:assert {m.name:dg(tar.extractfile(m).read()) for m in tar if m.isfile()}==record['source_inputs']
        record['container_sources_verified']=True
        cmd=['docker','exec','--workdir',REMOTE,'-e','CARGO_HOME=/home/moa/.cargo','-e','CARGO_NET_OFFLINE=true','-e','CARGO_TARGET_DIR='+TARGET,'-e','FURIOSA_OPT_OUT_DIR='+REMOTE+'/kernels','moa2026-dev','cargo','furiosa-opt','compile','stream_probe::down','--exact','--dump-schedule','schedule.json','--dump-graph','graph.json']
        record['commands'].append(cmd)
        with (EVID/'compile.log').open('x') as log: result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        record['exact_compile_exit_code']=result.returncode
        for n in ['schedule.json','graph.json']:
            if subprocess.run(['docker','exec','moa2026-dev','test','-f',REMOTE+'/'+n],stdout=subprocess.DEVNULL).returncode==0:execute(['docker','cp','moa2026-dev:'+REMOTE+'/'+n,str(EVID/n)])
        if result.returncode==0:
            execute(['docker','cp','moa2026-dev:'+REMOTE+'/kernels',str(ART/'kernels')]);bins=list((ART/'kernels').rglob('*.bin'));assert len(bins)==1;record['exact_image']=fr(bins[0])
        record['status']='compiled_exact_only' if result.returncode==0 else 'exact_compile_failed'
    except Exception as error:record['error']=repr(error);raise
    finally:
        record['evidence']={str(p.relative_to(EVID)):fr(p) for p in EVID.rglob('*') if p.is_file()}
        with (EVID/'record.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps({k:record[k] for k in ['label','status','exact_compile_exit_code','host_built','npu_submitted']},indent=2))
if __name__=='__main__':main()
