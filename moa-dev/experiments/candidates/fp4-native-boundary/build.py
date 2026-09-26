#!/usr/bin/env python3
"""Build immutable diagnostic sources; never modify the adopted crate."""
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / "moa-dev/experiments/artifacts/fp4-native-boundary"

def digest(data):
    return {"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}

def run(args, log=None):
    if log:
        with log.open("w") as f: subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
    else: subprocess.run(args,stdout=subprocess.DEVNULL,check=True)

if __name__ == "__main__":
    variant,label=sys.argv[1:3]
    assert re.fullmatch(r"[a-z0-9-]+",label)
    source=HERE/variant
    recipe=json.loads((source/'recipe.json').read_text())
    commit=recipe['base_commit']
    out=ARTIFACTS/label;out.mkdir(parents=True,exist_ok=False)
    crate=out/'crate';crate.mkdir()
    files={}
    for name in ['Cargo.toml','rust-toolchain.toml','src/axes.rs']:
        files[name]=subprocess.check_output(['git','show',commit+':furiosa-opt-gemma4-12B/'+name],cwd=ROOT)
    files['Cargo.lock']=(ROOT/'moa-dev/locks/gemma4-Cargo.lock').read_bytes()
    for dest,name in {'src/lib.rs':'lib.rs','src/probe.rs':'probe.rs','src/bin/boundary_probe.rs':'boundary_probe.rs'}.items():
        files[dest]=(source/name).read_bytes()
    for dest,name in recipe.get('extra_sources',{}).items():
        assert dest not in files
        files[dest]=(source/name).read_bytes()
    for name,data in files.items():
        p=crate/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    remote='/tmp/moa-fp4-boundary/'+label
    run(['docker','exec','moa2026-dev','mkdir','-p',remote])
    run(['docker','cp',str(crate)+'/.','moa2026-dev:'+remote])
    archive=subprocess.check_output(['docker','exec','--workdir',remote,'moa2026-dev','tar','-cf','-',*files])
    with tarfile.open(fileobj=io.BytesIO(archive)) as a:
        assert {m.name:digest(a.extractfile(m).read()) for m in a if m.isfile()}=={k:digest(v) for k,v in files.items()}
    env=['docker','exec','--workdir',remote,'-e','CARGO_NET_OFFLINE=true','-e','CARGO_TARGET_DIR=/tmp/moa-c01-probe/target','-e',f'FURIOSA_OPT_OUT_DIR={remote}/kernels','moa2026-dev']
    record={'variant':variant,'label':label,'recipe':digest((source/'recipe.json').read_bytes()),'base_commit':commit,
            'source_inputs':{k:digest(v) for k,v in files.items()},'container_inputs_verified':True,'diagnostic_only':True}
    evidence=HERE/'evidence'/label;evidence.mkdir(parents=True,exist_ok=False)
    try:
        run(env+['cargo','clean','--release','-p','furiosa-opt-gemma4'])
        print(label+': building three diagnostic images',flush=True)
        run(env+['cargo','furiosa-opt','build','--release','--locked','--bin','boundary_probe'],out/'build.log')
        record['build_exit_code']=0
        package=out/'package';package.mkdir()
        run(['docker','cp','moa2026-dev:/tmp/moa-c01-probe/target/release/boundary_probe',str(package/'boundary_probe')])
        runner=['#!/bin/sh','set -u','cd "$(dirname "$0")"','chmod +x boundary_probe','failed=0']
        for mode in ['decode','materialized','fused']:
            runner += [f'echo BOUNDARY_BEGIN_{mode}',f'./boundary_probe {mode}; result=$?',f'echo BOUNDARY_EXIT_{mode} "$result"','if [ "$result" -ne 0 ]; then failed=1; fi']
        runner += ['echo BOUNDARY_SCREEN_COMPLETE','exit "$failed"']
        (package/'run.sh').write_text('\n'.join(runner)+'\n')
        record['package_files']={p.name:digest(p.read_bytes()) for p in package.iterdir()}
        run(['docker','cp','moa2026-dev:'+remote+'/kernels',str(out/'kernels')])
        host=(package/'boundary_probe').read_bytes()
        record['images']={}
        for p in (out/'kernels').rglob('*.bin'):
            data=p.read_bytes();assert host.count(data)==1
            record['images'][p.name]={**digest(data),'host_occurrences':1}
        assert len(record['images'])==3
        run(['docker','cp',str(package),'moa2026-dev:'+remote+'/package'])
        print('Package ready: '+remote+'/package',flush=True)
    except subprocess.CalledProcessError as e:
        record.update(failed_exit_code=e.returncode,failed_command=e.cmd)
        raise
    finally:
        record['logs']={p.name:digest(p.read_bytes()) for p in out.glob('*.log')}
        for name,data in files.items():
            p=evidence/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        for p in out.glob('*.log'):shutil.copyfile(p,evidence/p.name)
        with (HERE/f'build-{label}.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
