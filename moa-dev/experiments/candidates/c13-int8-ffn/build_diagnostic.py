#!/usr/bin/env python3
"""Build an immutable C13 correctness diagnostic, distinct from competition kernels."""
import argparse, hashlib, importlib.util, io, json, re, shutil, subprocess, tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('c13_diag_common',HERE.parent/'c02-c03-e55/build.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
digest,run=common.digest,common.run

def build(variant,label):
    assert re.fullmatch(r'[a-z0-9-]+',label)
    record_path=HERE/f'build-{label}.json'
    assert not record_path.exists()
    out=ROOT/'moa-dev/experiments/artifacts/c13-int8-ffn'/label
    out.mkdir(parents=True,exist_ok=False)
    crate=out/'crate'
    original=ROOT/'moa-dev/experiments/artifacts/c13-int8-ffn/c13-integer-v6/crate'
    mapping={'src/lib.rs':HERE/variant/'lib.rs','src/integer_probe.rs':HERE/variant/'integer_probe.rs','src/bin/probe.rs':HERE/variant/'probe.rs',**{f:original/f for f in ('Cargo.toml','Cargo.lock','rust-toolchain.toml')}}
    for name,path in mapping.items():
        dst=crate/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
    inputs={name:digest(path.read_bytes()) for name,path in mapping.items()}
    remote='/tmp/moa-c13/'+label
    run(['docker','exec','moa2026-dev','mkdir','-p',remote])
    run(['docker','cp',str(crate)+'/.','moa2026-dev:'+remote])
    raw=subprocess.check_output(['docker','exec','--workdir',remote,'moa2026-dev','tar','-cf','-',*mapping])
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        assert {m.name:digest(t.extractfile(m).read()) for m in t if m.isfile()}==inputs
    env=['docker','exec','--workdir',remote,'-e','CARGO_HOME=/home/moa/.cargo','-e','CARGO_NET_OFFLINE=true','-e','CARGO_TARGET_DIR=/tmp/moa-c01-probe/target','-e',f'FURIOSA_OPT_OUT_DIR={remote}/kernels','moa2026-dev']
    record={'variant':variant,'source_inputs':inputs,'source_inputs_verified_in_container':True,'scope':'Diagnostic images only; not whole-FFN correctness or performance evidence. Quantizer bodies copied from stated source variant.','builder':digest(Path(__file__).read_bytes())}
    try:
        run(env+['cargo','clean','--release','-p','furiosa-opt-gemma4'])
        run(env+['cargo','furiosa-opt','build','--release','--locked','--bin','probe'],out/'build.log')
        record['build_exit_code']=0
        package=out/'package';package.mkdir()
        run(['docker','cp','moa2026-dev:/tmp/moa-c01-probe/target/release/probe',str(package/'probe')])
        (package/'run.sh').write_text('#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nchmod +x probe\nexec ./probe\n')
        for p in package.iterdir():p.chmod(0o755)
        run(['docker','cp','moa2026-dev:'+remote+'/kernels',str(out/'kernels')])
        host=(package/'probe').read_bytes()
        record['images']={}
        for p in (out/'kernels').rglob('*.bin'):
            raw=p.read_bytes();assert host.count(raw)==1
            record['images'][p.name]=dict(digest(raw),host_occurrences=1)
        assert len(record['images'])==3
        record['package_files']={p.name:digest(p.read_bytes()) for p in package.iterdir()}
        run(['docker','cp',str(package),'moa2026-dev:'+remote+'/package'])
        record['container_package']=remote+'/package'
        print(record['container_package'],flush=True)
    except subprocess.CalledProcessError as error:
        record.update(failed_exit_code=error.returncode,failed_command=error.cmd);raise
    finally:
        e=HERE/'evidence'/label;e.mkdir(parents=True,exist_ok=False)
        record['frozen_files']={}
        for name in inputs:
            dst=e/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(crate/name,dst)
            record['frozen_files'][name]=str(dst.relative_to(ROOT))
        record['logs']={}
        for name in ('build.log',):
            if (out/name).exists():
                shutil.copyfile(out/name,e/name);record['logs'][name]=digest((e/name).read_bytes());record['frozen_files'][name]=str((e/name).relative_to(ROOT))
        record_path.write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('variant');p.add_argument('--label',required=True);a=p.parse_args();build(a.variant,a.label)
