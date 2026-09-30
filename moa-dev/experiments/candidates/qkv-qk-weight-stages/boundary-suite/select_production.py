#!/usr/bin/env python3
"""After parent selection, freeze one successful E146 build for both boundary suites.

No compile, NPU, or submission. Refuses to overwrite an existing selection.
"""
import argparse,json,sys
from pathlib import Path
from verify_preparation import ROOT,HERE,record,check_preparation
CANDIDATE=HERE.parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build-label',required=True);a=p.parse_args()
    assert a.build_label.replace('-','').isalnum()
    assert not (HERE/'production-selection.json').exists(), 'selection is immutable; preserve prior attempts'
    assert not check_preparation()['production_build_selected']
    bp=CANDIDATE/f'build-{a.build_label}-repro.json';b=json.loads(bp.read_text())
    assert b['label']==a.build_label and b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container']
    sys.path.insert(0,str(CANDIDATE));from package import checked_gate
    checked_gate(a.build_label)
    artifact=ROOT/'moa-dev/experiments/artifacts'/CANDIDATE.name/a.build_label
    crate=artifact/'crate';files={str(p.relative_to(crate)):p for p in crate.rglob('*') if p.is_file() and (p.is_relative_to(crate/'src') or p.name in {'Cargo.toml','Cargo.lock','rust-toolchain.toml'})}
    assert len(files)==46 and {k:{n:record(v)[n] for n in ('bytes','sha256')} for k,v in files.items()}==b['source_inputs']
    production=ROOT/b['source_plan']['source_overrides']['src/device/sliding/projection.rs']['path']
    assert production.read_bytes()==files['src/device/sliding/projection.rs'].read_bytes()
    image=artifact/'exact-kernel.bin';assert {k:record(image)[k] for k in ('bytes','sha256')}==b['exact_image']
    gate=CANDIDATE/f"graph-verification-{a.build_label}.json"
    result={'status':'SELECTED_FOR_BOUNDARY_VALIDATION','label':a.build_label,'build':record(bp),'projection':record(production),'ops':record(files['src/ops.rs']),'exact_image':b['exact_image'],'exact_image_file':record(image),'gate':record(gate),'selector':record(Path(__file__)),'source_inputs':b['source_inputs'],'scope':'Binding only. No diagnostic build or NPU result; diagnostic image must be compiled from these 46 unchanged production inputs plus the frozen wrapper. Full2 must embed exactly this whole image.'}
    with (HERE/'production-selection.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(check_preparation(require_selected=True),indent=2))
if __name__=='__main__':main()
