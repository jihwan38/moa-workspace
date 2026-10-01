#!/usr/bin/env python3
"""Bind frozen E155 direct packed routing to actual E154 source/images; no adoption."""
import argparse,subprocess,sys
from pathlib import Path
from package import ARTIFACTS,KERNELS,checked_compile,checked_package,digest,file_record,read,write
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];LABEL='e155-direct-v1'
def outputs():
    cp,b,control,_,cr=checked_compile(LABEL);pp,p,pd,_=checked_package(LABEL)
    assert len(b['source_inputs'])==46
    changed=sorted(k for k,v in b['source_inputs'].items() if v!=control['source_inputs'][k])
    assert changed==['src/device/shared/mlp.rs'] and b['base_commit']=='3f384063f5231e00ad286f365209806e5043fc83'
    v=HERE/b['variant'];recipe=read(v/'recipe.json');contract=read(v/'source-contract.json')
    origin=ROOT/'moa-dev/experiments/candidates/ffn-sharded-scale-replicated/replicated-v1'
    names={'ops.rs':'src/ops.rs','rmsnorm.rs':'src/device/shared/rmsnorm.rs','mlp.rs':'src/device/shared/mlp.rs'}
    assert set(contract['source_inputs'])==set(contract['prepared_sources'])==set(names)
    for name,relative in names.items():
        raw=(origin/name).read_bytes()
        assert digest(raw)==contract['source_inputs'][name]==control['source_inputs'][relative]
        assert raw==subprocess.check_output(['git','show',b['base_commit']+':furiosa-opt-gemma4-12B/'+relative],cwd=ROOT)
        assert digest((v/name).read_bytes())==contract['prepared_sources'][name]==b['source_inputs'][relative]
        assert (v/name).read_bytes()==(ARTIFACTS/LABEL/'crate'/relative).read_bytes()
    subprocess.run([sys.executable,str(v/'prepare.py'),'--source',str(origin),'--check'],check=True,stdout=subprocess.DEVNULL)
    assert all(p['images'][k]==control['images'][k] for k in KERNELS[:2])
    measured=dict(b)
    measured.update(record_kind='derived_measured_source_package_binding',original_compile_record=file_record(cp),original_package_record=file_record(pp),original_compile_only=b['compile_only'],compile_only=False,host_build_exit_code=p['host_build_exit_code'],images=p['images'],package_files=p['package_files'],artifact_root=str((ARTIFACTS/LABEL).relative_to(ROOT)),package_path=str(pd.relative_to(ROOT)),standard_image_files=p['image_files'],host_build_commands=p['commands'],host_build_logs=p['logs'],target_dir=p['target_dir'],binding_scope='Existing exact source and actual standard host only; no rebuilding or live adoption.',adoption_status='Final evidence/root adoption decision is separate.')
    binding={'candidate':'E155','valid':True,'builder':file_record(Path(__file__)),'actual_start_commit':recipe['actual_start_commit'],'source_base_commit':b['base_commit'],'performance_control':cr,'original_compile_record':file_record(cp),'original_package_record':file_record(pp),'descriptor_verification':p['graph_gate'],'recipe':file_record(v/'recipe.json'),'source_count':46,'changed_compile_files':changed,'production_source':file_record(v/'mlp.rs'),'composition':{'e154_other_compile_inputs_unchanged':45,'frozen_e154_origin_matches_git':True,'exact_preparation_reproduced':True,'preparation_verifier':file_record(v/'prepare.py'),'source_contract':file_record(v/'source-contract.json'),'source_patch':'Only private broadcast helper CL0 source ownership and sole caller removing late CL2 packed intermediate. Inverse-source proof retains every other byte, arithmetic/native/down and destination ring geometry.'},'images':p['images'],'standard_image_files':p['image_files'],'qkv_o_images_identical_to_e154':True,'standard_host_uses_exact_candidate_image':True,'standard_fixture_runner_unchanged':True,'adopted':False,'live_sources_changed':False,'official_submission':False,'scope':'Source/image binding only. Standard9, rawupgate8 actual helper and fullFFN regression are separate finite evidence; no generic BF16/fullmodel claim. Raw upgate excludes production preRMS.'}
    return {HERE/'compiled-source-manifest.json':b['source_inputs'],HERE/'measured-build-e155-adoption.json':measured,HERE/'source-build-binding.json':binding}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');a=ap.parse_args();values=outputs()
    if not a.check:assert all(not p.exists() for p in values)
    for p,v in values.items():
        if a.check:assert read(p)==v,p
        else:write(p,v)
    print('PASS E155 source46/oneMLP/45unchanged/currentE154/images, no adoption')
