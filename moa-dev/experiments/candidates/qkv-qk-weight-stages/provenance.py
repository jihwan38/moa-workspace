#!/usr/bin/env python3
"""Bind the frozen E146 Q/K stage source and measured standard images; no adoption."""
import argparse
from pathlib import Path
import subprocess
from package import ARTIFACTS, KERNELS, checked_compile, checked_package, digest, file_record, read, write
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LABEL='e146-qk-half-v4'


def function(source,name):
    start=source.index('fn '+name+'(');begin=source.index('{',start);depth=1;end=begin+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]


def outputs():
    compile_path,build,control,_,control_record=checked_compile(LABEL)
    package_path,pkg,package_dir,_=checked_package(LABEL)
    assert len(build['source_inputs'])==46
    changed=sorted(n for n in build['source_inputs'] if build['source_inputs'][n]!=control['source_inputs'][n])
    assert changed==['src/device/sliding/projection.rs']
    assert build['base_commit']=='3586c417e43f4ea3afc7df7c428d381b7dcbcc5e'
    art=ARTIFACTS/LABEL;variant=HERE/build['variant'];source=variant/'projection.rs'
    recipe_path=variant/'recipe.json';recipe=read(recipe_path)
    assert source.read_bytes()==(art/'crate/src/device/sliding/projection.rs').read_bytes()
    assert recipe['base_commit']==build['base_commit'] and recipe['comparison_recipe']=='baseline/e144.json'
    base=subprocess.check_output(['git','show',build['base_commit']+':furiosa-opt-gemma4-12B/src/device/sliding/projection.rs'],cwd=ROOT)
    marker=b'pub(crate) fn prepare_output_terms('
    assert source.read_bytes().split(marker,1)[1]==base.split(marker,1)[1]
    assert function(source.read_text(),'prepare_qkv_terms')==function(base.decode(),'prepare_qkv_terms')
    assert all(pkg['images'][k]==control['images'][k] for k in KERNELS[1:])
    measured=dict(build)
    measured.update(record_kind='derived_measured_source_package_binding',original_compile_record=file_record(compile_path),
        original_package_record=file_record(package_path),original_compile_only=build['compile_only'],compile_only=False,
        host_build_exit_code=pkg['host_build_exit_code'],images=pkg['images'],package_files=pkg['package_files'],
        artifact_root=str(art.relative_to(ROOT)),package_path=str(package_dir.relative_to(ROOT)),
        standard_image_files=pkg['image_files'],host_build_commands=pkg['commands'],host_build_logs=pkg['logs'],target_dir=pkg['target_dir'],
        binding_scope='Existing exact build and actual measured standard host; no rebuild, execution or live source adoption.',
        adoption_status='Source/image binding only; final evidence and root adoption decision remain separate.')
    binding={'candidate':'E146','valid':True,'builder':file_record(Path(__file__).resolve()),
        'actual_start_commit':recipe['actual_start_commit'],'source_base_commit':build['base_commit'],
        'performance_control':control_record,'original_compile_record':file_record(compile_path),
        'original_package_record':file_record(package_path),'descriptor_verification':pkg['graph_gate'],
        'recipe':file_record(recipe_path),'source_count':46,'changed_compile_files':changed,'production_source':file_record(source),
        'composition':{'e144_other_compile_inputs_unchanged':45,'e144_output_suffix_byte_identical':True,
            'e144_prepare_qkv_terms_byte_identical':True,'qk_stage_memory_layout_and_v_preservation_scope':'New actual graph gate and separate original truth/whole validation; not inferred from this source ledger.'},
        'images':pkg['images'],'standard_image_files':pkg['image_files'],
        'o_ffn_images_identical_to_e144':True,'standard_host_uses_exact_candidate_image':True,
        'standard_fixture_runner_unchanged':True,'adopted':False,'live_sources_changed':False,'official_submission':False,
        'scope':'Actual source and complete standard image binding only. Finite original-standard, raw9 and whole2 numerical checks and root adoption decision are separate; no arbitrary BF16 or end-to-end model claim.'}
    return {HERE/'compiled-source-manifest.json':build['source_inputs'],HERE/'measured-build-e146-adoption.json':measured,HERE/'source-build-binding.json':binding}


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();values=outputs()
    if not a.check:assert all(not path.exists() for path in values)
    for path,value in values.items():
        if a.check:assert read(path)==value,path
        else:write(path,value)
    print('PASS E146 46-file composition, original standard host and exact three images; no live adoption')


if __name__=='__main__':main()
