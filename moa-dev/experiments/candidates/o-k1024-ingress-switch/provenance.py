#!/usr/bin/env python3
"""Bind the frozen E149 O K1024 ingress/Main Switch source and measured standard images; no adoption."""
import argparse
from pathlib import Path
import subprocess
from package import ARTIFACTS, KERNELS, checked_compile, checked_package, digest, file_record, read, write
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LABEL='e149-ingress-v1'


def function(source,name):
    start=source.index('fn '+name+'(');begin=source.index('{',start);depth=1;end=begin+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]


def outputs():
    compile_path,build,control,_,control_record=checked_compile(LABEL)
    package_path,pkg,package_dir,_=checked_package(LABEL)
    if 'original_wait_attempt' in pkg:
        old_path=ROOT/pkg['original_wait_attempt']['path']
        assert file_record(old_path)==pkg['original_wait_attempt']
        prior=read(old_path)
        assert prior['status']=='failed' and prior['host_build_exit_code']==0
        assert prior['error']=="TimeoutError('Independent graph gate not ready after host build')"
        assert file_record(ROOT/pkg['coordinator']['path'])==pkg['coordinator']
        for key in ('compile_record','source_inputs','control','images','image_files','package_files','logs'):
            assert prior[key]==pkg[key],key
    assert len(build['source_inputs'])==46
    changed=sorted(n for n in build['source_inputs'] if build['source_inputs'][n]!=control['source_inputs'][n])
    assert changed==['src/device/sliding/projection.rs']
    assert build['base_commit']=='fb45202bba3c24537f74b4bf463276b76d823773'
    art=ARTIFACTS/LABEL;variant=HERE/build['variant'];source=variant/'projection.rs'
    recipe_path=variant/'recipe.json';recipe=read(recipe_path)
    assert source.read_bytes()==(art/'crate/src/device/sliding/projection.rs').read_bytes()
    assert recipe['base_commit']==build['base_commit'] and recipe['comparison_recipe']=='baseline/e147.json'
    base=subprocess.check_output(['git','show',build['base_commit']+':furiosa-opt-gemma4-12B/src/device/sliding/projection.rs'],cwd=ROOT)
    unchanged_functions=('prepare_output_terms','apply_output_channel_scale_f32','add_output_residual')
    assert all(function(source.read_text(),name)==function(base.decode(),name) for name in unchanged_functions)
    old_type='DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 120 = 60, Qs % 256]>'
    new_type='DmTensor<f8e4m3, Chip, OutputCluster, m![H / 120 % 16, Qs / 1024, H % 120 = 60 % 4], m![H % 120 = 60 / 4, Qs % 1024]>'
    old_fetch='.fetch::<m![H % 120 = 60, Qs / 32 % 8], m![Qs % 32]>()'
    new_fetch='.fetch::<m![Qs / 256 % 4, H % 120 = 60 / 4], m![Qs % 256]>()\n        .switch::<OutputColumns, m![H % 120 = 60]>(SwitchConfig::InterTranspose { slice1: 4, slice0: 1, time0: 15 })'
    origin=base.decode();assert origin.count(old_type)==3 and origin.count(old_fetch)==1
    assert source.read_text()==origin.replace(old_type,new_type).replace(old_fetch,new_fetch)
    assert all(pkg['images'][k]==control['images'][k] for k in (KERNELS[0],KERNELS[2]))
    measured=dict(build)
    measured.update(record_kind='derived_measured_source_package_binding',original_compile_record=file_record(compile_path),
        original_package_record=file_record(package_path),original_compile_only=build['compile_only'],compile_only=False,
        host_build_exit_code=pkg['host_build_exit_code'],images=pkg['images'],package_files=pkg['package_files'],
        artifact_root=str(art.relative_to(ROOT)),package_path=str(package_dir.relative_to(ROOT)),
        standard_image_files=pkg['image_files'],host_build_commands=pkg['commands'],host_build_logs=pkg['logs'],target_dir=pkg['target_dir'],
        binding_scope='Existing exact build and actual measured standard host; no rebuild, execution or live source adoption.',
        adoption_status='Source/image binding only; final evidence and root adoption decision remain separate.')
    binding={'candidate':'E149','valid':True,'builder':file_record(Path(__file__).resolve()),
        'actual_start_commit':recipe['actual_start_commit'],'source_base_commit':build['base_commit'],
        'performance_control':control_record,'original_compile_record':file_record(compile_path),
        'original_package_record':file_record(package_path),'descriptor_verification':pkg['graph_gate'],
        'recipe':file_record(recipe_path),'source_count':46,'changed_compile_files':changed,'production_source':file_record(source),
        'composition':{'e147_other_compile_inputs_unchanged':45,'e147_qkv_prefix_byte_identical':True,
            'e147_quantizer_and_epilogue_functions_byte_identical':list(unchanged_functions),
            'source_patch':'Only three weight DmTensor type occurrences and one Main fetch/InterTranspose sequence differ from E147. Original stage0/60 producer offsets, native body, quantizer and epilogue remain byte-identical.'},
        'images':pkg['images'],'standard_image_files':pkg['image_files'],
        'qkv_ffn_images_identical_to_e147':True,'standard_host_uses_exact_candidate_image':True,
        'standard_fixture_runner_unchanged':True,'adopted':False,'live_sources_changed':False,'official_submission':False,
        'scope':'Actual source and complete standard image binding only. Finite original-standard, raw9 O and original whole-O case7 numerical checks and root adoption decision are separate; no arbitrary BF16 or end-to-end model claim.'}
    return {HERE/'compiled-source-manifest.json':build['source_inputs'],HERE/'measured-build-e149-adoption.json':measured,HERE/'source-build-binding.json':binding}


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();values=outputs()
    if not a.check:assert all(not path.exists() for path in values)
    for path,value in values.items():
        if a.check:assert read(path)==value,path
        else:write(path,value)
    print('PASS E149 46-file composition, original standard host and exact three images; no live adoption')


if __name__=='__main__':main()
