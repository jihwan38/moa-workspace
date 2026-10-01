#!/usr/bin/env python3
"""Bind the frozen E151 FFN down scale ingress/SubSwitch source and measured standard images; no adoption."""
import argparse
from pathlib import Path
import subprocess
from package import ARTIFACTS, KERNELS, checked_compile, checked_package, digest, file_record, read, write
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LABEL='e151-ingress-v1'


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
    assert changed==['src/device/shared/mlp.rs']
    assert build['base_commit']=='ad1ecb3d3868c7e0ca2e4d1428bbd66bf1111821'
    art=ARTIFACTS/LABEL;variant=HERE/build['variant'];source=variant/'mlp.rs'
    recipe_path=variant/'recipe.json';recipe=read(recipe_path)
    assert source.read_bytes()==(art/'crate/src/device/shared/mlp.rs').read_bytes()
    assert recipe['base_commit']==build['base_commit'] and recipe['comparison_recipe']=='baseline/e150.json'
    base=subprocess.check_output(['git','show',build['base_commit']+':furiosa-opt-gemma4-12B/src/device/shared/mlp.rs'],cwd=ROOT)
    unchanged_functions=('prepare_down_fp8_split','prepare_ffn_fp8_split','project_rows_with_scale_contraction','project_down_native_half','finish_down','geglu')
    assert all(function(source.read_text(),name)==function(base.decode(),name) for name in unchanged_functions)
    original=base.decode();candidate=source.read_text()
    a=original.index('fn project_down_tile(');ca=candidate.index('fn project_down_tile(')
    oldbody=function(original,'project_down_tile');newbody=function(candidate,'project_down_tile')
    assert original[:a]==candidate[:ca] and original[a+len(oldbody):]==candidate[ca+len(newbody):]
    old_scale='    let scales: DmTensor<f8e4m3, Chip, UpGateCluster, DownColumns, m![H % 60, TileL / 16 % 120]> = scale.to_dm(&mut device.tdma);'
    new_scale='\n'.join([
        "    // Source slice 4*g+b owns rows 60*g+4*r+b, r<15, with each",
        "    // cluster's full 480-byte scale row. No intermediate DM scatter.",
        '    let scales: DmTensor<f8e4m3, Chip, UpGateCluster, m![H / 60, H % 4], m![H / 4 % 15, TileL / 16]> = scale.to_dm(&mut device.tdma);'])
    old_stream='\n'.join([
        '            .begin(scales.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, TileL / 16 % 120]>(12 * i))',
        '            .fetch::<m![H % 60 = 12], m![TileL / 16 % 120]>()',
        '            .fetch_cast::<f32>()'])
    new_stream='\n'.join([
        '            .begin(scales.view().tile::<m![H / 4 % 15], 3, m![H / 4 % 15 = 3 # 15, TileL / 16]>(3 * i))',
        '            .fetch::<m![TileL / 1920, H / 4 % 15 = 3], m![TileL / 16 % 120]>()',
        '            .fetch_cast::<f32>()',
        '            // Actual InterTranspose order is (row-high3,row-low4).',
        '            // The four old source slices supply 12 consecutive output rows.',
        '            .switch::<DownColumns, m![H % 60 = 12]>(SwitchConfig::InterTranspose { slice1: 4, slice0: 1, time0: 3 })'])
    assert original.count(old_scale)==original.count(old_stream)==candidate.count(new_scale)==candidate.count(new_stream)==1
    assert original.replace(old_scale,new_scale).replace(old_stream,new_stream)==candidate
    assert all(pkg['images'][k]==control['images'][k] for k in KERNELS[:2])
    measured=dict(build)
    measured.update(record_kind='derived_measured_source_package_binding',original_compile_record=file_record(compile_path),
        original_package_record=file_record(package_path),original_compile_only=build['compile_only'],compile_only=False,
        host_build_exit_code=pkg['host_build_exit_code'],images=pkg['images'],package_files=pkg['package_files'],
        artifact_root=str(art.relative_to(ROOT)),package_path=str(package_dir.relative_to(ROOT)),
        standard_image_files=pkg['image_files'],host_build_commands=pkg['commands'],host_build_logs=pkg['logs'],target_dir=pkg['target_dir'],
        binding_scope='Existing exact build and actual measured standard host; no rebuild, execution or live source adoption.',
        adoption_status='Source/image binding only; final evidence and root adoption decision remain separate.')
    binding={'candidate':'E151','valid':True,'builder':file_record(Path(__file__).resolve()),
        'actual_start_commit':recipe['actual_start_commit'],'source_base_commit':build['base_commit'],
        'performance_control':control_record,'original_compile_record':file_record(compile_path),
        'original_package_record':file_record(package_path),'descriptor_verification':pkg['graph_gate'],
        'recipe':file_record(recipe_path),'source_count':46,'changed_compile_files':changed,'production_source':file_record(source),
        'composition':{'e150_other_compile_inputs_unchanged':45,'source_outside_two_scale_supply_regions_identical':True,
            'e150_preserved_functions_byte_identical':list(unchanged_functions),
            'source_patch':'Only scale ingress slice ownership and five Sub fetch/cast/Switch-to-original-VRF streams change. Original two native/weight producers, weighted quantizer, blockscale arithmetic, partials, up/gate and output epilogue remain byte-identical.'},
        'images':pkg['images'],'standard_image_files':pkg['image_files'],
        'qkv_o_images_identical_to_e150':True,'standard_host_uses_exact_candidate_image':True,
        'standard_fixture_runner_unchanged':True,'adopted':False,'live_sources_changed':False,'official_submission':False,
        'scope':'Actual source and complete standard image binding only. Finite original-standard, rawdown9 and full-FFN regression numerical checks and root adoption decision are separate; no arbitrary BF16 or end-to-end model claim.'}
    return {HERE/'compiled-source-manifest.json':build['source_inputs'],HERE/'measured-build-e151-adoption.json':measured,HERE/'source-build-binding.json':binding}


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();values=outputs()
    if not a.check:assert all(not path.exists() for path in values)
    for path,value in values.items():
        if a.check:assert read(path)==value,path
        else:write(path,value)
    print('PASS E151 46-file composition, original standard host and exact three images; no live adoption')


if __name__=='__main__':main()
