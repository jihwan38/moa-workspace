#!/usr/bin/env python3
"""Bind the frozen E154 FFN pre-RMS shard/quantizer ownership source and measured standard images; no adoption."""
import argparse
from pathlib import Path
import subprocess
import contextlib,importlib.util,io,sys,tempfile
from package import ARTIFACTS, KERNELS, checked_compile, checked_package, digest, file_record, read, write
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LABEL='e154-replicated-v1'


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
    assert changed==['src/device/shared/mlp.rs','src/device/shared/rmsnorm.rs','src/ops.rs']
    assert build['base_commit']=='8adbbae1776aa8080a96348a9d9a64022a0cf6d0'
    art=ARTIFACTS/LABEL;variant=HERE/build['variant'];source=variant/'mlp.rs'
    recipe_path=variant/'recipe.json';recipe=read(recipe_path)
    assert source.read_bytes()==(art/'crate/src/device/shared/mlp.rs').read_bytes()
    assert recipe['base_commit']==build['base_commit'] and recipe['comparison_recipe']=='baseline/e151.json'
    origin153=ROOT/'moa-dev/experiments/candidates/ffn-prerms-sharded-consumer/sharded-v1'
    prior_build=ROOT/'moa-dev/experiments/candidates/ffn-prerms-sharded-consumer/build-e153-sharded-v1-repro.json'
    prior=read(prior_build)
    assert prior['base_commit']==build['base_commit'] and prior['exact_compile_exit_code']==0
    prepare_path=variant/'prepare.py';contract_path=variant/'source-contract.json';contract=read(contract_path)
    assert set(contract['prepared_sources'])=={'ops.rs','rmsnorm.rs','mlp.rs'}
    original153_contract=ROOT/'moa-dev/experiments/candidates/ffn-prerms-sharded-consumer/source-contract.json'
    original153=read(original153_contract)
    for relative in changed:
        name=Path(relative).name
        assert digest((origin153/name).read_bytes())==contract['source_inputs'][name]==prior['source_inputs'][relative]==original153['prepared_sources'][relative]
        assert digest((variant/name).read_bytes())==contract['prepared_sources'][name]==build['source_inputs'][relative]
        assert (variant/name).read_bytes()==(art/'crate'/relative).read_bytes()
    assert all(prior['source_inputs'][n]==control['source_inputs'][n]==build['source_inputs'][n] for n in build['source_inputs'] if n not in changed)
    subprocess.run([sys.executable,str(prepare_path),'--source',str(origin153),'--check'],check=True,stdout=subprocess.DEVNULL)
    unchanged_functions=tuple(original153['unchanged_functions'])
    for name in unchanged_functions:
        assert function(source.read_text(),name)==function((origin153/'mlp.rs').read_text(),name)
    assert all(pkg['images'][k]==control['images'][k] for k in KERNELS[:2])
    measured=dict(build)
    measured.update(record_kind='derived_measured_source_package_binding',original_compile_record=file_record(compile_path),
        original_package_record=file_record(package_path),original_compile_only=build['compile_only'],compile_only=False,
        host_build_exit_code=pkg['host_build_exit_code'],images=pkg['images'],package_files=pkg['package_files'],
        artifact_root=str(art.relative_to(ROOT)),package_path=str(package_dir.relative_to(ROOT)),
        standard_image_files=pkg['image_files'],host_build_commands=pkg['commands'],host_build_logs=pkg['logs'],target_dir=pkg['target_dir'],
        binding_scope='Existing exact build and actual measured standard host; no rebuild, execution or live source adoption.',
        adoption_status='Source/image binding only; final evidence and root adoption decision remain separate.')
    binding={'candidate':'E154','valid':True,'builder':file_record(Path(__file__).resolve()),
        'actual_start_commit':recipe['actual_start_commit'],'source_base_commit':build['base_commit'],
        'performance_control':control_record,'original_compile_record':file_record(compile_path),
        'original_package_record':file_record(package_path),'descriptor_verification':pkg['graph_gate'],
        'recipe':file_record(recipe_path),'source_count':46,'changed_compile_files':changed,'production_source':file_record(source),
        'composition':{'e151_other_compile_inputs_unchanged':43,'frozen_e153_three_file_origin_bound':True,'e154_repair_preparation_reproduced':True,'e153_original_build':file_record(prior_build),'e153_original_contract':file_record(original153_contract),'preparation_verifier':file_record(prepare_path),'source_contract':file_record(contract_path),
            'e151_preserved_functions_byte_identical':list(unchanged_functions),
            'source_patch':'E151 plus frozen E153 three-file ownership source, then only named Max8 replica output, same pow2 on eight live shards/direct Sub scale VRF and equal-replica restore scale relabel. E153 ops/rmsnorm are byte-identical; original localMax/residual Subsum/480Bcopies and downstream projection/GeGLU/E151down/ring functions remain unchanged.'},
        'images':pkg['images'],'standard_image_files':pkg['image_files'],
        'qkv_o_images_identical_to_e151':True,'standard_host_uses_exact_candidate_image':True,
        'standard_fixture_runner_unchanged':True,'adopted':False,'live_sources_changed':False,'official_submission':False,
        'scope':'Actual source and complete standard image binding only. Finite original-standard, rawupgate8 and full-FFN/changed pre-RMS boundary numerical checks and root adoption decision are separate; raw compatibility wrapper covers new quantizer but excludes production pre-RMS; no arbitrary BF16 or end-to-end model claim.'}
    return {HERE/'compiled-source-manifest.json':build['source_inputs'],HERE/'measured-build-e154-adoption.json':measured,HERE/'source-build-binding.json':binding}


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();values=outputs()
    if not a.check:assert all(not path.exists() for path in values)
    for path,value in values.items():
        if a.check:assert read(path)==value,path
        else:write(path,value)
    print('PASS E154 46-file composition, original standard host and exact three images; no live adoption')


if __name__=='__main__':main()
