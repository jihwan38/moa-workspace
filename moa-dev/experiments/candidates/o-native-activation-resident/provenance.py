#!/usr/bin/env python3
"""Bind existing measured E137 bytes; never build, run, or adopt live code."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LABEL='e137-native-v3'
sys.path.insert(0,str(HERE))
from package import ARTIFACTS,KERNELS,checked_package,checked_compile,digest,file_record,read,write


def extract(source,symbol):
    start=source.index('fn '+symbol+'(');begin=source.index('{',start);depth=1;end=begin+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end].encode()


def outputs():
    compile_path,build,control,_,control_record=checked_compile(LABEL)
    package_path,packaged,package_dir,_=checked_package(LABEL)
    root=ARTIFACTS/LABEL
    assert len(build['source_inputs'])==46
    changed=[p for p in build['source_inputs'] if build['source_inputs'][p]!=control['source_inputs'][p]]
    assert changed==['src/device/sliding/projection.rs']
    source=root/'crate/src/device/sliding/projection.rs'
    origin=build['source_plan']['source_base_commit']
    original=subprocess.check_output(['git','show',origin+':furiosa-opt-gemma4-12B/src/device/sliding/projection.rs'],cwd=ROOT).decode()
    candidate=source.read_text();preserved={}
    for symbol in ('project_query','project_one_kv_matrix','project_key_value','add_output_residual'):
        raw=extract(candidate,symbol);assert raw==extract(original,symbol),symbol;preserved[symbol]=digest(raw)
    assert source.read_bytes()==(HERE/'native-v3/projection.rs').read_bytes()
    unchanged={k:packaged['images'][k]==control['images'][k] for k in (KERNELS[0],KERNELS[2])}
    assert all(unchanged.values())
    measured=dict(build)
    measured.update(record_kind='derived_measured_source_package_binding',
        original_compile_record=file_record(compile_path),original_package_record=file_record(package_path),
        original_compile_only=build['compile_only'],compile_only=False,
        host_build_exit_code=packaged['host_build_exit_code'],images=packaged['images'],
        package_files=packaged['package_files'],artifact_root=str(root.relative_to(ROOT)),
        package_path=str(package_dir.relative_to(ROOT)),standard_image_files=packaged['image_files'],
        host_build_commands=packaged['commands'],host_build_logs=packaged['logs'],target_dir=packaged['target_dir'],
        binding_scope='Original exact compile and standard host, actual measured files; no rebuild, run or source adoption.',
        adoption_status='Validation passed; root owns final live source/tag/baseline adoption.')
    binding={'valid':True,'candidate':'E137','builder':file_record(Path(__file__)),
        'source_origin':origin,'actual_start_commit':'5c94b51dfd4e6149e6680c93b428da7577c0edeb',
        'performance_control':control_record,'original_compile_record':file_record(compile_path),
        'original_package_record':file_record(package_path),'descriptor_verification':packaged['graph_gate'],
        'source_count':46,'changed_compile_files':changed,'production_source':file_record(HERE/'native-v3/projection.rs'),
        'public_device_entrypoints_unchanged':True,'unchanged_source_sections':preserved,
        'images':packaged['images'],'standard_image_files':packaged['image_files'],
        'qkv_ffn_images_identical_to_e134':all(unchanged.values()),
        'standard_host_uses_exact_candidate_image':True,'standard_fixture_runner_unchanged':True,
        'adopted':False,'live_sources_changed':False,'official_submission':False,
        'scope':'Source/package binding only. Runtime source is unchanged until root adoption; performance and accuracy records are separate.'}
    return {HERE/'compiled-source-manifest.json':build['source_inputs'],
        HERE/'measured-build-e137-adoption.json':measured,HERE/'source-build-binding.json':binding}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    values=outputs()
    if not a.check:assert all(not p.exists() for p in values),'Do not overwrite frozen provenance'
    for path,value in values.items():
        if a.check:assert read(path)==value,path
        else:write(path,value)
    print('E137 exact standard image/package and 46-file source binding PASS; no adoption performed')


if __name__=='__main__':main()
