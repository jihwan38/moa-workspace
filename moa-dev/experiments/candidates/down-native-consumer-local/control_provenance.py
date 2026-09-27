#!/usr/bin/env python3
"""Bind the measured E137 copy-only correction; no build, NPU or live adoption."""
import argparse
from pathlib import Path
from package import ARTIFACTS, KERNELS, checked_compile, checked_package, digest, file_record, read, write
import verify_control_copy as gate

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LABEL='e140-control-copy-fix'

def outputs():
    compile_path,build,control,_,control_record=checked_compile(LABEL)
    package_path,packaged,package_dir,_=checked_package(LABEL)
    root=ARTIFACTS/LABEL
    assert len(build['source_inputs'])==46
    changed=[p for p in build['source_inputs'] if build['source_inputs'][p]!=control['source_inputs'][p]]
    assert changed==['src/device/shared/mlp.rs']
    source=HERE/'control-copy-fix/mlp.rs'
    assert source.read_bytes()==gate.expected_source()==(root/'crate/src/device/shared/mlp.rs').read_bytes()
    assert gate.output()==read(HERE/f'graph-verification-{LABEL}.json')
    assert all(packaged['images'][k]==control['images'][k] for k in KERNELS[:2])
    measured=dict(build)
    measured.update(record_kind='derived_measured_source_package_binding',
        original_compile_record=file_record(compile_path),original_package_record=file_record(package_path),
        original_compile_only=build['compile_only'],compile_only=False,
        host_build_exit_code=packaged['host_build_exit_code'],images=packaged['images'],
        package_files=packaged['package_files'],artifact_root=str(root.relative_to(ROOT)),
        package_path=str(package_dir.relative_to(ROOT)),standard_image_files=packaged['image_files'],
        host_build_commands=packaged['commands'],host_build_logs=packaged['logs'],target_dir=packaged['target_dir'],
        binding_scope='Original exact compile and measured standard host; no rebuild, run or source adoption.',
        adoption_status='Evidence binding for root review; root owns final source/tag/baseline adoption.')
    binding={'valid':True,'candidate':'E140','variant':'control-copy-fix','builder':file_record(Path(__file__).resolve()),
        'source_origin':gate.BASE,'actual_start_commit':'e35ea350a805bae11570032a62d150252a98d494',
        'performance_control':control_record,'original_compile_record':file_record(compile_path),
        'original_package_record':file_record(package_path),'descriptor_verification':packaged['graph_gate'],
        'source_count':46,'changed_compile_files':changed,'production_source':file_record(source),
        'literal_source_reconstruction_verified':True,
        'source_change':'Only three up-quantizer padded DM copies replaced with exact480B Sub streams; every other E137 byte, including the down path, unchanged.',
        'images':packaged['images'],'standard_image_files':packaged['image_files'],
        'qkv_o_images_identical_to_e137':True,'standard_host_uses_exact_candidate_image':True,
        'standard_fixture_runner_unchanged':True,'adopted':False,'live_sources_changed':False,'official_submission':False,
        'scope':'Binds measured source/package bytes; independent adoption validation and root decision remain separate.'}
    return {HERE/'compiled-copy-source-manifest.json':build['source_inputs'],
            HERE/'measured-build-e140-copy-adoption.json':measured,HERE/'control-source-build-binding.json':binding}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    values=outputs()
    if not a.check:assert all(not p.exists() for p in values),'Do not overwrite frozen provenance'
    for path,value in values.items():
        if a.check:assert read(path)==value,path
        else:write(path,value)
    print('E140 copy-only exact standard image and46-file source binding PASS; no adoption performed')

if __name__=='__main__':main()
