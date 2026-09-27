#!/usr/bin/env python3
"""Bind the measured E133 exact image and standard package; never adopt live code."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LABEL = 'e133-down-e5-v2'
sys.path.insert(0, str(HERE))
from package import ARTIFACTS, KERNELS, checked_package, checked_compile, file_record, read, write


def outputs(check=False):
    subprocess.run([sys.executable,str(HERE/'verify_graph.py'),LABEL,'--check'],check=True,stdout=subprocess.DEVNULL)
    compile_path, build, control, control_dir, control_record = checked_compile(LABEL)
    package_path, packaged, package_dir, _ = checked_package(LABEL)
    root = ARTIFACTS/LABEL
    compat = root/'kernels'
    if not check:
        compat.mkdir(exist_ok=False)
        for kernel in KERNELS:
            name=f'furiosa_opt_gemma4::ops::{kernel}.bin'
            (compat/name).write_bytes((root/'standard-kernels'/name).read_bytes())
    compatibility = {}
    for kernel in KERNELS:
        name=f'furiosa_opt_gemma4::ops::{kernel}.bin'
        assert (compat/name).read_bytes() == (root/'standard-kernels'/name).read_bytes()
        compatibility[kernel]=file_record(compat/name)
    assert len(build['source_inputs'])==46
    assert all(build['source_inputs'][p]==control['source_inputs'][p]
               for p in build['source_inputs'] if p!='src/device/shared/mlp.rs')
    measured=dict(build)
    measured.update(record_kind='derived_measured_source_package_binding',
        original_compile_record=file_record(compile_path), original_package_record=file_record(package_path),
        original_compile_only=build['compile_only'], compile_only=False,
        host_build_exit_code=packaged['host_build_exit_code'], images=packaged['images'],
        package_files=packaged['package_files'], artifact_root=str(root.relative_to(ROOT)),
        package_path=str(package_dir.relative_to(ROOT)), standard_image_files=packaged['image_files'],
        compatibility_image_files=compatibility,
        binding_scope='Digest-bound original exact compile plus separately built original standard host; existing measured bytes only, no new compile/run or adoption. Compatibility kernels are byte-identical copies of standard-kernels.',
        adoption_status='Pending root correctness/provenance review; this binding neither changes live code nor declares adoption.')
    binding={'valid':True,'candidate':'E133','builder':file_record(Path(__file__)),
        'source_origin':'cc82b9ccff22795165faffd3ba9a68cc557da001',
        'performance_control':control_record,'original_compile_record':file_record(compile_path),
        'original_package_record':file_record(package_path),'descriptor_verification':packaged['graph_gate'],
        'source_count':46,'changed_compile_files':['src/device/shared/mlp.rs'],
        'public_device_entrypoints_unchanged':True,'upgate_and_post_reduction_source_unchanged':True,
        'images':packaged['images'],'compatibility_image_files':compatibility,
        'qkv_o_images_identical_to_e128':all(packaged['images'][k]==control['images'][k] for k in KERNELS[:2]),
        'standard_host_uses_exact_candidate_image':True,'standard_fixture_runner_unchanged':True,
        'adopted':False,'live_sources_changed':False,'official_submission':False,
        'scope':'Source/package/descriptor binding only. Performance and accuracy reports are separate; final live/tag/recipe adoption is owned by root.'}
    assert binding['qkv_o_images_identical_to_e128']
    return {HERE/'compiled-source-manifest.json':build['source_inputs'],
            HERE/'measured-build-e133-adoption.json':measured,
            HERE/'source-build-binding.json':binding}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    paths=[HERE/x for x in ('compiled-source-manifest.json','measured-build-e133-adoption.json','source-build-binding.json')]
    if not a.check:assert all(not p.exists() for p in paths),'Do not overwrite frozen provenance'
    for path,value in outputs(a.check).items():
        if a.check:assert read(path)==value,path
        else:write(path,value)
    print('E133 46-file manifest, exact/standard images, canonical package and compatibility copies PASS; no adoption performed')


if __name__=='__main__':main()
