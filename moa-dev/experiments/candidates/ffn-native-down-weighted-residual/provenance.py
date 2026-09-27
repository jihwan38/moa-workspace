#!/usr/bin/env python3
"""Bind the measured E134 exact image and standard package; never adopt live code."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LABEL = 'e134-weighted-v3'
sys.path.insert(0, str(HERE))
from package import ARTIFACTS, KERNELS, checked_package, checked_compile, digest, file_record, read, write


def unchanged_functions(candidate, control):
    def extract(source, symbol):
        start = source.index('fn ' + symbol + '(')
        begin = source.index('{', start)
        depth = 1
        end = begin + 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        return source[start:end].encode()
    symbols = ('project_up_and_gate', 'project_rows_with_scale_contraction',
               'cast_gathered_rows', 'geglu', 'finish_down', 'feedforward',
               'feedforward_sharded', 'prepare_ffn_fp8_split', 'broadcast_ffn_quantized')
    verified = {}
    for symbol in symbols:
        raw = extract(candidate, symbol)
        assert raw == extract(control, symbol), symbol
        verified[symbol] = digest(raw)
    suffix = '    let mut down: DmTensor<f32, Chip, UpGateCluster, DownRows, m![H % 60]>'
    end = '\nfn finish_down('
    a = candidate[candidate.index(suffix):candidate.index(end)]
    b = control[control.index(suffix):control.index(end)]
    assert a == b, 'five block-scale/column-reduction streams and gather changed'
    verified['down_five_scale_reductions_and_gather'] = digest(a.encode())
    return verified


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
    source_path = root/'crate/src/device/shared/mlp.rs'
    origin = build['source_plan']['source_base_commit']
    control_source = subprocess.check_output(['git','show',origin +
        ':furiosa-opt-gemma4-12B/src/device/shared/mlp.rs'],cwd=ROOT).decode()
    preserved = unchanged_functions(source_path.read_text(), control_source)
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
    binding={'valid':True,'candidate':'E134','builder':file_record(Path(__file__)),
        'source_origin':'cc82b9ccff22795165faffd3ba9a68cc557da001',
        'performance_control':control_record,'original_compile_record':file_record(compile_path),
        'original_package_record':file_record(package_path),'descriptor_verification':packaged['graph_gate'],
        'source_count':46,'changed_compile_files':['src/device/shared/mlp.rs'],
        'public_device_entrypoints_unchanged':True,'upgate_and_post_reduction_source_unchanged':True,
        'unchanged_source_sections':preserved,
        'images':packaged['images'],'compatibility_image_files':compatibility,
        'qkv_o_images_identical_to_e128':all(packaged['images'][k]==control['images'][k] for k in KERNELS[:2]),
        'standard_host_uses_exact_candidate_image':True,'standard_fixture_runner_unchanged':True,
        'adopted':False,'live_sources_changed':False,'official_submission':False,
        'scope':'Source/package/descriptor binding only. Performance and accuracy reports are separate; final live/tag/recipe adoption is owned by root.'}
    assert binding['qkv_o_images_identical_to_e128']
    return {HERE/'compiled-source-manifest.json':build['source_inputs'],
            HERE/'measured-build-e134-adoption.json':measured,
            HERE/'source-build-binding.json':binding}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    paths=[HERE/x for x in ('compiled-source-manifest.json','measured-build-e134-adoption.json','source-build-binding.json')]
    if not a.check:assert all(not p.exists() for p in paths),'Do not overwrite frozen provenance'
    for path,value in outputs(a.check).items():
        if a.check:assert read(path)==value,path
        else:write(path,value)
    print('E134 46-file manifest, exact/standard images, canonical package and compatibility copies PASS; no adoption performed')


if __name__=='__main__':main()
