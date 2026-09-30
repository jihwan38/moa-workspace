#!/usr/bin/env python3
"""Bind frozen E144 full-H Switch source and measured standard image bytes; no adoption."""
import argparse
from pathlib import Path
import subprocess
import sys
from package import ARTIFACTS, KERNELS, checked_compile, checked_package, digest, file_record, read, write
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LABEL='e144-fullh-v2'


def outputs():
    compile_path,build,control,_,control_record=checked_compile(LABEL)
    package_path,pkg,package_dir,_=checked_package(LABEL)
    art=ARTIFACTS/LABEL
    assert len(build['source_inputs'])==46
    changed=sorted(n for n in build['source_inputs'] if build['source_inputs'][n]!=control['source_inputs'][n])
    assert changed==['src/device/sliding/projection.rs','src/ops.rs']
    recipe_path=HERE/'fullh-v2/recipe.json'; recipe=read(recipe_path)
    origins=recipe['source_origins']
    for name in ('origin_projection','origin_ops','source_projection','source_ops'):
        rec=origins[name]; assert file_record(ROOT/rec['path'])==rec
    source=ROOT/origins['source_projection']['path']
    ops=ROOT/origins['source_ops']['path']
    assert source.read_bytes()==(art/'crate/src/device/sliding/projection.rs').read_bytes()
    assert ops.read_bytes()==(art/'crate/src/ops.rs').read_bytes()==(ROOT/origins['origin_ops']['path']).read_bytes()
    # The independent checker reconstructs E138-v2 by undoing only the two
    # weight-owner types and fetch/Switch chains, then checks exact source equality.
    subprocess.run([sys.executable,str(HERE/'audit.py'),LABEL,'--check'],check=True,stdout=subprocess.DEVNULL)
    audit_path=HERE/f'audit-{LABEL}.json'; audit=read(audit_path)
    assert audit['passed'] and audit['row_quarter_bijection']==32768
    base=subprocess.check_output(['git','show',build['base_commit']+':furiosa-opt-gemma4-12B/src/device/sliding/projection.rs'],cwd=ROOT)
    marker=b'pub(crate) fn prepare_output_terms('
    assert source.read_bytes().split(marker,1)[1]==base.split(marker,1)[1]
    assert all(pkg['images'][k]==control['images'][k] for k in KERNELS[1:])
    assert build['source_inputs']['src/device/shared/mlp.rs']==control['source_inputs']['src/device/shared/mlp.rs']
    measured=dict(build)
    measured.update(record_kind='derived_measured_source_package_binding',original_compile_record=file_record(compile_path),
        original_package_record=file_record(package_path),original_compile_only=build['compile_only'],compile_only=False,
        host_build_exit_code=pkg['host_build_exit_code'],images=pkg['images'],package_files=pkg['package_files'],
        artifact_root=str(art.relative_to(ROOT)),package_path=str(package_dir.relative_to(ROOT)),
        standard_image_files=pkg['image_files'],host_build_commands=pkg['commands'],host_build_logs=pkg['logs'],target_dir=pkg['target_dir'],
        binding_scope='Existing exact build and actual measured standard host; no rebuild, execution or live source adoption.',
        adoption_status='Source/image binding only; final evidence and root adoption decision remain separate.')
    binding={'candidate':'E144','valid':True,'builder':file_record(Path(__file__).resolve()),
        'actual_start_commit':recipe['actual_start_commit'],'source_base_commit':build['base_commit'],
        'performance_control':control_record,'original_compile_record':file_record(compile_path),
        'original_package_record':file_record(package_path),'descriptor_verification':pkg['graph_gate'],
        'independent_source_coordinate_audit':file_record(audit_path),'recipe':file_record(recipe_path),
        'source_count':46,'changed_compile_files':changed,'production_source':file_record(source),
        'production_ops':file_record(ops),'composition':{
            'source_origins':origins,'e142_other_compile_inputs_unchanged':44,
            'only_e138_projection_changes':'Q/KV weight DM row ownership and pre-native fetch/Main Switch; independent source reconstruction equals E138-v2.',
            'e138_ops_byte_identical':True,'e142_output_suffix_byte_identical':True,
            'e142_ffn_compile_input_byte_identical':True,'native_arithmetic_head_cache_preserved_from_e138_v2':True},
        'images':pkg['images'],'standard_image_files':pkg['image_files'],
        'o_ffn_images_identical_to_e142':True,'standard_host_uses_exact_candidate_image':True,
        'standard_fixture_runner_unchanged':True,'adopted':False,'live_sources_changed':False,'official_submission':False,
        'scope':'Actual source composition and complete standard images only. Finite original-standard, raw9 and whole2 numerical checks and root adoption decision are separate; no general BF16 exactness claim.'}
    return {HERE/'compiled-source-manifest.json':build['source_inputs'],HERE/'measured-build-e144-adoption.json':measured,HERE/'source-build-binding.json':binding}


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();values=outputs()
    if not args.check:assert all(not p.exists() for p in values)
    for path,value in values.items():
        if args.check:assert read(path)==value,path
        else:write(path,value)
    print('PASS E144 46-file composition, original standard host and exact three images; no live adoption')


if __name__=='__main__':main()
