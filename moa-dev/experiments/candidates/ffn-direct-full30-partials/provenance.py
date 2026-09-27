#!/usr/bin/env python3
"""Bind measured E142 source composition and standard image bytes; no build/adoption."""
import argparse,re,subprocess
from pathlib import Path
from package import ARTIFACTS,KERNELS,checked_compile,checked_package,digest,file_record,read,write
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];LABEL='e142-full30-v1'

def span(source,symbol):
    start=source.index('fn '+symbol+'(');begin=source.index('{',start);depth=1;end=begin+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return start,end

def extract(source,symbol):
    a,b=span(source,symbol);return source[a:b]

def outputs():
    compile_path,build,control,_,control_record=checked_compile(LABEL)
    package_path,pkg,package_dir,_=checked_package(LABEL)
    art=ARTIFACTS/LABEL
    assert len(build['source_inputs'])==46
    changed=[n for n in build['source_inputs'] if build['source_inputs'][n]!=control['source_inputs'][n]]
    assert changed==['src/device/shared/mlp.rs']
    source=HERE/'full30-v1/mlp.rs';candidate=source.read_text()
    assert source.read_bytes()==(art/'crate/src/device/shared/mlp.rs').read_bytes()
    recipe=read(HERE/'full30-v1/recipe.json')
    parent=ROOT/recipe['mlp_source_origin']['path'];original=parent.read_text()
    assert digest(parent.read_bytes())=={k:recipe['mlp_source_origin'][k] for k in ('bytes','sha256')}
    symbol='project_rows_with_scale_contraction';a,b=span(original,symbol)
    assert original[:a]+extract(candidate,symbol)+original[b:]==candidate
    helper=ROOT/recipe['helper_origin']['path'];assert digest(helper.read_bytes())['sha256']==recipe['helper_origin']['sha256']
    def executable(src):return re.sub(r'\s+','',re.sub(r'//[^\n]*','',extract(src,symbol)))
    assert executable(candidate)==executable(helper.read_text())
    assert all(pkg['images'][k]==control['images'][k] for k in KERNELS[:2])
    base=subprocess.check_output(['git','show',build['base_commit']+':furiosa-opt-gemma4-12B/src/device/shared/mlp.rs'],cwd=ROOT,text=True)
    preserved={}
    for fn in ('prepare_down_fp8_split','project_down_tile','finish_down','geglu','cast_gathered_rows'):
        text=extract(candidate,fn);assert text==extract(base,fn),fn;preserved[fn]=digest(text.encode())
    measured=dict(build)
    measured.update(record_kind='derived_measured_source_package_binding',original_compile_record=file_record(compile_path),
        original_package_record=file_record(package_path),original_compile_only=build['compile_only'],compile_only=False,
        host_build_exit_code=pkg['host_build_exit_code'],images=pkg['images'],package_files=pkg['package_files'],
        artifact_root=str(art.relative_to(ROOT)),package_path=str(package_dir.relative_to(ROOT)),
        standard_image_files=pkg['image_files'],host_build_commands=pkg['commands'],host_build_logs=pkg['logs'],target_dir=pkg['target_dir'],
        binding_scope='Existing exact build and actual measured standard host; no rebuild, execution or live source adoption.',
        adoption_status='Source/image binding only; final evidence and root adoption decision remain separate.')
    binding={'candidate':'E142','valid':True,'builder':file_record(Path(__file__).resolve()),
        'actual_start_commit':recipe['actual_start_commit'],'source_base_commit':build['base_commit'],
        'performance_control':control_record,'original_compile_record':file_record(compile_path),
        'original_package_record':file_record(package_path),'descriptor_verification':pkg['graph_gate'],
        'source_count':46,'changed_compile_files':changed,'production_source':file_record(source),
        'composition':{'e141_origin':file_record(parent),'e103_helper_origin':file_record(helper),
            'only_changed_symbol':symbol,'outside_helper_byte_identical_to_e141':True,
            'helper_executable_tokens_identical_to_e103':True,'helper_comments_corrected':True,
            'e140_preserved_functions':preserved},
        'images':pkg['images'],'standard_image_files':pkg['image_files'],
        'qkv_o_images_identical_to_e140':True,'standard_host_uses_exact_candidate_image':True,
        'standard_fixture_runner_unchanged':True,'adopted':False,'live_sources_changed':False,'official_submission':False,
        'scope':'Actual source composition verified independently of numerical correctness. Historical E103 correctness is not inherited; E142 actual8case production oracle and whole checks are separate.'}
    return {HERE/'compiled-source-manifest.json':build['source_inputs'],HERE/'measured-build-e142-adoption.json':measured,HERE/'source-build-binding.json':binding}

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();values=outputs()
    if not a.check:assert all(not p.exists() for p in values)
    for path,value in values.items():
        if a.check:assert read(path)==value,path
        else:write(path,value)
    print('PASS E14246-file composition, original standard host and exact3images; no live adoption')

if __name__=='__main__':main()
