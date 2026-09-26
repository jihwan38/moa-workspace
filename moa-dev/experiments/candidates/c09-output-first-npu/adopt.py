#!/usr/bin/env python3
"""Adopt validated frozen C09 as E127, then bind a recipe to its source tag.

source: verify all preserved evidence, install the exact production source, and
        preserve its manifest/audit plus a derived standard-build binding.
        Commit that checkpoint and tag it baseline-e127 before the next action.
recipe: verify the immutable tag's sources and create E127/current recipes.

Neither action compiles, runs Arena, changes a frozen C09 record, or submits an
entry. Running this script is a live adoption action; authoring it is not.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREFIX = str(HERE.relative_to(ROOT))
CRATE = ROOT/'furiosa-opt-gemma4-12B'
LABEL = 'c09-first-npu-v1'
ARTIFACT = ROOT/'moa-dev/experiments/artifacts'/HERE.name/LABEL
PROJECTION = 'src/device/sliding/projection.rs'
COMPILE = PREFIX+'/build-'+LABEL+'-repro.json'
PACKAGE = PREFIX+'/package-'+LABEL+'.json'
BINDING = PREFIX+'/measured-build-c09-adoption.json'
MANIFEST = PREFIX+'/compiled-source-manifest.json'
AUDIT = PREFIX+'/source-build-audit.json'
VERIFY = PREFIX+'/validation/verification.json'
PERFORMANCE = 'docs/experiments/e127-c09-adoption-performance.json'
CONTROL = 'docs/experiments/e127-c09-e122-paired-control.json'
EXTRA = 'docs/experiments/e127-c09-extra-correctness.json'
PROBE = 'docs/experiments/e127-c09-probe-correctness.json'
TAG = 'baseline-e127'
KERNELS = ('sliding_project_qkv','sliding_attention_output','decoder_feedforward')
O = KERNELS[1]
O_SHA = 'fe90ed758c27fe40a11e78ff5132e7b98152fc8cd79208d49c22053672859495'
SOURCE_SHA = '5692369dd5b8f6ddbeecdfaa9289a9683fae18798b920a71775362f306560fe6'


def read(path):
    return json.loads((ROOT/path).read_text())


def digest(raw):
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def file(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT/path
    return {'path':str(path.relative_to(ROOT)),**digest(path.read_bytes())}


def preserved(path, value):
    """Never silently replace an existing adoption evidence record."""
    target = ROOT/path
    if target.exists():
        assert json.loads(target.read_text()) == value, 'Existing record differs: '+path
    else:
        with target.open('x') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2)
            stream.write('\n')


def verify_gate():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['python3',str(HERE/'validation_verify.py'),'--check'],
                   cwd=ROOT,env=env,check=True)
    checked = read(VERIFY)
    assert checked['valid'] is True
    assert checked['performance']['independent_allocations'] == 3
    assert checked['performance']['all_paired_o_improved'] is True
    assert checked['performance']['median_paired_delta_percent'] < 0
    assert checked['extra']['output_comparisons_passed'] == 30
    assert checked['extra']['distinct_inputs'] == 6
    assert checked['probe']['cases'] > 0 and checked['probe']['output_elements'] > 0
    perf = read(PERFORMANCE)
    assert perf['valid'] is True and perf['independent_arena_jobs'] == 3
    assert perf['all_paired_o_improved'] is True
    assert len(perf['paired_observations']) == 3
    assert len({p['job_id'] for p in perf['paired_observations']}) == 3
    assert all(p['delta_percent'] < 0 for p in perf['paired_observations'])
    assert perf['median_paired_delta_percent'] == checked['performance']['median_paired_delta_percent']
    index_path = ROOT/'moa-dev/experiment_index.py'
    index = types.ModuleType('c09_adoption_index'); index.__file__ = str(index_path)
    exec(compile(index_path.read_text(),str(index_path),'exec'),index.__dict__)
    for report in (PERFORMANCE,CONTROL):
        for kernel in index.KERNELS:
            index.measurement(report,kernel)
    index.check_comparable(PERFORMANCE,CONTROL)
    return checked,perf


def source():
    checked,perf = verify_gate()
    build,package = read(COMPILE),read(PACKAGE)
    control = read('baseline/e122.json')
    control_manifest = read(control['compile_inputs'])
    control_audit = read(control['source_build_audit'])
    manifest = build['source_inputs']
    assert build['label'] == package['label'] == LABEL
    assert build['exact_compile_exit_code'] == package['standard_host_build_exit_code'] == 0
    assert build['compile_only'] is True and package['status'] == 'ready'
    assert build['target_kernel'] == O and build['exact_image']['sha256'] == O_SHA
    assert package['compile_record'] == file(COMPILE)
    assert package['source_inputs'] == manifest
    assert {k:v for k,v in manifest.items() if k != PROJECTION} == {
        k:v for k,v in control_manifest.items() if k != PROJECTION}
    candidate = (HERE/'frozen-two-stage-v1/projection.rs').read_bytes()
    assert digest(candidate) == manifest[PROJECTION] and digest(candidate)['sha256'] == SOURCE_SHA
    assert digest((CRATE/PROJECTION).read_bytes()) in (control_manifest[PROJECTION], manifest[PROJECTION]), \
        'Live projection has unrelated edits; refusing to replace it'
    # Complete inventory plus candidate substitution prevents discarding unrelated
    # live edits while allowing the one reviewed production source replacement.
    assert {str(p.relative_to(CRATE)) for p in (CRATE/'src').rglob('*') if p.is_file()} == {
        k for k in manifest if k.startswith('src/')}
    for name,expected in manifest.items():
        raw = candidate if name == PROJECTION else (CRATE/name).read_bytes()
        assert digest(raw) == expected, 'Live input differs: '+name
        assert digest((ARTIFACT/'crate'/name).read_bytes()) == expected, 'Artifact input differs: '+name
    assert digest((ROOT/'moa-dev/locks/gemma4-Cargo.lock').read_bytes()) == manifest['Cargo.lock']
    assert digest((ARTIFACT/'exact-kernel.bin').read_bytes()) == build['exact_image']
    assert {p.name:digest(p.read_bytes()) for p in (ARTIFACT/'package').iterdir()} == package['package_files']
    host = (ARTIFACT/'package/test_runtime').read_bytes()
    image_records = {}
    for kernel in KERNELS:
        image_path = ARTIFACT/'standard-kernels'/f'furiosa_opt_gemma4::ops::{kernel}.bin'
        raw = image_path.read_bytes(); expected = package['images'][kernel]
        assert digest(raw) == {k:expected[k] for k in ('bytes','sha256')}
        assert host.count(raw) == expected['host_occurrences'] == 1
        if kernel != O:
            assert expected == control_audit['images'][kernel]
        else:
            assert digest(raw) == build['exact_image']
        image_records[kernel] = file(image_path)
    assert digest((ARTIFACT/'package/fixtures.safetensors').read_bytes()) == package['control']['package_files']['fixtures.safetensors']
    assert digest((ARTIFACT/'package/remote_entrypoint.sh').read_bytes()) == package['control']['package_files']['remote_entrypoint.sh']
    # A new compatibility directory lets existing control readers use the same
    # family/label/kernels path without rewriting frozen standard-kernels files.
    compat = ARTIFACT/'kernels'
    expected_names = {f'furiosa_opt_gemma4::ops::{k}.bin' for k in KERNELS}
    if compat.exists():
        assert {p.name for p in compat.iterdir()} == expected_names
    else:
        compat.mkdir()
        for name in sorted(expected_names):
            shutil.copyfile(ARTIFACT/'standard-kernels'/name,compat/name)
    compat_records = {}
    for kernel,rec in image_records.items():
        dest = compat/Path(rec['path']).name
        assert dest.read_bytes() == (ROOT/rec['path']).read_bytes()
        compat_records[kernel] = file(dest)
    binding = copy.deepcopy(build)
    binding.update(record_kind='derived_adopted_standard_build_binding',derived_binding=True,
        original_compile_record=file(COMPILE),original_package_record=file(PACKAGE),
        original_compile_only=build['compile_only'],compile_only=False,
        source_inputs_verified_in_container=package['source_inputs_verified_in_container'],
        host_build_exit_code=package['standard_host_build_exit_code'],
        images=package['images'],package_files=package['package_files'],
        artifact_root=str(ARTIFACT.relative_to(ROOT)),package_path=str((ARTIFACT/'package').relative_to(ROOT)),
        standard_image_files=image_records,compatibility_image_files=compat_records,
        adoption_verification=file(VERIFY),
        binding_scope='Derived binding of existing exact compile and standard host package, not a new compilation or runtime measurement. Original records remain frozen.')
    preserved(BINDING,binding)
    preserved(MANIFEST,manifest)
    audit = {'candidate':'E127','adopted_source_candidate':'C09 frozen-two-stage-v1',
        'source_origin':build['base_commit'],'performance_control':'baseline/e122.json',
        'measured_build':BINDING,'measured_build_digest':file(BINDING),
        'original_compile_record':file(COMPILE),'original_package_record':file(PACKAGE),
        'compile_inputs':len(manifest),'live_compile_inputs_match_frozen_build':True,
        'changed_runtime_files':[PROJECTION],'public_device_entrypoints_unchanged':True,
        'qkv_ffn_images_unchanged':True,'basic_and_extra_images_identical':True,
        'images':package['images'],'compatibility_image_files':compat_records,
        'validation':[PERFORMANCE,CONTROL,EXTRA,PROBE], 'verification':VERIFY,
        'descriptor_verification':PREFIX+'/descriptor-analysis-'+LABEL+'.json',
        'median_paired_o_delta_percent':perf['median_paired_delta_percent'],
        'adoption_policy':'An unchanged implementation with a reproducible positive gain may be adopted after bounded correctness/provenance validation. The5% exploration cutoff is not a minimum adoption gain.',
        'full_model_tested':False,'official_submission':False,
        'scope':'Exact frozen C09 staged sliding O combined with unchanged E122 QKV/M03 FFN. Three ordinary-input allocations, six additional standard inputs and separate projection-boundary diagnostics; no all-input/full-model equivalence claim.'}
    # All validation and possible conflicting record checks precede the live edit.
    if (ROOT/AUDIT).exists():
        assert read(AUDIT) == audit,'Existing source-build audit differs'
    (CRATE/PROJECTION).write_bytes(candidate)
    for name,expected in manifest.items():
        assert digest((CRATE/name).read_bytes()) == expected,name
    preserved(AUDIT,audit)
    print('E127 exact source checkpoint ready. Commit source/evidence and tag baseline-e127 before recipe.')


def recipe():
    verify_gate()
    commit = subprocess.check_output(['git','rev-parse',TAG+'^{commit}'],cwd=ROOT,text=True).strip()
    manifest = read(MANIFEST)
    assert manifest == read(BINDING)['source_inputs']
    for name,expected in manifest.items():
        path = 'moa-dev/locks/gemma4-Cargo.lock' if name == 'Cargo.lock' else 'furiosa-opt-gemma4-12B/'+name
        assert digest(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)) == expected,path
        assert digest((CRATE/name).read_bytes()) == expected,'Live input differs from tagged source: '+name
    data = copy.deepcopy(read('baseline/e122.json'))
    evidence = [PREFIX+'/README.md',PERFORMANCE,EXTRA,PROBE,PREFIX+'/descriptor-analysis-'+LABEL+'.json']
    data.update(id='E127',status='adopted',git={'tag':TAG,'commit':commit},compile_inputs=MANIFEST,
        source_build_audit=AUDIT,performance_report=PERFORMANCE,recent_control_report=CONTROL,
        correctness_evidence=list(dict.fromkeys([EXTRA,PROBE]+data['correctness_evidence'])),
        scope='E127 keeps E122 QKV and cluster-local M03 FFN and adopts frozen C09 staged sliding O: two compact60-row FP8 DM loads, shared BF16 x TRF, stage-local K16 F32 sum and disjoint F32 commits, canonical gather and scale-before-BF16. Public interfaces unchanged. Bounded standard-input and boundary validation; no full-model/all-input or official-score claim. Inherited native FFN input representation remains approximate.')
    for method in data['methods']:
        if method['id'] == 'O1':
            method.update(name='O projection의16-column 소유권과 두60행 weight/dot stage',
                behavior='출력 행을 두 cluster와16 row groups에 나누고 각120행을 두60행 stage로 계산한다. 원본 row-major FP8 HBM을 서로 다른 compact DM에 읽고 같은 BF16 activation TRF를 재사용한다. 각 stage가16개 K256 shard의 F32 합을 구해 공통120행 결과의 서로 다른 절반에 기록한다. 두 stage 완료 뒤 기존 gather로 연결한다.',
                parameters='h=1920*c+120*g+60*stage+t; slice=16*g+q; k=256*q+u. c<2,g<16,q<16,stage<2,t<60,u<256; stage DM15360B/slice; combined weight15728640B.',
                invariant='두 HBM view는 원본120행 pitch와 offset0/60을 유지하지만 목적 DM은 각60×256 compact다. F32 출력[0,240)/[240,480)B를 중복 없이 채우고 K16 합 전후에 새 BF16 경계를 넣지 않는다. 두 DMA·lookup/task와 mutable output handoff 비용을 포함하며 source 순서만으로 실제 중첩을 보장하지 않는다.',
                implementation=[{'path':'furiosa-opt-gemma4-12B/'+PROJECTION,'symbols':['project_output','project_output_stage','OutputCluster','OutputColumns']}],
                evidence=evidence)
        elif method['id'] == 'O2':
            method['behavior']='BF16 attention 결과와 FP8→BF16 weight lookup으로 두 stage의 dot을 수행한다. 각 stage가 완전한 K16 F32 합을 구해 disjoint output에 기록한다. 두 stage 결과를 기존32×480B F32 gather로 H480×8에 모은 뒤 별도 Main에서 channel scale을 곱하고 처음 BF16으로 변환한다.'
            method['implementation'][0]['symbols'] = ['project_output','project_output_stage','apply_output_channel_scale_f32']
            method['evidence'] = list(dict.fromkeys(method['evidence']+evidence))
        elif method['id'] == 'O3':
            method['evidence'] = list(dict.fromkeys(method['evidence']+evidence))
    preserved('baseline/e127.json',data)
    (ROOT/'baseline/current.json').write_text(json.dumps({'recipe':'baseline/e127.json'},indent=2)+'\n')
    print('E127 recipe points to source checkpoint',commit)
    print('Regenerate the index, check --check --check-live, then commit the recipe/records separately.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['source','recipe'])
    args = parser.parse_args()
    source() if args.action == 'source' else recipe()
