#!/usr/bin/env python3
"""Recompute the bounded C09 adoption gates without compiling or running NPU."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
VALID=HERE/'validation'
PERF=ROOT/'docs/experiments/e127-c09-adoption-performance.json'
CONTROL=ROOT/'docs/experiments/e127-c09-e122-paired-control.json'
EXTRA=ROOT/'docs/experiments/e127-c09-extra-correctness.json'
PROBE=ROOT/'docs/experiments/e127-c09-probe-correctness.json'
O='sliding_attention_output'
PROJ='src/device/sliding/projection.rs'
sys.path.insert(0,str(ROOT/'moa-dev'))
from extra_seeds import parse_log as parse_extra
from validation_extra import verify as verify_extra


def read(p): return json.loads(p.read_text())
def digest(raw): return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def file(p): return {'path':str(p.relative_to(ROOT)),**digest(p.read_bytes())}
def checked_file(record):
    p=ROOT/record['path'];assert all(file(p)[k]==record[k] for k in ('bytes','sha256')),str(p)
    return p

def command(script,*args):
    subprocess.run([sys.executable,str(HERE/script),*args],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)

def uploaded(report):
    archive=checked_file(report['archive'])
    with tarfile.open(archive) as tar:
        members=[m for m in tar if m.isfile()]
        assert len(members)==len({m.name for m in members})
        contents={m.name:digest(tar.extractfile(m).read()) for m in members}
    assert contents==report['package_files']
    assert report['actual_upload_matches'] is True


def verify():
    command('verify_descriptors.py','c09-first-npu-v1','--check')
    command('validation_performance.py','report','--check')
    command('validation_probe.py','--check')
    extra_evidence=verify_extra()
    assert extra_evidence==read(VALID/'extra-image-verification.json')
    compile_path=HERE/'build-c09-first-npu-v1-repro.json'
    package_path=HERE/'package-c09-first-npu-v1.json'
    compiled,package=read(compile_path),read(package_path)
    manifest=compiled['source_inputs']
    baseline=read(ROOT/'baseline/e122.json');old=read(ROOT/baseline['compile_inputs'])
    assert set(manifest)==set(old) and len(manifest)==46
    assert [k for k in manifest if manifest[k]!=old[k]]==[PROJ]
    production=HERE/'frozen-two-stage-v1/projection.rs'
    assert digest(production.read_bytes())==manifest[PROJ]
    assert package['compile_record']==file(compile_path)
    assert package['source_inputs']==manifest
    assert package['images'][O]['sha256']==compiled['exact_image']['sha256']=='fe90ed758c27fe40a11e78ff5132e7b98152fc8cd79208d49c22053672859495'
    old_images=read(ROOT/baseline['source_build_audit'])['images']
    assert all(package['images'][k]==old_images[k] for k in ('sliding_project_qkv','decoder_feedforward'))
    crate=ROOT/'furiosa-opt-gemma4-12B'
    for name,expected in manifest.items():
        # Stable before and after adopting the reviewed projection; unrelated
        # live edits cannot be silently replaced during adoption.
        raw=(crate/name).read_bytes()
        assert digest(raw) in ([old[name],expected] if name==PROJ else [expected]),name
    perf,control=read(PERF),read(CONTROL)
    assert perf['valid'] and control['valid']
    assert perf['independent_arena_jobs']==control['independent_arena_jobs']==3
    paired=perf['paired_observations'];assert len(paired)==len({p['job_id'] for p in paired})==3
    assert perf['all_paired_o_improved'] and all(p['delta_percent']<0 for p in paired)
    assert statistics.median(p['delta_percent'] for p in paired)==perf['median_paired_delta_percent']
    assert all(r['comparisons_passed']==15 for r in perf['runs'])
    extra=read(EXTRA);assert extra==read(VALID/'extra-results.json')
    assert extra['valid'] and extra['job']['exit_code']==0 and extra['job']['status']=='SUCCEEDED'
    assert extra['run_ids']==list(range(3,9)) and extra['output_comparisons_passed']==30
    raw=checked_file(extra['log']).read_text();parsed=parse_extra(raw)
    assert all(extra[k]==v for k,v in parsed.items())
    uploaded(extra)
    probe=read(PROBE)
    assert probe['valid'] and probe['case_count']==8 and probe['output_elements']==30720
    assert probe['exact_bitwise'] is True and probe['signed_zero_equivalent'] is True
    prod_path=checked_file(probe['production_source'])
    assert digest(prod_path.read_bytes())==manifest[PROJ]
    assert probe['production_source_full_file_byte_identical'] is True
    compiled_projection=checked_file(probe['production_compile_source'])
    assert digest(compiled_projection.read_bytes())==manifest[PROJ]
    checked_file(probe['raw_log']);uploaded(probe)
    result={
        'valid':True,'candidate':'E127 / frozen C09','source_unchanged_since_E126':True,
        'performance':{'independent_allocations':3,'all_paired_o_improved':True,
            'median_paired_delta_percent':perf['median_paired_delta_percent'],
            'aggregate_median_delta_percent':perf['aggregate_median_delta_percent'],
            'paired_observations':paired,'candidate_standard_comparisons':45,
            'distinct_standard_inputs':3,'report':file(PERF),'control_report':file(CONTROL)},
        'extra':{'valid':True,'distinct_inputs':6,'output_comparisons_passed':30,
            'device_images_identical':True,'report':file(EXTRA),'verification':file(VALID/'extra-image-verification.json')},
        'probe':{'valid':True,'cases':8,'output_elements':30720,'exact_bitwise':True,
            'signed_zero_equivalent':True,'report':file(PROBE)},
        'distinct_standard_inputs':9,'candidate_standard_output_comparisons':75,
        'projection_source':file(production),'compiled_manifest':manifest,
        'compile_record':file(compile_path),'package_record':file(package_path),
        'qkv_ffn_images_unchanged':True,'public_entrypoints_unchanged':True,
        'policy':'Bounded adoption of a reproduced positive gain;5% limits further tuning, not adoption. No new tile/architecture or official submission.',
        'scope':'Original three inputs repeated across3 allocations, six additional standard inputs and eight separate pre-RMS projection cases. Not all-input/full-model equivalence or an official score.',
        'verifiers':{p.name:file(p) for p in [Path(__file__).resolve(),HERE/'validation_performance.py',HERE/'validation_extra.py',HERE/'validation_probe.py']}}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    data=verify();target=VALID/'verification.json'
    if a.check: assert read(target)==data
    else:
        with target.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
    print('E127 adoption gates PASS: 3 allocations,9 distinct standard inputs,8 pre-RMS cases/30720 outputs')
