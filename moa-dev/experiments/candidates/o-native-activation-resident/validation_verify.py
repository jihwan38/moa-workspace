#!/usr/bin/env python3
"""Recheck finite E137 validation and measured-source provenance without execution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE))
from package import read,write,file_record,digest
import validation_extra
sys.path.insert(0,str(ROOT/'moa-dev'))
from extra_seeds import parse_log
from experiment_index import measurement,check_comparable


def run(script,*args):
    subprocess.run([sys.executable,str(HERE/script),*args],check=True,stdout=subprocess.DEVNULL)


def verify():
    run('verify_graph_v3.py','e137-native-v3','--check')
    run('provenance.py','--check')
    run('validation_performance.py','--check')
    run('validation/oracle_v3.py','--check')
    run('validation/full_stress_v3.py','--check')
    e=read(HERE/'adoption/extra-results.json');ev=validation_extra.verify()
    assert ev==read(HERE/'adoption/extra-image-verification.json')
    assert e['valid'] and e['actual_upload_matches']
    assert e['verification']==file_record(HERE/'adoption/extra-image-verification.json')
    for key in ('log','archive'):assert file_record(ROOT/e[key]['path'])==e[key]
    with tarfile.open(ROOT/e['archive']['path']) as tar:
        members=[m for m in tar if m.isfile()]
        assert len({m.name for m in members})==len(members)
        actual={m.name:digest(tar.extractfile(m).read()) for m in members}
    assert actual==e['package_files'] and {k:v for k,v in actual.items() if k!='job.yaml'}==ev['package_files']
    parsed=parse_log((ROOT/e['log']['path']).read_text())
    assert all(e[k]==v for k,v in parsed.items())
    assert e['output_comparisons_passed']==30 and e['run_ids']==list(range(3,9))
    assert read(ROOT/'docs/experiments/e137-extra-correctness.json')==e
    p=read(HERE/'adoption/performance-summary.json')
    assert p['valid'] and p['independent_allocations']==3 and p['all_paired_o_improved']
    assert p['median_paired_delta_percent']<0 and p['outer_batch_payloads_identical']
    candidate='docs/experiments/e137-adoption-performance.json';control='docs/experiments/e137-e134-paired-control.json'
    c=measurement(candidate,'O');r=measurement(control,'O');check_comparable(candidate,control)
    assert set(c[1])==set(r[1]) and len(c[1])==3
    probe=read(HERE/'validation/oracle-results-v3.json')
    assert probe['valid'] and probe['actual_upload_matches'] and probe['exact_bitwise']
    assert probe['case_count']==9 and probe['output_elements']==34560
    assert probe['production_source_full_file_byte_identical']
    stress=read(HERE/'validation/stress-results-v3.json')
    assert stress['valid'] and stress['actual_upload_matches'] and stress['production_source_inputs_match']
    assert stress['case_count']==1 and stress['output_elements']==3840
    assert stress['cases'][0]['exact_f32']=='3840' and stress['cases'][0]['failures']=='0'
    binding=read(HERE/'source-build-binding.json')
    assert binding['qkv_ffn_images_identical_to_e134'] and binding['source_count']==46
    paths={'performance':ROOT/candidate,'paired_control':ROOT/control,
        'performance_summary':HERE/'adoption/performance-summary.json','extra':HERE/'adoption/extra-results.json',
        'source_binding':HERE/'source-build-binding.json','measured_build':HERE/'measured-build-e137-adoption.json',
        'manifest':HERE/'compiled-source-manifest.json','graph_gate':HERE/'graph-verification-e137-native-v3.json',
        'projection_oracle':HERE/'validation/oracle-results-v3.json','full_o_stress':HERE/'validation/stress-results-v3.json',
        'accuracy_summary':HERE/'validation/verification-summary-v3.json'}
    return {'valid':True,'candidate':'E137','adoption_ready':True,'status':'validation_passed_pending_root_adoption',
        'verifier':file_record(Path(__file__)),'inputdigests':{k:file_record(v) for k,v in paths.items()},
        'performance_allocations':3,'standard_input_cases':9,
        'performance':{'independent_allocations':3,'median_paired_delta_percent':p['median_paired_delta_percent'],
            'aggregate_median_delta_percent':p['aggregate_median_delta_percent'],'all_paired_o_improved':True,
            'candidate_cycles':c[0],'paired_control_cycles':r[0],'canonical_packages_index_compatible':True},
        'standard_correctness':{'distinct_inputs':9,'base_distinct_inputs':3,'extra_distinct_inputs':6,
            'three_allocation_comparisons':135,'candidate_three_allocation_comparisons':45,
            'extra_output_comparisons_passed':30,'extra_images_identical_to_standard':True},
        'projection_oracle':{'cases':9,'outputs':34560,'exact_bitwise':True,'signed_zero_equivalent':True,
            'actual_production_quantizer':True,'before_rms':True,'job':probe['job']['id']},
        'full_o_stress':{'cases':1,'outputs':3840,'exact_bitwise':True,'actual_public_image':True,
            'job':stress['job']['id'],'original_reference_and_tolerance_unchanged':True,'performance_excluded':True},
        'unchanged_images':{'qkv':True,'ffn':True},
        'remaining_limits':['Four weighted E4 terms are not an exact encoding of every BF16 value.',
            'Pointwise normalized error bounds do not prove arbitrary cancellation, projected or post-RMS error bounds.',
            'Nine standard inputs, nine selected raw projection cases and one full O construction are finite validation; no full-model claim.'],
        'adoption_owner':'Root owns live source/tag/baseline adoption after reviewing this evidence.',
        'adopted':False,'live_sources_changed':False,'full_model_tested':False,'official_submission':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');a=ap.parse_args()
    value=verify();path=HERE/'validation/final-verification.json'
    if a.check:assert read(path)==value,'Final verification is stale'
    else:write(path,value)
    print(json.dumps({k:value[k] for k in ('valid','candidate','adoption_ready','status','performance')}))


if __name__=='__main__':main()
