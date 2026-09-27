#!/usr/bin/env python3
"""Recheck immutable E134 evidence; check the coupled precision repair without generalizing beyond tested inputs."""
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
    run('verify_graph.py','e134-weighted-v3','--check')
    run('package.py','e134-weighted-v3','--check')
    run('provenance.py','--check')
    run('validation_performance.py','--check')
    run('down-oracle/probe.py','--check')
    run('numerical_bound.py','--check')
    run('full-ffn-stress/verify_results.py','--check')
    e=read(HERE/'validation/extra-results.json')
    ev=validation_extra.verify()
    assert ev==read(HERE/'validation/extra-image-verification.json')
    assert e['valid'] and e['actual_upload_matches'] and e['verification']==file_record(HERE/'validation/extra-image-verification.json')
    for key in ('log','archive'):assert file_record(ROOT/e[key]['path'])==e[key]
    with tarfile.open(ROOT/e['archive']['path']) as tar:
        members=[m for m in tar if m.isfile()]
        assert len({m.name for m in members})==len(members)
        actual={m.name:digest(tar.extractfile(m).read()) for m in members}
    assert actual==e['package_files'] and {k:v for k,v in actual.items() if k!='job.yaml'}==ev['package_files']
    parsed=parse_log((ROOT/e['log']['path']).read_text())
    assert all(e[k]==v for k,v in parsed.items())
    assert e['output_comparisons_passed']==30 and e['run_ids']==list(range(3,9))
    assert read(ROOT/'docs/experiments/e134-extra-correctness.json')==e
    p=read(HERE/'validation/performance-summary.json')
    assert p['valid'] and p['independent_allocations']==3 and p['all_paired_ffn_improved']
    assert p['median_paired_delta_percent']<0 and p['outer_batch_payloads_identical']
    candidate='docs/experiments/e134-adoption-performance.json';control='docs/experiments/e134-e128-paired-control.json'
    c=measurement(candidate,'FFN');r=measurement(control,'FFN');check_comparable(candidate,control)
    assert set(c[1])==set(r[1]) and len(c[1])==3
    probe=read(HERE/'down-oracle/results.json')
    assert probe['valid'] and probe['mapping_cases_exact_bitwise'] and probe['exact_mapping_case_count']==9
    assert probe['exact_mapping_output_elements']==34560 and probe['output_elements']==34560
    assert probe['prior_e133_underflow_case4_exact'] and probe['cpu_construction_down_boundary_case8_exact']
    assert read(ROOT/'docs/experiments/e134-probe-correctness.json') == probe
    stress=read(HERE/'full-ffn-stress/e134-full-ffn-stress-results.json')
    assert stress['valid'] and stress['actual_upload_matches'] and stress['performance_excluded']
    assert stress['stages']['e128-recovered']['exact'] == stress['stages']['e134']['exact'] ==3840
    assert stress['stages']['e128-recovered']['passed'] and stress['stages']['e134']['passed']
    assert not stress['stages']['e133']['passed'] and stress['stages']['e133']['mismatches']==1920
    pending=False
    paths={'performance':ROOT/candidate,'paired_control':ROOT/control,
           'performance_summary':HERE/'validation/performance-summary.json',
           'extra':HERE/'validation/extra-results.json','probe':HERE/'down-oracle/results.json',
           'source_binding':HERE/'source-build-binding.json','measured_build':HERE/'measured-build-e134-adoption.json',
           'manifest':HERE/'compiled-source-manifest.json','graph_gate':HERE/'graph-verification-e134-weighted-v3.json',
           'full_ffn_stress':HERE/'full-ffn-stress/e134-full-ffn-stress-results.json',
           'full_ffn_stress_verification':HERE/'full-ffn-stress/verification.json','numerical_bound':HERE/'numerical-bound.json'}
    return {'valid':True,'candidate':'E134','adoption_ready':not pending,
            'status':'evidence_verified_pending_root_adversarial_resolution' if pending else 'validation_passed_pending_root_adoption',
            'inputdigests':{k:file_record(v) for k,v in paths.items()},'verifier':file_record(Path(__file__)),
            'performance':{'independent_allocations':3,'median_paired_delta_percent':p['median_paired_delta_percent'],
                           'aggregate_median_delta_percent':p['aggregate_median_delta_percent'],'all_paired_ffn_improved':True,
                           'candidate_cycles':c[0],'paired_control_cycles':r[0],'canonical_packages_index_compatible':True},
            'standard_correctness':{'distinct_inputs':9,'base_distinct_inputs':3,'extra_distinct_inputs':6,
                                    'three_allocation_comparisons':135,'extra_output_comparisons_passed':30,
                                    'extra_images_identical_to_standard':True},
            'down_oracle':{'cases':9,'output_elements':34560,'exact_mapping_cases':9,'exact_mapping_outputs':34560,
                           'prior_e133_underflow_case4_exact':True,'cpu_construction_down_boundary_case8_exact':True,
                           'note':'Original BF16 raw-down truth for nine chosen cases; separate wrapper image. All nine outputs exact including previous badcase. Not generalBF16 encoding or fullmodel proof.'},
            'full_ffn_stress':{'job':stress['job']['id'],'constructed_inputs':1,'actual_production_images':True,
                              'e128_exact_outputs':3840,'e133_tolerance_failures':1920,'e134_exact_outputs':3840,
                              'original_public_tolerance_unchanged':True,'actual_official_torch_reference':True,
                              'scope':stress['scope'],'performance_excluded':True},
            'remaining_limits':['Power-of-two weighted E5 terms are not generally exact for all BF16 values.',
                                'CPU normalized-domain bound does not bound arbitrary projected or post-RMS error.',
                                'Nine standard selected inputs, nine raw-down cases and one constructed full-FFN input are finite validation; no end-to-end model claim.'],
            'adoption_owner':'Root: final live source/tag/baseline adoption is outside this evidence verifier.',
            'adopted':False,'live_sources_changed':False,'full_model_tested':False,'official_submission':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');a=ap.parse_args()
    value=verify();path=HERE/'validation/final-verification.json'
    if a.check:assert read(path)==value,'Final verification is stale'
    else:write(path,value)
    print(json.dumps({'valid':value['valid'],'adoption_ready':value['adoption_ready'],'status':value['status']}))


if __name__=='__main__':main()
