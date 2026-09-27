#!/usr/bin/env python3
"""Recheck frozen copy-only correctness/adoption evidence; no build, NPU or adoption."""
import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import control_provenance as provenance
import control_validation_performance as performance
import control_validation_extra as extra
import verify_control_copy as copy_gate
from package import digest,file_record,read,write

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
VALID=HERE/'control-adoption'

def verify():
    assert copy_gate.output()==read(HERE/'graph-verification-e140-control-copy-fix.json')
    bound=provenance.outputs()
    for path,value in bound.items():assert read(path)==value,path
    reports,markdown=performance.outputs()
    for path,value in reports.items():assert read(path)==value,path
    assert (VALID/'performance-summary.md').read_text()==markdown
    perf=reports[VALID/'performance-summary.json']
    assert perf['independent_allocations']==3 and perf['candidate_comparisons_passed']==45
    assert perf['all_paired_ffn_improved']
    extra_source=extra.verify()
    assert extra_source==read(VALID/'extra-image-verification.json')
    er_path=VALID/'extra-results.json';er=read(er_path)
    assert er==read(ROOT/'docs/experiments/e140-control-copy-extra-correctness.json')
    assert er['valid'] and er['actual_upload_matches']
    assert er['job']['status'] in ('COMPLETED','SUCCEEDED') and er['job']['exit_code']==0
    assert er['job']['id']==read(VALID/'extra-submission.json')['job_id']
    for key in ('log','verification','archive'):assert file_record(ROOT/er[key]['path'])==er[key],key
    parsed=extra.parse_log((ROOT/er['log']['path']).read_text())
    assert all(er[k]==v for k,v in parsed.items())
    assert parsed['output_comparisons_passed']==30 and parsed['run_ids']==list(range(3,9))
    with tarfile.open(ROOT/er['archive']['path']) as tar:
        members=[m for m in tar if m.isfile()]
        assert len(members)==len({m.name for m in members})
        files={m.name:digest(tar.extractfile(m).read()) for m in members}
    assert files==er['package_files']
    assert {k:v for k,v in files.items() if k!='job.yaml'}==extra_source['package_files']
    stress=HERE/'full-ffn-stress'
    subprocess.run([sys.executable,str(stress/'verify_results.py'),'--check'],check=True,stdout=subprocess.DEVNULL)
    sr=read(stress/'e140-full-ffn-stress-results.json');sv=read(stress/'verification.json')
    assert sr['valid'] and sv['valid'] and sv['new_candidate_pass']
    assert sr['stages']['e140']['exact']==sr['stages']['e140']['checked']==3840
    sb=read(stress/'build-e140.json')
    assert sb['production_build']==file_record(HERE/'build-e140-control-copy-fix-repro.json')
    binding=bound[HERE/'control-source-build-binding.json']
    inputs=[HERE/'graph-verification-e140-control-copy-fix.json',HERE/'independent-control-copy-review.json',
        HERE/'build-e140-control-copy-fix-repro.json',HERE/'package-e140-control-copy-fix.json',
        *bound,*reports,VALID/'performance-summary.md',VALID/'extra-image-verification.json',
        VALID/'extra-submission.json',er_path,ROOT/'docs/experiments/e140-control-copy-extra-correctness.json',
        stress/'verification.json',stress/'e140-full-ffn-stress-results.json',stress/'build-e140.json',
        stress/'verify_results.py',HERE/'control_provenance.py',HERE/'control_validation_performance.py',
        HERE/'control_validation_extra.py',HERE/'verify_control_copy.py']
    for row in perf['paired_observations']:
        inputs += [HERE/f"{row['screen_label']}-job-{row['job_id']}-results.json"]
    return {'candidate':'E140','variant':'control-copy-fix','valid':True,'adoption_ready':True,
        'verifier':file_record(Path(__file__).resolve()),
        'inputdigests':{str(path.relative_to(ROOT)):file_record(path) for path in inputs},
        'source':{'origin':copy_gate.BASE,'count':46,'changed_files':['src/device/shared/mlp.rs'],
            'production_source':binding['production_source'],'exact_three_copy_change_only':True,
            'qkv_o_images_identical_to_e137':True,'all_actual_uploads_verified':True},
        'performance':{'independent_allocations':3,'jobs':[p['job_id'] for p in perf['paired_observations']],
            'paired_delta_percent':[p['delta_percent'] for p in perf['paired_observations']],
            'median_paired_delta_percent':perf['median_paired_delta_percent'],
            'aggregate_median_delta_percent':perf['aggregate_median_delta_percent'],
            'interpretation':'Neutral performance supported for a necessary memory-bound correction; no statistically resolved speedup claim.'},
        'correctness':{'distinct_standard_inputs':9,'standard_input_indices':list(range(9)),
            'candidate_standard_comparisons_passed':75,'candidate_ffn_comparisons_passed':15,
            'repeat_inputs_are_not_new_cases':True,'extra6_job':er['job']['id'],
            'actual_whole_ffn_stress':{'job':sr['job']['id'],'cases':1,'outputs':3840,'exact_bf16':True,
                'same_original_13_inputs_and_torch_reference':True,'performance_excluded':True},
            'raw_down_oracle_inherited':False,
            'scope':'Original three standard cases reused in three allocations plus six different standard cases; one constructed full-FFN precision regression. This does not prove arbitrary-input or end-to-end correctness.'},
        'decision':{'adoption':'Evidence ready for root correctness-repair adoption review; runtime untouched here.',
            'further_tuning':'Stop; no parameter changes or new diagnostic needed for this exact copy fix.',
            'major_research':'Local-down architecture remains unadopted; this repairs the inherited up-prefix copy contract only.'},
        'adopted':False,'live_sources_changed':False,'official_submission':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    value=verify();dest=VALID/'final-verification.json'
    if a.check:assert read(dest)==value
    else:write(dest,value)
    print('PASS E140 copy-only correction:46 source inputs,3 allocations,9 standard inputs,3840 exact whole-FFN regression outputs; root adoption separate')

if __name__=='__main__':main()
