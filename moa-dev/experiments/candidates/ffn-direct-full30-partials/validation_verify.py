#!/usr/bin/env python3
"""Recheck E142 adoption evidence without build, NPU execution or live adoption."""
import argparse,importlib.util,subprocess,sys,tarfile
from pathlib import Path
import provenance
import validation_performance as performance
import validation_extra as extra
from package import digest,file_record,read,write
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];VALID=HERE/'validation';ADOPT=HERE/'adoption'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def verify():
    subprocess.run([sys.executable,str(HERE/'independent-audit.py'),'e142-full30-v1','--check'],check=True,stdout=subprocess.DEVNULL)
    bound=provenance.outputs()
    for path,value in bound.items():assert read(path)==value,path
    reports,markdown=performance.outputs()
    for path,value in reports.items():assert read(path)==value,path
    assert (ADOPT/'performance-summary.md').read_text()==markdown
    perf=reports[ADOPT/'performance-summary.json']
    assert perf['independent_allocations']==3 and perf['candidate_comparisons_passed']==45
    assert perf['all_paired_ffn_improved'] and perf['median_paired_delta_percent']<0 and perf['aggregate_median_delta_percent']<0
    extra_source=extra.verify();assert extra_source==read(ADOPT/'extra-image-verification.json')
    er_path=ADOPT/'extra-results.json';er=read(er_path)
    assert er==read(ROOT/'docs/experiments/e142-extra-correctness.json')
    assert er['valid'] and er['actual_upload_matches']
    assert er['job']['status'] in ('COMPLETED','SUCCEEDED') and er['job']['exit_code']==0
    assert er['job']['id']==read(ADOPT/'extra-submission.json')['job_id']
    for key in ('log','verification','archive'):assert file_record(ROOT/er[key]['path'])==er[key],key
    parsed=extra.parse_log((ROOT/er['log']['path']).read_text());assert all(er[k]==v for k,v in parsed.items())
    assert parsed['output_comparisons_passed']==30 and parsed['run_ids']==list(range(3,9))
    with tarfile.open(ROOT/er['archive']['path']) as tar:
        members=[m for m in tar if m.isfile()];assert len(members)==len({m.name for m in members})
        files={m.name:digest(tar.extractfile(m).read()) for m in members}
    assert files==er['package_files'] and {k:v for k,v in files.items() if k!='job.yaml'}==extra_source['package_files']
    oracle=module('e142_oracle_verifier',VALID/'verify_oracle.py').verify()
    assert oracle==read(VALID/'verification-summary.json')
    assert oracle['valid'] and oracle['case_count']==8 and oracle['output_elements']==245760 and oracle['exact_bitwise']
    stress=HERE/'full-ffn-stress'
    subprocess.run([sys.executable,str(stress/'verify_results.py'),'--check'],check=True,stdout=subprocess.DEVNULL)
    sr=read(stress/'e142-full-ffn-stress-results.json');sv=read(stress/'verification.json')
    assert sr['valid'] and sv['valid'] and sv['new_candidate_pass']
    assert sr['stages']['e142']['exact']==sr['stages']['e142']['checked']==3840
    assert read(stress/'build-e142.json')['production_build']==file_record(HERE/'build-e142-full30-v1-repro.json')
    binding=bound[HERE/'source-build-binding.json']
    inputs=[HERE/'independent-audit.json',HERE/'independent-audit.py',HERE/'build-e142-full30-v1-repro.json',
        HERE/'package-e142-full30-v1.json',*bound,*reports,ADOPT/'performance-summary.md',
        ADOPT/'extra-image-verification.json',ADOPT/'extra-submission.json',er_path,
        ROOT/'docs/experiments/e142-extra-correctness.json',VALID/'verification-summary.json',VALID/'verify_oracle.py',
        VALID/'probe-results-v1.json',VALID/'probe-build-v1.json',VALID/'independent-probe-review.json',
        stress/'verification.json',stress/'e142-full-ffn-stress-results.json',stress/'build-e142.json',stress/'verify_results.py',
        HERE/'provenance.py',HERE/'validation_performance.py',HERE/'validation_extra.py']
    for row in perf['paired_observations']:inputs.append(HERE/f"{row['screen_label']}-job-{row['job_id']}-results.json")
    return {'candidate':'E142','variant':'full30-v1','valid':True,'adoption_ready':True,
        'verifier':file_record(Path(__file__).resolve()),'inputdigests':{str(p.relative_to(ROOT)):file_record(p) for p in inputs},
        'source':{'origin':binding['source_base_commit'],'count':46,'changed_files':['src/device/shared/mlp.rs'],
            'production_source':binding['production_source'],'e141_outside_helper_byte_identical':True,
            'e103_executable_helper_reused_with_new_bounds_and_numerical_proof':True,
            'qkv_o_images_identical_to_e140':True,'all_actual_uploads_verified':True},
        'performance':{'independent_allocations':3,'jobs':[r['job_id'] for r in perf['paired_observations']],
            'paired_delta_percent':[r['delta_percent'] for r in perf['paired_observations']],
            'median_paired_delta_percent':perf['median_paired_delta_percent'],
            'aggregate_median_delta_percent':perf['aggregate_median_delta_percent'],
            'all_paired_ffn_improved':True,'scope':'Three independent allocations with contemporary E140 brackets; unchanged source and original three standard inputs. No ranking or statistical certainty promise.'},
        'correctness':{'distinct_standard_inputs':9,'standard_input_indices':list(range(9)),
            'candidate_standard_comparisons_passed':75,'candidate_ffn_comparisons_passed':15,
            'repeat_inputs_are_not_new_cases':True,'extra6_job':er['job']['id'],
            'actual_projection_oracle':{'job':oracle['job'],'cases':8,'outputs':245760,'exact_bf16':True,'signed_zero_equivalent':True,
                'nonfinite_values':0,'actual_quantizer_and_full30_partial_consumer':True,'historical_inputs_wrapper_truth_byte_identical':True},
            'actual_whole_ffn_stress':{'job':sr['job']['id'],'cases':1,'outputs':3840,'exact_bf16':True,
                'same_original_13_inputs_and_torch_reference':True,'performance_excluded':True},
            'scope':'Nine distinct standard inputs, eight targeted first-BF16 projection inputs, one full-FFN regression. Original criteria preserved. Not arbitrary-input or end-to-end equivalence.'},
        'decision':{'adoption':'Validation complete; root owns source-only commit/tag, live baseline and final adoption decision.',
            'further_tuning':'Frozen source; stop row/stage/precision tuning. Positive below5% still qualifies for adoption review.',
            'major_research':'Recovery of the existing direct30/F32-partial family, not a new family or globally correct/fast direct-FP4 proof.'},
        'adopted':False,'live_sources_changed':False,'official_submission':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args();v=verify();dest=VALID/'final-verification.json'
    if a.check:assert read(dest)==v
    else:write(dest,v)
    print('PASS E142 source46/3alloc/9standard/8oracle245760/fullFFN3840; no live adoption')

if __name__=='__main__':main()
