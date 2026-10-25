#!/usr/bin/env python3
"""Replay preserved job323014 evidence only; no device or network operations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
HERE=Path(__file__).resolve().parent
ELF="1cb160656f2cc0c1006184f6d7fafde0d61a3f7a49e2659cd3cee72eb5f69e5b"
PREVIOUS="c0835bbf4fcd57cc2611b3c39deb566ec26a107efa157f2854d9420f161e1ab1"
LOCK="a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e"
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fields(line):return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)",line))
def stats(values):
    assert values and all(math.isfinite(x) and x>0 for x in values)
    return {"block_means":values,"mean":statistics.mean(values),"minimum":min(values),"maximum":max(values)}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=HERE/'actual-323014')
    parser.add_argument('--verify-local-artifacts',action='store_true')
    args=parser.parse_args();raw=args.raw_dir;repo=HERE.parents[1]
    hashes=load(raw/'raw-files.json')
    for name,v in hashes.items():assert sha(raw/name)==v['sha256'] and (raw/name).stat().st_size==v['bytes']
    status=load(raw/'arena-status.json');run=load(raw/'run.json');inputs=load(raw/'source-hashes.json')
    build=load(HERE/'a002-build-manifest.json');parity=load(HERE/'a002-embedded-parity.json')
    assert status['id']==323014 and status['status']=='SUCCEEDED' and status['exit_code']==0
    assert status['duration_sec']==9.142186017
    assert inputs==load(raw/'build-source-hashes.json') and len(inputs)==404 and inputs['Cargo.lock']==LOCK
    assert run['submitted_artifact']['sha256']==build['artifact']['sha256']==parity['right']['elf_sha256']==ELF
    assert parity['left']['elf_sha256']==PREVIOUS
    assert parity['common_count']==parity['equal_common_count']==4 and parity['different_common_count']==0
    assert not parity['left_only'] and not parity['right_only']
    assert parity['left']['section_sha256']==parity['right']['section_sha256']
    assert run['command'][run['command'].index('--timeout')+1]=='70'
    assert not any('FURIOSA_OPT_PROFILE' in x for x in run['command'])
    for image in parity['right']['records']:
        p=HERE/'kernel-out-a002/furiosa-opt-gemma4'/f"{image['path']}.bin"
        assert sha(p)==image['sha256'] and p.stat().st_size==image['bytes']
    if args.verify_local_artifacts:
        for p,v in inputs.items():assert sha(repo/p)==v,p
        for artifact in [run['submitted_artifact'],build['artifact']]:assert sha(Path(artifact['path']))==ELF
    lines=(raw/'arena.log').read_text().splitlines()
    def records(prefix):return [fields(x) for x in lines if x.startswith(prefix+' ')]
    bound,=records('PASS_I4_GROUP960_BOUND');projection,=records('PASS_I4_GROUP960_PROJECTION')
    expected={'raw_output_bf16_values':35389440,'raw_permutation_values':17694720,'prelaunch_negative_checks':52}
    for key,n in expected.items():assert int(bound[key])==n
    assert bound['zero_device_input_copy']==bound['same_typed_kernel_image']=='true'
    counts={'layer':0,'output_bf16_values':35389440,'independent_i64_F64_samples':3072,'candidate_permutation_values':17694720,'immutable_bytes_compared':571922440}
    for key,n in counts.items():assert int(projection[key])==n
    assert projection['down_resource_HOLD_preserved']==projection['no_full_FFN_or_model_quality_claim']=='true'
    exact=records('I4_GROUP960_EXACT');assert [int(x['round']) for x in exact]==[0,1]
    assert sum(int(x['output_bf16_values']) for x in exact)==counts['output_bf16_values']
    assert all(x['all_output_bits']==x['finite']==x['device_and_supplied_sources_immutable']=='true' for x in exact)
    samples=records('I4_GROUP960_F64');assert len(samples)==6 and all(int(x['samples'])==512 for x in samples)
    assert {(int(x['round']),x['projection']) for x in samples}=={(r,p) for r in [0,1] for p in ['UP','GATE','DOWN']}
    weights=records('I4_GROUP960_WEIGHT');assert len(weights)==3
    assert {x['name'] for x in weights}=={'UP','GATE','DOWN'}
    previous=load(HERE.parent/'int4-group960-probe/actual-result.json')['weight_code_fingerprints']
    assert weights==previous,'same provided weight/code fingerprints as initial projection run'
    timelines={}
    for prefix,flag,scope in [('I4_GROUP960_TIME','candidate','projection_with_all_output_stores'),('I4_GROUP960_BOUND_TIME','bound','typed_I4_vs_checked_public_binding')]:
        data=records(prefix);assert len(data)==8
        for r in [0,1]:
            group=[x for x in data if int(x['round'])==r]
            assert [int(x['block']) for x in group]==[0,1,2,3]
            assert [x[flag] for x in group]==['true','false','false','true']
        assert all(int(x['samples'])==8 and x['scope']==scope for x in data)
        exclusions='device_quant_pack_GeGLU_upload_download' if flag=='bound' else 'activation_quant_pack_GeGLU_upload_download'
        assert all(x['excludes']==exclusions for x in data)
        if flag=='candidate':assert all(x['reference_extra_row_mul']=='unity' for x in data)
        stage={}
        for name in ['upgate','down']:
            a=stats([float(x[name+'_us']) for x in data if x[flag]=='true'])
            b=stats([float(x[name+'_us']) for x in data if x[flag]=='false'])
            stage[name]={'arm_true_us':a,'arm_false_us':b,'true_minus_false_mean_us':a['mean']-b['mean'],'true_over_false_change_percent':100*(a['mean']/b['mean']-1),'timed_calls_per_arm':32}
        timelines[prefix]={'true_arm':'checked_raw_I4' if flag=='bound' else 'typed_I4','false_arm':'typed_I4' if flag=='bound' else 'dense_E4_intended_reference','projections':stage}
    startup=records('I4_GROUP960_STARTUP');assert len(startup)==2
    assert int(startup[0]['values'])==176947200 and int(startup[0]['scale_groups'])==184320
    words=records('I4_GROUP960_WORDS');assert len(words)==2 and [int(x['round']) for x in words]==[0,1]
    assert all(int(x['bytes'])==4915200 for x in words)
    assert counts['immutable_bytes_compared']-562092040==2*4915200
    assert load(raw/'arena-records.json')==[] and not any(x.startswith('PROFILE_') for x in lines)
    out={'job':status,'source_commit':build['source_commit'],'submitted_evidence_commit':run['commit'],'ELF_sha256':ELF,'previous_ELF_sha256':PREVIOUS,
         'source_input_count':len(inputs),'build_submit_inputs_identical':True,'Cargo.lock_sha256':LOCK,'unchanged_embedded_images':4,'whole_kernel_section_equal':True,
         'raw_input_hashes':hashes,'binding_counts':expected,'intended_reference_counts':counts,'negative_check_breakdown':'(12 UP/GATE +14 DOWN)*2; no invalid case reaches lookup or launch',
         'timings':timelines,'cpu_weight_reconstruct_quantize_audit_ms':float(startup[0]['cpu_load_reconstruct_quantize_audit_ms']),
         'weight_upload_ms':float(startup[1]['weight_upload_ms']),'independent_word_encoder':words,'input_setup_records':records('I4_GROUP960_INPUT'),
         'estimated_live_device_payload_bytes_not_measured':int(startup[1]['estimated_peak_live_device_payload_bytes']),
         'sample_radius_records':samples,'weight_code_fingerprints':weights,'profile_enabled':False,
         'decision':'Public packed-I32-to-I4 binding correctness PASS; DOWN resource HOLD remains',
         'limits':['No new device arithmetic; same four actual-passed projection images.',
                   'Comparisons overlap and are not summed as unique outputs or bytes.',
                   'Device quantizer/packer, GeGLU and complete FFN/model remain outside this gate.',
                   'Timing excludes CPU encoding, allocation, transfer, validation-negative-case setup and readback; valid adapter validation is inside raw call timing.',
                   'Observed single-job ABI overhead is not a zero-overhead or model-throughput guarantee.',
                   'Dense E4 reference is intended quantized arithmetic, not production baseline or original model accuracy.',
                   'No synthetic overflow Buffer/address was constructed; checked overflow guard is source-reviewed.']}
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
