#!/usr/bin/env python3
"""Replay job323172 intended-code projection evidence; no device/network calls."""
import argparse,hashlib,json,math,re,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
ELF='4fb53b95157ed9a32afa91f0f78c7eea2d62fd8e08074d8e31c86ebc7f3085cd'
LOCK='a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fields(line):return dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)',line))
def stats(v):
    assert v and all(math.isfinite(x) and x>0 for x in v)
    return {'block_means_us':v,'mean_us':statistics.mean(v),'min_us':min(v),'max_us':max(v)}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--verify-local-artifacts',action='store_true');args=ap.parse_args()
    raw=HERE/'actual-323172';repo=HERE.parents[1];hashes=load(raw/'raw-files.json')
    for name,v in hashes.items():assert sha(raw/name)==v['sha256'] and (raw/name).stat().st_size==v['bytes']
    status=load(raw/'arena-status.json');run=load(raw/'run.json');inputs=load(raw/'source-hashes.json');build=load(HERE/'build-result.json');registry=load(HERE/'embedded-registry.json')
    assert status['id']==323172 and status['status']=='SUCCEEDED' and status['exit_code']==0 and status['duration_sec']==9.306930598
    assert inputs==load(raw/'build-source-hashes.json') and len(inputs)==418 and inputs['Cargo.lock']==LOCK
    assert run['submitted_artifact']['sha256']==build['captured_elf']['sha256']==registry['elf_sha256']==ELF
    assert run['submitted_artifact']['size']==build['captured_elf']['size']==4339808
    assert run['command'][run['command'].index('--timeout')+1]=='70'
    assert not any('FURIOSA_OPT_PROFILE' in x for x in run['command'])
    assert registry['record_count']==4 and len(build['embedded_images'])==4
    for image in registry['records']:
        p=HERE/'kernel-out-a004/furiosa-opt-gemma4'/f"{image['path']}.bin"
        assert sha(p)==image['sha256'] and p.stat().st_size==image['bytes']
    assert sum(x.get('resource_bin_equal_embedded',False) for x in build['embedded_images'])==2
    if args.verify_local_artifacts:
        for p,v in inputs.items():assert sha(repo/p)==v,p
        for a in [run['submitted_artifact'],build['captured_elf']]:assert sha(Path(a['path']))==ELF
    lines=(raw/'arena.log').read_text().splitlines()
    def records(tag):return [fields(x) for x in lines if x.startswith(tag+' ')]
    bound,=records('PASS_I4_GROUPED_ACTIVATION_BOUND');projection,=records('PASS_I4_GROUPED_ACTIVATION_PROJECTION')
    p,h,l=512,3840,15360
    outputs=2*p*(2*l+h);permutation=p*(2*l+h)
    weights=3*h*l*3//2+3*h*l//960*4+4
    input_per_round=p*(h+l)*3//2+p*(h+l)//960*4
    words_per_round=p*(h+l)//2
    immutable=2*(weights+input_per_round+words_per_round)
    expected={'raw_output_bf16_values':outputs,'raw_permutation_values':permutation,'prelaunch_negative_checks':2*(13+15)}
    for k,n in expected.items():assert int(bound[k])==n
    assert bound['zero_device_input_copy']==bound['same_typed_kernel_image']=='true'
    counts={'layer':0,'output_bf16_values':outputs,'independent_i64_F64_samples':2*3*512,'candidate_permutation_values':permutation,'immutable_bytes_compared':immutable}
    for k,n in counts.items():assert int(projection[k])==n
    assert immutable==571719688 and outputs==35389440 and permutation==17694720
    assert projection['down_resource_HOLD_preserved']==projection['no_full_FFN_or_model_quality_claim']=='true'
    exact=records('I4_GROUPED_ACTIVATION_EXACT');assert [int(x['round']) for x in exact]==[0,1]
    assert all(int(x['output_bf16_values'])==permutation and x['all_output_bits']==x['finite']==x['device_and_supplied_sources_immutable']=='true' for x in exact)
    samples=records('I4_GROUPED_ACTIVATION_F64');assert len(samples)==6
    assert {(int(x['round']),x['projection']) for x in samples}=={(r,s) for r in [0,1] for s in ['UP','GATE','DOWN']}
    assert all(int(x['samples'])==512 and x['local_dot']=='i64_exact' and x['bound']=='gamma_3Nplus4_plus_group_amplified_FTZ_plus_F64_error' and math.isfinite(float(x['max_preBF16_radius'])) and float(x['max_preBF16_radius'])>=0 for x in samples)
    weights_records=records('I4_GROUPED_ACTIVATION_WEIGHT');assert len(weights_records)==3 and {x['name'] for x in weights_records}=={'UP','GATE','DOWN'}
    previous=load(HERE.parent/'int4-group960-probe/actual-result.json')['weight_code_fingerprints']
    assert weights_records==previous,'provided weights/encoding identical to initial per960 projection'
    timing={}
    for tag,flag,scope in [('I4_GROUPED_ACTIVATION_TIME','candidate','projection_with_all_output_stores'),('I4_GROUPED_ACTIVATION_BOUND_TIME','bound','typed_I4_vs_checked_public_binding')]:
        data=records(tag);assert len(data)==8
        for r in [0,1]:
            group=[x for x in data if int(x['round'])==r]
            assert [int(x['block']) for x in group]==[0,1,2,3] and [x[flag] for x in group]==['true','false','false','true']
        assert all(int(x['samples'])==8 and x['scope']==scope for x in data)
        exclusions='device_quant_pack_GeGLU_upload_download' if flag=='bound' else 'activation_quant_pack_GeGLU_upload_download'
        assert all(x['excludes']==exclusions for x in data)
        if flag=='candidate':assert all(x['reference']=='dense_E4_integer_codes_group_scales_no_extra_row_mul' for x in data)
        stages={}
        for name in ['upgate','down']:
            a=stats([float(x[name+'_us']) for x in data if x[flag]=='true']);b=stats([float(x[name+'_us']) for x in data if x[flag]=='false'])
            stages[name]={'true_arm':a,'false_arm':b,'true_minus_false_mean_us':a['mean_us']-b['mean_us'],'true_over_false_change_percent':100*(a['mean_us']/b['mean_us']-1),'timed_calls_per_arm':32}
        timing[tag]={'records':data,'true_arm':'checked_raw_I4' if flag=='bound' else 'typed_I4','false_arm':'typed_I4' if flag=='bound' else 'dense_E4_intended_reference','stages':stages}
    startup=records('I4_GROUPED_ACTIVATION_STARTUP');assert len(startup)==2
    assert int(startup[0]['values'])==3*h*l and int(startup[0]['scale_groups'])==3*h*l//960
    assert startup[0]['code_rule']=='independent_u128_quotient_remainder'
    assert int(startup[1]['declared_weight_payload_bytes'])==weights and startup[1]['allocator_overhead_and_probe_scratch_excluded']=='true'
    word_records=records('I4_GROUPED_ACTIVATION_WORDS');assert [int(x['round']) for x in word_records]==[0,1] and all(int(x['bytes'])==words_per_round for x in word_records)
    setups=records('I4_GROUPED_ACTIVATION_INPUT');assert len(setups)==4
    scales=[x for x in setups if 'input_scales' in x];assert [int(x['round']) for x in scales]==[0,1]
    assert all(int(x['input_scales'])==2048 and int(x['hidden_scales'])==8192 and x['all_token_group_scales_unique']=='true' for x in scales)
    assert load(raw/'arena-records.json')==[] and not any(x.startswith('PROFILE_') for x in lines)
    out={'job':status,'source_commit':build['functional_source'],'build_commit':build['build_commit'],'submitted_evidence_commit':run['commit'],'ELF_sha256':ELF,'ELF_bytes':4339808,
         'source_input_count':418,'build_submit_inputs_identical':True,'Cargo.lock_sha256':LOCK,'embedded_images':build['embedded_images'],'raw_input_hashes':hashes,
         'binding_counts':expected,'intended_reference_counts':counts,'negative_check_breakdown':'(13 UP/GATE+15 DOWN)*2; includes old2048B scale rejection before lookup/launch',
         'immutable_formula':{'provided_weight_code_and_scale_owners_per_round':weights,'typed_I4_E4_inputs_and_group_scales_per_round':input_per_round,'packed_I32_inputs_per_round':words_per_round,'rounds':2},
         'timings':timing,'cpu_weight_reconstruct_quantize_audit_ms':float(startup[0]['cpu_load_reconstruct_quantize_audit_ms']),'weight_upload_ms':float(startup[1]['weight_upload_ms']),
         'declared_weight_payload_bytes_not_allocator_peak':weights,'independent_word_encoder':word_records,'input_setup_records':setups,'sample_radius_records':samples,'weight_code_fingerprints':weights_records,
         'decision':'Grouped activation/weight intended-code projection and public raw binding correctness PASS; no fullpath speed/model promotion',
         'limits':['Dense-E4 timing is an intended integer-code oracle, not current production E4 or previous row-A4 arithmetic.',
                   'All-output comparisons overlap; counts are comparisons, not unique tensor elements across arms.',
                   'Future group quantizer, packer, GeGLU, fullFFN and model task quality are outside this job.',
                   'CPU weight conversion, upload, allocations, negative checks and readback are startup/outside timed intervals; valid raw binding validation and completed awaits are inside ABI timing.',
                   'Absolute F64 radii span the wide preregistered synthetic token-group scale range; they are supplemental to strict all-output device reference equality, not model-error measurements.',
                   'No actual per-op trace or isolated per-launch constant; cross-job timings do not establish a controlled speed delta.',
                   'Historical complete323067 failure and old DOWN resource HOLD remain preserved.']}
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
