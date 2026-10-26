#!/usr/bin/env python3
"""Strict saved-log replay for job323407. No build, network, or device action."""
import argparse, hashlib, json, math
from pathlib import Path
import audit_build

HERE=Path(__file__).resolve().parent
RAW=HERE/'actual-323407'
P,H,L=512,3840,15360
ARMS=('provided_bf16','ideal_f64_rotation','f32_bf16_rotation_only','original_A4W4','rotated_A4W4',
      'original_A4_only','original_W4_only','rotated_A4_only','rotated_W4_only')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def finite(v):
    if isinstance(v,dict):
        for x in v.values():finite(x)
    elif isinstance(v,list):
        for x in v:finite(x)
    elif isinstance(v,float):assert math.isfinite(v)
def fields(s):
    out={}
    for token in s.split():
        k,v=token.split('=',1);assert k not in out
        out[k]=True if v=='true' else False if v=='false' else int(v)
    return out
def analyze(verify_local=False):
    assert sha(RAW/'raw-hashes.json')=='38a5188f047f3a974a585577c23a9f5a147c0596cf2424e862513fc99895b980'
    hashes=read(RAW/'raw-hashes.json')
    for n,s in hashes.items():assert sha(RAW/n)==s,n
    status,run=read(RAW/'arena-status.json'),read(RAW/'run.json')
    assert status['id']==323407 and status['status']=='SUCCEEDED' and status['exit_code']==0
    assert run['name']=='int4-hadamard64-actual-20261026' and run['result']=='command_succeeded' and run['exit_code']==0
    assert run['commit']=='516382ad21cbf3aa3eaf463ab635ed19dda41d52' and run['status']==''
    assert (RAW/'source.diff').read_bytes()==b'' and read(RAW/'arena-records.json')==[]
    build=read(HERE/'build-audit.json');inputs=read(RAW/'source-hashes.json')
    assert len(inputs)==429 and inputs==read(HERE/'a005-device-build/source-hashes.json')
    artifact=run['submitted_artifact']
    assert artifact['sha256']==build['artifact']['sha256']=='c591b27a69a73e3b6d9a58e596d6ea2fc6612200fb5e8290cbc889f4965cf5fd'
    assert artifact['size']==build['artifact']['size']==4859280
    assert build['all8_prior_and_fresh_equal'] and build['q020_control_exact'] and len(build['comparisons'])==8
    if verify_local:
        assert audit_build.main()==build
        for field in ('path','source_path'):
            p=Path(artifact[field]);assert sha(p)==artifact['sha256'] and p.stat().st_size==artifact['size']
    lines=(RAW/'arena.log').read_text().splitlines();assert len(lines)==18
    at=0
    def take(marker):
        nonlocal at
        prefix=marker+' ';assert lines[at].startswith(prefix),(at,marker)
        payload=lines[at][len(prefix):];at+=1
        result=json.loads(payload) if payload.startswith('{') else fields(payload)
        finite(result);return result
    setup=take('I4_H64_SETUP')
    assert setup['layer']==0 and setup['sampled_weight_rows']==1536 and setup['sampled_weight_values']==11796480
    assert setup['seed']=='0x243f6a8885a308d3' and setup['quantizer']=='unchanged rational per960, no MSE'
    assert setup['no_all48_startup_or_runtime_speed_claim'] is True
    assert setup['CPU_sampled_decode_rotate_quantize_ms']>0
    assert len(setup['source_fingerprints'])==3
    records=[];first={};all_gates=[]
    for order in range(2):
        for step in range(2):
            for projection,cols,rows in [('up_proj',H,L),('gate_proj',H,L),('down_proj',L,H)]:
                r=take('I4_H64_DOTS');assert (r['round'],r['step'],r['projection'])==(order,step,projection)
                assert (r['pairs'],r['terms_per_arm'],r['output_rows'])==(P,P*cols,rows)
                assert r['scope']=='pre_global_FP64_projection_dots' and r['sample_row']=='(317*canonical_owner+19)%rows'
                assert r['feedback']=='original q020 only; DOWN input is production GeGLU'
                assert all(r[k] is True for k in ['matched_inputs','finite_normal_or_zero_scope_passed','not_device_reduction_or_FFN_output','no_task_quality_or_speed_claim'])
                assert r['global_reported_not_applied']!=0 and set(r['arms'])==set(ARMS)
                norm=r['arms']['provided_bf16']['reference_ss'];assert norm>0
                for n in ARMS:
                    e=r['arms'][n];assert set(e)=={'samples','sse','reference_ss','relative_l2','maxabs'}
                    assert e['samples']==P and e['reference_ss']==norm and min(e['sse'],e['relative_l2'],e['maxabs'])>=0
                    assert math.isclose(e['relative_l2'],math.sqrt(e['sse']/norm),rel_tol=2e-15,abs_tol=1e-30)
                    assert e['maxabs']**2<=e['sse']*(1+1e-12) and e['sse']<=P*e['maxabs']**2*(1+1e-12)
                assert all(r['arms']['provided_bf16'][k]==0 for k in ['sse','relative_l2','maxabs'])
                assert r['ideal_identity_maxabs']==r['arms']['ideal_f64_rotation']['maxabs']
                assert 0<=r['ideal_identity_maxabs']<=r['ideal_identity_max_bound']
                for k in ['original_activation_zero_fraction','rotated_activation_zero_fraction','original_weight_zero_fraction','rotated_weight_zero_fraction']:
                    assert 0<=r[k]<=1
                assert r['gate_rotation_only_max_l2']==0.01 and r['gate_rotated_over_original_A4W4']==0.75
                gate=(r['arms']['f32_bf16_rotation_only']['relative_l2']<=0.01 and
                      r['arms']['rotated_A4W4']['relative_l2']<=0.75*r['arms']['original_A4W4']['relative_l2'])
                assert r['numerical_gate_passed'] is gate
                canonical={k:v for k,v in r.items() if k!='round'}
                if order==0:first[(step,projection)]=canonical
                else:assert canonical==first[(step,projection)],'every reported metric must match canonical-owner order'
                all_gates.append(gate);records.append(r)
            s=take('I4_H64_STEP_PASS')
            assert s=={'round':order,'step':step,'original_production_feedback':True,'full_owner_reuse':True}
    final=take('PASS_I4_H64_DIAGNOSTIC');assert at==len(lines)
    assert (final['pairs_per_arm'],final['arms'],final['terms_per_arm'])==(6144,9,47185920)
    # Existing compare_rows returns BF16 ELEMENTS. The frozen marker incorrectly
    # adds these to pair-record BYTES; preserve it rather than alter raw evidence.
    state_elements=2*P*(3*H+L)
    pair_bytes=2*3*P*11*8
    assert final['permutation_bytes']==state_elements+pair_bytes==27795456
    production_bytes=P*(H*2+H+4+2*L*2+L+4+H*2)
    activation_bytes=P*(H+L+H)*2
    assert final['reuse_bytes']==4*(production_bytes+activation_bytes)==290996224
    assert final['immutable_bytes_compared']==4*(production_bytes+activation_bytes+P*H*2)==306724864
    assert all(final[k] is True for k in ['no_promotion','no_task_quality_or_speed_claim','device_rotation_unimplemented'])
    assert final['numerical_gate_passed'] is all(all_gates)
    assert final['decision']==('SAMPLED_NUMERICAL_GATE_PASS_ONLY' if all(all_gates) else 'HOLD')
    weights=final['production_weights_full_read_fingerprints']
    assert len(weights)==21 and len({w[0] for w in weights})==21
    assert all(n>0 and 0<=h<2**64 for _,n,h in weights)
    # Supplied packed/blockscale fingerprints must match the retained originals.
    byname={name:(n,h) for name,n,h in weights}
    for group,short in zip(setup['source_fingerprints'],['up','gate','down']):
        assert len(group)==3
        assert tuple(group[0][1:])==byname[short+'_packed']
        assert tuple(group[1][1:])==byname[short+'_blockscale']
        assert group[2][1]==4
    summary=[]
    for r in records[:6]:
        a=r['arms'];summary.append({'step':r['step'],'projection':r['projection'],
            'original_A4W4_l2':a['original_A4W4']['relative_l2'],'rotated_A4W4_l2':a['rotated_A4W4']['relative_l2'],
            'rotated_over_original':a['rotated_A4W4']['relative_l2']/a['original_A4W4']['relative_l2'],
            'rotation_only_l2':a['f32_bf16_rotation_only']['relative_l2'],'gate':r['numerical_gate_passed'],
            'activation_zero_original':r['original_activation_zero_fraction'],'activation_zero_rotated':r['rotated_activation_zero_fraction']})
    return {'decision':final['decision'],'no_promotion':True,'job':status,'records':18,
        'raw_hashes':hashes,'raw_hash_manifest_sha256':sha(RAW/'raw-hashes.json'),
        'lineage':{'build_audit_sha256':sha(HERE/'build-audit.json'),'submitted_artifact':artifact,'inputs':429,'old_images':8,'q020_control_exact':True},
        'counts':{'pairs_per_arm':6144,'arms':9,'terms_per_arm':47185920,'unique_sampled_weight_rows':1536,
            'raw_mixed_permutation_counter_named_bytes':final['permutation_bytes'],'permuted_bf16_values':state_elements,
            'permuted_pair_record_bytes':pair_bytes,'total_permutation_compared_bytes':2*state_elements+pair_bytes,
            'reuse_bytes':final['reuse_bytes'],'immutable_exact_bytes':final['immutable_bytes_compared'],
            'retained_production_fingerprinted_bytes_per_pass':sum(w[1] for w in weights)},
        'summary':summary,'setup':setup,'dot_records':records,'final_marker':final,
        'limitations':['Provided layer0 weights with synthetic incoming512-row fixture; bothsteps share unchanged q020 feedback.',
            'Pre-global sampled F64 dots only. Rotation and quantization run onCPU; no rotated device graph, FFN output, whole-model/task quality or speed result.',
            'Nine arms use exactF64 code*storedF32scale reconstruction followed by F64 dots; not actual localF32 multiplication/reduction/BF16 behavior.',
            'Ideal rotation is finite F64 evaluation with a per-pair absolute normwise bound; not an all-element exact-real execution claim.',
            'Raw permutation_bytes mixes BF16 element counts and pair bytes; split/corrected source-derived counts are provided.',
            'Diagnostic duration and sampled CPU preparation time are not kernel/model runtime speed or all48 startup estimates.']}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify-local-artifacts',action='store_true');args=ap.parse_args()
    print(json.dumps(analyze(args.verify_local_artifacts),indent=2))
