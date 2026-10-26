#!/usr/bin/env python3
"""Replay actual323102; reads frozen evidence only, no builds/device invocation."""
import argparse,hashlib,importlib.util,json,math,re,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
ELF='9a50b33d754f4eeeb85af7c3ea917e35f1ff2400f9aaf93152fd7cb9df8f1551'
STAGES=['prefix_norm_residual_BF16','input_quantize_pack','raw_UP_GATE','global_GeGLU_BF16_bridge','hidden_quantize_pack','raw_DOWN','postnorm_residual_layer_gate']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def kv(line):
    out={}
    for k,v in re.findall(r'(\w+)=([^\s]+)',line):
        if v in ('true','false'):out[k]=v=='true'
        else:
            try:out[k]=int(v)
            except ValueError:
                try:out[k]=float(v)
                except ValueError:out[k]=v
    return out
def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)
def summary(v):
    x=sorted(v);return {'min':x[0],'p50':x[(len(x)-1)//2],'p90':x[math.ceil(.9*(len(x)-1))],'max':x[-1],'mean':statistics.mean(x)}
def analyze(local):
    raw=HERE/'actual-323102';hashes=read(raw/'hashes.json')
    for name,h in hashes.items():assert sha(raw/name)==h['sha256'] and (raw/name).stat().st_size==h['bytes'],name
    run=read(raw/'run.json');status=read(raw/'arena-status.json');art=run['submitted_artifact']
    assert status['id']==323102 and status['status']=='SUCCEEDED' and status['exit_code']==0
    assert run['exit_code']==0 and run['commit']=='466d28006ee46b1f9536518defcd6f946877b8b5'
    assert run['command'][-4:]==['--timeout','70','--name','beef-int4-drift-attribution-actual-20261026']
    assert art['sha256']==ELF and art['size']==6_878_176
    submitted=read(raw/'source-hashes.json');built=read(HERE/'a004-device-build/source-hashes.json')
    assert submitted==built and len(built)==413
    prior=read(HERE.parent/'int4-complete-ffn/a005-int4-complete-ffn-device-build/source-hashes.json')
    assert len(prior)==411 and all(submitted[p]==h for p,h in prior.items())
    audit=read(HERE/'a004-image-audit.json');assert audit['current']['elf_sha256']==ELF
    assert audit['current']['record_count']==21 and len(audit['comparisons'])==21
    assert sum(x['status']=='equal_to_actual323067' for x in audit['comparisons'])==19
    assert sum(x['status']=='equal_to_q020_frozen_image' for x in audit['comparisons'])==1
    assert sum(x['status'].startswith('not_in_prior_complete_or_q020_ELF') for x in audit['comparisons'])==1
    if local:
        tree=HERE.parents[1];root=tree.parent
        for p,h in submitted.items():assert sha(tree/p)==h,p
        for p in ['path','source_path']:assert sha(Path(art[p]))==ELF,p
        reader=root/'b16-f4-mse-lm-fp8/experiments/sdk09-cache-audit/audit_registry.py'
        assert sha(reader)==audit['parser_sha256']
        spec=importlib.util.spec_from_file_location('registry',reader);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        current=m.read_artifact(art['path']);assert current['records']==audit['current']['records']
        old=m.read_artifact(root/'runs/a005-int4-complete-ffn-device-build/built/target/release/test_int4_complete_ffn')
        ci={r['path']:r for r in current['records']};assert len(old['records'])==19
        for r in old['records']:assert ci[r['path']]==r
        q020=m.read_artifact(root/'runs/a003-q020-mse-parallel-server-build/built/target/release/server')
        name='furiosa_opt_gemma4::ops_native_fp8_p512_blocks::quantize_hidden'
        prior_image,=[r for r in q020['records']if r['path']==name and not r['key']];assert ci[name]==prior_image
        name='furiosa_opt_gemma4::ops_native_fp8_p512_blocks::quantize_ffn'
        fresh=HERE/'kernel-out-a004/furiosa-opt-gemma4'/f'{name}.bin';assert sha(fresh)==ci[name]['sha256'] and fresh.stat().st_size==ci[name]['bytes']
    lines=(raw/'arena.log').read_text().splitlines()
    def records(prefix):return [kv(l) for l in lines if l.startswith(prefix+' ')]
    def js(prefix):return [json.loads(l.split(' ',1)[1]) for l in lines if l.startswith(prefix+' ')]
    total,=records('PASS_I4_ATTRIBUTION');drift=records('I4_COMPLETE_DRIFT');anchors=records('I4_ATTR_ANCHOR');f64=records('I4_COMPLETE_F64')
    weights=js('I4_ATTR_WEIGHT');activations=js('I4_ATTR_ACTIVATION');wall=js('I4_ATTR_WALL');interaction=js('I4_ATTR_INTERACTION')
    assert len(weights)==3 and len(activations)==4 and len(wall)==16 and len(interaction)==8 and len(anchors)==4 and len(drift)==46 and len(f64)==6
    nh=512*3840;nl=512*15360;weight_values=3*3840*15360
    candidate_bytes=6*(nh+nl)+(nh+nl)//2+4096
    production_bytes=5*(nh+nl)+4096
    x_and_h_owners=3*(nh+nl)//2+4096
    weights_checked=weight_values//2+weight_values+(weight_values//960+2*15360+3840+1)*4
    extra_inputs=nh+3*nl+4096+(2*15360*4+3840*16)*4
    expected={
        'exact_bf16_comparisons':2*(49_152_000+4*nl+2*nl+3*nh),
        'integer_F64_samples':2*(3*512+512),
        'permutation_values':(4*nh+4*nl+1024)+4*(3*nl+2*nh)+(2*nl+nh),
        'direct_immutable_bytes':2*(2*x_and_h_owners+candidate_bytes+production_bytes+2*nh+4160+weights_checked+extra_inputs),
        'replay_bytes':2*(4*2*(4*nl)+4*2*(2*nh)+candidate_bytes+production_bytes),
        'original_EE_anchors_required':True,'rows':512,'orders':2,'layer':0,'synthetic':True,'no_model_quality_or_speed_claim':True}
    assert total==expected,(total,expected)
    # Independent count reconciliation and required negative-control anchors.
    for r in range(2):
        a=[x for x in anchors if x['round']==r];assert len(a)==2 and all(x['passed'] for x in a)
        assert next(x for x in a if 'prefix_E4_codes'in x)['prefix_E4_codes']==nh
        assert next(x for x in a if 'hidden_E4_codes'in x)['hidden_E4_codes']==nl
        assert sum(x.get('prefix_scale_bits',x.get('hidden_scale_bits',0))for x in a)==1024
        for stage in ['factor_upgate_0_UP','factor_upgate_0_GATE','factor_geglu_0','factor_down_0','factor_tail_0']:
            row,=[x for x in drift if x['round']==r and x['stage']==stage]
            assert row['changed']==0 and row['relative_l2']==0 and row['maxabs']==0
    old=[kv(l)for l in (HERE.parent/'int4-complete-ffn/actual-323067/arena.log').read_text().splitlines()if l.startswith('I4_COMPLETE_DRIFT ')]
    for row in drift:
        assert row['step']==0 and row['no_model_quality_claim']
        if row['stage']in ['joined','down','final_residual']:
            oldrow,=[x for x in old if x['round']==row['round']and x['step']==0 and x['stage']==row['stage']]
            assert row==oldrow
    # All actual rows and full column arrays remain in the raw JSON log.
    weight_summary=[]
    for w in weights:
        assert w['native_codes_exact']==3840*15360 and w['native_scale_bits_exact']==w['rows']
        assert w['rows']*w['cols']==3840*15360
        assert len(w['row_errors'])==w['rows'] and [x[0]for x in w['row_errors']]==list(range(w['rows']))
        assert len(w['column_ranges_provided_min_max_E4absmax_I4absmax'])==w['cols']
        for kind in ['E4','I4']:
            e=w[kind];assert e['values']==3840*15360;close(e['relative_l2'],math.sqrt(e['error_squared']/e['reference_squared']));close(e['zero_fraction'],e['zeros']/e['values'])
        assert sum(x[3]for x in w['row_errors'])==w['E4']['zeros'] and sum(x[4]for x in w['row_errors'])==w['I4']['zeros']
        columns=w['column_ranges_provided_min_max_E4absmax_I4absmax'];assert all(a<=b and c>=0 and d>=0 for a,b,c,d in columns)
        weight_summary.append({k:w[k]for k in ['projection','rows','cols','native_codes_exact','native_scale_bits_exact','E4','I4','metric','scope']}|
                              {'provided_column_absmax':summary([max(abs(x[0]),abs(x[1]))for x in columns]),'full_column_ranges_in_raw_log':True})
    assert sum(x['native_codes_exact']for x in weights)==weight_values
    assert sum(x['native_scale_bits_exact']for x in weights)==2*15360+3840
    activation_summary=[]
    for a in activations:
        assert len(a['rows'])==512 and len(a['column_absmax'])==a['width']
        assert [x['p']for x in a['rows']]==list(range(512))
        for kind in ['E4','A4']:
            e=a[kind];assert e['values']==512*a['width'];close(e['relative_l2'],math.sqrt(e['error_squared']/e['reference_squared']));close(e['zero_fraction'],e['zeros']/e['values'])
            assert sum(x[kind]['zeros']for x in a['rows'])==e['zeros']
            assert sum(x[kind]['reference_zeros']for x in a['rows'])==e['reference_zeros']
            close(sum(x[kind]['error_squared']for x in a['rows']),e['error_squared']);close(sum(x[kind]['reference_squared']for x in a['rows']),e['reference_squared'])
        if a['round']==1:
            base,=[b for b in activations if b['round']==0 and b['label']==a['label']]
            for i,row in enumerate(a['rows']):assert {k:v for k,v in row.items()if k!='p'}=={k:v for k,v in base['rows'][(73*i+127)%512].items()if k!='p'}
            assert a['column_absmax']==base['column_absmax']
        activation_summary.append({k:a[k]for k in ['label','round','width','E4','A4','metric','scope']}|
                                  {'row_amax':summary([x['amax']for x in a['rows']]),'column_absmax':summary(a['column_absmax']),'all512_rows_and_full_columns_in_raw_log':True})
    # Eight candidate and eight one-call control samples. No pruning warm outliers.
    for r in range(2):
        rows=[x for x in wall if x['round']==r]
        assert {(x['block'],x['repeat'],x['candidate'])for x in rows}=={(b,i,b in [0,3])for b in range(4)for i in range(2)}
    for x in wall:
        if x['candidate']:
            assert len(x['stages_us'])==7;close(sum(x['stages_us']),x['sum_us']);close(x['sum_us']+x['clock_gaps_us'],x['total_us']);assert x['clock_gaps_us']>=0
    timing=[]
    for r in [0,1,None]:
        rows=[x for x in wall if r is None or x['round']==r];c=[x for x in rows if x['candidate']];p=[x for x in rows if not x['candidate']]
        cm=statistics.mean(x['total_us']for x in c);pm=statistics.mean(x['total_us']for x in p)
        timing.append({'round':r,'samples_per_arm':len(c),'candidate_total_us':summary([x['total_us']for x in c]),'production_onecall_us':summary([x['total_us']for x in p]),
                       'candidate_slower_percent':100*(cm/pm-1),'stage_means_us':{name:statistics.mean(x['stages_us'][i]for x in c)for i,name in enumerate(STAGES)},
                       'clock_gap_us':summary([x['clock_gaps_us']for x in c])})
    records=read(raw/'arena-records.json');assert records==[]
    return {'job':status,'source_commit':run['commit'],'submitted_artifact':art,'raw_hashes':hashes,'source_hashes':413,'original411_unchanged':True,
      'image_parity':{'prior_complete_exact':19,'quantize_hidden':'q020 image exact; same-prefix code/scale actual PASS','quantize_ffn':'newly bound existing source; same-production-hidden code/scale actual PASS'},
      'counts':expected,'native_weight_code_checks':weight_values,'native_weight_scale_bit_checks':2*15360+3840,'activation_E4_anchor_code_checks':2*(nh+nl),'activation_E4_anchor_scale_bits':2048,
      'anchors':anchors,'original_step0_reproduced':True,'weight_reconstruction':weight_summary,'activation_reconstruction':activation_summary,
      'all_matched_step0_drift_records':drift,'factor_interaction':interaction,'timing_samples':wall,'timing_summary':timing,
      'scope':['Supplied layer0 weights; synthetic finite activations; not model accuracy.','EE exact anchors passed before factorial interpretation.','DOWN factors share production GeGLU BF16 and identical midpoint residual; they are not complete-chain endpoints.',
               'Representation metrics use RN32(code*scale), F64 SSE; projection group-scaled sums can round differently.','Nonzero factorial interaction is not an additive L2 decomposition.',
               'Completed-await phase wall clocks include raw binding; no intermediate downloads; no device-op trace.', 'Prior full-FFN performance HOLD remains; no model promotion, calibration or balancing performed.'],
      'device_trace_payload':False,'independent_review':'round2_path reconciled raw hashes, all counters, row permutation metrics and timing means; source unchanged'}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify-local-artifacts',action='store_true');args=ap.parse_args()
    print(json.dumps(analyze(args.verify_local_artifacts),indent=2))
