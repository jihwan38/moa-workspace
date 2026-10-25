#!/usr/bin/env python3
"""Read completed local diagnostic logs; PASS here is not task accuracy."""
import hashlib, json, pathlib, re, statistics, argparse
parser=argparse.ArgumentParser()
parser.add_argument("--identity",action="store_true")
IDENTITY=parser.parse_args().identity
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
def fields(s):
    return dict(x.split('=',1) for x in s.split() if '=' in x)
cases=[]
runs=[('model','r1275-b16-f4-mse-identity-model'),('guards','r1276-b16-f4-mse-identity-guards')] if IDENTITY else [('model','r1260-b16-f4-mse-model'),('guards','r1261-b16-f4-mse-guards')]
for mode, run in runs:
    p=ROOT/'runs'/run/'arena.log'; raw=p.read_bytes(); lines=raw.decode().splitlines()
    prefixes=['B16_F4_PREPARE','B16_F4_MEMORY','B16_F4_STEP','B16_F4_STAGE','B16_F4_DRIFT']
    parsed={prefix:[fields(s) for s in lines if s.startswith(prefix+' ')] for prefix in prefixes}
    startup=next(s for s in lines if s.startswith('B16_F4_STARTUP '))
    elapsed=float(re.search(r'elapsed_ms=([0-9.]+)',startup)[1])
    prepare=parsed['B16_F4_PREPARE']; assert len(prepare)==144
    sums={k:sum(float(x[k]) for x in prepare) for k in ['read_ms','encode_ms','upload_ms']}
    memory={x['label']:x for x in parsed['B16_F4_MEMORY']}
    shadow_bytes=int(memory['all_original_workspaces_before_shadow']['available'])-int(memory['all_original_workspaces_and_complete_shadow']['available'])
    assert shadow_bytes==4246732800+35389440
    summary=dict(mode=mode,run=run,sha256=hashlib.sha256(raw).hexdigest(),startup_elapsed_ms=elapsed,
                 timed_startup_ms=sums,startup_outside_reported_spans_ms=elapsed-sum(sums.values()),
                 measured_shadow_allocation_bytes=shadow_bytes,records=parsed,
                 pass_lines=[s for s in lines if s.startswith('PASS B16_F4_')])
    if IDENTITY:
        identity=[fields(s) for s in lines if s.startswith('B16_F4_STORAGE_IDENTITY ')]
        assert len(identity)==1
        expected={'physical_rings':'32','admission':'32','query_rows':'64','full_pages':'256','context':'2048','sliding_layers':'40','full_layers':'8','allocated_kv_bytes':'12884901888','verified_owning_buffers':'true'}
        assert identity[0]==expected
        summary['actual_buffer_identity']=identity[0]
    if mode=='model':
        eligible={2,3,4,5,6,7,9}
        drift=parsed['B16_F4_DRIFT']; assert len(drift)==71
        for name, rows in [('all',drift),('f4_calls',[x for x in drift if int(x['step']) in eligible]),('fallback_after_changed_history',[x for x in drift if x['step']=='8'])]:
            l2=[float(x['relative_l2']) for x in rows]
            summary[name]=dict(rows=len(rows),top1_differences=sum(x['same_top1']=='false' for x in rows),
                               relative_l2_min=min(l2),relative_l2_mean=statistics.mean(l2),relative_l2_max=max(l2),
                               max_abs=max(float(x['max_abs']) for x in rows),bitdifferences=sum(int(x['bitdiff']) for x in rows))
        summary['eligible_timing_by_pass']={}
        for arm in ['0','1','2']:
            rows=[x for x in parsed['B16_F4_STEP'] if x['pass']==arm and int(x['step']) in eligible]
            stages=[x for x in parsed['B16_F4_STAGE'] if x['pass']==arm and int(x['step']) in eligible and x['stage']=='ffn.post_attention_epilogue_fused']
            assert len(rows)==len(stages)==7
            summary['eligible_timing_by_pass'][arm]=dict(model_mean_ms=statistics.mean(float(x['model_ms']) for x in rows),ffn_mean_ms=statistics.mean(float(x['total_ms']) for x in stages))
        summary['strict_logits_breakdown']={'same_candidate_replay':71*262144,'unchanged_scalar_and_Automatic':3*262144,'unchanged_P128':2*262144,'total':76*262144}
        summary['candidate_replay_cache_bytes_one_side']=2952790016*2
    cases.append(summary)
result={'decision':'Structural/state gates passed; original model quality NOT established; default remains OFF',
        'source':'2509e19f','elf_sha256':'e17b5c4e339b8a86e9fb90b509e0818b74d752b6686e2911b7a88a09c7761d8a','identity_status':'Initial artifact; explicit physical-buffer identity fresh build pending','cases':cases}
reference=json.loads((ROOT/'b16-f4-model/experiments/b16-f4-model/actual-power2-r1256-r1257.json').read_text())
old=reference['cases'][0]['records']['B16_F4_DRIFT']
new=cases[0]['records']['B16_F4_DRIFT']
assert [(x['step'],x['owner']) for x in old]==[(x['step'],x['owner']) for x in new]
assert all(a['original_top1']==b['original_top1'] for a,b in zip(old,new))
paired=[(a,b) for a,b in zip(old,new) if int(a['step']) in {2,3,4,5,6,7,9}]
result['paired_to_power2']={
 'eligible_rows':len(paired),
 'l2_improved_rows':sum(float(b['relative_l2'])<float(a['relative_l2']) for a,b in paired),
 'regained_original_top1':sum(a['same_top1']=='false' and b['same_top1']=='true' for a,b in paired),
 'lost_original_top1':sum(a['same_top1']=='true' and b['same_top1']=='false' for a,b in paired),
 'logged_original_top1_matches_all_rows':True,
 'cross_job_original_all_logit_bitproof':False,
 'power2_eligible':reference['cases'][0]['f4_calls'],
 'mse_eligible':cases[0]['f4_calls']}
if IDENTITY:
    result.update(source='2d0bb5a85f62858e67fa84bd6db2551681ee82ad',elf_sha256='6ab8428702954445c9fb2773e7b04d964d9abbe208048effc0911f121d40d3e4',identity_status='Fresh-package owning-buffer identity verified in both actual jobs')
    initial=json.loads((HERE/'actual-mse-r1260-r1261.json').read_text())
    assert initial['cases'][0]['records']['B16_F4_DRIFT']==cases[0]['records']['B16_F4_DRIFT']
    result['initial_and_fresh_reported_drift_records_identical']=True
    output='actual-mse-identity-r1275-r1276.json'
else:output='actual-mse-r1260-r1261.json'
(HERE/output).write_text(json.dumps(result,indent=2)+'\n')
for c in cases:
    print(json.dumps({k:v for k,v in c.items() if k!='records'},indent=2))

print(json.dumps(result['paired_to_power2'],indent=2))
