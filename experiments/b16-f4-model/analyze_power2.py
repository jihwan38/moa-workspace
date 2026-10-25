#!/usr/bin/env python3
"""Read completed local diagnostic logs; PASS here is not task accuracy."""
import hashlib, json, pathlib, re, statistics
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
def fields(s):
    return dict(x.split('=',1) for x in s.split() if '=' in x)
cases=[]
for mode, run in [('model','r1256-b16-f4-power2-model'),('guards','r1257-b16-f4-power2-guards')]:
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
        'source':'72786a5b16caf5d11522a2089882e0ee962a893c','elf_sha256':'2a2ffd0cb64d4e9b8993070e5a758934797147c47eb7ef66e8670df533eca088','cases':cases}
(HERE/'actual-power2-r1256-r1257.json').write_text(json.dumps(result,indent=2)+'\n')
for c in cases:
    print(json.dumps({k:v for k,v in c.items() if k!='records'},indent=2))
