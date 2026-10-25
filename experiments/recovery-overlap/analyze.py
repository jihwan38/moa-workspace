#!/usr/bin/env python3
"""Read collected evidence only; never infer HTTP↔worker IDs from arrival order."""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import re


def pct(xs, p=90):
    xs=sorted(x for x in xs if x is not None)
    return xs[max(0,math.ceil(len(xs)*p/100)-1)] if xs else None


def fields(line):
    return dict(re.findall(r'(\w+)=([^ ]*)',line))


def members(raw, ready):
    result=[]
    for item in raw.split(',') if raw else []:
        parts=item.split(':')
        if len(parts)==2 and parts[1]=='gone':
            result.append({'id':int(parts[0]),'gone':True})
        else:
            assert len(parts)==(4 if ready else 3),item
            result.append(dict(zip(['id','ring','remaining']+(['ready'] if ready else []),map(int,parts))))
    return result


def strip_record(data):
    keys=['request_index','suite','passed','finish_reason','usage','ttft_ms','latency_ms',
          'estimated_tpot_ms','dispatch_offset_ms','completion_offset_ms','delta_offsets_ms','error']
    row={key:data[key] for key in keys if key in data}
    if 'dispatch_offset_ms' in row and 'ttft_ms' in row:
        row['first_offset_ms']=row['dispatch_offset_ms']+row['ttft_ms']
    return row


def summarize_subset(rows):
    good=[r for r in rows if r.get('passed')]
    return {'record_count':len(rows),'passed_count':len(good),
            'completed_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in good),
            'observed_ttft_p90_ms':pct([r.get('ttft_ms') for r in good]),
            'observed_tpot_p90_ms':pct([r.get('estimated_tpot_ms') for r in good]),
            'over4s_count':sum(r.get('ttft_ms',0)>4000 for r in good),
            'over100ms_tpot_count':sum(r.get('estimated_tpot_ms',0)>100 for r in good),
            'last_completion_offset_ms':max((r.get('completion_offset_ms',0) for r in good),default=None)}


def gap_report(row):
    offsets=row.get('delta_offsets_ms')
    if offsets is None:return {'available':False}
    assert offsets==sorted(offsets),'Nonmonotone content arrivals'
    gaps=[b-a for a,b in zip(offsets,offsets[1:])]
    return {'available':True,'content_event_count':len(offsets),
            'generated_token_count':row.get('usage',{}).get('completion_tokens'),
            'content_gap_p50_ms':pct(gaps,50),'content_gap_p90_ms':pct(gaps),
            'content_gap_max_ms':max(gaps,default=None),
            'content_gaps_over150ms':sum(g>150 for g in gaps),
            'note':'Nonempty content event gaps; one SSE event need not equal one generated token.'}


def analyze(path, parse_recovery=True):
    raw=(path/'arena.log').read_bytes(); lines=raw.decode(errors='replace').splitlines()
    status=json.loads((path/'arena-status.json').read_text())
    records=json.loads((path/'arena-records.json').read_text())
    candidate=json.loads((path/'tested-candidate.json').read_text())
    summaries={}; http={'warmup':[],'measured':[]}
    for rec in records:
        kind=rec['type']; data=rec['data']
        if not kind.startswith('STAGE2_PERFORMANCE'):continue
        phase='warmup' if '_WARMUP_' in kind else 'measured'
        if kind.endswith('_SUMMARY'):summaries[phase]=data
        elif kind.endswith('_RESULT'):http[phase].append(strip_record(data))
    workers=[]; cohorts=[]; recovery=[]; totals=[]
    for num,line in enumerate(lines,1):
        if line.startswith('SCHED_REQ '):workers.append({'line':num,**fields(line)})
        elif line.startswith('SCHED_COHORT '):cohorts.append({'line':num,**fields(line)})
        elif parse_recovery and line.startswith('SCHED_RECOVERY '):
            f=fields(line)
            recovery.append({'line':num,'action':f['action'],
                'members':members(f['members'],True),'passthrough':members(f['passthrough'],False),
                'oldest_ms':float(f['oldest_ms']),'left_ms':float(f['left_ms'])})
        elif line.startswith('SCHED_TOTAL '):totals.append({'line':num,'raw':line})
    expected_match=re.search(r'Sending (\d+) requests for .*?, (\d+) at a time',raw.decode(errors='replace'))
    expected=int(expected_match[1]) if expected_match else None
    concurrency=int(expected_match[2]) if expected_match else None
    observed=[r['request_index'] for r in http['measured']]
    missing=sorted(set(range(1,expected+1))-set(observed)) if expected is not None else None
    complete=(expected is not None and not missing and len(observed)==len(set(observed))
              and len(observed)==expected and 'measured' in summaries)
    cycles=[]; errors=[]; current=None; participated=set()
    for event in recovery:
        ids=[r['id'] for r in event['members']]
        if len(ids)!=len(set(ids)):errors.append('duplicate member ID')
        if set(ids)&{r['id'] for r in event['passthrough']}:errors.append('member passed through')
        if event['action']=='start':
            if current:errors.append('new snapshot before previous release')
            if participated&set(ids):errors.append('previous member re-held')
            participated.update(ids)
            if len(ids)<2:errors.append('snapshot smaller than2')
            if any(r.get('ready')!=0 for r in event['members']):errors.append('start member already ready')
            if abs(event['oldest_ms']+event['left_ms']-3750)>0.003:errors.append('receipt budget changed')
            current=event
        else:
            if current is None:
                errors.append('release without start (possibly truncated log)');continue
            if ids!=[r['id'] for r in current['members']]:errors.append('snapshot membership/order changed')
            begin={r['id']:r for r in current['members']}
            for row in event['members']:
                if not row.get('gone') and row['remaining']>begin[row['id']]['remaining']:
                    errors.append('member remaining count increased')
            if event['action']=='ready' and any(not(r.get('gone') or r.get('ready')) for r in event['members']):
                errors.append('ready release has unfinished member')
            if event['action']=='deadline' and event['oldest_ms']<3750:errors.append('early deadline')
            pass_ids={r['id'] for r in current['passthrough'] if r['remaining']==0}
            completed_inside=[w for w in workers if int(w['id']) in pass_ids and current['line']<w['line']<event['line']]
            cycles.append({'start':current,'release':event,
                'duration_from_oldest_clock_ms':event['oldest_ms']-current['oldest_ms'],
                'ready_count':sum(bool(r.get('ready') or r.get('gone')) for r in event['members']),
                'remaining_prompt_tokens':sum(r.get('remaining',0) for r in event['members']),
                'start_passthrough_decoder_ids':sorted(pass_ids),
                'start_passthrough_prior_prompt_ids':[r['id'] for r in current['passthrough'] if r['remaining']>0],
                'old_decoder_completions_inside':completed_inside,
                'execution_evidence':bool(completed_inside)})
            current=None
    decode=None
    if totals:
        widths={int(w):{'count':int(n),'mean_ms':float(mean),'max_ms':float(mx)}
                for w,n,mean,mx in re.findall(r'\bd(\d+)=(\d+):([\d.]+)/([\d.]+)',totals[-1]['raw'])}
        count=sum(x['count'] for x in widths.values()); rows=sum(w*x['count'] for w,x in widths.items())
        decode={'widths':widths,'dispatch_count':count,'token_rows':rows,
                'mean_width':rows/count if count else None,
                'complete_time_from_rounded_means_ms':sum(x['count']*x['mean_ms'] for x in widths.values()),
                'rounding_bound_ms':0.005*count,'includes_warmup':True}
    bands=[]
    for start in range(1,(expected or max(observed,default=0))+1,concurrency or 16):
        end=min(start+(concurrency or 16)-1,expected or max(observed))
        rows=[r for r in http['measured'] if start<=r['request_index']<=end]
        bands.append({'index_range':[start,end],'scope':'HTTP submission band, not worker/cohort membership',**summarize_subset(rows)})
    longer=[r for r in http['measured'] if r.get('passed') and (r.get('usage',{}).get('completion_tokens',0)>64 or ('overlap' in path.name and r['request_index']==1))]
    overlap=[]
    for old in longer:
        a=old['first_offset_ms']; b=old['completion_offset_ms']
        replacements=[r for r in http['measured'] if r['request_index']!=old['request_index']
                      and r.get('dispatch_offset_ms',-1)>a and r['dispatch_offset_ms']<b]
        active=[r for r in replacements if r.get('first_offset_ms',b+1)<b]
        long_workers=[w for w in workers if int(w['n'])==old['usage']['completion_tokens']
                      and int(w['p'])==old['usage']['prompt_tokens']]
        same_http_tuple=[r for rows in http.values() for r in rows if r.get('usage')==old.get('usage')]
        unique_id=int(long_workers[0]['id']) if len(long_workers)==1 and len(same_http_tuple)==1 else None
        events=[old['dispatch_offset_ms']+x for x in old.get('delta_offsets_ms',[])]
        replacement_begin=min((r['dispatch_offset_ms'] for r in replacements),default=None)
        overlap_gaps=[y-x for x,y in zip(events,events[1:]) if replacement_begin is not None and y>=replacement_begin]
        overlap.append({'http_index':old['request_index'],'first_ms':a,'completed_ms':b,
            'content_gaps':gap_report(old),'new_dispatch_while_visible_ids':[r['request_index'] for r in replacements],
            'new_first_while_visible_ids':[r['request_index'] for r in active],
            'replacement_overlap_content_gap_max_ms':max(overlap_gaps,default=None),
            'unique_worker_id_by_length_and_token_count':unique_id,
            'mapping_note':'Only unique observed (prompt_tokens, completion_tokens) tuple; no arrival-order join. If more than one match, unset.',
            'matching_recovery_passthrough_cycles':[i for i,c in enumerate(cycles) if unique_id in c['start_passthrough_decoder_ids']]})
    return {'run':path.name,'status':status,'source_commit':candidate.get('source_commit'),
            'binary_sha256':candidate.get('binary_sha256'),'arena_log_sha256':hashlib.sha256(raw).hexdigest(),
            'expected_measured':expected,'concurrency':concurrency,'full_measured_summary_present':complete,
            'missing_measured_ids':missing,'duplicate_measured_ids':[k for k,v in collections.Counter(observed).items() if v>1],
            'record_counts':{k:summarize_subset(v) for k,v in http.items()},'published_summaries':summaries,
            'incomplete_rule':'No full-run throughput/SLO score is derived unless all expected records and a final summary exist. Observed subset values remain incomplete.',
            'scheduler_trace':{'request_count':len(workers),'cohort_count':len(cohorts),
                'recovery_event_count':len(recovery),'recovery_cycle_count':len(cycles),'structural_errors':errors,
                'open_snapshot':current,'cycles':cycles,'cohorts':cohorts,'final_decode_totals':decode,
                'worker_requests':workers,'total_lines':totals,
                'observability':'No scheduler lines means unknown, not no work; server tail is printed only during harness cleanup.'},
            'http_submission_bands':bands,'long_stream_overlap':overlap,'http_records':http,
            'limitations':['HTTP and worker clocks/IDs are different; no unconditional id=index+warmup join.',
                           'Nonempty passthrough proves eligibility, not service. A matching completion between start/release proves execution.',
                           'Real content-event gaps are not necessarily per-token intervals.',
                           'Overlap harness changes output length and is not rank-comparable.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=pathlib.Path,default=pathlib.Path('/home/jihwan38/moa-worktrees/round2'))
    parser.add_argument('runs',nargs='+')
    args=parser.parse_args()
    print(json.dumps({name:analyze(args.root/'runs'/name) for name in args.runs},indent=2))

if __name__=='__main__':main()
