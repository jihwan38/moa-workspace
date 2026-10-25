#!/usr/bin/env python3
"""Collected-only c078/c079 attribution, reusing the reviewed join-aware reader."""
import collections, hashlib, importlib.util, json, re
from pathlib import Path
ROOT=Path('/home/jihwan38/moa-worktrees/round2'); HERE=Path(__file__).parent
reader=ROOT/'prefill-prefix512/experiments/late-admission-overlap/analyze.py'
spec=importlib.util.spec_from_file_location('joined',reader);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
RUNS=['r1152-c078-c16-warm-r128-submit','r1173-c079-c16-warm-r128-submit','r1150-c078-overlap-r96-submit','r1181-c079-overlap-r96-submit','r1180-c079-c16-warm-r144-submit']

def parse(line):
 return {k:(int(n),float(mean)) for k,n,mean in re.findall(r'\b((?:d|pc|prep|layer|final)\d+)=(\d+):([\d.]+)/',line)}

def account(name):
 p=ROOT/'runs'/name;r=mod.analyze(p);t=r['scheduler_trace'];warm=r['record_counts']['warmup']['record_count'];n=r['expected_measured'];assert warm==16
 assert r['status']['status']=='SUCCEEDED' and r['status']['exit_code']==0
 assert r['full_measured_summary_present'] and not r['missing_measured_ids'] and not r['duplicate_measured_ids']
 assert not t['structural_errors'] and t['open_snapshot'] is None
 before=[w for w in t['worker_requests'] if w['line']<t['total_lines'][0]['line']]
 assert len(before)==warm and sorted(int(w['id']) for w in before)==list(range(1,warm+1))
 assert len(t['worker_requests'])==warm+n and sorted(int(w['id']) for w in t['worker_requests'])==list(range(1,warm+n+1))
 for phase,rows in r['http_records'].items():
  for row in rows:
   assert row['passed'] and row['finish_reason']=='length' and not row.get('error')
   ds=row.get('delta_offsets_ms')
   if ds is not None:
    assert ds and ds==sorted(ds) and len(ds)==row['usage']['completion_tokens']
    assert ds[-1]<=row['latency_ms']+.01
   elif 'overlap' in name: raise AssertionError('Overlap event record missing')
 a,b=map(lambda x:parse(x['raw']),(t['total_lines'][0],t['total_lines'][-1]));groups={};widths={}
 for prefix in ['d','pc','prep','layer','final']:
  aa=[v for k,v in a.items() if re.fullmatch(prefix+r'\d+',k)];bb=[v for k,v in b.items() if re.fullmatch(prefix+r'\d+',k)]
  na,nb=sum(x[0] for x in aa),sum(x[0] for x in bb)
  groups[prefix]=dict(count=nb-na,cost_ms=sum(n*m for n,m in bb)-sum(n*m for n,m in aa),rounding_bound_ms=.005*(na+nb))
 for k,(nb,mb) in b.items():
  if re.fullmatch(r'd\d+',k):
   na,ma=a.get(k,(0,0));count=nb-na
   if count: widths[int(k[1:])]=dict(count=count,cost_ms=nb*mb-na*ma,mean_ms=(nb*mb-na*ma)/count)
 rows=sum(w*v['count'] for w,v in widths.items());expected_rows=sum(row['usage']['completion_tokens']-1 for row in r['http_records']['measured']);assert rows==expected_rows
 assert groups['layer']['count']==48*groups['prep']['count']
 own=lambda s:{int(w):int(n) for w,n in re.findall(r'\bp(\d+)=(\d+)\b',s)}
 oa,ob=map(lambda z:own(z['raw']),(t['total_lines'][0],t['total_lines'][-1]));owners={w:n-oa.get(w,0) for w,n in ob.items()};assert all(n%48==0 for n in owners.values())
 manifest=json.load(open(p/'tested-candidate.json'))
 return dict(source_commit=r['source_commit'],binary_sha256=r['binary_sha256'],arena_job=r['status']['id'],arena_duration_seconds=r['status']['duration_sec'],log_sha256=r['arena_log_sha256'],measured=n,warm=warm,all_pass=True,all_length_finishes=True,complete_delta_usage=True if 'overlap' in name else None,unique_worker_completions=warm+n,groups=groups,widths=widths,decode_rows=rows,mean_decode_width=rows/groups['d']['count'],prompt_owner_chunks={w:n//48 for w,n in owners.items()},metrics=r['published_summaries']['measured']['metrics'],over4s=r['record_counts']['measured']['over4s_count'],workload=sorted([list(k)+[count] for k,count in collections.Counter((x['suite'],x['usage']['prompt_tokens'],x['usage']['completion_tokens']) for x in r['http_records']['measured']).items()]),cohorts=[dict(initial=len(c['start']['members']),final=len(c['release']['members']),joins=len(c['joins']),reason=c['release']['action'],remaining=c['remaining_prompt_tokens'],old_decoder_completions=[int(w['id']) for w in c['old_decoder_completions_inside']]) for c in t['cycles']],join_count=t['join_count'],unique_participants=t['unique_participant_count'],long_streams=r['long_stream_overlap'],source_hashes=manifest['source_hashes'])

def compare(a,b):
 assert a['workload']==b['workload']
 d={k:dict(count=b['groups'][k]['count']-a['groups'][k]['count'],cost_ms=b['groups'][k]['cost_ms']-a['groups'][k]['cost_ms'],rounding_bound_ms=a['groups'][k]['rounding_bound_ms']+b['groups'][k]['rounding_bound_ms']) for k in a['groups']}
 wa,wb=a['widths'],b['widths'];common=set(wa)&set(wb);new=set(wb)-set(wa);gone=set(wa)-set(wb)
 mix=sum((wb.get(w,{'count':0})['count']-x['count'])*x['mean_ms'] for w,x in wa.items());within=sum(wb[w]['count']*(wb[w]['mean_ms']-wa[w]['mean_ms']) for w in common);newcost=sum(wb[w]['cost_ms'] for w in new)
 assert abs(mix+within+newcost-d['d']['cost_ms'])<.00001
 window=1000*(b['metrics']['measurement_window_seconds']-a['metrics']['measurement_window_seconds']);dispatch=d['d']['cost_ms']+d['pc']['cost_ms'];bound=d['d']['rounding_bound_ms']+d['pc']['rounding_bound_ms']
 changed=sorted(k for k in a['source_hashes']|b['source_hashes'] if a['source_hashes'].get(k)!=b['source_hashes'].get(k))
 for k in ['src/host/worker.rs','src/host/startup_cohort.rs','src/host/generate.rs','src/api/handlers.rs']:assert k not in changed,k
 return dict(deltas=d,window_delta_ms=window,dispatch_delta_ms=dispatch,dispatch_rounding_bound_ms=bound,unassigned_ms=window-dispatch,decode_width_composition_ms=mix,within_width_ms=within,new_width_cost_ms=newcost,new_widths=sorted(new),changed_source=changed,tps_change_fraction=b['metrics']['generated_tokens_per_second']/a['metrics']['generated_tokens_per_second']-1)

out={'reader_sha256':hashlib.sha256(reader.read_bytes()).hexdigest(),'runs':{n:account(n) for n in RUNS}}
out['comparisons']={label:compare(out['runs'][a],out['runs'][b]) for label,a,b in [('C16',RUNS[0],RUNS[1]),('overlap',RUNS[2],RUNS[3])]}
checks={}
for label,roots in [('C16',('c078-c16-warm-r128','c079-c16-warm-r128')),('overlap',('c078-c16-overlap-r96','c079-c16-overlap-r96'))]:
 checks[label]={}
 for rel in ['scripts/stage2/performance.py','scripts/stage2/performance_test.sh','scripts/stage2/entrypoint.sh','scripts/stage2/scoring.py']:
  hs=[hashlib.sha256((ROOT/'evaluations'/x/rel).read_bytes()).hexdigest() for x in roots];assert hs[0]==hs[1],(label,rel);checks[label][rel]=hs[0]
out['matched_client_hashes']=checks
binary=out['runs'][RUNS[1]]['binary_sha256'];quality=[]
for num in range(1174,1180):
 p=next((ROOT/'runs').glob('r'+str(num)+'-c079-quality-*'));s=json.load(open(p/'arena-status.json'));m=json.load(open(p/'tested-candidate.json'));recs=json.load(open(p/'arena-records.json'));assert s['status']=='SUCCEEDED' and s['exit_code']==0 and m['binary_sha256']==binary
 rows=[r['data'] for r in recs if r['type']=='STAGE2_RESULT'];sums=[r['data'] for r in recs if r['type']=='STAGE2_SUMMARY'];assert len(sums)==1 and sums[0]['failed']==0 and sums[0]['passed']==sums[0]['total']==len(rows) and all(x['passed'] for x in rows)
 quality.append(dict(run=p.name,job=s['id'],summary=sums[0],same_binary=True))
out['quality']=quality
long=out['runs'][RUNS[4]]; step16=long['widths'][16]['mean_ms']; tps=long['metrics']['generated_tokens_per_second']; cycle16=16*64/tps*1000; prompt16=cycle16-63*step16; target17=17*64/tps*1000
out['c17_screen']={'scope':'Illustrative steady-cohort bound, not an executed C17 prediction; historical R32 model time excludes selection.', 'current_step16_ms':step16,'observed_cycle16_equivalent_ms':cycle16,'inferred_nondecode_per_cohort_ms':prompt16,'step17_break_even_same_prompt_ms':(target17-prompt16)/63,'step17_break_even_prompt_scaled_17over16_ms':(target17-prompt16*17/16)/63,'historical_warm_r899_step17_model_ms':[64.144,64.189,64.179,64.319,64.371],'later_r1077_step17_model_ms':64.785,'optimistic_64ms_step17_with_scaled_prompt_tps':17*64/((prompt16*17/16+63*64)/1000),'decision':'No C17 run justified by existing width-cliff evidence.'}
for label,(aa,bb) in {'C16':(RUNS[0],RUNS[1]),'overlap':(RUNS[2],RUNS[3])}.items():
 a,b=out['runs'][aa],out['runs'][bb]
 out['comparisons'][label]['prompt_phase_decomposition']={}
 for st in ['pc','prep','layer','final']:
  x,y=a['groups'][st],b['groups'][st];am=x['cost_ms']/x['count'];bm=y['cost_ms']/y['count']
  out['comparisons'][label]['prompt_phase_decomposition'][st]={'control_mean_ms':am,'candidate_mean_ms':bm,'count_cost_ms':(y['count']-x['count'])*am,'within_call_cost_ms':y['count']*(bm-am)}

# Full manifests are retained in run directories; remove duplicate large maps from report.
for r in out['runs'].values():r.pop('source_hashes')
(HERE/'result.json').write_text(json.dumps(out,indent=2)+'\n')
for n,r in out['runs'].items():print(n,{'TPS':r['metrics']['generated_tokens_per_second'],'widths':{k:v['count'] for k,v in r['widths'].items()},'rows':r['decode_rows'],'groups':r['groups'],'cohorts':r['cohorts']})
print('COMPARISONS',json.dumps(out['comparisons'],indent=2));print('QUALITY',quality)
