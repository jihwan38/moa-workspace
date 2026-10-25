#!/usr/bin/env python3
"""Reproduce the bounded physical32/C40 runtime evidence; no model/policy changes."""
import argparse, collections, hashlib, json, re
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
lines=(args.run/'arena.log').read_text().splitlines()
def fields(line): return dict(part.split('=',1) for part in line.split() if '=' in part)
def tuples(value):return [tuple(map(int,x.split(':'))) for x in value.split(',') if x]
records=json.loads((args.run/'arena-records.json').read_text())
http=[x['data'] for x in records if x['type']=='STAGE2_PERFORMANCE_RESULT'];summaries=[x['data'] for x in records if x['type']=='STAGE2_PERFORMANCE_SUMMARY']
assert len(http)==40 and sorted(x['request_index'] for x in http)==list(range(1,41))
assert all(x['passed'] and x['finish_reason']=='length' and x['usage']=={'prompt_tokens':256,'completion_tokens':64,'total_tokens':320} for x in http)
assert len(summaries)==1 and summaries[0]['passed']==40 and summaries[0]['failed']==0
assert max(x['dispatch_offset_ms'] for x in http)<min(x['completion_offset_ms'] for x in http)
identity=[fields(l) for l in lines if l.startswith('startup.storage_identity ')];assert len(identity)==1;identity=identity[0]
expected={'physical_rings':32,'admission':32,'query_rows':64,'full_pages':256,'context':8192,'sliding_pool_bytes':134217728,'full_pool_bytes':134217728,'sliding_layers':40,'full_layers':8,'allocated_kv_bytes':12884901888}
assert all(int(identity[k])==v for k,v in expected.items()) and identity['verified_owning_buffers']=='true'
active={};owner_ring={};first={};done={};last_owner={};reuses=[];chunks=[];history=collections.defaultdict(list);cohorts=[];cohort=None;totals=[];requests={};max_active=0

def observe(owner,ring,line,source):
 global max_active
 assert 0<=ring<32 and owner not in done,(line,owner,ring)
 if owner in owner_ring:assert owner_ring[owner]==ring
 else:
  assert ring not in active,(line,source,owner,ring,active.get(ring))
  owner_ring[owner]=ring;first[owner]={'line':line,'source':source};active[ring]=owner
  if ring in last_owner:
   previous=last_owner[ring];assert previous in done and done[previous]<line
   reuses.append({'ring':ring,'previous_owner':previous,'previous_complete_line':done[previous],'next_owner':owner,'next_first_seen_line':line})
 assert active.get(ring)==owner
 max_active=max(max_active,len(active))

for lineno,line in enumerate(lines,1):
 if line.startswith('SCHED_RECOVERY '):
  f=fields(line);members=tuples(f.get('members',''));passthrough=tuples(f.get('passthrough',''))
  for owner,ring,remaining,ready in members+passthrough:observe(owner,ring,lineno,'recovery_'+f['action'])
  if f['action']=='start':
   assert cohort is None;cohort={'start_line':lineno,'ids':[x[0] for x in members],'start':f}
  elif f['action']=='join':
   assert cohort is not None;cohort['ids']=[x[0] for x in members]
  else:
   assert cohort is not None and cohort['ids']==[x[0] for x in members]
   assert f['action']=='ready' and all(x[2:]==(0,1) for x in members)
   cohort.update(release_line=lineno,release=f);cohorts.append(cohort);cohort=None
 elif line.startswith('SCHED_CHUNK '):
  f=fields(line);rows=tuples(f['rows']);pending=tuples(f.get('pending',''))
  for owner,ring,position,remaining,limit in pending:observe(owner,ring,lineno,'pending_snapshot')
  for owner,ring,position,count,remaining in rows:
   observe(owner,ring,lineno,'completed_chunk');assert remaining==256-position
   assert position==sum(x['count'] for x in history[owner]);assert count>0 and position+count<=256
   history[owner].append({'line':lineno,'seq':int(f['seq']),'position':position,'count':count})
  assert len(rows)==int(f['owners']) and len(set(x[0] for x in rows))==len(rows)
  assert sum(x[3] for x in rows)==int(f['tokens'])==512 and int(f['quanta'])==48
  chunks.append({'line':lineno,**{k:v for k,v in f.items() if k!='pending'}})
 elif line.startswith('SCHED_REQ '):
  f=fields(line);owner=int(f['id']);assert owner in owner_ring and owner not in done
  assert int(f['p'])==256 and int(f['n'])==64 and sum(x['count'] for x in history[owner])==256
  ring=owner_ring[owner];assert active.pop(ring)==owner;done[owner]=lineno;last_owner[ring]=owner;requests[owner]=f
 elif line.startswith('SCHED_TOTAL '):totals.append({'line':lineno,'fields':fields(line)})
assert cohort is None and not active
assert sorted(done)==list(range(1,41)) and sorted(history)==list(range(1,41))
assert [int(x['seq']) for x in chunks]==list(range(20)) and sum(int(x['tokens']) for x in chunks)==10240
assert len(reuses)==8 and sorted(x['next_owner'] for x in reuses)==list(range(33,41))
assert max(done[i] for i in range(1,33))<min(first[i]['line'] for i in range(33,41))
assert max_active==32 and len(cohorts)==2 and [len(c['ids']) for c in cohorts]==[32,8]
assert len(totals)==2
histograms=[]
for total in totals:
 d={int(k[1:]):int(v.split(':')[0]) for k,v in total['fields'].items() if re.fullmatch(r'd\d+',k)}
 histograms.append(d)
assert histograms==[{32:63},{8:63,32:63}]
decode_rows=sum(w*n for w,n in histograms[-1].items());assert decode_rows==2520 and decode_rows+len(done)==sum(x['usage']['completion_tokens'] for x in http)==2560
assert sum(int(v.split(':')[0]) for k,v in totals[-1]['fields'].items() if re.fullmatch(r'p\d+',k))==960
candidate=json.loads((args.run/'tested-candidate.json').read_text());assert candidate['source_commit'].startswith('3d497087') and candidate['binary_sha256']=='b1aa162208a29f2688de7926ec76eea5de419fa2ce0d04b553cc25a83fcf1705'
status=json.loads((args.run/'arena-status.json').read_text());assert status['status']=='SUCCEEDED' and status['exit_code']==0
late=sorted(http,key=lambda x:x['completion_offset_ms'])[32:]
result={'run':args.run.name,'job':status['id'],'duration_seconds':status['duration_sec'],'source_commit':candidate['source_commit'],'binary_sha256':candidate['binary_sha256'],'input_hashes':{f:hashlib.sha256((args.run/f).read_bytes()).hexdigest() for f in ['arena.log','arena-records.json','tested-candidate.json']},'runtime_identity':identity,'http':{'complete_unique_requests':40,'protocol_passes':40,'prompt_tokens':10240,'generated_tokens':2560,'dispatch_ms_minmax':[min(x['dispatch_offset_ms'] for x in http),max(x['dispatch_offset_ms'] for x in http)],'earliest_completion_ms':min(x['completion_offset_ms'] for x in http),'all40_dispatched_before_any_completion':True,'late8_client_indices':sorted(x['request_index'] for x in late),'late8_ttft_ms_minmax':[min(x['ttft_ms'] for x in late),max(x['ttft_ms'] for x in late)]},'trace':{'completed_worker_ids':sorted(done),'physical_rings':sorted(set(owner_ring.values())),'max_observed_concurrent_owners':max_active,'owner_to_ring':owner_ring,'completed_chunks':len(chunks),'total_prompt_tokens':10240,'prompt_layer_quanta':960,'reuses':reuses,'all_first32_completed_before_any_new_owner33to40_observed':True,'decode_cumulative_histograms':histograms,'decode_rows':decode_rows,'prompt_first_outputs':40,'token_accounting_exact':True,'cohorts':cohorts,'chunks':chunks,'owner_histories':history,'completion_lines':done,'worker_q_ms_minmax':[min(float(x['q']) for x in requests.values()),max(float(x['q']) for x in requests.values())]},'public_summary':summaries[0],'decision':'PASS physical32 capacity, allocation extent, complete overflow handling and observed slot lifetime. This C40 diagnostic violates4s TTFT and is not a mixed-workload/ranked performance claim.','limits':['FIFO is source/CPU-backed at the Worker mpsc/pending queue; the API does not correlate client request_index to worker IDs, so exact client-index FIFO order is not proved.','Trace observation is a lower bound on admission time; no unobserved absolute admission timestamps are inferred.','Small Worker q excludes time before mpsc dequeue. The extra8 HTTP requests show6.16s TTFT despite Worker q≈2.1ms.','Forced-fresh source-only artifact now proves actual32 behavior. The old a005 stale64-library explanation remains strongly supported provenance inference rather than a controlled compiler-cache reproduction. No c080 serving results are rehabilitated.']}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['trace','public_summary']},indent=2))
print('CHECKS PASS:40 HTTP,40 worker IDs,20 chunks,10240 prompt tokens,2520 decode rows+40 first outputs,8 legal ring reuses,32 maximum owners')
