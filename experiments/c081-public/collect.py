#!/usr/bin/env python3
"""Verify the one frozen c081 binary's public quality and completed HTTP runs."""
from pathlib import Path
import json, re, hashlib, datetime
root=Path(__file__).resolve().parents[3]
candidate='c081-physical32-clean'
m=json.loads((root/'candidates'/candidate/'candidate.json').read_text())
expected=m['binary_sha256']
result={'candidate':candidate,'source_commit':m['source_commit'],'binary_sha256':expected,'build_run':m['build_run'],'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_tests':131,'model_evidence':'physical-pool32/experiments/physical-pool32 matched32vs64 diagnostics; physical-pool32-export/experiments/physical-pool32-export/runtime-queue40.json','scope':'Exact current c079 math with32 physical sliding slots and FIFO overflow. Verified actual owning buffers and admission capacity; public quality does not prove private quality.','quality':[],'performance':[]}
def read_run(name):
    run=root/'runs'/name
    records=json.loads((run/'arena-records.json').read_text())
    status=json.loads((run/'arena-status.json').read_text())
    assert status['status']=='SUCCEEDED'
    assert json.loads((run/'tested-candidate.json').read_text())['binary_sha256']==expected
    jobs=set(re.findall(r'submitted job (\d+)',(run/'output.log').read_text()))
    assert len(jobs)==1
    return run,records,status,int(jobs.pop())
for x in json.loads((root/'state/c081-quality-plan.json').read_text()):
    run,records,status,job=read_run(x['name'])
    individual=[r['data']for r in records if r['type']=='STAGE2_RESULT']
    summaries=[r['data']for r in records if r['type']=='STAGE2_SUMMARY']
    assert individual and all(i['passed'] is True for i in individual)
    assert sum(s['total']for s in summaries)==len(individual) and all(s['failed']==0 for s in summaries)
    sched=[l for l in (run/'arena.log').read_text().splitlines()if l.startswith('SCHED_TOTAL')]
    assert sched
    snapshots=[{int(w):{'calls':int(n),'mean_ms':float(avg),'max_ms':float(mx)} for w,n,avg,mx in re.findall(r'\bd(\d+)=(\d+):([\d.]+)/([\d.]+)',line)}for line in sched]
    for previous,current in zip(snapshots,snapshots[1:]):
        assert all(current.get(w,{}).get('calls',0)>=v['calls']for w,v in previous.items())
    widths=snapshots[-1]
    result['quality'].append({'run':x['name'],'job':job,'status':status['status'],'duration_sec':status['duration_sec'],'individual_passed':len(individual),'summaries':summaries,'final_cumulative_decode_widths':widths,'b16_calls':sum(v['calls']for w,v in widths.items()if 2<=w<=16),'log_sha256':hashlib.sha256((run/'arena.log').read_bytes()).hexdigest()})
assert sum(x['individual_passed']for x in result['quality'])==124
for name,baseline in [('r1274-c081-c16-warm-r144-submit','r1180-c079-c16-warm-r144-submit')]:
    run,records,status,job=read_run(name)
    rs=[x['data']for x in records if x['type']=='STAGE2_PERFORMANCE_RESULT']
    warm=[x['data']for x in records if x['type']=='STAGE2_PERFORMANCE_WARMUP_RESULT']
    ss=[x['data']for x in records if x['type']=='STAGE2_PERFORMANCE_SUMMARY']
    assert len(ss)==1
    s=ss[0]
    assert len(rs)==s['total'] and len(warm)==s['concurrency'] and all(x['passed']for x in rs+warm)
    assert len({x['request_index']for x in rs})==len(rs)
    old=[x['data']for x in json.loads((root/'runs'/baseline/'arena-records.json').read_text())if x['type']=='STAGE2_PERFORMANCE_SUMMARY'][0]
    done=re.findall(r'^SCHED_REQ id=(\d+)',(run/'arena.log').read_text(),re.M)
    assert len(done)==len(set(done))==len(rs)+len(warm)
    result['performance'].append({'run':name,'job':job,'status':status['status'],'duration_sec':status['duration_sec'],'warmup':len(warm),'summary':s,'unique_worker_completions':len(done),'individual_ttft_over4s':sum(x['ttft_ms']>4000 for x in rs),'baseline_run':baseline,'baseline_tps':old['metrics']['generated_tokens_per_second'],'observed_relative_tps_gain':s['metrics']['generated_tokens_per_second']/old['metrics']['generated_tokens_per_second']-1,'log_sha256':hashlib.sha256((run/'arena.log').read_bytes()).hexdigest(),'limit':'One matched-size run, not official full240/private quality; no measured speed gain claimed for memory capacity change.'})
identity_log=(root/'runs/r1263-c081-c40-queue-submit/arena.log').read_text()
identity=next(line for line in identity_log.splitlines()if line.startswith('startup.storage_identity'))
for field in ['physical_rings=32','admission=32','query_rows=64','full_pages=256','context=8192','allocated_kv_bytes=12884901888','verified_owning_buffers=true']:
    assert field in identity
assert m['artifact_captured_at_build'] and m['fresh_package']['contents_changed'] is False
result['runtime_capacity_identity']={'run':'r1263-c081-c40-queue-submit','line':identity,'log_sha256':hashlib.sha256(identity_log.encode()).hexdigest(),'queue_proof':'physical-pool32-export/experiments/physical-pool32-export/runtime-queue40.json','limit':'C40 overflow correctness passed; C40 latency invalid, not ranked performance'}
p=root/'r64-server/evidence/c081-http-validation.json' 
p.write_text(json.dumps(result,indent=2)+'\n')
print('verified 124 quality checks',[(x['summary']['concurrency'],x['summary']['metrics']['generated_tokens_per_second'],x['observed_relative_tps_gain'])for x in result['performance']])
print('B16 quality calls',[(x['run'],x['b16_calls'])for x in result['quality']])
