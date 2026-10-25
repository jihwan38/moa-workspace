#!/usr/bin/env python3
"""Reproduce the strict r1192 integer ABI evidence, without inferred pack timing."""
from pathlib import Path
import ast, collections, hashlib, json, re, statistics
root=Path('/home/jihwan38/moa-worktrees/round2')
run=root/'runs/r1192-int4-raw-binding-submit'
log=(run/'arena.log').read_text()
rows=[]
for line in log.splitlines():
    if line.startswith('I4_RAW_CASE '):
        r={'case':int(re.search(r'case=(\d+)',line)[1])}
        for name in ['typed_us','raw_us','pack_plus_raw_us']:
            r[name]=ast.literal_eval(re.search(name+r'=(\[[^]]+\])',line)[1])
        rows.append(r)
assert [r['case'] for r in rows]==list(range(8))
assert 'PASS I4_RAW_ABI exact_outputs=4096' in log
assert 'runtime=slot 0 needs 30720 bytes on every device, the buffer has 30712' in log
stats={}
for name in ['typed_us','raw_us','pack_plus_raw_us']:
    vals=[x for r in rows for x in r[name]]
    stats[name]={'min':min(vals),'max':max(vals),'mean':statistics.mean(vals),'median':statistics.median(vals)}
extra=[joint-raw for r in rows for joint,raw in zip(r['pack_plus_raw_us'],r['raw_us'])]
spans=[json.loads(s.split(' ',1)[1]) for s in log.splitlines() if s.startswith('NPU_SPAN ')]
trace={}
for c in [0,1]:
    selected=[s for s in spans if s['tid']==c]
    by_name=collections.defaultdict(list)
    for s in selected:by_name[s['name']].append(s['dur'])
    trace[str(c)]=dict(by_name)
old=json.loads((root/'int4-pack-words/experiments/int4-pack-words/actual-r1185.json').read_text())
record=json.loads((run/'run.json').read_text())
result={
 'status':'PASS integer ABI and ownership/byte contract; no model adoption',
 'job':321147,'run':run.name,'elapsed_seconds_reported_by_root':0.2084,
 'source_commit':'a02a95ff761bf901bd2dd3f9125beee835d44930',
 'submitted_elf':record['submitted_artifact'],
 'log_sha256':hashlib.sha256((run/'arena.log').read_bytes()).hexdigest(),
 'counts':{'known_raw_outputs':4096,'joint_outputs':4096,'device_packed_bytes':245760,'reported_immutable_input_bytes':983040,'fixtures':8},
 'gates':['independent I64 integer dot','normal typed I4 vs raw Buffer','device pack bytes vs independent host encoding','all signed nibble codes/extremes/zero/sparse/random','true paired row/column permutation','immutable input owners and repeated use','short adapter and normal runtime rejection, no output writes'],
 'timings':rows,'timing_summary':stats,
 'joint_minus_raw_us':{'min':min(extra),'max':max(extra),'mean':statistics.mean(extra)},
 'prior_standalone_pack_us':{'mean':old['timing_us_mean'],'range':old['timing_us_range']},
 'timing_source':{'outer_modes':[0,1,2,2,1,0],'inner_samples':8,'mode0':'one awaited typed consumer','mode1':'one awaited raw consumer','mode2':'one awaited device pack, then one awaited raw consumer','readbacks_uploads_inside_timer':False,'profiling_enabled_during_timings':False,'profile_scope':'one separate case2 raw consumer only'},
 'trace_cycles':trace,
 'limits':['Joint minus raw is not an isolated pack measurement: mixed kernel sequence differs from prior standalone same-kernel loop.','No combined pair profile was collected; source proves one pack, but observed excess over separate warm sums is not attributed.','Raw-consumer trace has only consumer input/weight/output DMA, TRF ingress and dot; no producer or extra launch.','Cancellation lifetime source-audited, not stress tested.','No floating-point quantizer, model quality, CE throughput doubling or full-FFN speed proof.']}
Path(__file__).with_name('actual-r1192.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'summary':stats,'joint_minus_raw':result['joint_minus_raw_us'],'trace':trace},indent=2))
