#!/usr/bin/env python3
"""Strict read-only actual323310 provenance/markers/ABBA replay; no ELF invocation."""
import contextlib,io,json,math,re,statistics
from pathlib import Path
import audit_build
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RAW=HERE/'actual-323310'
SCOPE='complete_group_quant_replica_load_offset_write_read_pack_scale_no_GEMM'
TIME=re.compile(r'REPLICATED_LUT_TIME width=(\d+) case=(\d+) block=(\d+) variant=(\d+) samples=(\d+) us=([0-9.]+) scope=(\S+)')
EXACT=re.compile(r'REPLICATED_LUT_EXACT width=(\d+) case=(\d+) codes=(\d+) packed_bytes=(\d+) scales=(\d+) offsets=(\d+) postcheck=true')
PROFILE=re.compile(r'PROFILE_REPLICATED_LUT width=(\d+)')
PASS=re.compile(r'PASS_REPLICATED_LUT codes=(\d+) packed_bytes=(\d+) group_scales=(\d+) offsets=(\d+) code_scale_original_control_equal=true explicit_offset_abi=true table_bytes=65536 base_bytes=512 no_GEMM_complete_FFN_model_quality_claim')
WIDTHS=(3840,15360)

def sha(p):return audit_build.sha(p)
def load(p):return json.loads(p.read_text())
def main():
 manifest=load(RAW/'manifest.json');assert manifest['job']==323310
 for name,row in manifest['files'].items():
  p=RAW/name;assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],name
 assert sha(RAW/'arena.log')=='e689c6f1ba277a6b0008860b4796be8af89b2ba10d01cf9c3d1e00df460e113b'
 status=load(RAW/'arena-status.json');run=load(RAW/'run.json')
 assert status['id']==323310 and run['name']=='int4-lut-replicas-actual-20261026'
 assert run['commit']=='0488a75a9dc911097a2ba3450546172a2fbf3108'
 assert (RAW/'source.diff').read_bytes()==b'' and run['result']=='command_succeeded' and run['exit_code']==0
 assert set(run['status'].splitlines())=={'?? experiments/int4-grouped-lut-replicas/'+p for p in ['audit_build.py','build-result.json','device-build/','kernel-out-a004/']}
 assert run['command'][run['command'].index('--timeout')+1]=='70'
 assert 'submitted job 323310' in (RAW/'output.log').read_text()
 # Replay complete original-dependency/source/fresh-resource/prior-actual binding.
 captured=io.StringIO()
 with contextlib.redirect_stdout(captured):audit_build.main()
 build=json.loads(captured.getvalue());assert build==load(HERE/'build-result.json')
 hashes=load(RAW/'source-hashes.json');assert hashes==load(HERE/'device-build/source-hashes.json') and len(hashes)==421
 artifact=run['submitted_artifact'];assert artifact['sha256']==build['captured_artifact']['sha256'] and artifact['size']==3210264
 for name in ['path','source_path']:
  p=Path(artifact[name]);assert sha(p)==artifact['sha256'] and p.stat().st_size==artifact['size']
 result={'terminal_status':status,'provenance':{'build_commit':build['source_commit'],'functional_source':'78054937','source_inputs':421,'submitted_artifact':artifact,
  'build_replay_result_sha256':sha(HERE/'build-result.json'),'four_image_bindings':build['four_image_bindings'],'source_manifest_sha256':sha(RAW/'source-hashes.json'),
  'raw_manifest':manifest,'raw_manifest_sha256':sha(RAW/'manifest.json'),'run_record':run},'decision':'TERMINAL_NOT_SUCCESS; NO_CORRECTNESS_OR_SPEED_PASS'}
 log=(RAW/'arena.log').read_text()
 if status['status']!='SUCCEEDED' or status['exit_code']!=0:
  result['partial_log']=log;print(json.dumps(result,indent=2,sort_keys=True));return
 times={};exact={};profiles=[];passes=[];order=[]
 for line in log.splitlines():
  if m:=TIME.fullmatch(line):
   w,c,b,v,n=map(int,m.groups()[:5]);us=float(m[6]);key=(w,c,b)
   assert w in WIDTHS and c in range(5) and b in range(4) and v==[0,1,1,0][b] and n==10
   assert key not in times and math.isfinite(us) and us>0 and m[7]==SCOPE
   times[key]={'variant':v,'samples':n,'us':us};order.append(('time',w,c,b))
  elif m:=EXACT.fullmatch(line):
   w,c,n,p,s,o=map(int,m.groups());assert w in WIDTHS and c in range(5) and (w,c)not in exact
   assert (n,p,s,o)==(512*w,512*w//2,512*w//960,512*w//960)
   exact[w,c]={'codes':n,'packed_bytes':p,'scales':s,'offsets':o};order.append(('exact',w,c))
  elif m:=PROFILE.fullmatch(line):
   w=int(m[1]);assert w in WIDTHS and w not in profiles;profiles.append(w);order.append(('profile',w))
  elif m:=PASS.fullmatch(line):passes.append(tuple(map(int,m.groups())));order.append(('pass',))
  else:raise AssertionError('Unparsed/extra log line: '+line)
 expected=[]
 for w in WIDTHS:
  for c in range(5):
   expected.extend(('time',w,c,b)for b in range(4))
   if c==0:expected.append(('profile',w))
   expected.append(('exact',w,c))
 expected.append(('pass',));assert order==expected
 assert passes==[(49152000,24576000,51200,51200)]
 counts={k:sum(row[k]for row in exact.values())for k in ['codes','packed_bytes','scales','offsets']}
 assert tuple(counts.values())==passes[0]
 records=load(RAW/'arena-records.json');assert records==[]
 rows=[]
 for w in WIDTHS:
  for c in range(5):
   original=[times[w,c,b]['us']for b in [0,3]];candidate=[times[w,c,b]['us']for b in [1,2]]
   om,cm=statistics.mean(original),statistics.mean(candidate)
   rows.append({'width':w,'case':c,'blocks':[{'block':b,**times[w,c,b]}for b in range(4)],
    'original_two_block_means_us':original,'replicated_two_block_means_us':candidate,
    'original_mean_us':om,'replicated_mean_us':cm,'mean_saved_us':om-cm,'mean_gain_percent':100*(1-cm/om),
    'conservative_gain_percent':100*(1-max(candidate)/min(original)),
    'observed_gain_extremes_percent':[100*(1-max(candidate)/min(original)),100*(1-min(candidate)/max(original))],
    'mean_direction':'faster'if cm<om else'slower'})
 regressions=[{'width':x['width'],'case':x['case'],'mean_gain_percent':x['mean_gain_percent']}for x in rows if x['mean_direction']=='slower']
 assert [(x['width'],x['case'])for x in regressions]==[(3840,0),(3840,3)]
 result.update({'decision':'ACTUAL_EXACT_PASS_WITH_MIXED_FIXTURE_TIMING; NO_POLICY_SELECTION_OR_FULL_FFN_PROMOTION',
 'declared_candidate_counts':counts,'marker_counts':{'timings':len(times),'exact':len(exact),'profile':len(profiles),'pass':len(passes),'lines':len(order)},
 'source_bound_additional_checks':{
  'output_comparison_bytes_both_arms_initial_and_post':4*(counts['packed_bytes']+4*counts['scales']+4*counts['offsets']),
  'explicit_permutation_nibbles':2*512*sum(WIDTHS),'explicit_permutation_scales_and_offsets_each':2*512*sum(w//960 for w in WIDTHS),
  'immutable_read_bytes':5*512*sum(WIDTHS)*2+10*(65536+512+4096+32)+(65536+512+4096),
  'completed_kernel_calls':422,'timed_calls':400,'candidate_calls':212,'baseline_calls':210,
  'scope':'Frozen assertions reached before each EXACT/PASS marker; declared candidate counts are not doubled for baseline, repeated reads or permutations. No raw tensor dump is retained.'},
 'timings':rows,'mean_regressions':regressions,
 'timing_scope':{'variants':{'0':'unchanged grouped-store original actual323173 kernel','1':'adjacent16-replica lookup, explicit512B base load'},
  'block_order':[0,1,1,0],'unit':'microseconds','each_value':'mean of10 awaited complete calls, two block means per arm; individual-call distribution/confidence unavailable',
  'includes':SCOPE,'excludes':'host CPU oracle, input/table/base/shifts upload, readbacks; no GEMM, complete FFN or serving',
  'no_aggregate_score':'Five artificial fixtures are not a workload distribution; do not average them into a serving prediction.'},
 'fixtures':{'per_width':5,'permutations':{'3':0,'4':1},'mapping':'source row=(73p+17)%512, group=(3g+1)%G, channel=(73k+127)%960; candidate replica uses destinationp%16',
  'case2':'A=14*2^shift =>normalizedmantissa224,index96. Old everyoffset3072; candidate adjacent16offsets49152+32*(p%16),49152..49632. This is a source-backed address correlation, not proof of memory-bank/cache contention.'},
 'profile':{'markers':profiles,'NPU_SPAN_count':0,'arena_records':records,'scope':'No operation spans emitted; cannot attribute measured savings/regressions to gather versus DMA/VRF/scheduling.'},
 'memory':{'old_table_bytes':4096,'new_table_bytes':65536,'replica_base_bytes':512,'extra_retained_bytes':61952},
 'preserved_history':{'retained_axis_no_go':'8afe741d remains; this candidate uses an explicit complete indexed region and real replica input.',
  'baseline_actual323173':'Earlier unmatched timing unchanged; use current same-job ABBA for comparisons.',
  'original_resource_H_L':[29069,67271],'replica_resource_H_L':[29115,67239],'resource_scope':'Compiler spans only, no proportional latency or actual contention inference.'},
 'limits':['Exact codes/scales and both old/new offset ABIs passed on the fixed finite fixtures, including signed ties, extremes, permutations, reuse and all immutable bytes. No quantization rule changed.',
  'Runtime changes vary by fixture: Hcases0/3 regress, Hcase2 improves, Lallfive improve in this one ABBA run. No selective fixture/input dispatch policy is introduced or authorized.',
  'No model activation distribution, GEMM, complete FFN, logits/task quality or serving throughput was measured.',
  'Do not sum or multiply these gains with seven-to-three graph composition or infer contention cause. No model/default policy changes.']})
 print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
