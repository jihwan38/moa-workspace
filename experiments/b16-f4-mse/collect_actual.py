import pathlib,json,re,statistics,hashlib
root=pathlib.Path(__file__).resolve().parents[3]; out=pathlib.Path(__file__).parent
def convert(v):
 if v in ('true','false'):return v=='true'
 try:return int(v)
 except ValueError:
  try:return float(v)
  except ValueError:return v
def fields(line):return {k:convert(v)for k,v in re.findall(r'([\w]+)=([^ ]+)',line.strip())}
def union(spans):
 merged=[]
 for a,b in sorted(spans):
  if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
  else:merged.append([a,b])
 return sum(b-a for a,b in merged)
def parse(path):
 lines=path.read_text().splitlines(); result={'source_log':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'stages':[],'weights':[],'cpu_partials':[],'cpu_final':[],'times':[],'profiles':[]}
 profile=None
 for line in lines:
  if line.startswith('F4_STAGE '):result['stages'].append(fields(line))
  elif line.startswith('F4_WEIGHT '):result['weights'].append(fields(line))
  elif line.startswith('F4_CPU_PARTIAL '):result['cpu_partials'].append(fields(line))
  elif line.startswith('F4_CPU_FINAL '):result['cpu_final'].append(fields(line))
  elif line.startswith('F4_TIME '):result['times'].append(fields(line))
  elif line.startswith('F4_PROFILE '):
   profile=fields(line);profile['spans']=[];result['profiles'].append(profile)
  elif line.startswith('NPU_SPAN ') and profile is not None:profile['spans'].append(json.loads(line[len('NPU_SPAN '):]))
  elif line.startswith('PASS F4_PARTIALSCALE '):result['pass']=fields(line)
 for p in result['profiles']:
  summary={}
  for tid in sorted({s['tid']for s in p['spans']}):
   z=[s for s in p['spans']if s['tid']==tid];summary[str(tid)]={}
   for name in sorted({s['name']for s in z}):
    a=[s for s in z if s['name']==name];summary[str(tid)][name]={'count':len(a),'duration_sum':sum(s['dur']for s in a),'duration_union':union([(s['begin_cycle'],s['end_cycle'])for s in a])}
  p.pop('spans');p['engine_spans_raw_cycles']=summary
 return result
runs=[]
for n in [1252,1253]:
 p=next((root/'runs').glob(f'r{n}*/arena.log'));runs.append(parse(p));(out/f'r{n}-arena.log').write_bytes(p.read_bytes())
old=[]
for n in [1204,1205]:old+=parse(next((root/'runs').glob(f'r{n}*/arena.log')))['stages']
stages=sum((r['stages']for r in runs),[]);times=sum((r['times']for r in runs),[])
assert len(stages)==128 and all(s['intended_bitexact']for s in stages)
assert sum(s['bytes']for s in stages)==18679808
assert all(s['original_byte_diff']==0 for s in stages if s['stage']in(2,3,4))
comparisons=[]
for s in stages:
 if s['stage']==1 and not s['perm'] and s['feedback']==0:
  prior=next(a for a in old if a['case']==s['case']and a['stage']==1 and not a['perm']and a['feedback']==0)
  comparisons.append({'layer':s['layer'],'baseline_power2_relative_l2':prior['original_relative_l2'],'mse_relative_l2':s['original_relative_l2'],'mse_max_abs':s['original_max_abs']})
means={str(mode):statistics.mean(t['complete_us']for t in times if t['candidate']==mode)for mode in [False,True]}
report={'source':'844c738b66ac0a4bca463d0a735b65491af57fe5','runs':runs,'totals':{'intended_exact_bytes':sum(r['pass']['intended_exact_bytes']for r in runs),'immutable_bytes':sum(r['pass']['immutable_bytes']for r in runs),'F64_partial_samples':sum(a['samples']for r in runs for a in r['cpu_partials']),'F64_final_samples':sum(a['samples']for r in runs for a in r['cpu_final']),'complete_us':means,'fraction_lower_time':1-means['True']/means['False']},'stage_names':['projected','residual','norm','qinput','xscale','hidden','qhidden','hscale'],'same_fixture_first_step_residual_old_vs_mse':comparisons,'notes':['All current-E4 comparisons are approximate; intended dense fixed-code plus SAME group scale reference is strict bitwise.','F64 partial and final checks use frozen numeric tolerance, not bit equality.','Actual logs do not count non-power-of-two scales. All chosen MSE scales traverse these unchanged kernels; exhaustive CPU proof separately covers all 25 candidates.','Second-feedback input is candidate-generated and shared across the three arms in each experiment; it differs between old power2 and new MSE experiments. Only first-feedback old-vs-MSE rows are like-for-like.','Timing includes complete postattention/FFN/postnorm one-launch chain, excludes static encoding/upload, readbacks and resets.','Profile cycle sums overlap and are not wall time; do not map DMA ordinals across compiled graphs.']}
(out/'actual-r1252-r1253.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'totals':report['totals'],'residual':comparisons,'F64_partial_max_abs':max(a['max_abs']for r in runs for a in r['cpu_partials']),'F64_final_max_abs':max(a['max_abs']for r in runs for a in r['cpu_final'])},indent=2))
