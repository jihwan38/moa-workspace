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
  if line.startswith('R32_F4_STAGE '):result['stages'].append(fields(line))
  elif line.startswith('R32_F4_WEIGHT '):result['weights'].append(fields(line))
  elif line.startswith('R32_F4_CPU_PARTIAL '):result['cpu_partials'].append(fields(line))
  elif line.startswith('R32_F4_CPU_FINAL '):result['cpu_final'].append(fields(line))
  elif line.startswith('R32_F4_TIME '):result['times'].append(fields(line))
  elif line.startswith('R32_F4_PROFILE '):
   profile=fields(line);profile['spans']=[];result['profiles'].append(profile)
  elif line.startswith('NPU_SPAN ') and profile is not None:profile['spans'].append(json.loads(line[len('NPU_SPAN '):]))
  elif line.startswith('PASS R32_F4_NUMERICS '):result['pass']=fields(line)
 for p in result['profiles']:
  summary={}
  for tid in sorted({s['tid']for s in p['spans']}):
   z=[s for s in p['spans']if s['tid']==tid];summary[str(tid)]={}
   for name in sorted({s['name']for s in z}):
    a=[s for s in z if s['name']==name];summary[str(tid)][name]={'count':len(a),'duration_sum':sum(s['dur']for s in a),'duration_union':union([(s['begin_cycle'],s['end_cycle'])for s in a])}
  p.pop('spans');p['engine_spans_raw_cycles']=summary
 return result
runs=[]
for n in [1258,1259]:
 p=next((root/'runs').glob(f'r{n}*/arena.log'));runs.append(parse(p));(out/f'r{n}-arena.log').write_bytes(p.read_bytes())
stages=sum((r['stages']for r in runs),[]);times=sum((r['times']for r in runs),[])
assert len(stages)==128 and all(s['intended_bitexact']for s in stages)
assert sum(s['bytes']for s in stages)==49156096
assert all(s['original_byte_diff']==0 for s in stages if s['stage']in(2,3,4))
means={str(mode):statistics.mean(t['complete_us']for t in times if t['candidate']==mode)for mode in [False,True]}
report={'source':'6f6c0de5','runs':runs,'totals':{
 'intended_exact_bytes':sum(r['pass']['intended_exact_bytes']for r in runs),
 'immutable_bytes':sum(r['pass']['immutable_bytes']for r in runs),
 'preserved_projected_tail_bytes':sum(r['pass']['preserved_projected_tail_bytes']for r in runs),
 'F64_partial_samples':sum(a['samples']for r in runs for a in r['cpu_partials']),
 'F64_final_samples':sum(a['samples']for r in runs for a in r['cpu_final']),
 'complete_us':means,'fraction_lower_time':1-means['True']/means['False']},
 'stage_names':['projected','residual','norm','qinput','xscale','hidden','qhidden','hscale'],
 'notes':['Three complete current launches: pre-RMS, four-stage core, post-RMS/residual/gate. Static fit/upload, resets/readbacks/oracles excluded equally.',
 'Strict reference is dense unscaled fixed E4 codes plus SAME group960 scales before local inter-slice sums. Original current E4 comparison is approximate.',
 'R64 normalization still processes all64 rows. The live32 FFN leaves projected rows32..63 exactly intact; full-chain inactive residual/norm tails match current separately.',
 'F64 checks retain the B16 numeric tolerances; equality to the original E4 model/task quality is not claimed.',
 'Profile cycle sums can overlap and do not replace measured complete wall timing.']}
(out/'actual-r1258-r1259.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['totals'],indent=2))
print('F64 maxima',max(a['max_abs']for r in runs for a in r['cpu_partials']),max(a['max_abs']for r in runs for a in r['cpu_final']))
for stage in [0,1,5,6,7]:
 vals=[s['original_relative_l2']for s in stages if s['stage']==stage]
 print('stage',stage,'L2',min(vals),max(vals))
for r in runs:
 for p in r['profiles']:
  x=p['engine_spans_raw_cycles']['0'];print('profile',p['case'],p['candidate'],'Task',x.get('Task'),'DMA',x.get('DMA'),'TU',x.get('Renegade::TuExec'))
