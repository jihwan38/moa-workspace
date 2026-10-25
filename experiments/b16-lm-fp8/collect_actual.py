#!/usr/bin/env python3
"""Collect intended-algorithm validation and OLD-LM approximation separately."""
import argparse,ast,collections,hashlib,json,re,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('log',type=Path);a=ap.parse_args()
def fields(line):
 out={}
 for k,v in re.findall(r'(\w+)=(\[[^\]]*\]|[^ ]+)',line):
  try:out[k]=ast.literal_eval(v)
  except (ValueError,SyntaxError):out[k]=v
 return out
lines=a.log.read_text().splitlines();records=collections.defaultdict(list);profiles=collections.defaultdict(list);marker=None
for line in lines:
 if line.startswith('LM_'):
  records[line.split(' ',1)[0]].append(fields(line))
 elif line.startswith('PROFILE_VARIANT '):marker=int(line.split()[1])
 elif line.startswith('NPU_SPAN ')and marker is not None:profiles[marker].append(json.loads(line.split(' ',1)[1]))
result={'source':'3a877b1e9fa807d68515d2231ffc37c2a60e6f48','elf_sha256':'fbac12cf562b1c65c3586315f42f6985fefa828a4134916161473f1ad707abc5',
 'log':str(a.log.resolve()),'log_sha256':hashlib.sha256(a.log.read_bytes()).hexdigest(),
 'pass_line':next((s for s in lines if s.startswith('PASS intended quantization')),None),
 'records':dict(records),'timing':[],'trace':{},'limits':[
 'Micro only. No model, quality-suite or serving-throughput claim.',
 'Candidate includes the separate inputquant and complete LM; both use existing finalnorm output.',
 'Startup, CPU/readback checks and selection comparison are outside both ABBA arms.',
 'Old BF16 logit/selection differences describe approximation and do not pass an exactness gate.',
 'F64 verifies sampled intended algorithm, not F32 reduction bit identity.',
 'Shadow allocation delta is standalone; current full-model headroom remains untested.',
 'Device span durations overlap and use raw counter units; no clock or DMA ordinal inference.']}
for r in records['LM_ABBA']:
 c=statistics.mean(r['control_us']);n=statistics.mean(r['candidate_us'])
 result['timing'].append({**r,'control_mean_us':c,'candidate_mean_us':n,'saved_us':c-n,'fraction_reduction':1-n/c})
for variant,spans in profiles.items():
 result['trace'][variant]={}
 for lane in sorted({s['tid']for s in spans}):
  ss=[s for s in spans if s['tid']==lane];eng={}
  for name in sorted({s['name']for s in ss}):
   x=[s for s in ss if s['name']==name];eng[name]={'count':len(x),'sum_raw':sum(s['dur']for s in x)}
  result['trace'][variant][lane]={'engines':eng,'tasks':[s for s in ss if s['name']=='Task'],
   'largest_dma':sorted((s['dur']for s in ss if s['name']=='DMA'),reverse=True)[:20]}
p=Path(__file__).with_name('actual-result.json');p.write_text(json.dumps(result,indent=2)+'\n')
print('PASS intended',bool(result['pass_line']));print('summary',records['LM_APPROX_RESULT']);print('startup',records['LM_STARTUP'])
for r in result['timing']:print(r['case'],r['permutation'],r['control_us'],'->',r['candidate_us'],'reduction',r['fraction_reduction'])
if result['pass_line']:
 assert len(records['LM_ALL_VOCAB'])==6 and sum(x['values']for x in records['LM_ALL_VOCAB'])==25165824
 assert sum(x['selection_checks']for x in records['LM_ALL_VOCAB'])==384
 assert sum(x['tolerance_failures']for x in records['LM_F64'])==0
