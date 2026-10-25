#!/usr/bin/env python3
import json,re
from pathlib import Path
root=Path('/home/jihwan38/moa-worktrees/round2/runs')
result=[]
for prefix,r,job in [(512,952,319656),(1024,953,319655)]:
 log=root/f'r{r}-b16-attn-output-model{prefix}-submit/arena.log';text=log.read_text();steps={};stages={}
 for line in text.splitlines():
  if line.startswith('B16_ATTN_O_STEP '):
   p,s,n,t=re.search(r'pass=(\d+).*step=(\d+) live=(\d+) model_ms=([\d.]+)',line).groups();steps[(int(p),int(s))]={'live':int(n),'model_ms':float(t)}
  elif line.startswith('B16_ATTN_O_STAGE '):
   p,s,name,c,t=re.search(r'pass=(\d+) step=(\d+) stage=(\S+) calls=(\d+) total_ms=([\d.]+)',line).groups();stages[(int(p),int(s),name)]=float(t)
 pairs=[]
 for s in range(10):
  old=steps[(0,s)];new=steps[(1,s)];row={'step':s,'live':old['live'],'original_ms':old['model_ms'],'candidate_ms':new['model_ms'],'eligible':s not in [0,5,8]}
  if row['eligible']:
   row['original_attention_output_ms']=stages[(0,s,'sliding.attention')]+stages[(0,s,'sliding.output')]
   row['composed_attention_output_ms']=stages[(1,s,'sliding.attention_output_composed')]
  pairs.append(row)
 values=sum(map(int,re.findall(r'B16_ATTN_O_(?:PROMPT_)?CACHE[^\n]*bf16_values=(\d+)',text)))
 result.append({'job':job,'prefix':prefix,'log':str(log),'logits_exact':int(re.search(r'PASS .*checked=(\d+)',text)[1]),'kv_bf16_exact':values,'kv_bytes_exact':values*2,'pairs':pairs,'pass_line':[s for s in text.splitlines() if s.startswith('PASS ')][0]})
print(json.dumps({'runs':result,'total_logits_exact':sum(r['logits_exact'] for r in result),'total_kv_bytes_exact':sum(r['kv_bytes_exact'] for r in result),'scope':'Matched routes, candidate pass follows control; model wall excludes retained-logit read/selection. Scalar/R32/fallback deltas are not fusion gains. First B16 call can include cold effects.'},indent=2))
