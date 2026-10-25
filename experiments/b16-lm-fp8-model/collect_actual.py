#!/usr/bin/env python3
import argparse,ast,collections,hashlib,json,re,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('logs',nargs='+',type=Path);a=ap.parse_args()
def fields(line):
 out={}
 for k,v in re.findall(r'(\w+)=([^ ]+)',line):
  if v in ('true','false'):out[k]=v=='true';continue
  try:out[k]=ast.literal_eval(v)
  except (ValueError,SyntaxError):out[k]=v
 return out
result={'source':'e759c4fcb7dc8e761c310544497cdee175c20549','elf_sha256':'fc1b1c6d50cc0efea97bc7b390be455313a66ad32b314ff292543a4008cd8407','runs':[]}
for p in a.logs:
 lines=p.read_text().splitlines();r=collections.defaultdict(list)
 for line in lines:
  if line.startswith('B16_LM_FP8_'):r[line.split()[0]].append(fields(line))
 logits=r['B16_LM_FP8_LOGITS'];approx=[x for x in logits if not x['strict']];exact=[x for x in logits if x['strict']]
 samples=r['B16_LM_FP8_SELECTION'];sampling={}
 for strict in [False,True]:
  for temp in [0,0.8]:
   for pol in range(3):
    xs=[s for s in samples if s['strict']==strict and s['temperature']==temp and s['policy']==pol]
    sampling[f'strict{strict}_temp{temp}_policy{pol}']={'checks':len(xs),'changed':sum(not s['same']for s in xs)}
 timing=[]
 for step in range(10):
  xs=[s for s in r['B16_LM_FP8_STEP']if s['step']==step]
  assert len(xs)==2
  c,n=xs
  lm=[s for s in r['B16_LM_FP8_STAGE']if s['step']==step and s['stage']=='logits.projection']
  timing.append({'step':step,'width':c['live'],'control_ms':c['model_ms'],'native_ms':n['model_ms'],'lm':lm})
 x={'log':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pass_line':next((x for x in lines if x.startswith('PASS B16 FP8 LM host')),None),
 'startup':r['B16_LM_FP8_STARTUP'],'memory':r['B16_LM_FP8_MEMORY'],
 'approximation':{'rows':len(approx),'values':sum(x['values']for x in approx),'bf16_differences':sum(x['differences']for x in approx),'max_abs':max(x['max_abs']for x in approx),'max_row_relative_l2':max(x['relative_l2']for x in approx),'median_row_relative_l2':statistics.median(x['relative_l2']for x in approx)},
 'strict_unchanged':{'rows':len(exact),'values':sum(x['values']for x in exact),'bf16_differences':sum(x['differences']for x in exact)},
 'sampling':sampling,'sampling_changes':[s for s in samples if not s['same']],
 'kv':r['B16_LM_FP8_CACHE']+r['B16_LM_FP8_PROMPT_CACHE'],'timing_sequential_not_ABBA':timing,'row_records':logits}
 assert x['pass_line'] and len(x['kv'])==18 and x['strict_unchanged']['bf16_differences']==0
 result['runs'].append(x)
 print(p.parent.name,'approx',x['approximation'],'strict',x['strict_unchanged']);print('sampling',sampling)
 print('timing',[(t['step'],t['width'],t['control_ms'],t['native_ms'])for t in timing])
Path(__file__).with_name('actual-result.json').write_text(json.dumps(result,indent=2)+'\n')
