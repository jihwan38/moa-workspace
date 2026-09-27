#!/usr/bin/env python3
"""Revalidate frozen full-O profile evidence and compare same-cluster intervals."""
import argparse, collections, json, re, statistics, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE))
from profile_package import ARTIFACTS,bindings,digest,file_record,map_markers,read,write
from profile_screen import STAGES,parse_diagnostic
sys.path.insert(0,str(ROOT/'moa-dev'));from measure import parse_log
import screen as whole
SELECTED={
 'stage0_table_constant_dma':'5','stage0_table_prepare':'24','input_hbm':'25',
 'stage0_weight':'27','input_trf':'32','stage0_dot':'37','stage1_table_constant_dma':'38',
 'stage1_table_prepare':'41','stage1_weight':'42','stage1_dot':'53',
 'channel_scale_load':'54','channel_scale_vrf':'58','post_rms_weight_load':'59',
 'post_rms_weight_vrf':'65','residual_load':'66','residual_vrf':'72','gather':'81',
 'channel_scale':'89','rms_square':'94','rms_sum':'99','rms_sqrt':'104',
 'rms_scalar_transfer':'107','rms_scalar_vrf':'111','rms_normalize':'115',
 'residual_add':'120','output_store':'123'}

def stats(v):return {'count':len(v),'median':statistics.median(v),'min':min(v),'max':max(v)}
def overlap(a,b):return max(0,min(a['end_from_task'],b['end_from_task'])-max(a['begin_from_task'],b['begin_from_task']))

def observation(obs,names):
 by={r['schedule_index']:r for r in obs['operations']};assert set(by)==set(names.values())
 rows={name:by[idx] for name,idx in names.items()}
 b=lambda n:rows[n]['begin_from_task'];e=lambda n:rows[n]['end_from_task'];d=lambda n:e(n)-b(n)
 w={n+'_interval':d(n) for n in names}
 w.update({
  'task_cycles':obs['task_cycles'],'input_ready_from_task':e('input_trf'),
  'input_ready_to_first_dot':b('stage0_dot')-e('input_trf'),
  'first_weight_begin_from_task':b('stage0_weight'),'last_weight_end_from_task':e('stage1_weight'),
  'first_dot_begin_from_task':b('stage0_dot'),'last_dot_end_from_task':e('stage1_dot'),
  'weight_supply_envelope':e('stage1_weight')-b('stage0_weight'),
  'first_dot_second_weight_overlap':overlap(rows['stage0_dot'],rows['stage1_weight']),
  'first_dot_second_table_dma_overlap':overlap(rows['stage0_dot'],rows['stage1_table_constant_dma']),
  'first_dot_second_table_prepare_overlap':overlap(rows['stage0_dot'],rows['stage1_table_prepare']),
  'W0end_to_W1begin':b('stage1_weight')-e('stage0_weight'),
  'table1DMAend_to_W1begin':b('stage1_weight')-e('stage1_table_constant_dma'),
  'dot0end_to_dot1begin':b('stage1_dot')-e('stage0_dot'),
  'last_dot_end_to_gather_begin':b('gather')-e('stage1_dot'),
  'gather_begin_from_task':b('gather'),'gather_end_from_task':e('gather'),
  'gather_end_to_scale_begin':b('channel_scale')-e('gather'),
  'scale_begin_from_task':b('channel_scale'),
  'last_dot_end_to_task_end':obs['task_cycles']-e('stage1_dot'),
  'scale_begin_to_task_end':obs['task_cycles']-b('channel_scale')})
 readiness=[]
 for i in range(2):
  boundaries={'weight':e(f'stage{i}_weight'),'TRF':e('input_trf'),'table':e(f'stage{i}_table_prepare')}
  if i:boundaries['previous_dot']=e('stage0_dot')
  latest=max(boundaries.values());assert latest<=b(f'stage{i}_dot')
  readiness.append({'stage':i,'boundaries':boundaries,'latest_observed':max(boundaries,key=boundaries.get),'latest_to_dot':b(f'stage{i}_dot')-latest})
 return {'run':obs['run'],'cluster':obs['cluster'],'windows':w,'readiness':readiness}

def analyze(result_path):
 r=read(result_path);assert r['valid'] and r['actual_upload_matches'] and r['correctness_passes']==57
 bp=ROOT/r['build_record']['path'];assert file_record(bp)==r['build_record'];batch=read(bp)
 pp=ROOT/batch['candidate_package']['path'];assert file_record(pp)==batch['candidate_package'];package=read(pp)
 sides,control,origin=bindings();assert package['control']==control and package['origin']==origin
 raw=(ROOT/r['log']).read_bytes();assert digest(raw)==r['log_digest'] and file_record(ROOT/r['archive']['path'])==r['archive']
 archived=whole.archive_files(ROOT/r['archive']['path']);assert archived==r['package_files']
 assert {k:v for k,v in archived.items() if k!='job.yaml'}==batch['package_files']
 sections=re.findall(r'^DIAG_STAGE_BEGIN (\w+)\n(.*?)^DIAG_STAGE_END \1$',raw.decode(),re.M|re.S)
 assert [n for n,_ in sections]==[s['name'] for s in STAGES] and raw.decode().count('DIAG_SCREEN_COMPLETE')==1
 content=dict(sections);out={'status':'complete','result':file_record(result_path),'build':file_record(bp),'package':file_record(pp),'analyzer':file_record(Path(__file__)),'mappings':{},'sides':{},'whole_o':r['whole_o_comparison'],
 'scope':'Each interval is computed on one run/cluster Task clock. Interval includes waiting; overlap is not additive speedup. No physical bandwidth, bank-conflict or counter attribution.'}
 maps={}
 for side in ('control','candidate'):
  e=HERE/'evidence'/batch['build_label'];meta=read(e/f'{side}-ffn.image-profile.json');schedule=read(e/side/'schedule.json')
  mapping=map_markers(schedule,meta)
  mp=ARTIFACTS/batch['build_label']/'diagnostic'/f'{side}-marker-map.json'
  assert digest(mp.read_bytes())==package['diagnostic_files'][mp.name] and read(mp)==mapping
  assert mapping==read(e/f'{side}-marker-map.json')
  assert file_record(e/side/'schedule.json')==package['sides'][side]['schedule']
  assert file_record(e/side/'graph.json')==package['sides'][side]['graph']
  names=dict(SELECTED)
  if side=='candidate':names.update(stage0_weight='32',input_trf='30')
  assert len(mapping['operations'])==26 and {o['schedule']['index'] for o in mapping['operations']}==set(names.values())
  nodes={o['schedule']['index']:o['schedule'] for o in mapping['operations']}
  for i in range(2):
   for dep in (f'stage{i}_weight','input_trf',f'stage{i}_table_prepare'):
    assert set(nodes[names[dep]]['output_tensors'])&set(nodes[names[f'stage{i}_dot']]['input_tensors'])
  assert 'result.to_dm(&mut device.tdma)' in nodes[names['gather']]['description']
  out['mappings'][side]={'count':26,'semantic_indices':names,'metadata':file_record(e/f'{side}-ffn.image-profile.json'),'schedule':file_record(e/side/'schedule.json')}
  maps[side]=mapping
 for stage in STAGES:
  text=content[stage['name']];assert re.findall(r'^DIAG_STAGE_EXIT '+stage['name']+r' (\d+)$',text,re.M)==['0']
  saved=r['stages'][stage['name']];assert saved['valid'] and saved['exit_code']==0
  if stage['kind']=='diagnostic':
   parsed=parse_diagnostic(text,stage,maps[stage['side']]);assert all(saved[k]==v for k,v in parsed.items())
  else:assert parse_log(text)==saved['kernels']
 for side in ('control','candidate'):
  obs=r['stages'][side+'_trace']['trace_observations'];assert {(o['run'],o['cluster']) for o in obs}=={(i,j) for i in range(3) for j in range(2)}
  rows=[observation(o,out['mappings'][side]['semantic_indices']) for o in obs]
  agg={n:stats([row['windows'][n] for row in rows]) for n in rows[0]['windows']}
  out['sides'][side]={'observations':rows,'aggregate':agg,'info_cycles':r['stages'][side+'_info']['cycles'],'trace_cycles':r['stages'][side+'_trace']['cycles']}
 out['window_median_changes']={n:{'control':out['sides']['control']['aggregate'][n]['median'],'candidate':out['sides']['candidate']['aggregate'][n]['median'],'candidate_minus_control':out['sides']['candidate']['aggregate'][n]['median']-out['sides']['control']['aggregate'][n]['median']} for n in out['sides']['control']['aggregate']}
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('--check',action='store_true');a=p.parse_args();v=json.loads(json.dumps(analyze(Path(a.results).resolve())));path=HERE/'profile-analysis.json'
 if a.check:assert read(path)==v
 else:write(path,v)
 print(json.dumps({'whole_o':v['whole_o'],'windows':{n:v['window_median_changes'][n] for n in ['task_cycles','input_hbm_interval','input_ready_from_task','first_weight_begin_from_task','weight_supply_envelope','stage0_weight_interval','stage1_weight_interval','first_dot_begin_from_task','last_dot_end_from_task','gather_interval','gather_end_to_scale_begin','scale_begin_to_task_end']}},indent=2))
