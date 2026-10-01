#!/usr/bin/env python3
"""E151 independently inspect actual raw-down diagnostic graph, without NPU claims."""
import argparse,importlib.util,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;C=HERE.parent;ROOT=C.parents[3]
s=importlib.util.spec_from_file_location('e151_gate',C/'audit.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a);h=a.h

def review():
 B=C/'down-oracle';e=B/'evidence';b=a.load(B/'build.json');assert b['valid'] and b['exact_compile_exit_code']==0 and b['production_source_full_file_byte_identical_prefix']
 src=e/'production-mlp.rs';assert src.read_bytes()==(C/'ingress-v1/mlp.rs').read_bytes()
 g=h.Graph(a.load(e/'graph.json'));whole=h.Graph(a.load(C/'evidence/e151-ingress-v1/graph.json'))
 pred=lambda x:x['kind']=='TuExec'and x['context']=='Main'and'lookup_table:'in x['description']and'LReduce(+) [axes=FfnTerm]'in x['description']
 ns=[x for x in g.ops.values()if pred(x)];ws=[x for x in whole.ops.values()if pred(x)];assert len(ns)==len(ws)==2
 ns.sort(key=lambda n:g.view(n['inputs'][-1])['start']);ws.sort(key=lambda n:whole.view(n['inputs'][-1])['start']);roots=set();producers=[]
 for n,w,off in zip(ns,ws,[0,30]):
  v=g.view(n['inputs'][-1]);assert(v['stride_elements'],v['axis_extent'],v['window'],v['start'])==(120,60,30,off)
  roots.add(g.root_output(v['parent']));assert h.offsets(h.sequence(n['description'],'commit_projection'))==list(range(3600))
  # Full hardware reader/VE/commit description is identical apart from table tensor and operator names.
  norm=lambda d:re.sub(r'T\d+','T',re.sub(r'#O\d+','#O',d))
  assert norm(n['description'])==norm(w['description'])
  assert [g.tensor(t)['shape']for t in n['inputs'][:3]]==[whole.tensor(t)['shape']for t in w['inputs'][:3]]
  producers.append({'native':n['name'],'output_view':v,'partial_byte_bounds':[off*480,(off+30)*480],'whole_hardware_description_equal':True})
 assert len(roots)==1;root=next(iter(roots));assert g.tensor(root)['size']==512*28800
 reads=[]
 for n in g.ops.values():
  if n['kind']!='TuExec' or n['context']!='Main' or 'LReduce(+) [axes=TileL]'not in n['description'] or 'H_1=12'not in n['description']:continue
  v=g.view(n['inputs'][0]);assert g.root_output(v['parent'])==root and(v['stride_elements'],v['axis_extent'],v['window'])==(120,60,12)
  assert h.offsets(h.sequence(n['description'],'fetch_sequencer'))==list(range(1440));reads.append(v['start'])
 assert sorted(reads)==[0,12,24,36,48]
 switches=[o for o in g.ops.values()if o['context']=='Sub'and'InterTranspose'in o['description']];wswitches=[o for o in whole.ops.values()if o['context']=='Sub'and'InterTranspose'in o['description']];assert len(switches)==len(wswitches)==5
 switches.sort(key=lambda o:g.view(o['inputs'][0])['start']);wswitches.sort(key=lambda o:whole.view(o['inputs'][0])['start']);routes=[]
 for i,(o,w)in enumerate(zip(switches,wswitches)):
  v=g.view(o['inputs'][0]);assert(v['stride_elements'],v['axis_extent'],v['window'],v['start'])==(480,15,3,3*i)
  assert norm(o['description'])==norm(w['description']) and g.tensor(o['outputs'][0])['shape']==whole.tensor(w['outputs'][0])['shape']
  assert g.tensor(o['outputs'][0])['size']==512*5760
  assert h.offsets(h.sequence(o['description'],'fetch_sequencer'))==[a3*480+q*120+u for q in range(4)for a3 in range(3)for u in range(120)]
  readers=[x for x in g.ops.values()if o['outputs'][0]in x['inputs']];assert len(readers)==1
  reader=readers[0];assert g.view(reader['inputs'][0])['start']==12*i
  original_reader=[x for x in whole.ops.values()if w['outputs'][0]in x['inputs']];assert len(original_reader)==1 and norm(reader['description'])==norm(original_reader[0]['description'])
  dma=g.producer(v['parent'],'Dma.DtoS')[1];wdma=whole.producer(whole.view(w['inputs'][0])['parent'],'Dma.DtoS')[1];assert norm(dma['description'])==norm(wdma['description'])
  routes.append({'operator':o['name'],'source_view':v,'consumer':reader['name'],'consumer_row_start':12*i,'whole_descriptor_reader_switch_and_consumer_identical':True})
 sc=a.load(e/'schedule.json');ts={int(t['index']):t for t in sc['tensors']};vrfs=[t for t in ts.values()if t['buffer_type']=='Vrf'];consumers=[]
 for i,x in enumerate(vrfs):
  assert 0<=x['address']<x['address']+x['size']<=8192*512
  for y in vrfs[i+1:]:
   assert not(max(x['lifetime']['begin'],y['lifetime']['begin'])<min(x['lifetime']['end'],y['lifetime']['end'])and max(x['address'],y['address'])<min(x['address']+x['size'],y['address']+y['size']))
 for i in sc['instructions']:
  for tid in i['input_tensors']:
   t=ts.get(tid)
   if t and t['buffer_type']in('Trf','Vrf'):assert t['lifetime']['begin']<=i['lifetime']['begin']<=i['lifetime']['end']<=t['lifetime']['end']
  if i['tpe']=='Main' and '.contract_outer' in i['description'] and 'H % 60 = 30' in i['description']:
   rr=[ts[t]for t in i['input_tensors']if t in ts and ts[t]['buffer_type']in('Trf','Vrf')];assert sorted(t['buffer_type']for t in rr)==['Trf','Vrf'];consumers.append({'instruction':i['index'],'register_ids':[t['index']for t in rr]})
 assert len(consumers)==2 and consumers[0]['register_ids']==consumers[1]['register_ids']
 paths=[B/'build.json',e/'graph.json',e/'schedule.json',src,C/'ingress-v1/mlp.rs',C/'evidence/e151-ingress-v1/graph.json',C/'audit.py',Path(__file__)]
 return {'passed':True,'scope':'Actual diagnostic graph/source/register layout only; no NPU arithmetic PASS is inferred.','inputdigests':{str(p.relative_to(ROOT)):a.rec(p)for p in paths},'five_scale_routing':routes,'native_halves':producers,'partial_consumer_starts':sorted(reads),'whole_vrf_buffers_checked':len(vrfs),'shared_down_native_registers':consumers}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=json.loads(json.dumps(review()));f=HERE/'raw-graph-review.json'
 if args.check:assert a.load(f)==r
 else:f.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS E151 raw diagnostic graph only')
