#!/usr/bin/env python3
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path('/home/jihwan38/moa-ffn-direct-full30');BASE=ROOT/'moa-dev/experiments/candidates';P=BASE/'ffn-direct-full30-partials';E=P/'validation/probe-evidence-v1'
z=importlib.util.spec_from_file_location('prior',BASE/'ffn-independent-direct-copy-repair/independent-audit.py');m=importlib.util.module_from_spec(z);z.loader.exec_module(m);h=m.h;h.ROOT=ROOT
G=E/'graph.json';S=E/'schedule.json';SOURCE=E/'production-mlp.rs';FROZEN=P/'full30-v1/mlp.rs';IMAGE=ROOT/'moa-dev/experiments/artifacts/ffn-direct-full30-partials/e142-full30-probe-v1/kernels/furiosa-opt-gemma4/furiosa_opt_gemma4::stream_probe::upgate.bin'
assert SOURCE.read_bytes()==FROZEN.read_bytes();g=h.Graph(json.loads(G.read_text()));s=json.loads(S.read_text());prefix=m.prefix_and_bitmap(g,s,IMAGE)
partials={};native=[]
for o in g.ops.values():
 d=o['description']
 if o['kind']!='TuExec' or o['context']!='Main' or 'lookup_table:' not in d or 'FfnRow_1=30' not in d:continue
 assert len(o['inputs'])==4
 w,trf,xs,table=o['inputs'];t=o['outputs'][0]
 assert g.tensor(w)['element_type']=='Float4e2m1' and g.tensor(trf)['buffer_type']=='Trf' and g.tensor(trf)['size']==512*3*3840
 assert g.tensor(xs)['buffer_type']=='Vrf' and g.tensor(xs)['size']==512*32
 assert 'FpMul0:' in d and 'FpMul1:' not in d and 'cast_compaction:' not in d and 'LReduce(+) [axes=H]' not in d
 assert h.offsets(h.sequence(d,'fetch_sequencer'))==list(range(30*3840))
 assert g.tensor(t)['element_type']=='Float32' and g.tensor(t)['size']==512*28800
 assert h.offsets(h.sequence(d,'commit_projection'))==list(range(7200)) and 'commit_in_size: 16B (=4EA)' in d
 partials[t]=[];native.append({'operator':o['name'],'partial':t,'bytes_per_slice':28800,'commit_packets':1800,'commit_packet_bytes':16})
assert len(partials)==2
consumers=[]
for o in g.ops.values():
 d=o['description']
 if o['kind']!='TuExec' or o['context']!='Main' or 'FfnRow_1=6' not in d or 'LReduce(+) [axes=H]' not in d:continue
 assert len(o['inputs'])==3
 pv,vrf,dest=o['inputs'];v=g.view(pv);ov=g.view(dest);sp=g.producer(vrf,'TuExec')[1];sv=g.view(sp['inputs'][0])
 assert v['parent'] in partials and sp['context']=='Sub' and g.tensor(vrf)['size']==512*5760
 for view,stride in [(v,240),(sv,240),(ov,1)]:assert (view['stride_elements'],view['axis_extent'],view['window'])==(stride,30,6)
 assert v['start']==sv['start']==ov['start'] and v['start'] in [0,6,12,18,24]
 assert h.offsets(h.sequence(d,'fetch_sequencer'))==list(range(1440))
 assert d.index('FpMul0:')<d.index('LReduce(+) [axes=H]') and 'cast_compaction:' not in d
 assert h.offsets(h.sequence(d,'commit_projection'))==list(range(6)) and 'commit_in_size: 8B (=2EA)' in d
 partials[v['parent']].append(v['start']);consumers.append({'operator':o['name'],'partial':v['parent'],'row_start':v['start'],'byte_bounds':[960*v['start'],960*(v['start']+6)]})
assert len(consumers)==10 and all(sorted(v)==[0,6,12,18,24] for v in partials.values())
ts={int(t['index']):t for t in s['tensors']}
for i in s['instructions']:
 for tid in i['input_tensors']:
  t=ts.get(tid)
  if t and t['buffer_type'] in ['Vrf','Trf']:assert t['lifetime']['begin']<=i['lifetime']['begin']<=i['lifetime']['end']<=t['lifetime']['end']
paths={'source':SOURCE,'frozen_production':FROZEN,'graph':G,'schedule':S,'exact_image':IMAGE,'whole_checker':P/'independent-audit.py','prefix_checker':BASE/'ffn-independent-direct-copy-repair/independent-audit.py'}
out={'passed':True,'inputdigests':{k:h.record(v) for k,v in paths.items()},'prefix':prefix,'native_producers':native,'partial_consumers':consumers,'scope':'Actual production-prefix identity and emitted copy/bitmap/full30F32partial/five-sixrow-consumer/register lifetime contracts. No hardware numerical or performance result inherited.'}
Path('/tmp/moa-e142-probe-review.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS',len(native),'native,',len(consumers),'partial windows, exact3copies/bitmap')
