import json,re,hashlib,importlib.util
from pathlib import Path
r=Path('/home/jihwan38/moa-qkv-native-local');p=r/'moa-dev/experiments/candidates/qkv-native-consumer-local';e=p/'evidence/e138-native-v2';e1=p/'evidence/e138-native-v1'
hp=r/'moa-dev/experiments/candidates/ffn-native-down-weighted-residual/verify_graph.py';z=importlib.util.spec_from_file_location('h',hp);h=importlib.util.module_from_spec(z);z.loader.exec_module(h)
g=h.Graph(json.loads((e/'graph.json').read_text()));old=h.Graph(json.loads((e1/'graph.json').read_text()));ops={int(x['name'].split('#O')[-1]):x for x in g.ops.values()};oo={int(x['name'].split('#O')[-1]):x for x in old.ops.values()}
M=4194304
def sl(a,x):return a['pe']*64+a['dmn']*32+a['slice']+x//M
seen=set();seed=set()
for d in h.dma_descriptors(ops[8]['description']).values():
 a,b=d['src'],d['dst'];sa,da=a['address'],b['address'];aa=list(h.intervals(sa['in_slice'],a['dims']));bb=list(h.intervals(da['in_slice'],b['dims']));assert len(aa)==len(bb)
 for (lo,hi),(dl,dh) in zip(aa,bb):
  ss,ds=sl(sa,lo),sl(da,dl);q=ds%32
  assert sa['cluster']==da['cluster'] and 0<=ds<256 and q<4
  assert lo%M==0 and hi-lo==dh-dl==960 and dl%M in (0,960)
  assert ss==2*q+dl%M//960
  seen.add((da['cluster'],ds,dl%M));seed.add((da['cluster'],ds))
assert len(seen)==128 and len(seed)==64
sw=ops[10];assert sw['inputs'][0]==ops[8]['outputs'][0] and sw['context']=='Main'
bitmap=[int(x) for x in re.findall(r'SliceBitmap \[(\d+)\]',sw['description'])];assert bitmap==[32*(d//32)+d%4 for d in range(256)]
assert 'ring_size: 32 }, bf16' in sw['description'] and 'cast_compaction:' not in sw['description']
assert h.offsets(h.sequence(sw['description'],'fetch_sequencer'))==list(range(960));assert h.offsets(h.sequence(sw['description'],'commit_projection'))==list(range(960))
assert 'commit_in_size: 32B (=16EA)' in sw['description']
reshape=ops[11];assert reshape['kind']=='LowLevelReshape' and reshape['inputs']==sw['outputs']
assert g.mappings(reshape['outputs'][0])==old.mappings(oo[8]['outputs'][0]);assert g.tensor(reshape['outputs'][0])['size']==g.tensor(sw['outputs'][0])['size']==983040
for c in range(2):
 for a in range(8):
  for b in range(8):
   for q in range(4):
    ds=32*a+4*b+q;assert (c,bitmap[ds]) in seed and ds==4*(8*a+b)+q
# Source only import and input staging changed; whole downstream math frozen.
s1=(p/'native-v1/projection.rs').read_text();s2=(p/'native-v2/projection.rs').read_text();mark='    let (packed, scale, coefficients) = prepare_qkv_terms(device, &x);'
assert s1[s1.index(mark):]==s2[s2.index(mark):]
f='pub(crate) fn project_qkv_native('
assert s1[:s1.index(f)].replace('Ds, Dummy2, Gs','Ds, Dummy2, Dummy256, Gs')==s2[:s2.index(f)]
assert (p/'native-v1/ops.rs').read_bytes()==(p/'native-v2/ops.rs').read_bytes()
# Same emitted core descriptions and IO mappings after inserted ops; no source-only inheritance.
matched=[]
def clean(d):return re.sub(r'#O\d+','#O',d)
for n in [9,10,11,12,67,70,71,72,73,75,76,80,84,86,88,89,93,97,99,101,102,106,110,112]:
 x=oo[n];matches=[]
 for y in ops.values():
  if x['kind']!=y['kind'] or clean(x['description'])!=clean(y['description']):continue
  if len(x['inputs'])!=len(y['inputs']) or len(x['outputs'])!=len(y['outputs']):continue
  if all((old.tensor(a)['size'],old.tensor(a)['element_type'],old.tensor(a)['shape'])==(g.tensor(b)['size'],g.tensor(b)['element_type'],g.tensor(b)['shape']) for a,b in zip(x['inputs']+x['outputs'],y['inputs']+y['outputs'])):matches.append(y['name'])
 assert matches,(n,'missing actual matching op');matched.append({'v1':x['name'],'v2':matches})
s=json.loads((e/'schedule.json').read_text());ts={int(t['index']):t for t in s['tensors']};lifes=[]
for i in s['instructions']:
 if i['tpe']!='Main' or '.contract_outer::' not in i['description']:continue
 regs=[ts[t] for t in i['input_tensors'] if ts[t]['buffer_type'] in ['Trf','Vrf']];assert sorted(t['buffer_type'] for t in regs)==['Trf','Vrf']
 for t in regs:assert t['lifetime']['begin']<=i['lifetime']['begin']<=i['lifetime']['end']<=t['lifetime']['end']
 lifes.append({'index':i['index'],'lifetime':i['lifetime'],'registers':regs})
assert len(lifes)==3 and len({tuple(t['index'] for t in x['registers']) for x in lifes})==1
src=p/'native-v2/projection.rs';os=p/'native-v2/ops.rs';assert src.read_bytes()==(e/'src/device/sliding/projection.rs').read_bytes();assert os.read_bytes()==(e/'src/ops.rs').read_bytes()
def rec(f):b=f.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
out={'passed':True,'scope':'Actual changed seed DMA, emitted Switch bitmap, wire-order relabel, inherited core shape/description, source identity and shared-register lifetime review. No NPU arithmetic/liveness/performance guarantee.','inputdigests':{str(f.relative_to(r)):rec(f) for f in [src,os,e/'graph.json',e/'schedule.json',p/'native-v1/projection.rs',p/'native-v1/ops.rs',e1/'graph.json']},'seed':{'actual_packets':128,'packet_bytes':960,'initialized_anchor_slices':64,'logical_payload_bytes':122880,'destination':'same cluster, slice32*a+q, local byte960*j, q<4,j<2; source slice2*q+j local0'},'broadcast':{'bitmap_entries':256,'formula':'dst32*a+4*b+q reads source32*a+q in same cluster','type':'BF16, no conversion','each_consumer_elements':960,'destination_payload_bytes':983040,'config_hbm_dma_bytes':16384},'relabel':'32*a+4*b+q ==4*g+q, g=8*a+b; full allocation, same K shard/wire order','inherited_v1_graph_core_ops_exact_descriptions_and_shapes':matched,'schedule_register_lifetimes':lifes,'remaining_scope':'Whole and numerical boundary must bind v2 image; prior v1 PASS does not prove v2 runtime.'}
Path('/tmp/moa-e138-review-graph-v2.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS /tmp/moa-e138-review-graph-v2.json')
