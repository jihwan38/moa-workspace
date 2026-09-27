import json,hashlib,importlib.util,re
from pathlib import Path
root=Path('/home/jihwan38/moa-o-native-activation-resident');p=root/'moa-dev/experiments/candidates/o-native-activation-resident';ev=p/'evidence/e137-native-v2'
hp=root/'moa-dev/experiments/candidates/ffn-native-down-weighted-residual/verify_graph.py';sp=importlib.util.spec_from_file_location('helpers',hp);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
g=h.Graph(json.load(open(ev/'graph.json')));ops=g.ops
by={o['name']:o for o in ops.values()}
def named(s):return next(o for o in ops.values() if o['name'].endswith(s))
def packets(o):
 for sides in h.dma_descriptors(o['description']).values():
  a,b=sides['src'],sides['dst'];aa=list(h.intervals(a['address'].get('in_chip',a['address'].get('in_slice')),a['dims']));bb=list(h.intervals(b['address'].get('in_chip',b['address'].get('in_slice')),b['dims']))
  assert len(aa)==len(bb)
  for x,y in zip(aa,bb):yield a,b,x,y

def sl(a,x):return a['pe']*64+a['dmn']*32+a['slice']+x//4194304
# Whole HBM x is read once per duplicated cluster into exactly16 live K owners.
xload=named('#O1');xc={0:[],1:[]};xd=set()
for a,b,(lo,hi),(dl,dh) in packets(xload):
 da=b['address'];c=da['cluster'];q=sl(da,dl);off=dl%4194304
 assert 0<=q<16 and hi-lo==dh-dl==256 and off in (0,256)
 assert lo==q*512+off
 xc[c].append((lo,hi));xd.add((c,q,off))
assert len(xd)==64
for c in xc:h.full_coverage(xc[c],8192,'percluster whole x')
# Actual max reduction retains padded physical256way with16 logical valid shards.
local=named('#O2');gm=named('#O3');scale=named('#O4')
assert local['inputs']==xload['outputs'] and gm['inputs']==local['outputs'] and scale['inputs']==gm['outputs']
assert 'LReduce(max_f) [axes=Qs]' in local['description'] and 'GReduceMaxF [ratio=256]' in gm['description']
assert '7.8125e-3f' in scale['description']
# Activation terms are replicated with K and term coordinates unchanged.
trf=next(o for o in ops.values() if o['kind']=='StoTrf');rep=g.producer(trf['inputs'][0],'Dma.StoS')[1];seen=set()
for a,b,(lo,hi),(dl,dh) in packets(rep):
 sa,da=a['address'],b['address'];ss,ds=sl(sa,lo),sl(da,dl);so,do=lo%4194304,dl%4194304
 assert sa['cluster']==da['cluster'] and 0<=ss<16 and 0<=ds<256 and ds%16==ss
 assert hi-lo==dh-dl==256 and so==do and so in (0,256,512,768)
 seen.add((da['cluster'],ds,do//256))
assert len(seen)==512*4 and g.tensor(trf['outputs'][0])['size']==512*4*256
assert 'TrfMacRows: [OutputTerm_1=4]' in trf['description'] and 'InSlice: [Qs_1=256]' in trf['description']
# Four constant terms are written at literal indices then packed in two8B writes.
cp=named('#O60');assert cp['context']=='Sub'
assert h.sequence(cp['description'],'fetch_sequencer')==(0,[(4,8),(8,1)])
assert h.offsets(h.sequence(cp['description'],'commit_projection'))==[0,1,2,3]
assert 'commit_in_size: 8B (=2EA)' in cp['description']
cr=g.root_output(cp['inputs'][0]);constants=[]
for o in ops.values():
 if o['kind']!='IndexWrite' or g.root_output(o['outputs'][0])!=cr:continue
 idx=g.scalar(o['inputs'][1]);maker=g.producer(o['inputs'][2],'TuExec')[1]
 val=float(re.search(r'FpFma: %l \+ %r \[true\] %r=([^f, ]+)f',maker['description'])[1])
 assert abs(val/[1,2**-4,2**-22,2**-27][idx]-1)<1e-7
 constants.append((idx,val))
assert sorted(i for i,v in constants)==[0,1,2,3]
cb=named('#O63');assert cb['inputs']==cp['outputs'];cseen=set()
for a,b,(lo,hi),(dl,dh) in packets(cb):
 sa,da=a['address'],b['address'];assert sa['cluster']==da['cluster'] and sl(sa,lo)==0
 assert lo%4194304==dl%4194304==0 and hi-lo==dh-dl==16
 cseen.add((da['cluster'],sl(da,dl)))
assert len(cseen)==512
cv=named('#O65');assert cv['inputs']==cb['outputs'] and h.sequence(cv['description'],'fetch_sequencer')==(0,[(4,1)])
assert g.tensor(cv['outputs'][0])['size']==512*32
# Fused native packetTerm reduction precedes K16 and writes exactly60F32/owner/stage.
natives=[o for o in ops.values() if 'LReduce(+) [axes=OutputTerm]' in o['description']];assert len(natives)==2
starts=[]
for o in natives:
 d=o['description'];assert trf['outputs'][0] in o['inputs'] and cv['outputs'][0] in o['inputs']
 assert '[H_1=60, <ve_fp_way>OutputTerm_1=4]' in d
 assert d.index('FpMul0:')<d.index('LReduce(+) [axes=OutputTerm]')<d.index('GReduceAddF [ratio=16]')
 assert 'lookup_table:' not in d and 'cast_compaction:' not in d and 'commit_in_size: 8B (=2EA)' in d
 assert h.offsets(h.sequence(d,'commit_projection'))==list(range(60))
 v=g.view(o['inputs'][-1]);assert (v['stride_elements'],v['window'],v['axis_extent'])==(1,60,120);starts.append(v['start'])
assert sorted(starts)==[0,60]
# One common s from cluster0/slice0 is broadcast into canonical eight output shards.
sd=named('#O62');assert sd['inputs']==scale['outputs'];sdest=set()
for a,b,(lo,hi),(dl,dh) in packets(sd):
 sa,da=a['address'],b['address'];assert sa['cluster']==da['cluster']==0 and sl(sa,lo)==0
 assert lo%4194304==dl%4194304==0 and hi-lo==dh-dl==32
 sdest.add(sl(da,dl))
assert sdest==set(range(8))
sv=next(o for o in ops.values() if o['kind']=='StoVrf' and o['inputs']==sd['outputs'])
epi=named('#O92');assert sv['outputs'][0]==epi['inputs'][1]
assert epi['description'].index('FpMul0:')<epi['description'].index('FpMul1:')<epi['description'].index('output_element_type: bf16')
# Resource lifetimes use only schedule IDs, without joining graph numbers.
s=json.load(open(ev/'schedule.json'));ts={int(t['index']):t for t in s['tensors']};sr=[]
for i in s['instructions']:
 if i['tpe']!='Main' or '.contract_outer::' not in i['description']:continue
 regs=[ts[x] for x in i['input_tensors'] if x in ts and ts[x]['buffer_type'] in ('Trf','Vrf')]
 assert sorted(t['buffer_type'] for t in regs)==['Trf','Vrf']
 for t in regs:assert t['lifetime']['begin']<=i['lifetime']['begin'] and t['lifetime']['end']>=i['lifetime']['end']
 sr.append({'index':i['index'],'lifetime':i['lifetime'],'registers':regs})
assert len(sr)==2
src=p/'native-v2/projection.rs';assert src.read_bytes()==(ev/'src/device/sliding/projection.rs').read_bytes()
def rec(f):
 b=f.read_bytes();return {'path':str(f.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
r={'passed':True,'scope':'Independent read-only source and emitted mapping/order review; production gate owned by kernel agent. No NPU arithmetic or performance claim. Hidden native/VRF SFR indexes and runtime bank arbitration are not established.','inputdigests':{str(f.relative_to(root)):{k:v for k,v in rec(f).items() if k!='path'} for f in [src,ev/'graph.json',ev/'schedule.json']},'quantization':{'valid_shards_per_cluster':16,'clusters_duplicated':2,'hbm_read_bytes_total':16384,'source_hbm_extent':8192,'logical_global_k':4096,'emitted_max_ratio_with_padding':256},'replication':{'term_packets':len(seen),'packet_bytes':256,'destination_k':'256*(slice%16)+u','source_slice':'slice%16','same_cluster':True,'trf_lanes':4,'trf_element_k':256,'trf_bytes_per_slice':1024},'coefficients':{'ordered_graph_printed_constants':sorted(constants),'source_slots_bytes':128,'packed_source_bytes':16,'sub_commit_bytes':8,'commits':2,'broadcast_bytes_per_slice':16,'vrf_payload_bytes_per_slice':32},'native_stages':[{'name':o['name'],'inputs':o['inputs'],'outputs':o['outputs']} for o in natives],'order':'normalized coefficient Mul -> packet Term ReduceAccTree -> K16 GReduceAdd -> compact F32 pair commit/gather -> common s Mul0 -> channel scale Mul1 -> first BF16','canonical_scale_source':'cluster0,slice0 -> cluster0 slices0..7, eight32B copies','schedule_register_lifetimes':sr}
Path('/tmp/moa-e137-independent-review.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS /tmp/moa-e137-independent-review.json')
