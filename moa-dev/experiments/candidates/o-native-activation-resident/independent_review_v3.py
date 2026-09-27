import json,hashlib,importlib.util,re
from pathlib import Path
root=Path('/home/jihwan38/moa-o-native-activation-resident');p=root/'moa-dev/experiments/candidates/o-native-activation-resident';ev=p/'evidence/e137-native-v3'
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
# Every consumer group receives the same full input; enumerate actual descriptors.
xload=named('#O1');xc={(c,g):[] for c in range(2) for g in range(16)};xd=set()
for a,b,(lo,hi),(dl,dh) in packets(xload):
 da=b['address'];c=da['cluster'];ds=sl(da,dl);gr,q=divmod(ds,16);off=dl%4194304
 assert 0<=ds<256 and hi-lo==dh-dl==256 and off in (0,256)
 assert lo==q*512+off
 xc[c,gr].append((lo,hi));xd.add((c,ds,off))
assert len(xd)==1024
for cg in xc:h.full_coverage(xc[cg],8192,'pergroup whole x')
local=named('#O2');gm=named('#O3');scale=named('#O4')
assert local['inputs']==xload['outputs'] and gm['inputs']==local['outputs'] and scale['inputs']==gm['outputs']
assert 'LReduce(max_f) [axes=Qs]' in local['description'] and 'GReduceMaxF [ratio=16]' in gm['description']
assert '[H_120=16, <global_reduce>Qs_256=16<global_reduce>]' in gm['description']
assert '7.8125e-3f' in scale['description']
# Every q receives its own Hgroup scale from initialized q0, not another K shard.
qscale=named('#O5');assert qscale['inputs']==scale['outputs'];qseen=set()
for a,b,(lo,hi),(dl,dh) in packets(qscale):
 sa,da=a['address'],b['address'];ss,ds=sl(sa,lo),sl(da,dl)
 assert sa['cluster']==da['cluster'] and ss==(ds//16)*16
 assert lo%4194304==dl%4194304==0 and hi-lo==dh-dl==32
 qseen.add((da['cluster'],ds))
assert len(qseen)==512
# Full allocation relabels retain physical masks/strides; neither performs a DMA.
sreshape=named('#O61');assert sreshape['kind']=='LowLevelReshape' and sreshape['inputs']==scale['outputs']
sm=g.mappings(sreshape['inputs'][0]);dm=g.mappings(sreshape['outputs'][0]);want=json.loads(json.dumps(sm))
want['Partitioning']['Pair']['left']={'Broadcast':{'size':16}}
assert dm==want and g.tensor(sreshape['inputs'][0])['size']==g.tensor(sreshape['outputs'][0])['size']==16384
trf=next(o for o in ops.values() if o['kind']=='StoTrf');preshape=g.producer(trf['inputs'][0],'LowLevelReshape')[1]
pm=g.mappings(preshape['inputs'][0]);pd=g.mappings(preshape['outputs'][0]);want=json.loads(json.dumps(pm));want['Cluster']={'Symbol':{'symbol':'Dummy2','size':2}}
assert pd==want and g.tensor(preshape['inputs'][0])['size']==g.tensor(preshape['outputs'][0])['size']==524288
assert g.producer(preshape['inputs'][0])[1]['kind']=='IndexWrite'
assert not [o for o in ops.values() if o['kind']=='Dma.StoS' and any(g.tensor(t)['element_type']=='Float8e4m3' and g.tensor(t)['size']==524288 for t in o['outputs'])]
assert g.tensor(trf['outputs'][0])['size']==512*4*256
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
cb=named('#O64');assert cb['inputs']==cp['outputs'];cseen=set()
for a,b,(lo,hi),(dl,dh) in packets(cb):
 sa,da=a['address'],b['address'];assert sa['cluster']==da['cluster'] and sl(sa,lo)==(sl(da,dl)//16)*16
 assert lo%4194304==dl%4194304==0 and hi-lo==dh-dl==16
 cseen.add((da['cluster'],sl(da,dl)))
assert len(cseen)==512
cv=named('#O66');assert cv['inputs']==cb['outputs'] and h.sequence(cv['description'],'fetch_sequencer')==(0,[(4,1)])
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
sd=named('#O63');assert sd['inputs']==sreshape['outputs'];sdest=set()
for a,b,(lo,hi),(dl,dh) in packets(sd):
 sa,da=a['address'],b['address'];assert sa['cluster']==da['cluster']==0 and sl(sa,lo)==0
 assert lo%4194304==dl%4194304==0 and hi-lo==dh-dl==32
 sdest.add(sl(da,dl))
assert sdest==set(range(8))
sv=next(o for o in ops.values() if o['kind']=='StoVrf' and o['inputs']==sd['outputs'])
epi=named('#O93');assert sv['outputs'][0]==epi['inputs'][1]
assert epi['description'].index('FpMul0:')<epi['description'].index('FpMul1:')<epi['description'].index('output_element_type: bf16')
# Enumerate actual two-stage row-major HBM source and slice-local destination coordinates.
weightspans=[];wrecords=[]
for native in natives:
 wop=g.producer(native['inputs'][0],'Dma.DtoS')[1];view=g.view(wop['inputs'][0]);start=view['start']
 assert (view['stride_elements'],view['window'],view['axis_extent'])==(4096,60,120)
 assert start in (0,60);count=0
 for a,b,(lo,hi),(dl,dh) in packets(wop):
  da=b['address'];c=da['cluster'];ds=sl(da,dl);gr,q=divmod(ds,16);row,tail=divmod(dl%4194304,256)
  assert tail==0 and 0<=row<60 and hi-lo==dh-dl==256
  actual=lo+start*4096
  assert actual==(c*1920+gr*120+start+row)*4096+q*256
  weightspans.append((actual,hi+start*4096));count+=1
 wrecords.append({'dma':wop['name'],'row_start':start,'packets':count,'packet_bytes':256})
wcoverage=h.full_coverage(weightspans,3840*4096,'whole original E4 matrix')
# Resource lifetimes use only schedule IDs, without joining graph numbers.
s=json.load(open(ev/'schedule.json'));ts={int(t['index']):t for t in s['tensors']};sr=[]
for i in s['instructions']:
 if i['tpe']!='Main' or '.contract_outer::' not in i['description']:continue
 regs=[ts[x] for x in i['input_tensors'] if x in ts and ts[x]['buffer_type'] in ('Trf','Vrf')]
 assert sorted(t['buffer_type'] for t in regs)==['Trf','Vrf']
 for t in regs:assert t['lifetime']['begin']<=i['lifetime']['begin'] and t['lifetime']['end']>=i['lifetime']['end']
 sr.append({'index':i['index'],'lifetime':i['lifetime'],'registers':regs})
assert len(sr)==2
src=p/'native-v3/projection.rs';assert src.read_bytes()==(ev/'src/device/sliding/projection.rs').read_bytes()
def rec(f):
 b=f.read_bytes();return {'path':str(f.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
oldsrc=(p/'native-v2/projection.rs').read_text();newsrc=src.read_text()
assert oldsrc[oldsrc.index('\nfn project_output_stage('):]==newsrc[newsrc.index('\nfn project_output_stage('):]
r={'passed':True,'scope':'Independent source and actual descriptor/mapping/order review. No NPU arithmetic or performance claim; hidden SFR and runtime SRAM bank arbitration are not established.',
'inputdigests':{str(f.relative_to(root)):{k:v for k,v in rec(f).items() if k!='path'} for f in [src,ev/'graph.json',ev/'schedule.json']},
'quantization':{'groups':32,'valid_shards_per_group':16,'logical_global_k_per_group':4096,'hbm_extent_bytes':8192,'hbm_replicated_payload_bytes':262144,'input_packets':len(xd),'global_max_ratio':16,'group_own_scale_destinations':len(qseen),'scale_equality_basis':'Each c,g receives identical finite BF16 x[256q+u]; identical absmax/floor/Max/exponent mask/pow2 gives bit-identical scalar.'},
'reshapes':{'scale':{'name':sreshape['name'],'old':sm,'new':dm,'physical_bytes':16384,'valid_slice_q':0},'packed':{'name':preshape['name'],'old':pm,'new':pd,'bytes':524288},'packed_512kib_dma_absent':True},
'coefficients':{'ordered_graph_printed_constants':sorted(constants),'source_groups':32,'packed_source_bytes_per_group':16,'sub_commit_bytes':8,'commits_per_group':2,'broadcast_destinations':len(cseen),'vrf_bytes_per_slice':32},
'weight_ingress':{'coverage':wcoverage,'stages':wrecords},
'native_stages':[{'name':o['name'],'inputs':o['inputs'],'outputs':o['outputs']} for o in natives],
'order':'normalized coefficient Mul -> packet Term ReduceAccTree -> K16 GReduceAdd -> F32 pair commit/gather -> common s Mul0 -> channel scale Mul1 -> first BF16',
'canonical_scale_source':'initialized cluster0 group0 q0 -> canonical cluster0 slices0..7, eight32B copies',
'native_and_epilogue_source_suffix_identical_to_v2':True,'schedule_register_lifetimes':sr}
Path('/tmp/moa-e137-independent-v3-review.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS /tmp/moa-e137-independent-v3-review.json')
