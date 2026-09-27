import json,hashlib,importlib.util,re
from pathlib import Path
root=Path('/home/jihwan38/moa-qkv-native-local');p=root/'moa-dev/experiments/candidates/qkv-native-consumer-local';ev=p/'evidence/e138-native-v1'
hp=root/'moa-dev/experiments/candidates/ffn-native-down-weighted-residual/verify_graph.py';sp=importlib.util.spec_from_file_location('h',hp);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
g=h.Graph(json.loads((ev/'graph.json').read_text()));op={int(x['name'].split('#O')[-1]):x for x in g.ops.values()}
def packets(o):
 for sides in h.dma_descriptors(o['description']).values():
  a,b=sides['src'],sides['dst'];aa=list(h.intervals(a['address'].get('in_chip',a['address'].get('in_slice')),a['dims']));bb=list(h.intervals(b['address'].get('in_chip',b['address'].get('in_slice')),b['dims']));assert len(aa)==len(bb)
  for x,y in zip(aa,bb):yield a['address'],b['address'],x,y
M=4194304
def sl(a,x):return a['pe']*64+a['dmn']*32+a['slice']+x//M
# Actual normalized-input source H480 -> identical H3840 at each c,g.
xc={(c,gr):[] for c in range(2) for gr in range(64)};xd=set()
for sa,da,(lo,hi),(dl,dh) in packets(op[8]):
 ss,ds=sl(sa,lo),sl(da,dl);c=da['cluster'];gr,q=divmod(ds,4);so,do=lo%M,dl%M
 assert sa['cluster']==c and 0<=ss<8 and 0<=ds<256
 assert hi-lo==dh-dl==960 and so==0 and do in (0,960)
 assert ss*960+so==q*1920+do
 xc[c,gr].append((ss*960+so,ss*960+so+hi-lo));xd.add((c,ds,do))
assert len(xd)==1024
for v in xc.values():h.full_coverage(v,7680,'each group input')
assert op[9]['inputs']==op[8]['outputs'] and op[10]['inputs']==op[9]['outputs'] and op[11]['inputs']==op[10]['outputs']
assert 'LReduce(max_f) [axes=H]' in op[9]['description'] and 'GReduceMaxF [ratio=4]' in op[10]['description'] and '7.8125e-3f' in op[11]['description']
qdest=set()
for sa,da,(lo,hi),(dl,dh) in packets(op[12]):
 assert sa['cluster']==da['cluster'] and sl(sa,lo)==sl(da,dl)//4*4 and lo%M==dl%M==0 and hi-lo==dh-dl==32
 qdest.add((da['cluster'],sl(da,dl)))
assert len(qdest)==512
# Relabel changes only semantic group anonymity, preserving all q0 valid masks.
sm=g.mappings(29);dm=g.mappings(86);want=json.loads(json.dumps(sm));want['Partitioning']['Pair']['left']={'Broadcast':{'size':64}};assert dm==want
assert op[68]['inputs']==[29] and g.tensor(29)['size']==g.tensor(86)['size']==16384
pm=g.mappings(62);want=json.loads(json.dumps(pm));want['Cluster']={'Symbol':{'symbol':'Dummy2','size':2}};assert g.mappings(87)==want
assert op[69]['inputs']==[62] and g.tensor(62)['size']==g.tensor(87)['size']==1966080
sdest=set();ssources=set()
for sa,da,(lo,hi),(dl,dh) in packets(op[70]):
 assert sa['cluster']==da['cluster']==0 and sl(sa,lo)%4==0 and 0<=sl(sa,lo)<256
 assert lo%M==dl%M==0 and hi-lo==dh-dl==32
 sdest.add(sl(da,dl));ssources.add(sl(sa,lo))
assert sdest==set(range(8))
# Four constants use literal indexed32B slots; transpose writes only two8B packets.
cp=op[67];assert cp['context']=='Sub' and h.sequence(cp['description'],'fetch_sequencer')==(0,[(4,8),(8,1)])
assert h.offsets(h.sequence(cp['description'],'commit_projection'))==list(range(4)) and 'commit_in_size: 8B (=2EA)' in cp['description']
cr=g.root_output(cp['inputs'][0]);constants=[]
for o in g.ops.values():
 if o['kind']!='IndexWrite' or g.root_output(o['outputs'][0])!=cr:continue
 idx=g.scalar(o['inputs'][1]);maker=g.producer(o['inputs'][2],'TuExec')[1];val=float(re.search(r'FpFma: %l \+ %r \[true\] %r=([^f, ]+)f',maker['description'])[1]);assert abs(val/[1,2**-4,2**-22,2**-27][idx]-1)<1e-7;constants.append((idx,val))
assert sorted(i for i,v in constants)==list(range(4))
cdest=set()
for sa,da,(lo,hi),(dl,dh) in packets(op[71]):
 assert sa['cluster']==da['cluster'] and sl(sa,lo)==sl(da,dl)//4*4 and lo%M==dl%M==0 and hi-lo==dh-dl==16
 cdest.add((da['cluster'],sl(da,dl)))
assert len(cdest)==512 and op[73]['inputs']==op[71]['outputs']
assert op[72]['inputs']==[87] and op[72]['outputs']==[90] and 'TrfMacRows: [QkvTerm_1=4]' in op[72]['description'] and 'InSlice: [H_1=960]' in op[72]['description']
# Every actual weight descriptor maps the public row-major original once.
wr=[];grr=[];sr=[]
for name,rows,wn,nn,rn,gn,sn,scn,en in [('Q',32,75,76,77,80,84,85,86),('K',16,88,89,90,93,97,98,99),('V',16,101,102,103,106,110,111,112)]:
 spans=[];dseen=set();rowcount=128*rows;headrows=16*rows
 for sa,da,(lo,hi),(dl,dh) in packets(op[wn]):
  c=da['cluster'];ds=sl(da,dl);gr,q=divmod(ds,4);rr,u=divmod(dl%M,960)
  assert 0<=ds<256 and 0<=rr<rows and u==0 and hi-lo==dh-dl==960
  assert lo==(c*64*rows+gr*rows+rr)*3840+q*960
  spans.append((lo,hi));dseen.add((c,ds,rr))
 assert len(dseen)==512*rows
 wr.append({'projection':name,'dma':op[wn]['name'],'coverage':h.full_coverage(spans,rowcount*3840,name+'HBM'),'packet_bytes':960})
 n=op[nn];d=n['description'];assert n['inputs']==[op[wn]['outputs'][0],90,91]
 assert d.index('FpMul0:')<d.index('LReduce(+) [axes=QkvTerm]')<d.index('GReduceAddF [ratio=4]')
 assert 'cast_compaction:' not in d and 'lookup_table:' not in d and 'commit_in_size: 8B (=2EA)' in d
 assert h.offsets(h.sequence(d,'commit_projection'))==list(range(rows))
 assert op[rn]['inputs']==n['outputs'] and op[gn]['inputs'][0]==op[rn]['outputs'][0]
 reads=[];writes=[]
 for sa,da,(lo,hi),(dl,dh) in packets(op[gn]):
  ss,ds=sl(sa,lo),sl(da,dl);so,do=lo%M,dl%M
  assert ss%4==0 and 0<=ss<256 and da['cluster']==0 and 0<=ds<8
  assert so==0 and hi-lo==dh-dl==rows*4 and do%(rows*4)==0
  source_row=sa['cluster']*64*rows+(ss//4)*rows+so//4;dest_row=ds*headrows+do//4
  assert source_row==dest_row
  reads.append((source_row*4,source_row*4+hi-lo));writes.append((dest_row*4,dest_row*4+dh-dl))
 grr.append({'projection':name,'gather':op[gn]['name'],'source':h.full_coverage(reads,rowcount*4,name+'raw initialized rows'),'destination':h.full_coverage(writes,rowcount*4,name+'canonical head rows'),'packet_bytes':rows*4})
 scales=[]
 for sa,da,(lo,hi),(dl,dh) in packets(op[sn]):
  ds=sl(da,dl);do=dl%M;assert da['cluster']==0 and 0<=ds<8 and lo==ds*headrows*2+do and hi-lo==dh-dl==256 and do+dh-dl<=headrows*2;scales.append((lo,hi))
 sr.append({'projection':name,'channel_scale':h.full_coverage(scales,rowcount*2,name+'channel')})
 assert op[scn]['inputs']==op[sn]['outputs'];e=op[en];assert e['inputs'][1:]==[100,op[scn]['outputs'][0]]
 assert e['description'].index('FpMul0:')<e['description'].index('FpMul1:')<e['description'].index('output_element_type: bf16')
# Schedule namespaces inspected independently; each native consumes one same long-lived TRF/VRF.
s=json.loads((ev/'schedule.json').read_text());ts={int(t['index']):t for t in s['tensors']};regs=[]
for i in s['instructions']:
 if i['tpe']!='Main' or '.contract_outer::' not in i['description']:continue
 rr=[ts[t] for t in i['input_tensors'] if ts[t]['buffer_type'] in ('Trf','Vrf')];assert sorted(t['buffer_type'] for t in rr)==['Trf','Vrf']
 for t in rr:assert t['lifetime']['begin']<=i['lifetime']['begin']<=i['lifetime']['end']<=t['lifetime']['end']
 regs.append({'index':i['index'],'lifetime':i['lifetime'],'registers':rr})
assert len(regs)==3 and len({tuple(t['index'] for t in r['registers']) for r in regs})==1
src=p/'native-v1/projection.rs';opsrc=p/'native-v1/ops.rs'
assert src.read_bytes()==(ev/'src/device/sliding/projection.rs').read_bytes()
assert opsrc.read_bytes()==(ev/'src/ops.rs').read_bytes()
def digest(f):
 b=f.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
r={'passed':True,'scope':'Independent emitted descriptor/source/mapping/order and scheduled lifetime review only. No NPU arithmetic, latency, hidden SFR readidx or runtime bank-arbitration guarantee.','inputdigests':{str(f.relative_to(root)):digest(f) for f in [src,opsrc,ev/'graph.json',ev/'schedule.json']},'input':{'actual_packets':len(xd),'packet_bytes':960,'destination_payload_bytes':983040,'same_global_input_groups':128,'logical_bf16_h_per_group':3840,'coordinate':'slice=4*g+q; dstbyte=j*960; srcslice=2*q+j, srcbyte=0','global_max_ratio':4,'scale_destinations':len(qdest),'scale_equality_basis':'Each group receives identical complete finite BF16 H3840; same absmax/floor/max/exponentmask/power2 scale. Not independent perK scale.'},'coefficients':{'ordered_graph_printed_constants':sorted(constants),'copies':len(cdest),'compact_bytes':16,'vrf_bytes_per_slice':32},'reshapes':{'scale_group64_named_to_anonymous_only':True,'packed_cluster_only':True,'packed_payload_bytes':1966080,'no_packed_redistribution':'LowLevelReshape62to87 immediately feeds StoTrf72'},'canonical_scale':{'source_cluster':0,'initialized_source_q0_slices':sorted(ssources),'destination_head_slices':sorted(sdest),'packet_bytes':32},'weight_ingress':wr,'raw_gathers':grr,'channel_scales':sr,'order':'native normalized coefficient Mul -> packet Term reduce -> K4 reduce -> F32 pair commit/canonical gather -> common scale Mul0 -> channel Mul1 -> first BF16','schedule_register_lifetimes':regs,'limits':['Source and emitted graph lower the fixed four factors; actual native precision and whole head RMS/RoPE must pass separate NPU checks.','The q0 valid masks and semantic relabel proof assume finite inputs and source globalamax contract.','Scheduled register lifetime containment does not establish all hidden hardware hazards.']}
Path('/tmp/moa-e138-review-graph.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS /tmp/moa-e138-review-graph.json')
