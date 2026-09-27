#!/usr/bin/env python3
"""Frozen E140 local-v4 source/descriptor/lifetime gate; --check never writes."""
import argparse,hashlib,importlib.util,json,re,itertools
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3] if HERE.name=='down-native-consumer-local' else Path('/home/jihwan38/moa-down-native-local')
P=ROOT/'moa-dev/experiments/candidates/down-native-consumer-local'
HELPER=ROOT/'moa-dev/experiments/candidates/ffn-native-down-weighted-residual/verify_graph.py'
z=importlib.util.spec_from_file_location('oldhelpers',HELPER);h=importlib.util.module_from_spec(z);z.loader.exec_module(h)
def req(v,s):
 if not v:raise ValueError(s)
def record(f):
 b=f.read_bytes()
 try:path=str(f.relative_to(ROOT))
 except ValueError:path=str(f)
 return {'path':path,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def digest(f):return {k:v for k,v in record(f).items() if k!='path'}
def analyze(label):
 req(label=='e140-local-v4','Frozen local-v4 only')
 ev=P/'evidence'/label;art=ROOT/'moa-dev/experiments/artifacts/down-native-consumer-local'/label
 bp=P/f'build-{label}-repro.json';b=json.loads(bp.read_text());src=P/'local-v4/mlp.rs';original=P.parent/'ffn-native-down-weighted-residual/weighted-v3/mlp.rs'
 req(b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container'],'Exact compile gate')
 paths={'build_record':bp,'graph':art/'graph.json','schedule':art/'schedule.json','source':src,'exact_image':art/'exact-kernel.bin'}
 for k in ['graph','schedule','exact_image']:req(digest(paths[k])==b[k],k+' digest')
 for k in ['graph','schedule']:req(paths[k].read_bytes()==(ev/f'{k}.json').read_bytes(),k+' evidence identity')
 req(src.read_bytes()==(ev/'src/device/shared/mlp.rs').read_bytes(),'frozen source identity')
 req(digest(src)==b['source_inputs']['src/device/shared/mlp.rs'],'source build digest')
 plan=b['source_plan'];expected=dict(plan['source_origin_inputs']);expected['src/device/shared/mlp.rs']=digest(src)
 req(expected==b['source_inputs'] and b['changed_sources']==['src/device/shared/mlp.rs'],'only intended mlp source changed')
 for path,d in b['source_inputs'].items():req(digest(art/'crate'/path)==d,'compiled crate '+path)
 s=src.read_text();old=original.read_text();prefix=s[:s.index('// E140: prepare')].replace(', DownChunk = 4, DownPackedChunk = 12','')
 req(prefix==old[:old.index('// E133: quantize')],'up/gate and GeGLU prefix changed')
 req(s[s.index('fn project_down_tile('):]==old[old.index('fn project_down_tile('):],'native down/finish/public entries/up quantizer changed')
 g=h.Graph(json.loads(paths['graph'].read_text()));ops={int(o['name'].split('#O')[-1]):o for o in g.ops.values()}
 M=4194304
 def sl(a,x):return a['pe']*64+a['dmn']*32+a['slice']+x//M
 def packets(o):
  for pair in h.dma_descriptors(o['description']).values():
   a,d=pair['src'],pair['dst'];sa,da=a['address'],d['address'];aa=list(h.intervals(sa.get('in_chip',sa.get('in_slice')),a['dims']));dd=list(h.intervals(da.get('in_chip',da.get('in_slice')),d['dims']));req(len(aa)==len(dd),'DMA paired packet count')
   for x,y in zip(aa,dd):yield sa,da,x,y
 # Actual compact GeGLU gather remains same-cluster, initialized q0 of each4slice group.
 canon={c:[] for c in range(2)}
 for sa,da,(lo,hi),(dl,dh) in packets(ops[191]):
  ss,ds=sl(sa,lo),sl(da,dl)
  req(sa['cluster']==da['cluster'] and ss%4==0 and ds==0 and lo%M==0 and hi-lo==dh-dl==240,'canonical gather coordinate')
  req(dl%M==(ss//4)*240,'canonical originalL order');canon[da['cluster']].append((dl%M,dl%M+240))
 for spans in canon.values():h.full_coverage(spans,15360,'canonicalBF16half')
 # Canonical same-cluster L7680 -> Hgroups, eachK1920 compact consumer exactlyonce.
 dest={(c,ds):[] for c in range(2) for ds in range(256)};pc=0
 for sa,da,(lo,hi),(dl,dh) in packets(ops[192]):
  ss,ds=sl(sa,lo),sl(da,dl);do=dl%M
  req(sa['cluster']==da['cluster'] and ss==0 and 0<=ds<256 and hi-lo==dh-dl==256,'consumerDMA packet')
  req(lo%M==(ds%4)*3840+do and 0<=do<3840,'consumer originalK coordinate')
  dest[da['cluster'],ds].append((do,do+256));pc+=1
 for spans in dest.values():h.full_coverage(spans,3840,'consumer BF16 stripe')
 req(pc==7680 and ops[193]['inputs']==ops[192]['outputs'],'input packet/pipeline count')
 req('LReduce(max_f) [axes=TileL]' in ops[193]['description'] and '1.92593e-34f' in ops[193]['description'],'local amax floor')
 req(ops[194]['inputs']==ops[193]['outputs'] and 'GReduceMaxF [ratio=4]' in ops[194]['description'],'samegroup K4 maximum')
 req(ops[195]['inputs']==ops[194]['outputs'] and '6.1035156e-5f' in ops[195]['description'],'same power2 scale')
 def group_dma(n,packet):
  seen=set()
  for sa,da,(lo,hi),(dl,dh) in packets(ops[n]):
   ss,ds=sl(sa,lo),sl(da,dl)
   req(sa['cluster']==da['cluster'] and ss==ds//4*4 and lo%M==dl%M==0 and hi-lo==dh-dl==packet,'group scale transfer')
   seen.add((da['cluster'],ds))
  req(len(seen)==512,'group scale destinations');return len(seen)
 group_dma(196,32);req(ops[197]['inputs']==ops[196]['outputs'],'quant scalarVRF binding')
 # Two whole owned reshapes preserve in-slice physicalorder and ownership.
 a,bm=g.mappings(211),g.mappings(218);req(all(a[k]==bm[k] for k in ['Chip','Cluster','Partitioning']),'input owner reshape')
 req(g.tensor(211)['size']==g.tensor(218)['size']==1966080,'input reshape extent')
 # Twelve actual direct quantizer writes: only 480E5 bytes each, exact flatindex4*t+i.
 writes=[];rootbuf=217
 for o in g.ops.values():
  if o['kind']!='IndexWrite' or g.root_output(o['outputs'][0])!=rootbuf:continue
  idx=g.scalar(o['inputs'][1]);n=g.producer(o['inputs'][2],'TuExec')[1];term,chunk=divmod(idx,4)
  req('axis: stride 480 x 12, window_size: 1' in o['description'] and term<3,'packed indexed layout')
  view=g.view(n['inputs'][0]);req((view['parent'],view['stride_elements'],view['axis_extent'],view['window'],view['start'])==(218,480,4,1,chunk),'quant input chunk')
  req(h.offsets(h.sequence(n['description'],'fetch_sequencer'))==list(range(480)),'quant BF16 read480')
  req(h.offsets(h.sequence(n['description'],'commit_projection'))==list(range(480)) and 'commit_in_size: 8B (=8EA)' in n['description'],'quant E5 write480')
  req(216 in n['inputs'] and 'FpFpu: %l / %r' in n['description'],'quant original scale')
  if term:
   req('FpFma: %l - %r' in n['description'] and ('1.6e1f' if term==1 else '2.56e2f') in n['description'],'weighted residual formula')
  writes.append({'index':idx,'term':term,'chunk':chunk,'byte_bounds':[480*idx,480*(idx+1)],'quantizer':n['name']})
 req(sorted(x['index'] for x in writes)==list(range(12)),'complete exactlyonce packed coverage')
 # Every E5 packed-view read used by a VRF/lo+hi is bounded480; both residualbuffers are1920B.
 readers=[]
 for o in g.ops.values():
  if o['kind'] not in ['TuExec','StoVrf'] or not o['inputs']:continue
  prod=g.producers.get(o['inputs'][0])
  if not prod or prod[1]['kind']!='IndexAccess':continue
  v=g.view(o['inputs'][0])
  if v['axis_extent']!=12 or v['stride_elements']!=480:continue
  req(v['window']==1 and 0<=v['start']<12,'packed read view');req(h.offsets(h.sequence(o['description'],'fetch_sequencer'))==list(range(480)),'packed read480')
  readers.append({'operator':o['name'],'index':v['start']})
 req(len(readers)==8,'four hiVRF and four restoredsum reads')
 req(ops[312]['kind']=='LowLevelReshape' and g.tensor(329)['size']==g.tensor(331)['size']==2949120,'final packed reshape size')
 req(all(g.mappings(329)[k]==g.mappings(331)[k] for k in ['Chip','Cluster','Partitioning']),'packed owners preserved')
 coeff=[]
 for idx,n in enumerate([317,322,327]):
  o=ops[n];v=g.view(o['inputs'][-1]);req((v['stride_elements'],v['axis_extent'],v['window'],v['start'])==(8,3,1,idx),'termcoef slot')
  req(h.offsets(h.sequence(o['description'],'commit_projection'))==list(range(8)),'coef32B write')
  value=float(re.search(r'FpMul0:.*?%r=([^f, ]+)f',o['description'])[1]);req(abs(value/[1,2**-4,2**-8][idx]-1)<1e-7,'restore coefficient');coeff.append(value)
 group_dma(329,96);req(ops[330]['inputs']==ops[329]['outputs'] and ops[331]['inputs']==[331],'coefficient/nativeTRF binding')
 n=ops[338];req(n['inputs'][:3]==[352,350,349],'native weight/TRF/coefficient')
 d=n['description'];req(d.index('FpMul0:')<d.index('LReduce(+) [axes=FfnTerm]'),'restore-before-Termsum')
 req(h.offsets(h.sequence(d,'commit_projection'))==list(range(7200)) and 'commit_in_size: 16B (=4EA)' in d,'same fullF32blockpartial coverage')
 for i in range(5):
  scale=ops[342+10*i];red=ops[348+10*i];sv=g.view(scale['inputs'][0]);pv=g.view(red['inputs'][0]);ov=g.view(red['inputs'][-1])
  req((sv['start'],pv['start'],ov['start'])==(12*i,12*i,12*i),'down12row offsets')
  rd=red['description'];req(rd.index('FpMul0:')<rd.index('LReduce(+) [axes=TileL]')<rd.index('GReduceAddF [ratio=4]'),'unchanged blockscale/reductions')
  req(h.offsets(h.sequence(rd,'commit_projection'))==list(range(12)),'down12rows exactcommit')
 hbm=h.hbm_ingress(g)
 # All wholeFFN VRFs, including unrelated prefetches, fit physicalcapacity with no simultaneousrange overlap.
 sch=json.loads(paths['schedule'].read_text());ts={int(t['index']):t for t in sch['tensors']};vr=[t for t in ts.values() if t['buffer_type']=='Vrf']
 for t in vr:req(0<=t['address'] and t['address']+t['size']<=8192*512,'VRF physicalcapacity')
 for a,c in itertools.combinations(vr,2):
  live=max(a['lifetime']['begin'],c['lifetime']['begin'])<min(a['lifetime']['end'],c['lifetime']['end'])
  overlap=max(a['address'],c['address'])<min(a['address']+a['size'],c['address']+c['size'])
  req(not(live and overlap),'simultaneous VRF overlap '+str((a['index'],c['index'])))
 relevant=[t for t in vr if t['lifetime']['end']>=130000]
 for ins in sch['instructions']:
  for tid in ins['input_tensors']:
   t=ts.get(tid)
   if t is None:continue
   if t['buffer_type'] not in ['Vrf','Trf']:continue
   req(t['lifetime']['begin']<=ins['lifetime']['begin']<=ins['lifetime']['end']<=t['lifetime']['end'],'register consumer lifetime')
 # No source claim that package-level unchanged-image verification has already run.
 return {'passed':True,'scope':'Frozen source, actual emitted descriptor/index/sequence and full schedule register gate. Hardware arithmetic, liveness and wholeperformance remain separate.','inputdigests':{k:record(v) for k,v in paths.items()},'verifiers':[record(Path(__file__).resolve()),record(HELPER)],'source_manifest_files':len(b['source_inputs']),'source_changes':['src/device/shared/mlp.rs down preparation only'],'unchanged_source_prefix_suffix':True,'unchanged_image_requirement':{'declared_images':plan['expected_unchanged_images'],'scope':'QKV/O sources equal E137; actual host3image equality is separately required by package gate'},'consumer_input':{'canonical_gather_packets':128,'canonical_packet_bytes':240,'consumer_packets':pc,'consumer_packet_bytes':256,'logical_consumer_stripes':512,'logical_stripe_bytes':3840,'payload_bytes':1966080,'mapping':'samecluster sourceSlice0 byte3840q+u ->slice4g+q localbyteu;K4 spans all7680values'},'scale':{'owner':'Dummy2cluster independent; eachHgroup uses sameclusterK4max','floor':2**-112,'scale_factor':2**-14,'broadcast_destinations':512},'packed_writes':sorted(writes,key=lambda x:x['index']),'packed_reads':readers,'coefficients':coeff,'hbm_ingress':hbm,'native_partial_elements_per_slice':7200,'vrf_schedule':{'checked_buffers':len(vr),'overlapping_live_pairs':0,'late_buffers':relevant},'limitations':['v1 allocation cause is not retroactively proven by v4 successful allocation','No hiddenSFR/readidx/bank-arbitration guarantee','Full original-down and fullFFN precision validation still required']}
def main():
 a=argparse.ArgumentParser();a.add_argument('label');a.add_argument('--check',action='store_true');a.add_argument('--output',type=Path);q=a.parse_args();out=q.output or P/f'graph-verification-{q.label}.json'
 try:r=analyze(q.label)
 except Exception as e:
  if not q.check:out.write_text(json.dumps({'passed':False,'error':str(e)},indent=2)+'\n')
  raise
 r=json.loads(json.dumps(r))
 if q.check:req(json.loads(out.read_text())==r,'Frozen graph gate differs')
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS',out)
if __name__=='__main__':main()
