#!/usr/bin/env python3
"""Frozen E132 graph coverage, shape/numerical boundary and provenance audit; no compile/NPU."""
import argparse, collections, hashlib, json, re, subprocess, types
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
ART=ROOT/'moa-dev/experiments/artifacts'/HERE.name
HELPER=HERE.parent/'m08-output-pair4-ingress/verify_descriptors.py'
c=types.ModuleType('descriptor_helpers'); c.__file__=str(HELPER)
exec(compile(HELPER.read_text(),str(HELPER),'exec'),c.__dict__)
req,only=c.require,c.only

def record(p):
 raw=p.read_bytes(); return {'path':str(p.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def constant(g,t):
 o=g.producer[t];req(o['kind']=='CoreImpCmd','nonconstant offset')
 return int(only(re.findall(r'__scalar_out\[\] = (-?\d+)i32;',o['description']),'constant'))

def index(g,o,stride):
 m=re.search(r'axis: stride (\d+) x (\d+), window_size: (\d+)',o['description']);req(m is not None,'index syntax')
 a,n,w=map(int,m.groups());st=constant(g,o['inputs'][1]);req((a,n)==(stride,30) and (st,w) in ((0,16),(16,14)),'wrong stage window')
 return st,w

def audit(label):
 bpath=HERE/f'build-{label}-repro.json';b=json.loads(bpath.read_text());art=ART/label
 inp={'build_record':record(bpath)}
 for name in ('graph','schedule'):
  p=art/f'{name}.json';inp[name]=record(p);req({k:inp[name][k] for k in ('bytes','sha256')}==b[name],name+' changed')
 req(b['exact_compile_exit_code']==0 and b['target_kernel']=='sliding_attention_output','compile identity')
 req({k:record(art/'exact-kernel.bin')[k] for k in ('bytes','sha256')}==b['exact_image'],'image identity')
 for name,expected in b['source_inputs'].items(): req({k:record(art/'crate'/name)[k] for k in ('bytes','sha256')}==expected,'source changed '+name)
 req(record(ROOT/b['source_plan']['recipe']['path'])==b['source_plan']['recipe'],'recipe changed')
 source=(art/'crate/src/device/sliding/projection.rs').read_text()
 old=subprocess.check_output(['git','show',b['base_commit']+':furiosa-opt-gemma4-12B/src/device/sliding/projection.rs'],cwd=ROOT,text=True)
 req(source.split('axes![OutputRow = 30];')[0]==old.split('pub(crate) fn project_output(')[0],'QKV source changed')
 req(source.split('pub(crate) type OutputEpilogueSlices')[1].split('// O projection')[0]==old.split('pub(crate) type OutputEpilogueSlices')[1].split('// O projection')[0],'scale/RMS/residual helpers changed')
 g=c.Graph(json.loads((art/'graph.json').read_text())); stages=[]
 for o in g.ops:
  if o['kind']!='Dma.DtoS' or len(o['inputs'])!=1: continue
  view=g.producer.get(o['inputs'][0])
  if not view or view['kind']!='IndexAccess' or not g.from_io(view['inputs'][0],2):continue
  st,n=index(g,view,4096)
  dot=only([d for d in g.ops if d['kind']=='TuExec' and d['context']=='Main' and set(o['outputs'])&set(d['inputs']) and 'GReduceAdd:' in d['description']],'dot')
  ov=only([g.producer[t] for t in dot['inputs'] if t in g.producer and g.producer[t]['kind']=='IndexAccess'],'outputview')
  req(index(g,ov,1)==(st,n),'weight/output mismatch')
  wr=only([d for d in g.ops if d['kind']=='IndexWrite' and d['inputs'][2] in dot['outputs']],'write')
  req(index(g,wr,1)==(st,n) and wr['inputs'][0]==ov['inputs'][0],'write mismatch')
  stages.append(dict(start=st,rows=n,weight=o,dot=dot,view=ov,write=wr))
 stages.sort(key=lambda s:s['start']);req([(s['start'],s['rows']) for s in stages]==[(0,16),(16,14)],'stage split')
 req(stages[1]['view']['inputs'][0] in stages[0]['write']['outputs'],'stage output handoff')
 total=collections.Counter();wranges=[];weights=[]
 for s in stages:
  n,st,o=s['rows'],s['start'],s['weight'];t=g.tensors[only(o['outputs'],'weighttensor')]
  req(t['element_type']=='Float8e4m3' and t['size']==512*n*1024,'compactweight size/dtype')
  addr=int(t['optionals']['address']);wranges.append((addr,addr+n*1024))
  dest=set();sizes=collections.Counter();per=collections.Counter()
  for _,sd,dd,sa,da,length in c.paired_packets(o):
   sa+=st*4096;cl,sl,local=c.physical(dd,da)
   req(length==256 and local%256==0 and 0<=local<=n*1024-length,'weight local bounds')
   row,koff=divmod(local,1024)
   req(sa==(1920*cl+30*(sl//4)+st+row)*4096+1024*(sl%4)+koff,'weight coordinate')
   req(0<=sa<=3840*4096-length and sa%256==0,'HBM bounds')
   key=(cl,sl,local);req(key not in dest,'duplicate weight destination');dest.add(key);total[sa]+=1;per[cl,sl]+=length;sizes[length]+=1
  req(len(dest)==512*n*4 and len(per)==512 and set(per.values())=={n*1024},'weight slice coverage')
  weights.append({'stage':st,'rows':n,'payload_bytes':sum(per.values()),'live_slices':len(per),'packet_histogram':dict(sizes),'local_allocation':[addr,addr+n*1024]})
 req(set(total)==set(range(0,3840*4096,256)) and set(total.values())=={1},'completeweight exactonce')
 req(wranges[0][1]<=wranges[1][0] or wranges[1][1]<=wranges[0][0],'liveweight allocation overlap')
 act=g.dma_from_io(0); adest=set();acount=collections.Counter()
 for _,sd,dd,sa,da,n in c.paired_packets(act):
  cl,sl,local=c.physical(dd,da);req(0<=local<=2048-n and sa==(sl%4)*2048+local,'activation K bounds/owner')
  req(n%2==sa%2==local%2==0,'activation BF16alignment')
  for off in range(0,n,2):
   key=cl,sl,local+off;req(key not in adest,'activation duplicate');adest.add(key);acount[sa+off]+=1
 req(len(adest)==512*1024 and set(acount)==set(range(0,8192,2)) and set(acount.values())=={128},'activation fullcoverage')
 trf=only([o for o in g.ops if o['kind']=='StoTrf' and all(set(o['outputs'])&set(s['dot']['inputs']) for s in stages)],'sharedTRF')
 req(g.tensors[only(trf['outputs'],'TRF')]['element_type']=='Bfloat16' and 'Qs_1=1024' in trf['description'],'TRF dtype/extent')
 written=set(); arith=[]
 for s in stages:
  d=s['dot']['description'];n,st=s['rows'],s['start']
  for text in ('lookup_table:','broadcast_kind: Direct, bf16','GReduceAddF [ratio=4]','<global_reduce>Qs_1024=4<global_reduce>','commit_in_size: 8B (=2EA)','data_type: Float32'):req(text in d,'missing arithmetic '+text)
  req('cast_compaction:' not in d and 'FpMul0:' not in d,'prematurecast/scale')
  m=re.search(r'commit_projection: (\d+)\+\[None=(\d+):(\d+), None=(\d+):(\d+)\]',d);req(m is not None,'commit projection')
  origin,n0,p0,n1,p1=map(int,m.groups());seq=[origin+i*p0+j*p1 for i in range(n0) for j in range(n1)]
  req(seq==list(range(n)) and n1==2 and p1==1,'compact pair rowcommit')
  req(all(g.tensors[t]['element_type']=='Float32' for t in s['dot']['outputs']+s['write']['outputs']),'F32 boundary')
  for cl in range(2):
   for group in range(64):
    for row in seq:
     h=1920*cl+30*group+st+row;req(h not in written,'overlappingstageoutput');written.add(h)
  arith.append({'rows':n,'Kreduce':4,'output_byte_range':[4*st,4*(st+n)],'commit_bytes':8,'operator':s['dot']['name']})
 req(written==set(range(3840)),'all outputrows')
 gather=only([o for o in g.ops if o['kind']=='Dma.StoS' and any(set(stages[1]['write']['outputs'])&g.ancestry(t,True) for t in o['inputs'])],'gather')
 seen=set();gsizes=collections.Counter()
 for _,sd,dd,sa,da,n in c.paired_packets(gather):
  sc,ss,slo=c.physical(sd,sa);dc,ds,dlo=c.physical(dd,da)
  req(ss%4==0 and dc==0 and ds<8 and n%8==sa%8==da%8==0,'gather ownership/alignment')
  req(0<=slo<=120-n and 0<=dlo<=1920-n,'gather bounds')
  for off in range(0,n,4):
   sh=1920*sc+30*(ss//4)+(slo+off)//4;dh=480*ds+(dlo+off)//4
   req(sh==dh and sh not in seen,'gather canonical mapping');seen.add(sh)
  gsizes[n]+=1
 req(seen==set(range(3840)) and gsizes=={120:128},'gather fullcoverage/no scalarpack')
 scale_dma=g.dma_from_io(3);scalec=c.check_scale_dma(scale_dma)
 scale=only([o for o in g.ops if o['kind']=='TuExec' and o['context']=='Main' and 'FpMul0:' in o['description'] and any(set(gather['outputs'])&g.ancestry(t,True) for t in o['inputs'])],'scale')
 req(g.dependent(scale,scale_dma),'scale provenance')
 d=scale['description'];req('cast_compaction:' in d and d.index('FpMul0:')<d.index('cast_compaction:') and 'bf16' in d,'scale beforeBF16')
 req(all(g.tensors[t]['element_type']=='Float32' for t in gather['outputs']),'F32 gather')
 schedule=json.loads((art/'schedule.json').read_text())
 spans=[{'index':o['index'],'kind':o['tpe'],'lifetime':o['lifetime'],'source':o.get('description','').splitlines()[0] if o.get('description') else ''} for o in schedule['instructions'] if o['lifetime']['end']>o['lifetime']['begin'] and o['tpe'] in ('DmaLoad','DmaStos','Main','Sub')]
 return {'label':label,'passed':True,'status':'PASS_GRAPH_CONTRACT_ONLY','inputdigests':inp,'verifiers':[record(Path(__file__)),record(HELPER)],'weights':weights,'activation':{'unique_bytes':8192,'replicated_bytes':1048576,'copies_per_BF16':128,'TRF_per_slice_bytes':2048},'arithmetic':arith,'gather':{'bytes':15360,'packet_histogram':dict(gsizes),'operator':gather['name']},'scale_channels':scalec,'source_QKV_and_epilogue_unchanged':True,'static_cycles':b['static_cycles'],'static_intervals':spans,'limit':'Descriptor coverage, dtypes, numerical boundaries and dependencies only; not NPU arithmetic correctness, exact F32 summation equivalence, bandwidth or speed.'}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--check',action='store_true');a=p.parse_args()
 value=json.loads(json.dumps(audit(a.label)));path=HERE/f'graph-verification-{a.label}.json'
 if a.check: req(json.loads(path.read_text())==value,'saved result differs')
 else:
  with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
 print(value['status'],value['static_cycles'],value['gather'])
