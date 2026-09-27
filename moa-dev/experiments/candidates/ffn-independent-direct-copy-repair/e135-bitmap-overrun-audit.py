#!/usr/bin/env python3
"""Read-only descriptor/lifetime/control-bitmap audit of frozen E135 whole and oracle."""
import argparse,hashlib,importlib.util,json,re
from pathlib import Path
A=argparse.ArgumentParser();A.add_argument('--root',type=Path,default=Path('/home/jihwan38/moa-independent-direct-fused'));A.add_argument('--output',type=Path,default=Path('/tmp/moa-e135-bitmap-overrun-audit.json'));A.add_argument('--check',action='store_true');a=A.parse_args();ROOT=a.root.resolve();BASE=ROOT/'moa-dev/experiments/candidates';P=BASE/'ffn-independent-direct-fused';ART=ROOT/'moa-dev/experiments/artifacts/ffn-independent-direct-fused';HELPER=BASE/'ffn-native-down-weighted-residual/verify_graph.py'
s=importlib.util.spec_from_file_location('h',HELPER);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
def record(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def audit(kind,ev,source,build,image,log,job):
 gp=ev/'graph.json';sp=ev/'schedule.json';g=h.Graph(json.loads(gp.read_text()));s=json.loads(sp.read_text());ts={int(t['index']):t for t in s['tensors']}
 switch=next(o for o in g.ops.values() if 'custom_snoop_bitmap:' in o['description'] and 'FfnTerm' in o['description'] and 'ring_size: 32' in o['description'])
 bitmap_id=int(re.search(r'custom_snoop_bitmap:\s*\[\s*T(\d+)',switch['description'])[1]);bm=g.tensor(bitmap_id);base=int(bm['optionals']['address']);assert bm['size']==16384
 bitmap_load=g.producers[bitmap_id][1];assert bitmap_load['kind']=='Dma.DtoS';weight=g.tensor(bitmap_load['inputs'][0]);assert weight['size']==16384
 sc=next(i for i in s['instructions'] if i['tpe']=='Main' and 'CustomBroadcast { ring_size: 32 }' in i['description'] and 'm![FfnTerm]' in i['description'])
 sb=ts[sc['input_tensors'][1]];assert sb['address']//512==base and sb['size']==16384
 sl=next(i for i in s['instructions'] if int(sb['index']) in i['output_tensors']);assert sl['tpe']=='DmaLoad'
 expected=[i//32*32 for i in range(256)];actual=[int(x) for x in re.findall(r'SliceBitmap \[(\d+)\]',switch['description'])];assert actual==expected
 bits=b''.join((1<<owner).to_bytes(32,'little') for c in range(2) for owner in expected)
 ib=image.read_bytes();build_record=json.loads(build.read_text());image_record=build_record.get('exact_image',build_record.get('image'));assert image_record['bytes']==len(ib) and image_record['sha256']==hashlib.sha256(ib).hexdigest();at=ib.find(bits);assert at>=0 and ib.find(bits,at+1)<0
 copies=[]
 for o in g.ops.values():
  if o['kind']!='Dma.StoS' or '720=2:720' not in o['description'] or len(o['inputs'])!=2:continue
  v=g.view(o['inputs'][1]);assert (v['stride_elements'],v['axis_extent'],v['window'])==(480,3,1);term=v['start'];assert term in [0,1,2]
  root=g.root_output(v['parent']);rt=g.tensor(root);packedbase=int(rt['optionals']['address']);assert rt['size']==1440*512
  prod=g.producers[o['inputs'][0]][1];assert h.offsets(h.sequence(prod['description'],'commit_projection'))==list(range(480))
  for pair in h.dma_descriptors(o['description']).values():
   for side in ['src','dst']:
    d=pair[side];iv=list(h.intervals(d['address']['in_slice'],d['dims']));by={}
    for lo,hi in iv:by.setdefault(lo//4194304,[]).append((lo%4194304,hi%4194304))
    assert set(by)==set(range(8))
    for spans in by.values():h.full_coverage(spans,1440,'copy actualextent')
  ins=next(i for i in s['instructions'] if i['tpe']=='DmaStos' and ['hi','lo','third'][term]+'.view().to_dm_view' in i['description'])
  lo=packedbase+480*term;hi=lo+1440;over=max(lo,base)<min(hi,base+32)
  assert ins['lifetime']['end']<=sc['lifetime']['begin'];assert sl['lifetime']['end']<=ins['lifetime']['begin']
  copies.append({'operator':o['name'],'source_initialized_bytes_per_slice':480,'actual_read_and_write_bytes_per_slice':1440,'packed_root_bounds':[packedbase,packedbase+1440],'term':term,'write_bounds':[lo,hi],'schedule_index':ins['index'],'schedule_time':ins['lifetime'],'bitmap_intersection':[max(lo,base),min(hi,base+32)] if over else None,'affected_clusters':[0,1] if over else [],'affected_slices':list(range(8)) if over else [],'raw_descriptor':o['description']})
 assert sorted(c['term'] for c in copies)==[0,1,2]
 assert sorted(c['term'] for c in copies if c['bitmap_intersection'])==[1,2]
 txt=log.read_text();assert 'launch ran past its time limit' in txt
 return {'inputdigests':{k:record(v) for k,v in {'source':source,'build':build,'graph':gp,'schedule':sp,'exact_image':image,'raw_npu_log':log}.items()},'bitmap':{'role':'CustomBroadcast ring32 fetch-network custom_snoop_bitmap; not numeric FP4 lookup table','graph_tensor':bitmap_id,'graph_consumer':switch['name'],'graph_producer':bitmap_load['name'],'dram_constant_address':weight['optionals']['address'],'schedule_tensor':int(sb['index']),'bounds_per_slice':[base,base+32],'lifetime':sb['lifetime'],'schedule_load':sl,'schedule_consumer':sc,'embedded_constant':{'bytes':len(bits),'sha256':hashlib.sha256(bits).hexdigest(),'unique_file_offset':at,'encoding':'512 rows: each256-bit little-endian row is one-hot bit32*floor(slice/32), repeated for both clusters'},'producer_descriptor':bitmap_load['description']},'copies':sorted(copies,key=lambda c:c['term']),'npu_observation':{'job':job,'first_launch_timeout_code':4,'no_arithmetic_outputs':True,'localized_timeout_operator':None}}
whole=audit('whole',P/'evidence/e135-direct-six-row-v1',P/'direct-six-row-v1/mlp.rs',P/'build-e135-direct-six-row-v1-repro.json',ART/'e135-direct-six-row-v1/exact-kernel.bin',P/'evidence/arena-110314.log',110314)
probe=audit('probe',P/'validation/probe-evidence-v1',P/'validation/probe-v1/probe_module.rs',P/'validation/probe-build-v1.json',ART/'e135-direct-probe-v1/kernels/furiosa-opt-gemma4/furiosa_opt_gemma4::stream_probe::upgate.bin',P/'validation/probe-evidence-v1/arena-110316.log',110316)
out={'passed':True,'meaning':'Audit assertions passed; original kernel copies violate bounds and overwrite live control bitmap. This is not a kernel correctness PASS.','whole':whole,'standalone_upgate_oracle':probe,'conclusion':{'same_bad_up_quantizer_copies_in_both':True,'same_control_bitmap_role_corrupted_in_both':True,'affected_bitmap_entries_each_image':16,'numeric_lut_instead_of_switch_bitmap':False,'specific_source_hazard_supports_copy_only_rerun':True,'timeout_sole_cause_proven':False,'sdk_fault_proven':False},'minimal_next_diagnostic':'Freeze exact E135 source with only three up-quantizer copies changed to Sub480B bounded commits. Audit same bounds; run standard whole and unchanged upgate oracle. E137 O/source composition and control must be recorded explicitly; do not silently treat old E134 full source as current E137.','limitations':['No per-operator trace or corrupted SRAM capture exists; bitmap write overlap does not prove the exact hardware stall location.','The observed payload values after overread are unknown; do not infer particular bitmap bits become zero or a specific NoC failure.','No new compile/NPU was run by this audit.'],'helper':record(HELPER)}
out=json.loads(json.dumps(out))
if a.check:assert json.loads(a.output.read_text())==out
else:a.output.write_text(json.dumps(out,indent=2)+'\n')
print('PASS audit: whole and oracle both overwrite live ring32 custom_snoop_bitmap; timeout causal confirmation pending copy-only rerun.')
