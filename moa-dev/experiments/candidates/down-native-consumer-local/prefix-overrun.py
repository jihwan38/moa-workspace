#!/usr/bin/env python3
"""Read-only E140/E137 up-quantizer typed DMA bound audit, without NPU inference."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/'moa-dev/experiments/candidates'
p=BASE/'ffn-native-down-weighted-residual/verify_graph.py'
s=importlib.util.spec_from_file_location('h',p);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
def record(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def audit(family,label,source):
 d=BASE/family/'evidence'/label; gp=d/'graph.json';sp=d/'schedule.json';g=h.Graph(json.loads(gp.read_text()));sch=json.loads(sp.read_text());ops={int(o['name'].split('#O')[-1]):o for o in g.ops.values()};ts={int(t['index']):t for t in sch['tensors']}
 copies=[]
 for term,n in enumerate([30,35,40]):
  o=ops[n];view=g.view(o['inputs'][1]);assert (view['stride_elements'],view['axis_extent'],view['window'],view['start'])==(480,3,1,term)
  prod=g.producers[o['inputs'][0]][1];assert prod['kind']=='TuExec';assert h.offsets(h.sequence(prod['description'],'commit_projection'))==list(range(480))
  packetbounds=[]
  for pair in h.dma_descriptors(o['description']).values():
   for side in ['src','dst']:
    a=pair[side];addr=a['address'];ivs=list(h.intervals(addr['in_slice'],a['dims']))
    perslice={}
    for lo,hi in ivs:
     sl=lo//4194304;perslice.setdefault(sl,[]).append((lo%4194304,hi%4194304))
    assert set(perslice)==set(range(8))
    for spans in perslice.values():h.full_coverage(spans,1440,'actual DMA transfer')
   packetbounds.append({'cluster':pair['dst']['address']['cluster'],'source_slices':list(range(8)),'per_slice_transfer_bytes':1440,'packets_per_slice':2,'packet_bytes':720})
  inst=next(i for i in sch['instructions'] if i['tpe']=='DmaStos' and ('hi','lo','third')[term]+'.view().to_dm_view' in i['description'])
  src=ts[inst['input_tensors'][0]];root=g.root_output(view['parent']);base=int(g.tensor(root)['optionals']['address']);start=base+480*term
  copies.append({'graph_operator':o['name'],'source_tensor':o['inputs'][0],'source_initialized_bytes_per_slice':480,'source_allocated_bytes_per_slice':src['size']//512,'actual_read_bytes_per_slice':1440,'read_excess_bytes_per_slice':960,'root_destination_tensor':root,'root_base_per_slice':base,'root_size_per_slice':1440,'index_term':term,'view_stride_bytes':480,'absolute_write_bounds_per_slice':[start,start+1440],'logical_write_bounds_per_slice':[start-base,start-base+1440],'schedule_dma_index':inst['index'],'schedule_time':inst['lifetime'],'descriptor_summary':packetbounds,'actual_descriptor':o['description']})
 load=next(i for i in sch['instructions'] if i['tpe']=='DmaLoad' and 'down_global_scale.to_dm' in i['description']);scalet=ts[load['output_tensors'][0]];ab=scalet['address']//512;ae=ab+scalet['size']//512
 consumers=[i for i in sch['instructions'] if load['output_tensors'][0] in i['input_tensors']]
 overlaps=[]
 for c in copies:
  b,e=c['absolute_write_bounds_per_slice'];t=c['schedule_time'];st=scalet['lifetime']
  if b<ae and ab<e and t['begin']<st['end'] and st['begin']<t['end']:
   overlaps.append({'term':c['index_term'],'byte_intersection':[max(b,ab),min(e,ae)],'cluster':0,'slices':list(range(8)),'dma_time':t})
 return {'inputdigests':{k:record(v) for k,v in {'source':BASE/family/source,'graph':gp,'schedule':sp,'build':BASE/family/f'build-{label}-repro.json'}.items()},'copies':copies,'down_global_scale':{'schedule_tensor':scalet,'per_slice_bounds':[ab,ae],'load':load,'consumers':consumers},'live_scalar_write_overlaps':overlaps,'unsafe_source_contract':True}
out={'e140_local_v4':audit('down-native-consumer-local','e140-local-v4','local-v4/mlp.rs'),'e137_ffn_e134_weighted_v3':audit('ffn-native-down-weighted-residual','e134-weighted-v3','weighted-v3/mlp.rs'),'conclusions':{'compact_source_480_actual_dma_1440':True,'e140_live_down_global_scale_descriptor_overlap':True,'same_invalid_dma_pattern_present_in_e137':True,'e137_down_global_scale_overlap':False,'numeric_root_cause_confirmed_by_repaired_npu':False,'minimal_discriminating_repair':'Replace only the three up-quantizer typed DMAs with exact480B Sub packet copies. Compare original E137, repaired E137, and repaired local-v4; unchanged down oracle already passed all9cases.'},'limits':['IndexAccess pointer offsets are derived from actual stride/window/scalar dependencies; graph address text alone omits dynamic offsets.','Descriptor/lifetime overlap is established; repaired whole NPU has not yet validated causal attribution.','Raw down oracle excludes this up-quantizer and feedforward down_global_scale epilogue.']}
assert len(out['e140_local_v4']['live_scalar_write_overlaps'])==2
assert not out['e137_ffn_e134_weighted_v3']['live_scalar_write_overlaps']
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
report=HERE/'prefix-overrun.json'
if args.check:assert json.loads(report.read_text())==out,'Frozen prefix audit differs'
else:report.write_text(json.dumps(out,indent=2)+'\n')
print('E140 typed-copy bound violation and live scalar overlap established; numerical attribution pending repaired NPU.')
