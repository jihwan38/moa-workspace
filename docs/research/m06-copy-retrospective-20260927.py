#!/usr/bin/env python3
"""Read preserved M06 source/schedules/images; no builds or runtime calls."""
import argparse,hashlib,json,re
from pathlib import Path
SCRIPT=Path(__file__).resolve()
R=next((p for p in SCRIPT.parents if (p/'AGENTS.md').is_file() and (p/'moa-dev/experiments/candidates').is_dir()),None)
if R is None:raise RuntimeError('Cannot locate repository from this script path')
C=R/'moa-dev/experiments/candidates/m06-sharded-input-pipeline';A=R/'moa-dev/experiments/artifacts/m06-sharded-input-pipeline'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
def digest(p):
 b=p.read_bytes();return {'path':str(p.relative_to(R)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def audit(label,variant):
 source=C/variant/'mlp.rs'; ops=C/variant/'ops.rs'; schedule=C/'evidence'/label/'schedule.json';build=C/f'build-{label}-repro.json';image=A/label/'exact-kernel.bin'
 b=json.loads(build.read_text());s=json.loads(schedule.read_text());ts={int(t['index']):t for t in s['tensors']};text=source.read_text()
 assert digest(source)['sha256']==b['source_inputs']['src/device/shared/mlp.rs']['sha256']
 assert digest(schedule)['sha256']==b['schedule']['sha256']
 assert digest(image)['sha256']==b['exact_image']['sha256']
 copies=[i for i in s['instructions'] if i['tpe']=='DmaStos' and re.search(r'(hi|lo|third)\.view\(\)\.to_dm_view',i['description'])]
 switch=next(i for i in s['instructions'] if i['tpe']=='Main' and 'CustomBroadcast { ring_size: 32 }' in i['description'] and 'm![FfnTerm]' in i['description'])
 bitmap=ts[switch['input_tensors'][1]]; assert bitmap['size']==16384
 payload=b''.join((1<<(32*(i//32))).to_bytes(32,'little') for c in range(2) for i in range(256));raw=image.read_bytes();assert raw.count(payload)==1
 pack=next(t for t in s['tensors'] if 'Element: (FfnTerm, H % 480)' in t['shape']);base=pack['address']//512;assert pack['size']==1440*512
 scale=next(i for i in s['instructions'] if i['tpe']=='DmaStos' and 'scale.to_dm' in i['description']);scale_t=ts[scale['output_tensors'][0]];scale_base=scale_t['address']//512
 out={'inputs':[digest(p) for p in [source,ops,schedule,build,image]],'graph_or_visa_preserved':False,'packed':{'schedule_tensor':pack['index'],'base_per_slice':base,'bytes_per_slice':1440},'ring32_bitmap':{'schedule_tensor':bitmap['index'],'base_per_slice':bitmap['address']//512,'bytes_per_slice':32,'lifetime':bitmap['lifetime'],'switch_schedule_index':switch['index'],'embedded_constant_unique_offset':raw.index(payload)},'restored_scale':{'schedule_dma':scale['index'],'lifetime':scale['lifetime'],'base_per_slice':scale_base,'bytes_per_slice':32},'packing_dma_count':len(copies),'actual_dma_descriptors_available':False,'conditional_hazards':[]}
 if variant.endswith('v5'):
  assert len(copies)==3
  for term,i in enumerate(copies):
   st=ts[i['input_tensors'][0]];assert st['size']==480*512
   lo=base+480*term;hi=lo+1440
   out['conditional_hazards'].append({'term':term,'schedule_index':i['index'],'source_initialized_bounds':[st['address']//512,st['address']//512+480],'static_lifetime':i['lifetime'],'IF_actual_DMA_extent_matches_E140_1440B':{'write_bounds':[lo,hi],'past_packed_bytes':max(0,hi-(base+1440)),'ring32_bitmap_overlap':max(lo,bitmap['address']//512)<min(hi,bitmap['address']//512+32),'restored_scale_overlap':max(lo,scale_base)<min(hi,scale_base+32),'after_scale_dma_end':i['lifetime']['begin']>=scale['lifetime']['end']},'actual_extent_unverified':True})
  assert out['conditional_hazards'][2]['IF_actual_DMA_extent_matches_E140_1440B']['restored_scale_overlap']
  assert out['conditional_hazards'][2]['IF_actual_DMA_extent_matches_E140_1440B']['after_scale_dma_end']
 else:
  assert not copies
  out['source_direct_commits']={'lines':[521,537,559],'each_logical_elements':480,'packet_bytes':8,'packets':60,'term_offsets':[0,480,960],'logical_union_bounds':[base,base+1440],'actual_commit_descriptors_unavailable':True}
 return out
out={'scope':'read-only historical audit; schedule lifetimes are compiler estimates, not measured engine spans','v4':audit('m06-sharded-packed-v4','sharded-packed-v4'),'v5':audit('m06-sharded-dma-v5','sharded-dma-terms-v5'),'conclusion':{'same_three_packing_DMA_bug_in_both':False,'v5_same_risky_source_pattern':True,'v5_actual_1440B_extent_recovered':False,'verified_live_bitmap_overwrite':False,'new_timeout_sole_cause_proven':False,'e140_repair_alone_justifies_claiming_m06_recovered':False}}
target=SCRIPT.with_suffix('.json')
if args.check:assert json.loads(target.read_text())==out,'Frozen audit differs from current evidence'
else:target.write_text(json.dumps(out,indent=2)+'\n')
print('PASS M06 source/schedule/image audit; conditional extents remain unverified')
