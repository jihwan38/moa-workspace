#!/usr/bin/env python3
"""Independent source/finite-Max audit only. No compiler, NPU, or production writes."""
import argparse, hashlib, json, random, struct
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=HERE.parent/'sharded-v1'
R=HERE.parents[4]
OUT=Path(__file__).with_suffix('.json')
def record(p):
 b=p.read_bytes();return {'path':str(p.relative_to(R)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def fn(s,name):
 start=s.index('fn '+name); a=s.index('{',start); b=a+1; depth=1
 while depth:
  depth+=(s[b]=='{')-(s[b]=='}');b+=1
 return s[start:b]
def body(s):return s[s.index('{')+1:]
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def value(x):return struct.unpack('<f',struct.pack('<I',x))[0]
def run():
 paths={'ops.rs':'src/ops.rs','rmsnorm.rs':'src/device/shared/rmsnorm.rs','mlp.rs':'src/device/shared/mlp.rs'}
 old={k:(HERE/('base-'+k)).read_text() for k,v in paths.items()}
 new={k:(P/k).read_text() for k in paths}
 assert hashlib.sha256(old['mlp.rs'].encode()).hexdigest()=='e412ef9422a17cba47b94bbda8955cf45be98825e7dd3577efa09cab6d50f893'
 # Complete original function remains, and new helper shares its exact body through F32 DM.
 original=fn(old['rmsnorm.rs'],'normalize<');assert fn(new['rmsnorm.rs'],'normalize<')==original
 sharded=fn(new['rmsnorm.rs'],'normalize_ffn_input_sharded<')
 marker='    device\n        .main\n        .begin(normalized.view())'
 assert body(original).rsplit(marker,1)[0]==body(sharded).rsplit(marker,1)[0]
 oldtail=body(original).rsplit(marker,1)[1]
 newtail=body(sharded).rsplit(marker,1)[1]
 expected=oldtail.replace('.fetch::<m![1], m![H % 480]>()\n        .switch::<Slice, m![H / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })\n        .collect::<m![H / 8], m![H % 8]>()','.fetch::<m![H / 8 % 60], m![H % 8]>()\n        .collect::<m![H / 8 % 60], m![H % 8]>()')
 assert newtail==expected
 assert new['ops.rs']==old['ops.rs'].replace('shared::rmsnorm::normalize(device, &residual, pre_ff_rms_weight)','shared::rmsnorm::normalize_ffn_input_sharded(device, &residual, pre_ff_rms_weight)')
 unchanged=['project_rows_with_scale_contraction','project_up_and_gate','geglu','prepare_down_fp8_split','project_down_native_half','project_down_tile','finish_down','broadcast_ffn_quantized']
 for n in unchanged:assert fn(old['mlp.rs'],n+'(')==fn(new['mlp.rs'],n+'('),n
 production=fn(new['mlp.rs'],'feedforward_sharded(')
 wrapper=fn(new['mlp.rs'],'prepare_ffn_fp8_split(')
 quant=fn(new['mlp.rs'],'prepare_ffn_fp8_split_sharded(')
 assert 'prepare_ffn_fp8_split_sharded(device, &x)' in production
 assert 'prepare_ffn_fp8_split_sharded(device, &x)' in wrapper
 assert 'Cluster, FfnQuantSlices, m![H % 480]> = x.to_dm' in wrapper
 oq=fn(old['mlp.rs'],'prepare_ffn_fp8_split(')
 oldterms=oq[oq.index('    let hi:'):oq.index('    (broadcast_ffn_quantized')]
 newterms=quant[quant.index('    let hi:'):quant.index('    // Replicate only prepared')]
 assert newterms.replace('Chip, Cluster, FfnQuantSlices','Chip, m![2], FfnQuantSlices')==oldterms
 oldscale=oq[oq.index('        .main.begin(amax.view())'):oq.index('    let quant_scale:')]
 newscale=quant[quant.index('        .main.begin(amax.view())'):quant.index('    let quant_scale:')]
 assert oldscale==newscale
 probe=R/'moa-dev/experiments/candidates/ffn-direct-full30-partials/validation/probe-v1/probe_module.rs'
 assert 'let (x, scale) = prepare_ffn_fp8_split(device, &x);' in probe.read_text()
 layout=R/'furiosa-opt-gemma4-12B/src/device/layout.rs'
 assert 'type Cluster = m![1 # 2];' in layout.read_text()
 # Finite BF16 Max/floor equivalence: abs turns signed zero positive; Max is exact/associative.
 floor=f32(1.925929944387236e-34);assert floor==2.0**-112
 vals=[value(b<<16) for b in range(65536) if b&0x7f80!=0x7f80]
 assert len(vals)==65280
 def scale(m):return f32(value(bits(max(m,floor))&0x7f800000)*2.0**-14)
 for x in vals:
  direct=scale(abs(x))
  local=max(abs(x),floor)
  assert bits(direct)==bits(scale(max([local]+[floor]*7)))
 rng=random.Random(153);rng.shuffle(vals)
 for i in range(17):
  xs=vals[3840*i:3840*(i+1)]
  local=[max([floor]+[abs(v) for v in xs[480*q:480*(q+1)]]) for q in range(8)]
  assert bits(scale(max(abs(v) for v in xs)))==bits(scale(max(local)))
 assert scale(0.0)==2.0**-126 and scale(value(0x7f7f0000))==2.0**113
 # Source-coordinate proof only: original H order, packed term offsets, late replication.
 assert sorted(480*q+u for q in range(8) for u in range(480))==list(range(3840))
 assert sorted(480*t+u for t in range(3) for u in range(480))==list(range(1440))
 copied=[(c,q,t,u) for c in range(2) for q in range(8) for t in range(3) for u in range(480)]
 assert len(copied)==23040
 return {'valid':True,'scope':'source-only; not compiled or executed; no target lowering/liveness or general model guarantee',
 'inputs':{**{'base_'+k:record(HERE/('base-'+k)) for k,v in paths.items()},**{k:record(P/k) for k in paths},'raw_wrapper':record(probe),'layout':record(layout)},
 'source':{'original_rms_function_unchanged':True,'same_rms_arithmetic_and_normalized_f32_dm':True,'final_cast_identical_after_routing_removed':True,'unchanged_mlp_functions':unchanged,'same_quantizer_arithmetic_and_bounded_copies_after_cluster_relabel':True,'same_scale_formula':True,'raw_wrapper_calls_actual_sharded_quantizer':True,'raw_does_not_test_pre_rms':True},
 'finite_amax':{'bf16_patterns':65280,'all_scalar_patterns':True,'all_pattern_grouped_cases':17,'grouped_case_width':3840,'mathematical_basis':'max(max(abs(x_q)), floor) is exact and associative for finite values; floor commutes with max; exponent-mask and F32 power2 multiplication unchanged','zero_scale':2.0**-126,'maximum_finite_scale':2.0**113,'nonfinite_equivalence_claimed':False},
 'source_geometry':{'valid_input_cluster':0,'valid_input_slices':list(range(8)),'h_per_slice':480,'term_copy_bytes':480,'term_copy_offsets':[0,480,960],'packed_extent_per_slice':1440,'late_copy_destination_bytes':23040,'replication_mapping':'dst(c,q,t,u) <- src(0,q,t,u); original H=480q+u'},
 'logical_costs':{'original_early_bf16_dma_destination_bytes':[15360,15360],'new_late_packed_dma_destination_bytes':23040,'cross_cluster_payload_old_bf16':7680,'cross_cluster_payload_new_packed':11520,'retained_rms_local_cast_read_f32_bytes':15360,'retained_rms_local_cast_write_bf16_bytes':7680,'retained_ring_seed_destination_bytes':184320,'retained_ring_broadcast_destination_bytes':5898240,'added_global_max_streams':1,'not_measured_bus_traffic_or_cycle_savings':True},
 'required_actual_gate':['three source overrides and43 unchanged inputs; QKV/O image identity','same RMS operations/constants up to normalized F32 DM and local BF16 cast','valid CL0/slices0..7; inter-Max actual mask/ratio and scalar distribution','three bounded480B Sub copies and ring32 bitmap safe addresses/lifetimes','late packed23040B descriptor source/destination coverage and H/Term order','8KiB VRF/TRF actual lifetimes including gamma and unchanged native/scales','actual raw wrapper→new quantizer; original fixture/expected bits preserved; fullFFN tests preRMS']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');a=ap.parse_args();r=run();r=json.loads(json.dumps(r));
 if a.check:assert json.loads(OUT.read_text())==r,'Review differs'
 else:OUT.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({'valid':r['valid'],'scope':r['scope'],'finite_bf16_patterns':r['finite_amax']['bf16_patterns']}))
if __name__=='__main__':main()
