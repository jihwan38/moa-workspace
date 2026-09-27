"""CPU coordinate contract only; not emitted-descriptor/lowering validation."""
import json,hashlib
from pathlib import Path
root=Path('/home/jihwan38/moa-workspace');src=root/'furiosa-opt-gemma4-12B/src/device/shared/mlp.rs'
seen=set();direct_packets=0;canonical_stripes=0
for c in range(2):
 for group in range(64):
  covered=[]
  for q in range(4):
   canonical_stripes+=1
   for j in range(16):
    src_slice=4*(16*q+j);src_l=120*(src_slice//4)
    dst_slice=4*group+q;dst_off=240*j
    assert src_l==1920*q+120*j and src_slice%4==0
    assert 0<=src_slice<256 and 0<=dst_slice<256 and dst_off+240<=3840
    direct_packets+=1
    for u in range(120):
     l=1920*q+120*j+u
     assert src_l+u==l
     key=(c,group,l);assert key not in seen;seen.add(key);covered.append(l)
  assert sorted(covered)==list(range(7680))
b=src.read_bytes();r={'passed':True,'scope':'CPU proposed layout contract only; no candidate source, actual descriptor, compile or NPU result','source':{'path':str(src.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()},'same_cluster_groups':128,'distinct_inputs_per_group':7680,'canonical_source':'same cluster canonical slice0, byte=2*(1920*q+u)','destination':'same cluster slice=4*g+q, byte=2*u','canonical_logical_stripes':canonical_stripes,'canonical_stripe_bytes':3840,'canonical_actual_packet_geometry':'not compiled; old E134 E5 logical1920B leader stripe splits2x960B','direct_geglu_valid_packet_contract':{'count':direct_packets,'bytes':240},'bf16_replication_bytes':1966080,'e5_three_local_term_bytes':2949120,'largest_f32_residual_vrf_bytes_per_slice':7680,'scalar_vrf_bytes_per_slice':32,'native_term_coefficient_vrf_bytes_per_slice':96,'later_blockscale_vrf_bytes_per_slice':5760,'required_runtime_gates':['globalmax within cluster and each Hgroup over K4','no crosscluster or differentK scale substitution','source exact finite normalization arithmetic unchanged','actual fullschedule VRF allocation/lifetimes and nohiddenreload/spill assumption']}
Path('/tmp/moa-e140-down-local-mapping.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS proposed-coordinate contract (CPU only)')
