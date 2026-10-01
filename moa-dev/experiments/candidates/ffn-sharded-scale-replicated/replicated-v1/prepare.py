#!/usr/bin/env python3
"""Prepare E154's source-only replicated-Max/scale repair from frozen E153."""
from pathlib import Path
import argparse, hashlib, json, difflib
P=Path(__file__).resolve().parent
DEFAULT=P.parents[1]/'ffn-prerms-sharded-consumer/sharded-v1'
EXPECTED={'ops.rs':'59a4ab6414d362876255caa8f332a01a567efd5ded0afa2d43da7ffe34e3c61e','rmsnorm.rs':'e2aa2efffd7cd6e257772c679b8655f42041963ed568688693842a466546bfc2','mlp.rs':'ea39b36a37aac00f1f1293bbe3ed5bd776c552dfefc7fbb490068df570eda756'}
def dg(b):return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=DEFAULT);ap.add_argument('--check',action='store_true');a=ap.parse_args()
 base={n:(a.source/n).read_bytes() for n in EXPECTED}
 for n,b in base.items():assert dg(b)['sha256']==EXPECTED[n],n
 out=dict(base);s=base['mlp.rs'].decode()
 def one(old,new):
  nonlocal s
  assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new,1)
 one('use crate::axes::{Dummy2, Dummy256, H, L};','use crate::axes::{Dummy2, Dummy8, Dummy256, H, L};')
 one('    let amax: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = device.main\n        .begin(local_amax.view())','''    // Materialize the same global finite Max on the eight existing valid
    // cluster0 shards. Dummy8 uses the original RMS reduction output mapping.
    let amax: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = device.main
        .begin(local_amax.view())''')
 one('        .vector_inter_slice_reduce::<Slice, m![1]>(InterSliceReduceOpF32::Max)','        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Max)')
 anchor='    // Exponent mask produces 2^floor(log2(amax)); floor 2^-112 guarantees\n    // scale >= 2^-126, and finite BF16 inputs imply scale <= 2^113.\n    let scale: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = device'
 one(anchor,'''    // Whole-allocation relabel: physical slices0..7 stay valid, with no
    // permutation and no read from the remaining248 padded slice positions.
    let amax: DmTensor<f32, Chip, Cluster, FfnQuantSlices, m![1 # 8]> = unsafe { amax.reshape() };
'''+anchor)
 one('    let scale: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = device\n        .main.begin(amax.view())','    let scale: DmTensor<f32, Chip, Cluster, FfnQuantSlices, m![1 # 8]> = device\n        .main.begin(amax.view())')
 one('''    let quant_scale: DmTensor<f32, Chip, Cluster, FfnQuantSlices, m![1 # 8]> = scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, Cluster, FfnQuantSlices, m![1 # 8]> = device
        .sub.begin(quant_scale.view())''','''    // The scale producer already occupies the quantizer's eight H480 shards.
    // The existing Sub load needs no scalar redistribution DMA.
    let scale_vrf: VrfTensor<f32, Chip, Cluster, FfnQuantSlices, m![1 # 8]> = device
        .sub.begin(scale.view())''')
 one('    (broadcast_ffn_quantized(device, &packed), scale.to_dm(&mut device.tdma))','''    // Every active source slice contains the identical global scale. Relabel
    // H/480 as anonymous8 replicas without shrinking the physical/live mask.
    // Final all512-slice replication remains a real typed DMA; packet padding
    // is not additional useful scalar data.
    let shared_scale: DmTensor<f32, Chip, Cluster, m![1 # 32, 8], m![1 # 8]> = unsafe { scale.reshape() };
    (broadcast_ffn_quantized(device, &packed), shared_scale.to_dm(&mut device.tdma))''')
 out['mlp.rs']=s.encode()
 # Exact common prefix and term/copy/broadcast arithmetic remain byte identical.
 a0=base['mlp.rs'].decode();before='fn prepare_ffn_fp8_split_sharded('
 assert a0[:a0.index(before)].replace('Dummy2, Dummy256','Dummy2, Dummy8, Dummy256')==s[:s.index(before)]
 qstart='    let hi: DmTensor<f8e5m2, Chip, Cluster, FfnQuantSlices'
 qend='    (broadcast_ffn_quantized(device, &packed), scale.to_dm(&mut device.tdma))'
 unchanged=a0[a0.index(qstart):a0.index(qend)]
 assert s[s.index(qstart):s.index('    // Every active source slice')]==unchanged
 contract={
 'status':'source_prepared_not_compiled_not_measured','candidate':'E154 conditional repair of E153',
 'source_origin':'E151 plus frozen E153 three-file ownership change; E153 is not adopted',
 'performance_control':'E151','source_inputs':{n:dg(b) for n,b in base.items()},'prepared_sources':{n:dg(b) for n,b in out.items()},
 'changed_from_E153':['global Max output canonical1 -> namedDummy8 at physical0..7','wire-order reshape Dummy8 -> H/480 retains8valid/256physical','pow2 scale runs on the same8valid slices, existing Sub directly reads it; one quant_scale DMA removed','final scale wire-order reshape H/480 -> anonymous8, original typed all512 replication retained'],
 'unchanged':['ops.rs fullbytes','rmsnorm.rs fullbytes','all MLP outside sharded quantizer except Dummy8 import','local abs-Max/floor expression','power-of-two mask/multiplication expression','three E5 terms and existing Sub residual addition','three exact480B Subcopies','late23040B packed destination and original184320B seed/5898240B ring','native/weight/scale/reduction/postRMS/down arithmetic'],
 'mapping':{'input_cluster':0,'input_valid_physical_slices':list(range(8)),'physical_slices_per_cluster':256,'h_of_slice':'h=480*slice+u, 0<=u<480','replicated_amax_mapping':'[1#32,Dummy8]','quant_scale_mapping':'[1#32,H/480]','restore_source_mapping':'[1#32,8]','scalar_packet':'one logical F32 scalar in 32B padded packet; padding is not scalar replicas'},
 'source_api_evidence':['original RMS global Add uses [1#32,Dummy8] and H/480 owned reshape','M06v1/v2 contain same replicated Max but failed SFR; no attribution to this operation alone','adopted E137 O uses named equal scale -> anonymous replicas -> typed DMA','SDK0.8.1 DmTensor::reshape retains per-level size and wire order; DmTensor::to_dm validates coverage/alignment'],
 'required_emitted_gates':['Max reads original8shards and commits exactly8identical scalar positions; actual ratio/mask recorded','all relabels preserve offset/wire order/valid mask; no canonical-mask shrink','Sub scale reader has correct32B packet and quantizer owner','final scale typed DMA reads initialized replicas only and writes all512 existing native consumers; source extent/strides/bounds and padding tracked','late packed and all3copies match E153','whole VRF overlaps/lifetimes including RMS weights/hi/sum/native scale/blocks checked'],
 'numerical_scope':'Finite BF16 Max/floor replicated deterministically; no intermediate RMS NPU observation or arbitrary BF16 whole-model promise. Original boundary fixtures/tolerances remain required.',
 'cost_scope':'Remove one scalar TDMA boundary, increase globalMax/pow2 active scalar output from1 to8; final all512 scale DMA remains. Actual encoded bytes and runtime benefit require emitted graph/profile/whole measurement.',
 'prior_scope':'M06v1/v2 SFR868; v3 canonicalMax still failed, v4 moved Subsum toMain and compiled then timed out. E153 same Subsum+separate480copies already exact/NPU45PASS; new replicatedMax composition still uncompiled.'}
 files={**out,'source.diff':''.join(''.join(difflib.unified_diff(base[n].decode().splitlines(True),out[n].decode().splitlines(True),fromfile='frozen-E153/'+n,tofile='prepared-E154/'+n)) for n in base).encode(),'source-contract.json':(json.dumps(contract,indent=2)+'\n').encode()}
 for n,b in files.items():
  if a.check:assert (P/n).read_bytes()==b,n
  else:(P/n).write_bytes(b)
 print(json.dumps({'passed':True,'source_only':True,'prepared_sources':contract['prepared_sources']}))
if __name__=='__main__':main()
