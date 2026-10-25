//! Shared QKV with physically blocked outputs for a matching head consumer.
//! SharedRow64 names the same physical output-group indices in Qs/64 and Ps/32.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,H,Qs,Ps}};
axes![SharedRow64=64];
pub type QueryOutput=m![P128/64,Qs/64,P128%64,Qs%64];
pub type KeyOutput=m![P128/64,Ps/32,P128%64,Ps%32];
type Cluster=m![P128/64];
type SharedParts=m![SharedRow64,H/960];
type SharedRows=m![SharedRow64,1#4];
type QsParts=m![Qs/64,H/960];
type QsRows=m![Qs/64,1#4];
type PsParts=m![Ps/32,H/960];
type PsRows=m![Ps/32,1#4];
fn pack_scales(
    device: &mut Device,
    xscale: &DmTensor<f32, Chip, Cluster, SharedRows, m![P128 % 64,1 # 8]>,
) -> DmTensor<f32, Chip, Cluster, SharedRows, m![P128 % 64]> {
    // Keep the 32-byte scalar slot for aligned inter-slice DMA, then pack by TU.
    let xscale: DmTensor<f32, Chip, Cluster, SharedRows, m![P128 % 64]> = device
        .sub
        .begin(xscale.view())
        .fetch::<m![P128 % 64], m![1 # 8]>()
        .collect::<m![P128 % 64], m![1 # 8]>()
        .transpose::<m![P128 % 64 / 2], m![P128 % 64 % 2 # 8]>()
        .commit_trim::<m![P128 % 64 % 2]>()
        .commit();
    xscale
}

fn matrix_query(device:&mut Device,
 x:&TrfTensor<f8e4m3,Chip,Cluster,SharedParts,m![P128%8],m![P128/8%8,H%960]>,
 xs:&VrfTensor<f32,Chip,Cluster,SharedRows,m![P128%64]>,
 weight:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Qs%64,H%960]>)
 ->DmTensor<bf16,Chip,Cluster,SharedRows,m![Qs%64,P128%64]> {
    device
        .main
        .begin(weight.view())
        .fetch::<m![Qs % 64, H / 32 % 30], m![H % 32]>()
        .collect::<m![Qs % 64, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![Qs % 64, H / 64 % 15, P128 / 8 % 8], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 64, P128 / 8 % 8]>()
        .contract_lane::<m![Qs % 64, P128 / 8 % 8], m![P128 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SharedRows, m![Qs % 64, P128 / 8 % 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs % 64, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), xs)
        .vector_widen_concat::<m![Qs % 64, P128 / 8 % 8], m![P128 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128%8#16]>()
        .commit_trim::<m![P128 % 8]>()
        .commit()
}
fn matrix_sliding_kv(device:&mut Device,
 x:&TrfTensor<f8e4m3,Chip,Cluster,SharedParts,m![P128%8],m![P128/8%8,H%960]>,
 xs:&VrfTensor<f32,Chip,Cluster,SharedRows,m![P128%64]>,
 weight:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Ps%32,H%960]>)
 ->DmTensor<bf16,Chip,Cluster,SharedRows,m![Ps%32,P128%64]> {
    device
        .main
        .begin(weight.view())
        .fetch::<m![Ps % 32, H / 32 % 30], m![H % 32]>()
        .collect::<m![Ps % 32, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![Ps % 32, H / 64 % 15, P128 / 8 % 8], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 32, P128 / 8 % 8]>()
        .contract_lane::<m![Ps % 32, P128 / 8 % 8], m![P128 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SharedRows, m![Ps % 32, P128 / 8 % 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps % 32, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), xs)
        .vector_widen_concat::<m![Ps % 32, P128 / 8 % 8], m![P128 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128%8#16]>()
        .commit_trim::<m![P128 % 8]>()
        .commit()
}

fn finish_query(
    device: &mut Device,
    dot: &DmTensor<bf16, Chip, Cluster, QsRows, m![Qs % 64, P128 % 64]>,
    scale: &HbmTensor<bf16, Chip, m![Qs]>,
    output: &mut HbmTensor<bf16, Chip, QueryOutput>,
) {
    let scale: DmTensor<bf16, Chip, Cluster, QsRows, m![Qs % 64]> = scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, QsRows, m![Qs % 64]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Qs % 64]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 8], m![Qs % 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, QsRows, m![Qs % 64, P128 % 64]> = device
        .main
        .begin(dot.view())
        .fetch::<m![Qs % 64, P128 / 8 % 8], m![P128 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs % 64, P128 / 8 % 8], m![P128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs % 64, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Qs % 64, P128 / 8 % 8], m![P128 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128%8#16]>()
        .commit_trim::<m![P128 % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, QsRows, m![P128 % 64, Qs % 64]> = device
        .main
        .begin(result.view())
        .fetch::<m![P128 / 16 % 4, Qs % 64], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, Qs % 64], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, Qs / 4 % 16, P128 % 16], m![Qs%4#16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

fn finish_sliding_kv(
    device: &mut Device,
    dot: &DmTensor<bf16, Chip, Cluster, PsRows, m![Ps % 32, P128 % 64]>,
    scale: &HbmTensor<bf16, Chip, m![Ps]>,
    output: &mut HbmTensor<bf16, Chip, KeyOutput>,
) {
    let scale: DmTensor<bf16, Chip, Cluster, PsRows, m![Ps % 32]> = scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, PsRows, m![Ps % 32]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Ps % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Ps / 8 % 4], m![Ps % 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, PsRows, m![Ps % 32, P128 % 64]> = device
        .main
        .begin(dot.view())
        .fetch::<m![Ps % 32, P128 / 8 % 8], m![P128 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Ps % 32, P128 / 8 % 8], m![P128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps % 32, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Ps % 32, P128 / 8 % 8], m![P128 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128%8#16]>()
        .commit_trim::<m![P128 % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, PsRows, m![P128 % 64, Ps % 32]> = device
        .main
        .begin(result.view())
        .fetch::<m![P128 / 16 % 4, Ps % 32], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, Ps % 32], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, Ps / 4 % 8, P128 % 16], m![Ps%4#16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

fn project_one(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![P128,H]>,
 xscale:&HbmTensor<f32,Chip,m![P128,1#8]>,
 qw:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Qs%64,H%960]>,
 kw:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Ps%32,H%960]>,
 vw:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Ps%32,H%960]>,
 qs:&HbmTensor<bf16,Chip,m![Qs]>,ks:&HbmTensor<bf16,Chip,m![Ps]>,vs:&HbmTensor<bf16,Chip,m![Ps]>,
 qo:&mut HbmTensor<bf16,Chip,QueryOutput>,ko:&mut HbmTensor<bf16,Chip,KeyOutput>,vo:&mut HbmTensor<bf16,Chip,KeyOutput>) {
 // One schedule epoch releases the shared full TRF before the next independent group.
 for _epoch in 0..1 {
 let input:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![P128%64,H%960]>=input.to_dm(&mut device.tdma);
 let scales:DmTensor<f32,Chip,Cluster,SharedRows,m![P128%64,1#8]>=xscale.to_dm(&mut device.tdma);
 let scales=pack_scales(device,&scales);
 let xs:VrfTensor<f32,Chip,Cluster,SharedRows,m![P128%64]>=device.sub.begin(scales.view())
  .fetch::<m![P128/8%8],m![P128%8]>().collect::<m![P128/8%8],m![P128%8]>().to_vrf();
 let x:TrfTensor<f8e4m3,Chip,Cluster,SharedParts,m![P128%8],m![P128/8%8,H%960]>=device.sub.begin(input.view())
  .fetch::<m![P128%8,P128/8%8,H/32%30],m![H%32]>().collect::<m![P128%8,P128/8%8,H/32%30],m![H%32]>().to_trf();
 let q=matrix_query(device,&x,&xs,qw);
 let k=matrix_sliding_kv(device,&x,&xs,kw);
 let v=matrix_sliding_kv(device,&x,&xs,vw);
 // Only names change: group g owns Qs[64g..64g+64] or Ps[32g..32g+32].
 // Every cluster/slice/local element keeps its exact physical wire position.
 let q:DmTensor<bf16,Chip,Cluster,QsRows,m![Qs%64,P128%64]>=unsafe{q.reshape()};
 let k:DmTensor<bf16,Chip,Cluster,PsRows,m![Ps%32,P128%64]>=unsafe{k.reshape()};
 let v:DmTensor<bf16,Chip,Cluster,PsRows,m![Ps%32,P128%64]>=unsafe{v.reshape()};
 finish_query(device,&q,qs,qo);
 finish_sliding_kv(device,&k,ks,ko);
 finish_sliding_kv(device,&v,vs,vo);

 }
}

#[device(chip=1)]
pub fn project_sliding_qkv_four(device:&mut Device,
 a:&HbmTensor<f8e4m3,Chip,m![P128,H]>,sa:&HbmTensor<f32,Chip,m![P128,1#8]>,
 b:&HbmTensor<f8e4m3,Chip,m![P128,H]>,sb:&HbmTensor<f32,Chip,m![P128,1#8]>,
 c:&HbmTensor<f8e4m3,Chip,m![P128,H]>,sc:&HbmTensor<f32,Chip,m![P128,1#8]>,
 d:&HbmTensor<f8e4m3,Chip,m![P128,H]>,sd:&HbmTensor<f32,Chip,m![P128,1#8]>,
 qw:&HbmTensor<f8e4m3,Chip,m![Qs,H]>,kw:&HbmTensor<f8e4m3,Chip,m![Ps,H]>,vw:&HbmTensor<f8e4m3,Chip,m![Ps,H]>,
 qs:&HbmTensor<bf16,Chip,m![Qs]>,ks:&HbmTensor<bf16,Chip,m![Ps]>,vs:&HbmTensor<bf16,Chip,m![Ps]>,
 oa:(&mut HbmTensor<bf16,Chip,QueryOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>),
 ob:(&mut HbmTensor<bf16,Chip,QueryOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>),
 oc:(&mut HbmTensor<bf16,Chip,QueryOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>),
 od:(&mut HbmTensor<bf16,Chip,QueryOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>)) {
 let (qa,ka,va)=oa;let (qb,kb,vb)=ob;let (qc,kc,vc)=oc;let (qd,kd,vd)=od;
 let qw:DmTensor<f8e4m3,Chip,Cluster,QsParts,m![Qs%64,H%960]>=qw.to_dm(&mut device.tdma);
 let kw:DmTensor<f8e4m3,Chip,Cluster,PsParts,m![Ps%32,H%960]>=kw.to_dm(&mut device.tdma);
 let vw:DmTensor<f8e4m3,Chip,Cluster,PsParts,m![Ps%32,H%960]>=vw.to_dm(&mut device.tdma);
 let qw:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Qs%64,H%960]>=unsafe{qw.reshape()};
 let kw:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Ps%32,H%960]>=unsafe{kw.reshape()};
 let vw:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Ps%32,H%960]>=unsafe{vw.reshape()};
 project_one(device,a,sa,&qw,&kw,&vw,qs,ks,vs,qa,ka,va);
 project_one(device,b,sb,&qw,&kw,&vw,qs,ks,vs,qb,kb,vb);
 project_one(device,c,sc,&qw,&kw,&vw,qs,ks,vs,qc,kc,vc);
 project_one(device,d,sd,&qw,&kw,&vw,qs,ks,vs,qd,kd,vd);
}
