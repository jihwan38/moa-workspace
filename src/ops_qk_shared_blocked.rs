//! Shared full Q/K with blocked output for a matching head consumer.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,H,Qf,Pf}};
axes![SharedFullRow64=64];
pub type QueryOutput=m![P128/64,Qf/128,P128%64,Qf%128];
pub type KeyOutput=m![P128/64,Pf/8,P128%64,Pf%8];
type Cluster=m![P128/64];
type SharedParts=m![SharedFullRow64,H/960];
type SharedRows=m![SharedFullRow64,1#4];
type QfParts=m![Qf/128,H/960];
type QfRows=m![Qf/128,1#4];
type PfParts=m![Pf/8,H/960];
type PfRows=m![Pf/8,1#4];
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
fn matrix_full_query(device:&mut Device,
 x:&TrfTensor<f8e4m3,Chip,Cluster,SharedParts,m![P128%8],m![P128/8%8,H%960]>,
 xs:&VrfTensor<f32,Chip,Cluster,SharedRows,m![P128%64]>,
 weight:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Qf%128,H%960]>)
 ->DmTensor<bf16,Chip,Cluster,SharedRows,m![Qf%128,P128%64]> {
    device
        .main
        .begin(weight.view())
        .fetch::<m![Qf % 128, H / 32 % 30], m![H % 32]>()
        .collect::<m![Qf % 128, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![Qf % 128, H / 64 % 15, P128 / 8 % 8], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qf % 128, P128 / 8 % 8]>()
        .contract_lane::<m![Qf % 128, P128 / 8 % 8], m![P128 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SharedRows, m![Qf % 128, P128 / 8 % 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qf % 128, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), xs)
        .vector_widen_concat::<m![Qf % 128, P128 / 8 % 8], m![P128 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128%8#16]>()
        .commit_trim::<m![P128 % 8]>()
        .commit()
}
fn finish_full_query(
    device: &mut Device,
    dot: &DmTensor<bf16, Chip, Cluster, QfRows, m![Qf % 128, P128 % 64]>,
    scale: &HbmTensor<bf16, Chip, m![Qf]>,
    output: &mut HbmTensor<bf16, Chip, QueryOutput>,
) {
    let scale: DmTensor<bf16, Chip, Cluster, QfRows, m![Qf % 128]> = scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, QfRows, m![Qf % 128]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Qf % 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Qf / 8 % 16], m![Qf % 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, QfRows, m![Qf % 128, P128 % 64]> = device
        .main
        .begin(dot.view())
        .fetch::<m![Qf % 128, P128 / 8 % 8], m![P128 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qf % 128, P128 / 8 % 8], m![P128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qf % 128, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Qf % 128, P128 / 8 % 8], m![P128 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128%8#16]>()
        .commit_trim::<m![P128 % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, QfRows, m![P128 % 64, Qf % 128]> = device
        .main
        .begin(result.view())
        .fetch::<m![P128 / 16 % 4, Qf % 128], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, Qf % 128], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, Qf / 4 % 32, P128 % 16], m![Qf%4#16]>()
        .commit_trim::<m![Qf % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}
fn matrix_full_key(device:&mut Device,
 x:&TrfTensor<f8e4m3,Chip,Cluster,SharedParts,m![P128%8],m![P128/8%8,H%960]>,
 xs:&VrfTensor<f32,Chip,Cluster,SharedRows,m![P128%64]>,
 weight:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Pf%8,H%960]>)
 ->DmTensor<bf16,Chip,Cluster,SharedRows,m![Pf%8,P128%64]> {
    device
        .main
        .begin(weight.view())
        .fetch::<m![Pf % 8, H / 32 % 30], m![H % 32]>()
        .collect::<m![Pf % 8, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![Pf % 8, H / 64 % 15, P128 / 8 % 8], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Pf % 8, P128 / 8 % 8]>()
        .contract_lane::<m![Pf % 8, P128 / 8 % 8], m![P128 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SharedRows, m![Pf % 8, P128 / 8 % 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Pf % 8, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), xs)
        .vector_widen_concat::<m![Pf % 8, P128 / 8 % 8], m![P128 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128%8#16]>()
        .commit_trim::<m![P128 % 8]>()
        .commit()
}
fn finish_full_key(
    device: &mut Device,
    dot: &DmTensor<bf16, Chip, Cluster, PfRows, m![Pf % 8, P128 % 64]>,
    scale: &HbmTensor<bf16, Chip, m![Pf]>,
    output: &mut HbmTensor<bf16, Chip, KeyOutput>,
) {
    let scale: DmTensor<bf16, Chip, Cluster, PfRows, m![Pf % 8]> = scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, PfRows, m![Pf % 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Pf % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Pf / 8 % 1], m![Pf % 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, PfRows, m![Pf % 8, P128 % 64]> = device
        .main
        .begin(dot.view())
        .fetch::<m![Pf % 8, P128 / 8 % 8], m![P128 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Pf % 8, P128 / 8 % 8], m![P128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Pf % 8, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Pf % 8, P128 / 8 % 8], m![P128 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128%8#16]>()
        .commit_trim::<m![P128 % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, PfRows, m![P128 % 64, Pf % 8]> = device
        .main
        .begin(result.view())
        .fetch::<m![P128 / 16 % 4, Pf % 8], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, Pf % 8], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, Pf / 4 % 2, P128 % 16], m![Pf%4#16]>()
        .commit_trim::<m![Pf % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}
fn project_one(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![P128,H]>,
 xscale:&HbmTensor<f32,Chip,m![P128,1#8]>,
 qw:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Qf%128,H%960]>,
 kw:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Pf%8,H%960]>,
 qs:&HbmTensor<bf16,Chip,m![Qf]>,ks:&HbmTensor<bf16,Chip,m![Pf]>,
 qo:&mut HbmTensor<bf16,Chip,QueryOutput>,ko:&mut HbmTensor<bf16,Chip,KeyOutput>) {
 // A schedule epoch releases this full TRF before another independent group.
 for _epoch in 0..1 {
  let input:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![P128%64,H%960]>=input.to_dm(&mut device.tdma);
  let scales:DmTensor<f32,Chip,Cluster,SharedRows,m![P128%64,1#8]>=xscale.to_dm(&mut device.tdma);
  let scales=pack_scales(device,&scales);
  let xs:VrfTensor<f32,Chip,Cluster,SharedRows,m![P128%64]>=device.sub.begin(scales.view())
   .fetch::<m![P128/8%8],m![P128%8]>().collect::<m![P128/8%8],m![P128%8]>().to_vrf();
  let x:TrfTensor<f8e4m3,Chip,Cluster,SharedParts,m![P128%8],m![P128/8%8,H%960]>=device.sub.begin(input.view())
   .fetch::<m![P128%8,P128/8%8,H/32%30],m![H%32]>().collect::<m![P128%8,P128/8%8,H/32%30],m![H%32]>().to_trf();
  let q=matrix_full_query(device,&x,&xs,qw);
  let k=matrix_full_key(device,&x,&xs,kw);
  // Same physical group order: Qf group128 and Pf group8 each index0..63.
  let q:DmTensor<bf16,Chip,Cluster,QfRows,m![Qf%128,P128%64]>=unsafe{q.reshape()};
  let k:DmTensor<bf16,Chip,Cluster,PfRows,m![Pf%8,P128%64]>=unsafe{k.reshape()};
  finish_full_query(device,&q,qs,qo);
  finish_full_key(device,&k,ks,ko);
 }
}
#[device(chip=1)]
pub fn project_full_qk_four(device:&mut Device,
 a:&HbmTensor<f8e4m3,Chip,m![P128,H]>,sa:&HbmTensor<f32,Chip,m![P128,1#8]>,
 b:&HbmTensor<f8e4m3,Chip,m![P128,H]>,sb:&HbmTensor<f32,Chip,m![P128,1#8]>,
 c:&HbmTensor<f8e4m3,Chip,m![P128,H]>,sc:&HbmTensor<f32,Chip,m![P128,1#8]>,
 d:&HbmTensor<f8e4m3,Chip,m![P128,H]>,sd:&HbmTensor<f32,Chip,m![P128,1#8]>,
 qw:&HbmTensor<f8e4m3,Chip,m![Qf,H]>,kw:&HbmTensor<f8e4m3,Chip,m![Pf,H]>,
 qs:&HbmTensor<bf16,Chip,m![Qf]>,ks:&HbmTensor<bf16,Chip,m![Pf]>,
 oa:(&mut HbmTensor<bf16,Chip,QueryOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>),
 ob:(&mut HbmTensor<bf16,Chip,QueryOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>),
 oc:(&mut HbmTensor<bf16,Chip,QueryOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>),
 od:(&mut HbmTensor<bf16,Chip,QueryOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>)) {
 let (qa,ka)=oa;let(qb,kb)=ob;let(qc,kc)=oc;let(qd,kd)=od;
 let qw:DmTensor<f8e4m3,Chip,Cluster,QfParts,m![Qf%128,H%960]>=qw.to_dm(&mut device.tdma);
 let kw:DmTensor<f8e4m3,Chip,Cluster,PfParts,m![Pf%8,H%960]>=kw.to_dm(&mut device.tdma);
 let qw:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Qf%128,H%960]>=unsafe{qw.reshape()};
 let kw:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Pf%8,H%960]>=unsafe{kw.reshape()};
 project_one(device,a,sa,&qw,&kw,qs,ks,qa,ka);
 project_one(device,b,sb,&qw,&kw,qs,ks,qb,kb);
 project_one(device,c,sc,&qw,&kw,qs,ks,qc,kc);
 project_one(device,d,sd,&qw,&kw,qs,ks,qd,kd);
}
