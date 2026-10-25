//! Blocked output shared native-FP8 input/TRF for one R64 group; original H960 arithmetic.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{R64,H,Qs,Ps,Qf,Pf}};
axes![SharedDecodeRow64=64];
pub type QueryOutput=m![R64/32,Qs/64,R64%32,Qs%64];
pub type KeyOutput=m![R64/32,Ps/32,R64%32,Ps%32];
pub type FullQueryOutput=m![R64/32,Qf/128,R64%32,Qf%128];
pub type FullKeyOutput=m![R64/32,Pf/8,R64%32,Pf%8];
type Cluster=m![R64/32];
type SharedParts=m![SharedDecodeRow64,H/960];
type SharedRows=m![SharedDecodeRow64,1#4];
type QsParts=m![Qs/64,H/960];
type QsRows=m![Qs/64,1#4];
type PsParts=m![Ps/32,H/960];
type PsRows=m![Ps/32,1#4];
type QfParts=m![Qf/128,H/960];
type QfRows=m![Qf/128,1#4];
type PfParts=m![Pf/8,H/960];
type PfRows=m![Pf/8,1#4];
fn pack_scales(
    device: &mut Device,
    xscale: &DmTensor<f32, Chip, Cluster, SharedRows, m![R64 % 32,1 # 8]>,
) -> DmTensor<f32, Chip, Cluster, SharedRows, m![R64 % 32]> {
    // Keep the 32-byte scalar slot for aligned inter-slice DMA, then pack by TU.
    let xscale: DmTensor<f32, Chip, Cluster, SharedRows, m![R64 % 32]> = device
        .sub
        .begin(xscale.view())
        .fetch::<m![R64 % 32], m![1 # 8]>()
        .collect::<m![R64 % 32], m![1 # 8]>()
        .transpose::<m![R64 % 32 / 2], m![R64 % 32 % 2 # 8]>()
        .commit_trim::<m![R64 % 32 % 2]>()
        .commit();
    xscale
}
fn matrix_query(device:&mut Device,
 x:&TrfTensor<f8e4m3,Chip,Cluster,SharedParts,m![R64%8],m![R64/8%4,H%960]>,
 xs:&VrfTensor<f32,Chip,Cluster,SharedRows,m![R64%32]>,
 weight:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Qs%64,H%960]>)
 ->DmTensor<bf16,Chip,Cluster,SharedRows,m![Qs%64,R64%32]> {
    device
        .main
        .begin(weight.view())
        .fetch::<m![Qs % 64, H / 32 % 30], m![H % 32]>()
        .collect::<m![Qs % 64, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![Qs % 64, H / 64 % 15, R64 / 8 % 4], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 64, R64 / 8 % 4]>()
        .contract_lane::<m![Qs % 64, R64 / 8 % 4], m![R64 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SharedRows, m![Qs % 64, R64 / 8 % 4]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs % 64, R64 / 4 % 8], m![R64 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), xs)
        .vector_widen_concat::<m![Qs % 64, R64 / 8 % 4], m![R64 % 8]>()
        .vector_final()
        .cast::<bf16, m![R64%8#16]>()
        .commit_trim::<m![R64 % 8]>()
        .commit()
}
fn finish_query(
    device: &mut Device,
    dot: &DmTensor<bf16, Chip, Cluster, QsRows, m![Qs % 64, R64 % 32]>,
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
    let result: DmTensor<bf16, Chip, Cluster, QsRows, m![Qs % 64, R64 % 32]> = device
        .main
        .begin(dot.view())
        .fetch::<m![Qs % 64, R64 / 8 % 4], m![R64 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs % 64, R64 / 8 % 4], m![R64 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs % 64, R64 / 4 % 8], m![R64 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Qs % 64, R64 / 8 % 4], m![R64 % 8]>()
        .vector_final()
        .cast::<bf16, m![R64%8#16]>()
        .commit_trim::<m![R64 % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, QsRows, m![R64 % 32, Qs % 64]> = device
        .main
        .begin(result.view())
        .fetch::<m![R64 / 16 % 2, Qs % 64], m![R64 % 16]>()
        .collect::<m![R64 / 16 % 2, Qs % 64], m![R64 % 16]>()
        .transpose::<m![R64 / 16 % 2, Qs / 4 % 16, R64 % 16], m![Qs%4#16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}
fn matrix_sliding_kv(device:&mut Device,
 x:&TrfTensor<f8e4m3,Chip,Cluster,SharedParts,m![R64%8],m![R64/8%4,H%960]>,
 xs:&VrfTensor<f32,Chip,Cluster,SharedRows,m![R64%32]>,
 weight:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Ps%32,H%960]>)
 ->DmTensor<bf16,Chip,Cluster,SharedRows,m![Ps%32,R64%32]> {
    device
        .main
        .begin(weight.view())
        .fetch::<m![Ps % 32, H / 32 % 30], m![H % 32]>()
        .collect::<m![Ps % 32, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![Ps % 32, H / 64 % 15, R64 / 8 % 4], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 32, R64 / 8 % 4]>()
        .contract_lane::<m![Ps % 32, R64 / 8 % 4], m![R64 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SharedRows, m![Ps % 32, R64 / 8 % 4]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps % 32, R64 / 4 % 8], m![R64 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), xs)
        .vector_widen_concat::<m![Ps % 32, R64 / 8 % 4], m![R64 % 8]>()
        .vector_final()
        .cast::<bf16, m![R64%8#16]>()
        .commit_trim::<m![R64 % 8]>()
        .commit()
}
fn finish_sliding_kv(
    device: &mut Device,
    dot: &DmTensor<bf16, Chip, Cluster, PsRows, m![Ps % 32, R64 % 32]>,
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
    let result: DmTensor<bf16, Chip, Cluster, PsRows, m![Ps % 32, R64 % 32]> = device
        .main
        .begin(dot.view())
        .fetch::<m![Ps % 32, R64 / 8 % 4], m![R64 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Ps % 32, R64 / 8 % 4], m![R64 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps % 32, R64 / 4 % 8], m![R64 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Ps % 32, R64 / 8 % 4], m![R64 % 8]>()
        .vector_final()
        .cast::<bf16, m![R64%8#16]>()
        .commit_trim::<m![R64 % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, PsRows, m![R64 % 32, Ps % 32]> = device
        .main
        .begin(result.view())
        .fetch::<m![R64 / 16 % 2, Ps % 32], m![R64 % 16]>()
        .collect::<m![R64 / 16 % 2, Ps % 32], m![R64 % 16]>()
        .transpose::<m![R64 / 16 % 2, Ps / 4 % 8, R64 % 16], m![Ps%4#16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}
fn matrix_full_query(device:&mut Device,
 x:&TrfTensor<f8e4m3,Chip,Cluster,SharedParts,m![R64%8],m![R64/8%4,H%960]>,
 xs:&VrfTensor<f32,Chip,Cluster,SharedRows,m![R64%32]>,
 weight:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Qf%128,H%960]>)
 ->DmTensor<bf16,Chip,Cluster,SharedRows,m![Qf%128,R64%32]> {
    device
        .main
        .begin(weight.view())
        .fetch::<m![Qf % 128, H / 32 % 30], m![H % 32]>()
        .collect::<m![Qf % 128, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![Qf % 128, H / 64 % 15, R64 / 8 % 4], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qf % 128, R64 / 8 % 4]>()
        .contract_lane::<m![Qf % 128, R64 / 8 % 4], m![R64 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SharedRows, m![Qf % 128, R64 / 8 % 4]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qf % 128, R64 / 4 % 8], m![R64 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), xs)
        .vector_widen_concat::<m![Qf % 128, R64 / 8 % 4], m![R64 % 8]>()
        .vector_final()
        .cast::<bf16, m![R64%8#16]>()
        .commit_trim::<m![R64 % 8]>()
        .commit()
}
fn finish_full_query(
    device: &mut Device,
    dot: &DmTensor<bf16, Chip, Cluster, QfRows, m![Qf % 128, R64 % 32]>,
    scale: &HbmTensor<bf16, Chip, m![Qf]>,
    output: &mut HbmTensor<bf16, Chip, FullQueryOutput>,
) {
    let scale: DmTensor<bf16, Chip, Cluster, QfRows, m![Qf % 128]> = scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, QfRows, m![Qf % 128]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Qf % 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Qf / 8 % 16], m![Qf % 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, QfRows, m![Qf % 128, R64 % 32]> = device
        .main
        .begin(dot.view())
        .fetch::<m![Qf % 128, R64 / 8 % 4], m![R64 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qf % 128, R64 / 8 % 4], m![R64 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qf % 128, R64 / 4 % 8], m![R64 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Qf % 128, R64 / 8 % 4], m![R64 % 8]>()
        .vector_final()
        .cast::<bf16, m![R64%8#16]>()
        .commit_trim::<m![R64 % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, QfRows, m![R64 % 32, Qf % 128]> = device
        .main
        .begin(result.view())
        .fetch::<m![R64 / 16 % 2, Qf % 128], m![R64 % 16]>()
        .collect::<m![R64 / 16 % 2, Qf % 128], m![R64 % 16]>()
        .transpose::<m![R64 / 16 % 2, Qf / 4 % 32, R64 % 16], m![Qf%4#16]>()
        .commit_trim::<m![Qf % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}
fn matrix_full_key(device:&mut Device,
 x:&TrfTensor<f8e4m3,Chip,Cluster,SharedParts,m![R64%8],m![R64/8%4,H%960]>,
 xs:&VrfTensor<f32,Chip,Cluster,SharedRows,m![R64%32]>,
 weight:&DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Pf%8,H%960]>)
 ->DmTensor<bf16,Chip,Cluster,SharedRows,m![Pf%8,R64%32]> {
    device
        .main
        .begin(weight.view())
        .fetch::<m![Pf % 8, H / 32 % 30], m![H % 32]>()
        .collect::<m![Pf % 8, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![Pf % 8, H / 64 % 15, R64 / 8 % 4], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Pf % 8, R64 / 8 % 4]>()
        .contract_lane::<m![Pf % 8, R64 / 8 % 4], m![R64 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SharedRows, m![Pf % 8, R64 / 8 % 4]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Pf % 8, R64 / 4 % 8], m![R64 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), xs)
        .vector_widen_concat::<m![Pf % 8, R64 / 8 % 4], m![R64 % 8]>()
        .vector_final()
        .cast::<bf16, m![R64%8#16]>()
        .commit_trim::<m![R64 % 8]>()
        .commit()
}
fn finish_full_key(
    device: &mut Device,
    dot: &DmTensor<bf16, Chip, Cluster, PfRows, m![Pf % 8, R64 % 32]>,
    scale: &HbmTensor<bf16, Chip, m![Pf]>,
    output: &mut HbmTensor<bf16, Chip, FullKeyOutput>,
) {
    let scale: DmTensor<bf16, Chip, Cluster, PfRows, m![Pf % 8]> = scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, PfRows, m![Pf % 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Pf % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Pf / 8 % 1], m![Pf % 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, PfRows, m![Pf % 8, R64 % 32]> = device
        .main
        .begin(dot.view())
        .fetch::<m![Pf % 8, R64 / 8 % 4], m![R64 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Pf % 8, R64 / 8 % 4], m![R64 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Pf % 8, R64 / 4 % 8], m![R64 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Pf % 8, R64 / 8 % 4], m![R64 % 8]>()
        .vector_final()
        .cast::<bf16, m![R64%8#16]>()
        .commit_trim::<m![R64 % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, PfRows, m![R64 % 32, Pf % 8]> = device
        .main
        .begin(result.view())
        .fetch::<m![R64 / 16 % 2, Pf % 8], m![R64 % 16]>()
        .collect::<m![R64 / 16 % 2, Pf % 8], m![R64 % 16]>()
        .transpose::<m![R64 / 16 % 2, Pf / 4 % 2, R64 % 16], m![Pf%4#16]>()
        .commit_trim::<m![Pf % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}
fn input_trf(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![R64,H]>,
 xscale:&HbmTensor<f32,Chip,m![R64,1#8]>)
 ->(TrfTensor<f8e4m3,Chip,Cluster,SharedParts,m![R64%8],m![R64/8%4,H%960]>,
 VrfTensor<f32,Chip,Cluster,SharedRows,m![R64%32]>) {
 let input:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![R64%32,H%960]>=input.to_dm(&mut device.tdma);
 let scales:DmTensor<f32,Chip,Cluster,SharedRows,m![R64%32,1#8]>=xscale.to_dm(&mut device.tdma);
 let scales=pack_scales(device,&scales);
 let xs=device.sub.begin(scales.view()).fetch::<m![R64/8%4],m![R64%8]>()
  .collect::<m![R64/8%4],m![R64%8]>().to_vrf();
 let x=device.sub.begin(input.view()).fetch::<m![R64%8,R64/8%4,H/32%30],m![H%32]>()
  .collect::<m![R64%8,R64/8%4,H/32%30],m![H%32]>().to_trf();
 (x,xs)
}
#[device(chip=1)]
pub fn project_sliding_qkv(device:&mut Device,
 input:&HbmTensor<f8e4m3,Chip,m![R64,H]>,xscale:&HbmTensor<f32,Chip,m![R64,1#8]>,
 qw:&HbmTensor<f8e4m3,Chip,m![Qs,H]>,
 kw:&HbmTensor<f8e4m3,Chip,m![Ps,H]>,
 vw:&HbmTensor<f8e4m3,Chip,m![Ps,H]>,
 qs:&HbmTensor<bf16,Chip,m![Qs]>,
 ks:&HbmTensor<bf16,Chip,m![Ps]>,
 vs:&HbmTensor<bf16,Chip,m![Ps]>,
 qo:&mut HbmTensor<bf16,Chip,QueryOutput>,
 ko:&mut HbmTensor<bf16,Chip,KeyOutput>,
 vo:&mut HbmTensor<bf16,Chip,KeyOutput>) {
 let qw:DmTensor<f8e4m3,Chip,Cluster,QsParts,m![Qs%64,H%960]>=qw.to_dm(&mut device.tdma);
 let qw:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Qs%64,H%960]>=unsafe{qw.reshape()};
 let kw:DmTensor<f8e4m3,Chip,Cluster,PsParts,m![Ps%32,H%960]>=kw.to_dm(&mut device.tdma);
 let kw:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Ps%32,H%960]>=unsafe{kw.reshape()};
 let vw:DmTensor<f8e4m3,Chip,Cluster,PsParts,m![Ps%32,H%960]>=vw.to_dm(&mut device.tdma);
 let vw:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Ps%32,H%960]>=unsafe{vw.reshape()};
 let (x,xs)=input_trf(device,input,xscale);
 let q=matrix_query(device,&x,&xs,&qw);
 let k=matrix_sliding_kv(device,&x,&xs,&kw);
 let v=matrix_sliding_kv(device,&x,&xs,&vw);
 // Same physical group g: channel tile starts at g*local_channel_count.
 let q:DmTensor<bf16,Chip,Cluster,QsRows,m![Qs%64,R64%32]>=unsafe{q.reshape()};
 let k:DmTensor<bf16,Chip,Cluster,PsRows,m![Ps%32,R64%32]>=unsafe{k.reshape()};
 let v:DmTensor<bf16,Chip,Cluster,PsRows,m![Ps%32,R64%32]>=unsafe{v.reshape()};
 finish_query(device,&q,qs,qo);
 finish_sliding_kv(device,&k,ks,ko);
 finish_sliding_kv(device,&v,vs,vo);
}
#[device(chip=1)]
pub fn project_full_qk(device:&mut Device,
 input:&HbmTensor<f8e4m3,Chip,m![R64,H]>,xscale:&HbmTensor<f32,Chip,m![R64,1#8]>,
 qw:&HbmTensor<f8e4m3,Chip,m![Qf,H]>,
 kw:&HbmTensor<f8e4m3,Chip,m![Pf,H]>,
 qs:&HbmTensor<bf16,Chip,m![Qf]>,
 ks:&HbmTensor<bf16,Chip,m![Pf]>,
 qo:&mut HbmTensor<bf16,Chip,FullQueryOutput>,
 ko:&mut HbmTensor<bf16,Chip,FullKeyOutput>) {
 let qw:DmTensor<f8e4m3,Chip,Cluster,QfParts,m![Qf%128,H%960]>=qw.to_dm(&mut device.tdma);
 let qw:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Qf%128,H%960]>=unsafe{qw.reshape()};
 let kw:DmTensor<f8e4m3,Chip,Cluster,PfParts,m![Pf%8,H%960]>=kw.to_dm(&mut device.tdma);
 let kw:DmTensor<f8e4m3,Chip,Cluster,SharedParts,m![Pf%8,H%960]>=unsafe{kw.reshape()};
 let (x,xs)=input_trf(device,input,xscale);
 let q=matrix_full_query(device,&x,&xs,&qw);
 let k=matrix_full_key(device,&x,&xs,&kw);
 // Same physical group g: channel tile starts at g*local_channel_count.
 let q:DmTensor<bf16,Chip,Cluster,QfRows,m![Qf%128,R64%32]>=unsafe{q.reshape()};
 let k:DmTensor<bf16,Chip,Cluster,PfRows,m![Pf%8,R64%32]>=unsafe{k.reshape()};
 finish_full_query(device,&q,qs,qo);
 finish_full_key(device,&k,ks,ko);
}
