//! Exact production P512 up/gate: shard activation HBM ingress by token and H,
//! then collect/broadcast to the existing contraction TRF. No arithmetic change.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P512,H,L}};
use crate::ops_batch_layouts::{P512H960,P512L240};
use crate::ops_native_fp8_blocked::UpWeight;
type Cluster=m![P512/256];
type UpParts=m![L/240,H/960];
type UpRows=m![L/240,1#4];
type InputRows=m![P512%256=64%8,P512%256=64/8,H/960];
#[device(chip=1)]
pub fn project_up_scaled(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512H960>,
 xscale:&HbmTensor<f32,Chip,m![P512]>,weight:&HbmTensor<f8e4m3,Chip,UpWeight>,
 weight_dequant_scale:&HbmTensor<f32,Chip,m![L]>,output:&mut HbmTensor<bf16,Chip,P512L240>) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![L % 240,H % 960]>=weight.to_dm(&mut device.tdma);
    let ws:DmTensor<f32,Chip,Cluster,UpRows,m![L % 240]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,Cluster,UpRows,m![L % 240]>=device.sub.begin(ws.view())
        .fetch::<m![L / 8 % 30],m![L % 8]>().collect::<m![L / 8 % 30],m![L % 8]>().to_vrf();
    macro_rules! chunk {($chunk:expr)=>{{
        let chunk=$chunk;
        let input:DmTensor<f8e4m3,Chip,Cluster,InputRows,m![H%960]>=input.view()
            .tile::<m![P512%256],64,m![P512/256,H/960,P512%256=64#256,H%960]>(64*chunk).to_dm(&mut device.tdma);
        let xscale:DmTensor<f32,Chip,Cluster,UpRows,m![P512%256=64]>=xscale.view()
            .tile::<m![P512%256],64,m![P512/256,P512%256=64#256]>(64*chunk).to_dm(&mut device.tdma);
    let xs:VrfTensor<f32,Chip,Cluster,UpRows,m![P512%256=64]>=device.sub.begin(xscale.view())
        .fetch::<m![P512%256=64/8],m![P512%256=64%8]>().collect::<m![P512%256=64/8],m![P512%256=64%8]>().to_vrf();
    let x:TrfTensor<f8e4m3,Chip,Cluster,UpParts,m![P512%256=64%8],m![P512%256=64/8,H%960]>=device.sub.begin(input.view())
        .fetch::<m![1],m![H%960]>()
        .switch::<UpParts,m![P512%256=64%8,P512%256=64/8]>(SwitchConfig::CustomBroadcast{ring_size:256})
        .collect::<m![P512%256=64%8,P512%256=64/8,H/32%30],m![H%32]>().to_trf();
    let result:DmTensor<bf16,Chip,Cluster,UpRows,m![L%240,P512%256=64]>=device.main.begin(weight.view())
        .fetch::<m![L%240,H/32%30],m![H%32]>().collect::<m![L%240,H/32%30],m![H%32]>()
        .contract_outer::<m![L%240,H/64%15,P512%256=64/8],m![H%64],_,_,_>(&x)
        .contract_packet::<m![1]>().contract_time::<m![L%240,P512%256=64/8]>()
        .contract_lane::<m![L%240,P512%256=64/8],m![P512%256=64%8]>(LaneMode::Interleaved)
        .vector_init().vector_inter_slice_reduce::<UpRows,m![L%240,P512%256=64/8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L%240,P512%256=64/4],m![P512%256=64%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![L%240,P512%256=64/8],m![P512%256=64%8]>()
        .vector_final().cast::<bf16,m![P512%256=64%8#16]>().commit_trim::<m![P512%256=64%8]>().commit();
        let result:DmTensor<bf16,Chip,Cluster,UpRows,m![P512%256=64,L%240]>=device.main.begin(result.view())
            .fetch::<m![P512%256=64/16,L%240],m![P512%256=64%16]>().collect::<m![P512%256=64/16,L%240],m![P512%256=64%16]>()
            .transpose::<m![P512%256=64/16,L/4%60,P512%256=64%16],m![L%4#16]>().commit_trim::<m![L%4]>().commit();
        result.view().to_hbm_view(&mut device.tdma,output.view_mut()
            .tile::<m![P512%256],64,m![P512/256,L/240,P512%256=64#{!}256,L%240]>(64*chunk));

    }};}
    chunk!(0usize);
    chunk!(1usize);
    chunk!(2usize);
    chunk!(3usize);
}

/// Diagnostic materialization of the exact same input routing, independent of weights.
#[device(chip=1)]
pub fn debug_ingress(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512H960>,
 output:&mut HbmTensor<f8e4m3,Chip,m![L/240,P512/256,H/960,P512%256,H%960]>) {
 for chunk in 0..4 {
  let input:DmTensor<f8e4m3,Chip,Cluster,InputRows,m![H%960]>=input.view()
   .tile::<m![P512%256],64,m![P512/256,H/960,P512%256=64#256,H%960]>(64*chunk).to_dm(&mut device.tdma);
  let copies:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![P512%256=64%8,P512%256=64/8,H%960]>=device.sub.begin(input.view())
   .fetch::<m![1],m![H%960]>()
   .switch::<UpParts,m![P512%256=64%8,P512%256=64/8]>(SwitchConfig::CustomBroadcast{ring_size:256})
   .collect::<m![P512%256=64%8,P512%256=64/8,H/32%30],m![H%32]>()
   .commit_trim::<m![H%32]>().commit();
  copies.view().to_hbm_view(&mut device.tdma,output.view_mut()
   .tile::<m![P512%256],64,m![L/240,P512/256,H/960,P512%256=64#{!}256,H%960]>(64*chunk));
 }
}
