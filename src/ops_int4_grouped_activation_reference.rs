//! Independent P64 dense-E4 integer-code reference for grouped activation scales.
//! Derived from the old dense reference, with its P64 mapping/FP8 contraction.
//! Every -7..7 code and its local integer dot is exactly representable in F32.
//! Weight Mul0 and activation Mul1 precede the original sum; no row-dequant.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P512,H,L}};
use crate::ops_batch_layouts::{P512H960,P512L960};
use crate::ops_native_fp8_blocked::DownWeight;
type Cluster=m![P512/256];
type UpParts=m![L/240,H/960];
type UpRows=m![L/240,1#4];
type InputRows=m![P512%256=64%8,P512%256=64/8,H/960];
type DownParts=m![H/240,L/960];
type DownRows=m![H/240,1#16];
pub use crate::ops_ffn_projection_clusters::{FfnProjection,JoinedWeight,JoinedScale,JoinedOutput};
pub type InputScale=m![P512/256,H/960,P512%256];
pub type HiddenScale=m![P512/256,L/960,P512%256];
pub type JoinedBlockScale=m![FfnProjection,L/240,H/960,L%240];
pub type DownBlockScale=m![H/240,L/960,H%240];
type ProjectionCluster=m![FfnProjection];
fn stage_upgate(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512H960>,
 xscale:&HbmTensor<f32,Chip,InputScale>,weight:&HbmTensor<f8e4m3,Chip,JoinedWeight>,
 block_scale:&HbmTensor<f32,Chip,JoinedBlockScale>,output:&mut HbmTensor<bf16,Chip,JoinedOutput>) {
    let weight:DmTensor<f8e4m3,Chip,ProjectionCluster,UpParts,m![L % 240,H % 960]>=weight.to_dm(&mut device.tdma);
    let block:DmTensor<f32,Chip,ProjectionCluster,UpParts,m![L%240]>=block_scale.to_dm(&mut device.tdma);
    let block:VrfTensor<f32,Chip,ProjectionCluster,UpParts,m![L%240]>=device.sub.begin(block.view())
        .fetch::<m![L/8%30],m![L%8]>().collect::<m![L/8%30],m![L%8]>().to_vrf();
    macro_rules! chunk {($chunk:expr)=>{{
        let chunk=$chunk;
        let half=chunk/4; let chunk=chunk%4;
        let input:DmTensor<f8e4m3,Chip,ProjectionCluster,InputRows,m![H%960]>=input.view()
            .tile::<m![P512/256],1,m![P512/256=1#2,H/960,P512%256,H%960]>(half)
            .tile::<m![P512%256],64,m![P512/256=1#2,H/960,P512%256=64#256,H%960]>(64*chunk).to_dm(&mut device.tdma);
        let xscale:DmTensor<f32,Chip,ProjectionCluster,UpParts,m![P512%256=64]>=xscale.view()
            .tile::<m![P512/256],1,m![P512/256=1#2,H/960,P512%256]>(half)
            .tile::<m![P512%256],64,m![P512/256=1#2,H/960,P512%256=64#256]>(64*chunk).to_dm(&mut device.tdma);
    let xs:VrfTensor<f32,Chip,ProjectionCluster,UpParts,m![P512%256=64]>=device.sub.begin(xscale.view())
        .fetch::<m![P512%256=64/8],m![P512%256=64%8]>().collect::<m![P512%256=64/8],m![P512%256=64%8]>().to_vrf();
    let x:TrfTensor<f8e4m3,Chip,ProjectionCluster,UpParts,m![P512%256=64%8],m![P512%256=64/8,H%960]>=device.sub.begin(input.view())
        .fetch::<m![1],m![H%960]>()
        .switch::<UpParts,m![P512%256=64%8,P512%256=64/8]>(SwitchConfig::CustomBroadcast{ring_size:256})
        .collect::<m![P512%256=64%8,P512%256=64/8,H/32%30],m![H%32]>().to_trf();
    let result:DmTensor<bf16,Chip,ProjectionCluster,UpRows,m![L%240,P512%256=64]>=device.main.begin(weight.view())
        .fetch::<m![L%240],m![H%960]>()
        .collect::<m![L%240,H/32%30],m![H%32]>()
        .contract_outer::<m![L%240,H/64%15,P512%256=64/8],m![H%64],_,_,_>(&x)
        .contract_packet::<m![1]>().contract_time::<m![L%240,P512%256=64/8]>()
        .contract_lane::<m![L%240,P512%256=64/8],m![P512%256=64%8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L%240,P512%256=64/4],m![P512%256=64%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&block)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&xs)
        .vector_widen_concat::<m![L%240,P512%256=64/8],m![P512%256=64%8]>()
        .vector_inter_slice_reduce::<UpRows,m![L%240,P512%256=64/8]>(InterSliceReduceOpF32::Add)
        .vector_final().cast::<bf16,m![P512%256=64%8#16]>().commit_trim::<m![P512%256=64%8]>().commit();
        let result:DmTensor<bf16,Chip,ProjectionCluster,UpRows,m![P512%256=64,L%240]>=device.main.begin(result.view())
            .fetch::<m![P512%256=64/16,L%240],m![P512%256=64%16]>().collect::<m![P512%256=64/16,L%240],m![P512%256=64%16]>()
            .transpose::<m![P512%256=64/16,L/4%60,P512%256=64%16],m![L%4#16]>().commit_trim::<m![L%4]>().commit();
        result.view().to_hbm_view(&mut device.tdma,output.view_mut()
            .tile::<m![P512/256],1,m![FfnProjection,P512/256=1#{!}2,L/240,P512%256,L%240]>(half)
            .tile::<m![P512%256],64,m![FfnProjection,P512/256=1#{!}2,L/240,P512%256=64#{!}256,L%240]>(64*chunk));

    }};}
    chunk!(0usize);
    chunk!(1usize);
    chunk!(2usize);
    chunk!(3usize);
    chunk!(4usize);
    chunk!(5usize);
    chunk!(6usize);
    chunk!(7usize);
}

fn stage_down(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512L960>,
 xscale:&HbmTensor<f32,Chip,HiddenScale>,weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
 block_scale:&HbmTensor<f32,Chip,DownBlockScale>,global_scale:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,crate::ops_native_fp8_p512_downblocked::Output>) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![H % 240,L % 960]>=weight.to_dm(&mut device.tdma);
    let block:DmTensor<f32,Chip,Cluster,DownParts,m![H%240]>=block_scale.to_dm(&mut device.tdma);
    let block:VrfTensor<f32,Chip,Cluster,DownParts,m![H%240]>=device.sub.begin(block.view())
        .fetch::<m![H/8%30],m![H%8]>().collect::<m![H/8%30],m![H%8]>().to_vrf();
    let global:DmTensor<f32,Chip,Cluster,DownRows,m![1#8]>=global_scale.to_dm(&mut device.tdma);
    let global:VrfTensor<f32,Chip,Cluster,DownRows,m![1#8]>=device.sub.begin(global.view())
        .fetch::<m![1],m![1#8]>().collect::<m![1],m![1#8]>().to_vrf();
    macro_rules! chunk {($chunk:expr)=>{{
        let chunk=$chunk;
        let input:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![P512%256=64,L%960]>=input.view()
            .tile::<m![P512%256],64,m![P512/256,L/960,P512%256=64#256,L%960]>(64*chunk).to_dm(&mut device.tdma);
        let xscale:DmTensor<f32,Chip,Cluster,DownParts,m![P512%256=64]>=xscale.view()
            .tile::<m![P512%256],64,m![P512/256,L/960,P512%256=64#256]>(64*chunk).to_dm(&mut device.tdma);
    let xs:VrfTensor<f32,Chip,Cluster,DownParts,m![P512%256=64]>=device.sub.begin(xscale.view())
        .fetch::<m![P512%256=64/8],m![P512%256=64%8]>().collect::<m![P512%256=64/8],m![P512%256=64%8]>().to_vrf();
    let x:TrfTensor<f8e4m3,Chip,Cluster,DownParts,m![P512%256=64%8],m![P512%256=64/8,L%960]>=device.sub.begin(input.view())
        .fetch::<m![P512%256=64%8,P512%256=64/8,L/32%30],m![L%32]>()
        .collect::<m![P512%256=64%8,P512%256=64/8,L/32%30],m![L%32]>().to_trf();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H%240,P512%256=64]>=device.main.begin(weight.view())
        .fetch::<m![H%240],m![L%960]>()
        .collect::<m![H%240,L/32%30],m![L%32]>()
        .contract_outer::<m![H%240,L/64%15,P512%256=64/8],m![L%64],_,_,_>(&x)
        .contract_packet::<m![1]>().contract_time::<m![H%240,P512%256=64/8]>()
        .contract_lane::<m![H%240,P512%256=64/8],m![P512%256=64%8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H%240,P512%256=64/4],m![P512%256=64%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&block)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&xs)
        .vector_widen_concat::<m![H%240,P512%256=64/8],m![P512%256=64%8]>()
        .vector_inter_slice_reduce::<DownRows,m![H%240,P512%256=64/8]>(InterSliceReduceOpF32::Add)
        .vector_final().cast::<bf16,m![P512%256=64%8#16]>().commit_trim::<m![P512%256=64%8]>().commit();
        let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H%240,P512%256=64]>=device.main.begin(result.view())
            .fetch::<m![H%240,P512%256=64/8],m![P512%256=64%8]>().fetch_cast::<f32>()
            .collect::<m![H%240,P512%256=64/8],m![P512%256=64%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H%240,P512%256=64/4],m![P512%256=64%4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&global)
            .vector_widen_concat::<m![H%240,P512%256=64/8],m![P512%256=64%8]>()
            .vector_final().cast::<bf16,m![P512%256=64%8#16]>().commit_trim::<m![P512%256=64%8]>().commit();
        let result:DmTensor<bf16,Chip,Cluster,DownRows,m![P512%256=64,H%240]>=device.main.begin(result.view())
            .fetch::<m![P512%256=64/16,H%240],m![P512%256=64%16]>().collect::<m![P512%256=64/16,H%240],m![P512%256=64%16]>()
            .transpose::<m![P512%256=64/16,H/4%60,P512%256=64%16],m![H%4#16]>().commit_trim::<m![H%4]>().commit();
        result.view().to_hbm_view(&mut device.tdma,output.view_mut()
            .tile::<m![P512%256],64,m![P512/256,H/240,P512%256=64#{!}256,H%240]>(64*chunk));

    }};}
    chunk!(0usize);
    chunk!(1usize);
    chunk!(2usize);
    chunk!(3usize);
}

#[device(chip=1)]
pub fn project_upgate(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512H960>,
 xscale:&HbmTensor<f32,Chip,InputScale>,weight:&HbmTensor<f8e4m3,Chip,JoinedWeight>,
 block_scale:&HbmTensor<f32,Chip,JoinedBlockScale>,
 output:&mut HbmTensor<bf16,Chip,JoinedOutput>) {
 stage_upgate(device,input,xscale,weight,block_scale,output);
}
#[device(chip=1)]
pub fn project_down(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512L960>,
 xscale:&HbmTensor<f32,Chip,HiddenScale>,weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
 block_scale:&HbmTensor<f32,Chip,DownBlockScale>,
 global_scale:&HbmTensor<f32,Chip,m![1]>,
 output:&mut HbmTensor<bf16,Chip,crate::ops_native_fp8_p512_downblocked::Output>) {
 stage_down(device,input,xscale,weight,block_scale,global_scale,output);
}
