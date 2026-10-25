//! Experimental exact UP/GATE projection ownership across two clusters.
//! One unchanged weight per cluster; all512 rows stream through eightP64 tiles.
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
type Shards=m![L/960,P512%256=64/4];
type MaxParts=m![P512%256=64/4,L/960];
type MaxRows=m![P512%256=64/4,1#16];
axes![FfnProjection = 2];
pub type JoinedWeight=m![FfnProjection,L/240,H/960,L%240,H%960];
pub type JoinedScale=m![FfnProjection,L];
pub type JoinedOutput=m![FfnProjection,P512/256,L/240,P512%256,L%240];
type ProjectionCluster=m![FfnProjection];
fn stage_upgate(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512H960>,
 xscale:&HbmTensor<f32,Chip,m![P512]>,weight:&HbmTensor<f8e4m3,Chip,JoinedWeight>,
 weight_dequant_scale:&HbmTensor<f32,Chip,JoinedScale>,output:&mut HbmTensor<bf16,Chip,JoinedOutput>) {
    let weight:DmTensor<f8e4m3,Chip,ProjectionCluster,UpParts,m![L % 240,H % 960]>=weight.to_dm(&mut device.tdma);
    let ws:DmTensor<f32,Chip,ProjectionCluster,UpRows,m![L % 240]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,ProjectionCluster,UpRows,m![L % 240]>=device.sub.begin(ws.view())
        .fetch::<m![L / 8 % 30],m![L % 8]>().collect::<m![L / 8 % 30],m![L % 8]>().to_vrf();
    macro_rules! chunk {($chunk:expr)=>{{
        let chunk=$chunk;
        let half=chunk/4; let chunk=chunk%4;
        let input:DmTensor<f8e4m3,Chip,ProjectionCluster,InputRows,m![H%960]>=input.view()
            .tile::<m![P512/256],1,m![P512/256=1#2,H/960,P512%256,H%960]>(half)
            .tile::<m![P512%256],64,m![P512/256=1#2,H/960,P512%256=64#256,H%960]>(64*chunk).to_dm(&mut device.tdma);
        let xscale:DmTensor<f32,Chip,ProjectionCluster,UpRows,m![P512%256=64]>=xscale.view()
            .tile::<m![P512/256],1,m![P512/256=1#2,P512%256]>(half)
            .tile::<m![P512%256],64,m![P512/256=1#2,P512%256=64#256]>(64*chunk).to_dm(&mut device.tdma);
    let xs:VrfTensor<f32,Chip,ProjectionCluster,UpRows,m![P512%256=64]>=device.sub.begin(xscale.view())
        .fetch::<m![P512%256=64/8],m![P512%256=64%8]>().collect::<m![P512%256=64/8],m![P512%256=64%8]>().to_vrf();
    let x:TrfTensor<f8e4m3,Chip,ProjectionCluster,UpParts,m![P512%256=64%8],m![P512%256=64/8,H%960]>=device.sub.begin(input.view())
        .fetch::<m![1],m![H%960]>()
        .switch::<UpParts,m![P512%256=64%8,P512%256=64/8]>(SwitchConfig::CustomBroadcast{ring_size:256})
        .collect::<m![P512%256=64%8,P512%256=64/8,H/32%30],m![H%32]>().to_trf();
    let result:DmTensor<bf16,Chip,ProjectionCluster,UpRows,m![L%240,P512%256=64]>=device.main.begin(weight.view())
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

pub(crate) fn global_dm(device: &mut Device, global: &HbmTensor<f32, Chip, m![1]>)
 -> DmTensor<f32, Chip, Cluster, Shards, m![1 # 8]> {
 global.to_dm(&mut device.tdma)
}
fn scale_tile(
    device: &mut Device,
    input: DmTensorView<'_,bf16,Chip,Cluster,Shards,m![P512%256=64%4=2#4,L%960]>,
    scale: &DmTensor<f32, Chip, Cluster, Shards, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, Shards, m![P512%256=64%4=2,L%960]> {
    let scale: VrfTensor<f32, Chip, Cluster, Shards, m![1 # 8]> = device.sub
        .begin(scale.view()).fetch::<m![1],m![1 # 8]>()
        .collect::<m![1],m![1 # 8]>().to_vrf();
    device
        .main
        .begin(input)
        .fetch::<m![1], m![P512%256=64%4=2,L%960]>()
        .fetch_cast::<f32>()
        .collect::<m![P512%256=64%4=2,L/8%120], m![L%8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P512%256=64%4=2,L/4%240], m![L%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![P512%256=64%4=2,L/8%120], m![L%8]>()
        .vector_final()
        .cast::<bf16, m![L%8#16]>()
        .commit_trim::<m![L%8]>()
        .commit()
}
pub(crate) fn geglu_tile_into(
    device: &mut Device,
    up: DmTensorView<'_,bf16,Chip,Cluster,Shards,m![P512%256=64%4=2#4,L%960]>,
    gate: DmTensorView<'_,bf16,Chip,Cluster,Shards,m![P512%256=64%4=2#4,L%960]>,
    up_global_scale: &DmTensor<f32, Chip, Cluster, Shards, m![1 # 8]>,
    gate_global_scale: &DmTensor<f32, Chip, Cluster, Shards, m![1 # 8]>,
    output:DmTensorViewMut<'_,bf16,Chip,Cluster,Shards,m![P512%256=64%4=2#{!}4,L%960]>,
) {
    let up = scale_tile(device, up, up_global_scale);
    let gate = scale_tile(device, gate, gate_global_scale);
    let gelu: DmTensor<f32, Chip, Cluster, Shards, m![P512%256=64%4=2,L%960]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![P512%256=64%4=2,L/8%120], m![L%8]>()
        .fetch_cast::<f32>()
        .collect::<m![P512%256=64%4=2,L/8%120], m![L%8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P512%256=64%4=2,L/4%240], m![L%4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.70710678118f32)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![P512%256=64%4=2,L/8%120], m![L%8]>()
        .vector_final()
        .commit_trim::<m![L%8]>()
        .commit();
    let gelu: VrfTensor<f32, Chip, Cluster, Shards, m![P512%256=64%4=2,L%960]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![P512%256=64%4=2,L/8%120], m![L%8]>()
        .collect::<m![P512%256=64%4=2,L/8%120], m![L%8]>()
        .to_vrf();
    device
        .main
        .begin(up.view())
        .fetch::<m![P512%256=64%4=2,L/8%120], m![L%8]>()
        .fetch_cast::<f32>()
        .collect::<m![P512%256=64%4=2,L/8%120], m![L%8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P512%256=64%4=2,L/4%240], m![L%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![P512%256=64%4=2,L/8%120], m![L%8]>()
        .vector_final()
        .cast::<bf16, m![L%8#16]>()
        .commit_trim::<m![L%8]>()
        .commit_view(output);
}


fn stage_geglu_quantize(device:&mut Device,joined:&HbmTensor<bf16,Chip,JoinedOutput>,up_global:&HbmTensor<f32,Chip,m![1]>,gate_global:&HbmTensor<f32,Chip,m![1]>,
    output:&mut HbmTensor<f8e4m3,Chip,P512L960>,scales:&mut HbmTensor<f32,Chip,m![P512]>) {
    let up=joined.view().tile::<m![FfnProjection],1,m![FfnProjection=1#2,P512/256,L/240,P512%256,L%240]>(0);
    let gate=joined.view().tile::<m![FfnProjection],1,m![FfnProjection=1#2,P512/256,L/240,P512%256,L%240]>(1);
    let up_global = global_dm(device,up_global);
    let gate_global = global_dm(device,gate_global);
    for chunk in 0..4 {
        let up:DmTensor<bf16,Chip,Cluster,Shards,m![P512%256=64%4,L%960]>=up
            .tile::<m![P512%256],64,m![FfnProjection=1#2,P512/256,L/240,P512%256=64#256,L%240]>(64*chunk).to_dm(&mut device.tdma);
        let gate:DmTensor<bf16,Chip,Cluster,Shards,m![P512%256=64%4,L%960]>=gate
            .tile::<m![P512%256],64,m![FfnProjection=1#2,P512/256,L/240,P512%256=64#256,L%240]>(64*chunk).to_dm(&mut device.tdma);
        let mut hidden:DmTensor<bf16,Chip,Cluster,Shards,m![P512%256=64%4,L%960]>=DmTensor::new();
        for tile in 0..2 {
            geglu_tile_into(device,
                up.view().tile::<m![P512%256=64%4],2,m![P512%256=64%4=2#4,L%960]>(2*tile),
                gate.view().tile::<m![P512%256=64%4],2,m![P512%256=64%4=2#4,L%960]>(2*tile),&up_global,&gate_global,
                hidden.view_mut().tile::<m![P512%256=64%4],2,m![P512%256=64%4=2#{!}4,L%960]>(2*tile));
        }
        let amax:DmTensor<f32,Chip,Cluster,Shards,m![P512%256=64%4,1#8]>=device.main.begin(hidden.view())
            .fetch::<m![P512%256=64%4,L/16%60],m![L%16]>().fetch_cast::<f32>()
            .collect::<m![P512%256=64%4,L/8%120],m![L%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32).vector_reinterpret::<f32>()
            .vector_narrow_split::<m![P512%256=64%4,L/4%240],m![L%4]>()
            .vector_intra_slice_reduce::<L,m![P512%256=64%4],m![1#4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_pad::<m![1#8]>().vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
            .vector_final().commit_trim::<m![1#8]>().commit();
        let amax:DmTensor<f32,Chip,Cluster,MaxParts,m![P512%256=64%4,1#8]>=amax.to_dm(&mut device.tdma);
        let amax:DmTensor<f32,Chip,Cluster,MaxRows,m![P512%256=64%4,1#8]>=device.main.begin(amax.view())
            .fetch::<m![P512%256=64%4],m![1#8]>().collect::<m![P512%256=64%4],m![1#8]>()
            .vector_init().vector_inter_slice_reduce::<MaxRows,m![P512%256=64%4]>(InterSliceReduceOpF32::Max)
            .vector_final().commit_trim::<m![1#8]>().commit();
        let scale:DmTensor<f32,Chip,Cluster,MaxRows,m![P512%256=64%4,1#8]>=device.main.begin(amax.view())
            .fetch::<m![P512%256=64%4],m![1#8]>().collect::<m![P512%256=64%4],m![1#8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7f80_0000i32).vector_reinterpret::<f32>()
            .vector_narrow_trim::<m![1#4]>().vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),0.0078125f32)
            .vector_widen_pad::<m![1#8]>().vector_final().commit_trim::<m![1#8]>().commit();
        let broadcast:DmTensor<f32,Chip,Cluster,Shards,m![P512%256=64%4,1#8]>=scale.to_dm(&mut device.tdma);
        let packed:DmTensor<f32,Chip,Cluster,Shards,m![P512%256=64%4]>=device.main.begin(broadcast.view())
            .fetch::<m![P512%256=64%4],m![1#8]>().collect::<m![P512%256=64%4],m![1#8]>()
            .transpose::<m![P512%256=64%4/2],m![P512%256=64%4%2#8]>().commit_trim::<m![P512%256=64%4%2]>().commit();
        let sv:VrfTensor<f32,Chip,Cluster,Shards,m![P512%256=64%4#8]>=device.sub.begin(packed.view())
            .fetch::<m![1],m![P512%256=64%4#8]>().collect::<m![1],m![P512%256=64%4#8]>().to_vrf();
        let q:DmTensor<f8e4m3,Chip,Cluster,Shards,m![P512%256=64%4,L%960]>=device.main.begin(hidden.view())
            .fetch::<m![P512%256=64%4,L/16%60],m![L%16]>().fetch_cast::<f32>()
            .collect::<m![P512%256=64%4,L/8%120],m![L%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![P512%256=64%4,L/4%240],m![L%4]>()
            .vector_fp_binary(FpBinaryOp::DivF,&sv)
            .vector_widen_concat::<m![P512%256=64%4,L/8%120],m![L%8]>().vector_final()
            .cast::<f8e4m3,m![L%8#32]>().commit_trim::<m![L%8]>().commit();
        q.view().to_hbm_view(&mut device.tdma,output.view_mut()
            .tile::<m![P512%256],64,m![P512/256,L/960,P512%256=64#{!}256,L%960]>(64*chunk));
        let scale:DmTensor<f32,Chip,Cluster,MaxRows,m![P512%256=64%4]>=device.main.begin(scale.view())
            .fetch::<m![P512%256=64%4],m![1#8]>().collect::<m![P512%256=64%4],m![1#8]>()
            .transpose::<m![P512%256=64%4/2],m![P512%256=64%4%2#8]>().commit_trim::<m![P512%256=64%4%2]>().commit();
        scale.view().to_hbm_view(&mut device.tdma,scales.view_mut()
            .tile::<m![P512%256],64,m![P512/256,P512%256=64#{!}256]>(64*chunk));
    }
}

fn stage_down(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512L960>,
 xscale:&HbmTensor<f32,Chip,m![P512]>,weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
 weight_dequant_scale:&HbmTensor<f32,Chip,m![H]>,global_scale:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,crate::ops_native_fp8_p512_downblocked::Output>) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![H % 240,L % 960]>=weight.to_dm(&mut device.tdma);
    let ws:DmTensor<f32,Chip,Cluster,DownRows,m![H % 240]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 240]>=device.sub.begin(ws.view())
        .fetch::<m![H / 8 % 30],m![H % 8]>().collect::<m![H / 8 % 30],m![H % 8]>().to_vrf();
    let global:DmTensor<f32,Chip,Cluster,DownRows,m![1#8]>=global_scale.to_dm(&mut device.tdma);
    let global:VrfTensor<f32,Chip,Cluster,DownRows,m![1#8]>=device.sub.begin(global.view())
        .fetch::<m![1],m![1#8]>().collect::<m![1],m![1#8]>().to_vrf();
    macro_rules! chunk {($chunk:expr)=>{{
        let chunk=$chunk;
        let input:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![P512%256=64,L%960]>=input.view()
            .tile::<m![P512%256],64,m![P512/256,L/960,P512%256=64#256,L%960]>(64*chunk).to_dm(&mut device.tdma);
        let xscale:DmTensor<f32,Chip,Cluster,DownRows,m![P512%256=64]>=xscale.view()
            .tile::<m![P512%256],64,m![P512/256,P512%256=64#256]>(64*chunk).to_dm(&mut device.tdma);
    let xs:VrfTensor<f32,Chip,Cluster,DownRows,m![P512%256=64]>=device.sub.begin(xscale.view())
        .fetch::<m![P512%256=64/8],m![P512%256=64%8]>().collect::<m![P512%256=64/8],m![P512%256=64%8]>().to_vrf();
    let x:TrfTensor<f8e4m3,Chip,Cluster,DownParts,m![P512%256=64%8],m![P512%256=64/8,L%960]>=device.sub.begin(input.view())
        .fetch::<m![P512%256=64%8,P512%256=64/8,L/32%30],m![L%32]>()
        .collect::<m![P512%256=64%8,P512%256=64/8,L/32%30],m![L%32]>().to_trf();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H%240,P512%256=64]>=device.main.begin(weight.view())
        .fetch::<m![H%240,L/32%30],m![L%32]>().collect::<m![H%240,L/32%30],m![L%32]>()
        .contract_outer::<m![H%240,L/64%15,P512%256=64/8],m![L%64],_,_,_>(&x)
        .contract_packet::<m![1]>().contract_time::<m![H%240,P512%256=64/8]>()
        .contract_lane::<m![H%240,P512%256=64/8],m![P512%256=64%8]>(LaneMode::Interleaved)
        .vector_init().vector_inter_slice_reduce::<DownRows,m![H%240,P512%256=64/8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H%240,P512%256=64/4],m![P512%256=64%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![H%240,P512%256=64/8],m![P512%256=64%8]>()
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

pub(crate) fn ffn_core_stage(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512H960>,xscale:&HbmTensor<f32,Chip,m![P512]>,
 joined_weight:&HbmTensor<f8e4m3,Chip,JoinedWeight>,joined_scale:&HbmTensor<f32,Chip,JoinedScale>,down:&HbmTensor<f8e4m3,Chip,DownWeight>,
 ds:&HbmTensor<f32,Chip,m![H]>,ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
 joined_output:&mut HbmTensor<bf16,Chip,JoinedOutput>,hidden_quant:&mut HbmTensor<f8e4m3,Chip,P512L960>,hidden_scale:&mut HbmTensor<f32,Chip,m![P512]>,
 output:&mut HbmTensor<bf16,Chip,crate::ops_native_fp8_p512_downblocked::Output>) {
 stage_upgate(device,input,xscale,joined_weight,joined_scale,joined_output);
 stage_geglu_quantize(device,joined_output,ug,gg,hidden_quant,hidden_scale);
 stage_down(device,hidden_quant,hidden_scale,down,ds,dg,output);
}

#[device(chip=1)]
pub fn ffn_core(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512H960>,xscale:&HbmTensor<f32,Chip,m![P512]>,
 joined_weight:&HbmTensor<f8e4m3,Chip,JoinedWeight>,joined_scale:&HbmTensor<f32,Chip,JoinedScale>,down:&HbmTensor<f8e4m3,Chip,DownWeight>,
 ds:&HbmTensor<f32,Chip,m![H]>,ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
 joined_output:&mut HbmTensor<bf16,Chip,JoinedOutput>,hidden_quant:&mut HbmTensor<f8e4m3,Chip,P512L960>,hidden_scale:&mut HbmTensor<f32,Chip,m![P512]>,
 output:&mut HbmTensor<bf16,Chip,crate::ops_native_fp8_p512_downblocked::Output>) {
 ffn_core_stage(device,input,xscale,joined_weight,joined_scale,down,ds,ug,gg,dg,joined_output,hidden_quant,hidden_scale,output);
}
