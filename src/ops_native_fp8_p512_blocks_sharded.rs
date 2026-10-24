//! Same P512 blocked FFN ABI, with 256 live L960/token4 GeGLU consumer slices.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P512,L}};
use crate::ops_batch_layouts::{P512L960,P512L240};
pub use crate::ops_native_fp8_p512_blocks::{quantize_hidden,project_up_scaled,project_down_scaled};
type Cluster=m![P512/256];
type Shards=m![L/960,P512%256=64/4];
type MaxParts=m![P512%256=64/4,L/960];
type MaxRows=m![P512%256=64/4,1#16];
fn scale_tile(
    device: &mut Device,
    input: DmTensorView<'_,bf16,Chip,Cluster,Shards,m![P512%256=64%4=2#4,L%960]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, Shards, m![P512%256=64%4=2,L%960]> {
    let scale: DmTensor<f32, Chip, Cluster, Shards, m![1 # 8]> =
        global_scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, Shards, m![1 # 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
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
fn geglu_tile_into(
    device: &mut Device,
    up: DmTensorView<'_,bf16,Chip,Cluster,Shards,m![P512%256=64%4=2#4,L%960]>,
    gate: DmTensorView<'_,bf16,Chip,Cluster,Shards,m![P512%256=64%4=2#4,L%960]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
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


#[device(chip=1)]
pub fn geglu_quantize(device:&mut Device,up:&HbmTensor<bf16,Chip,P512L240>,
    gate:&HbmTensor<bf16,Chip,P512L240>,up_global:&HbmTensor<f32,Chip,m![1]>,gate_global:&HbmTensor<f32,Chip,m![1]>,
    output:&mut HbmTensor<f8e4m3,Chip,P512L960>,scales:&mut HbmTensor<f32,Chip,m![P512]>) {
    for chunk in 0..4 {
        let up:DmTensor<bf16,Chip,Cluster,Shards,m![P512%256=64%4,L%960]>=up.view()
            .tile::<m![P512%256],64,m![P512/256,L/240,P512%256=64#256,L%240]>(64*chunk).to_dm(&mut device.tdma);
        let gate:DmTensor<bf16,Chip,Cluster,Shards,m![P512%256=64%4,L%960]>=gate.view()
            .tile::<m![P512%256],64,m![P512/256,L/240,P512%256=64#256,L%240]>(64*chunk).to_dm(&mut device.tdma);
        let mut hidden:DmTensor<bf16,Chip,Cluster,Shards,m![P512%256=64%4,L%960]>=DmTensor::new();
        for tile in 0..2 {
            geglu_tile_into(device,
                up.view().tile::<m![P512%256=64%4],2,m![P512%256=64%4=2#4,L%960]>(2*tile),
                gate.view().tile::<m![P512%256=64%4],2,m![P512%256=64%4=2#4,L%960]>(2*tile),up_global,gate_global,
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

/// Quantize already-computed blocked hidden rows using the same sharded amax.
#[device(chip=1)]
pub fn quantize_ffn(device:&mut Device,input:&HbmTensor<bf16,Chip,P512L240>,
    output:&mut HbmTensor<f8e4m3,Chip,P512L960>,scales:&mut HbmTensor<f32,Chip,m![P512]>) {
    for chunk in 0..4 {
        let hidden:DmTensor<bf16,Chip,Cluster,Shards,m![P512%256=64%4,L%960]>=input.view()
            .tile::<m![P512%256],64,m![P512/256,L/240,P512%256=64#256,L%240]>(64*chunk).to_dm(&mut device.tdma);
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
