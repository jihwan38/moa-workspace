//! Token-cluster P512 fused up/gate, retaining both blocked weight tiles across TRF32 chunks.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P512,H,L}};
pub use crate::ops_native_fp8_blocked::{UpWeight,DownWeight};
pub use crate::ops_native_fp8_p512_blocks::{quantize_hidden,project_down_scaled};
pub use crate::ops_native_fp8_p512_blocks_sharded::quantize_ffn;
use crate::ops_batch_layouts::{P512H960,P512L240};
type Cluster=m![P512/256];
type UpParts=m![L/240,H/960];
type UpRows=m![L/240,1#4];
fn matrix_tile(device:&mut Device,
    input:&TrfTensor<f8e4m3,Chip,Cluster,UpParts,m![P512%256=32%8],m![P512%256=32/8,H%960]>,
    weight:&DmTensor<f8e4m3,Chip,Cluster,UpParts,m![L%240,H%960]>,
    xs:&DmTensor<f32,Chip,Cluster,UpRows,m![P512%256=32]>,
    ws:&DmTensor<f32,Chip,Cluster,UpRows,m![L%240]>)
    ->DmTensor<bf16,Chip,Cluster,UpRows,m![L%240,P512%256=32]> {
    let xs:VrfTensor<f32,Chip,Cluster,UpRows,m![P512%256=32]>=device.sub.begin(xs.view())
        .fetch::<m![P512%256=32/8],m![P512%256=32%8]>().collect::<m![P512%256=32/8],m![P512%256=32%8]>().to_vrf();
    let ws:VrfTensor<f32,Chip,Cluster,UpRows,m![L%240]>=device.sub.begin(ws.view())
        .fetch::<m![L/8%30],m![L%8]>().collect::<m![L/8%30],m![L%8]>().to_vrf();
    device.main.begin(weight.view())
        .fetch::<m![L%240,H/32%30],m![H%32]>().collect::<m![L%240,H/32%30],m![H%32]>()
        .contract_outer::<m![L%240,H/64%15,P512%256=32/8],m![H%64],_,_,_>(input)
        .contract_packet::<m![1]>().contract_time::<m![L%240,P512%256=32/8]>()
        .contract_lane::<m![L%240,P512%256=32/8],m![P512%256=32%8]>(LaneMode::Interleaved)
        .vector_init().vector_inter_slice_reduce::<UpRows,m![L%240,P512%256=32/8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L%240,P512%256=32/4],m![P512%256=32%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![L%240,P512%256=32/8],m![P512%256=32%8]>()
        .vector_final().cast::<bf16,m![P512%256=32%8#16]>().commit_trim::<m![P512%256=32%8]>().commit()
}

fn scale_tile(
    device: &mut Device,
    input: DmTensorView<'_,bf16,Chip,Cluster,UpRows,m![L % 240,P512%256=32=4 # 32]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpRows, m![P512%256=32=4,L % 240]> {
    let input:DmTensor<bf16,Chip,Cluster,UpRows,m![P512%256=32=4,L%240]>=device.main.begin(input)
        .fetch::<m![L%240],m![P512%256=32=4#16]>().collect::<m![L%240],m![P512%256=32=4#16]>()
        .transpose::<m![L/4%60,P512%256=32=4],m![L%4#16]>().commit_trim::<m![L%4]>().commit();
    let scale: DmTensor<f32, Chip, Cluster, UpRows, m![1 # 8]> =
        global_scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, UpRows, m![1 # 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    device
        .main
        .begin(input.view())
        .fetch::<m![P512%256=32=4,L /8 %30], m![L %8]>()
        .fetch_cast::<f32>()
        .collect::<m![P512%256=32=4,L /8 %30], m![L %8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P512%256=32=4,L /4 %60], m![L %4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![P512%256=32=4,L /8 %30], m![L %8]>()
        .vector_final()
        .cast::<bf16, m![L %8 #16]>()
        .commit_trim::<m![L %8]>()
        .commit()
}
fn geglu_tile_into(
    device: &mut Device,
    up: DmTensorView<'_,bf16,Chip,Cluster,UpRows,m![L % 240,P512%256=32=4 # 32]>,
    gate: DmTensorView<'_,bf16,Chip,Cluster,UpRows,m![L % 240,P512%256=32=4 # 32]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    output:DmTensorViewMut<'_,bf16,Chip,Cluster,UpRows,m![L % 240,P512%256=32=4 #{!} 32]>,
) {
    let up = scale_tile(device, up, up_global_scale);
    let gate = scale_tile(device, gate, gate_global_scale);
    let gelu: DmTensor<f32, Chip, Cluster, UpRows, m![P512%256=32=4,L % 240]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![P512%256=32=4,L /8 %30], m![L %8]>()
        .fetch_cast::<f32>()
        .collect::<m![P512%256=32=4,L /8 %30], m![L %8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P512%256=32=4,L /4 %60], m![L %4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.70710678118f32)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![P512%256=32=4,L /8 %30], m![L %8]>()
        .vector_final()
        .commit_trim::<m![L %8]>()
        .commit();
    let gelu: VrfTensor<f32, Chip, Cluster, UpRows, m![P512%256=32=4,L %240]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![P512%256=32=4,L /8 %30], m![L %8]>()
        .collect::<m![P512%256=32=4,L /8 %30], m![L %8]>()
        .to_vrf();
    let result:DmTensor<bf16,Chip,Cluster,UpRows,m![P512%256=32=4,L%240]>=device
        .main
        .begin(up.view())
        .fetch::<m![P512%256=32=4,L /8 %30], m![L %8]>()
        .fetch_cast::<f32>()
        .collect::<m![P512%256=32=4,L /8 %30], m![L %8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P512%256=32=4,L /4 %60], m![L %4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![P512%256=32=4,L /8 %30], m![L %8]>()
        .vector_final()
        .cast::<bf16, m![L %8 #16]>()
        .commit_trim::<m![L %8]>()
        .commit();
    device.main.begin(result.view())
        .fetch::<m![L/16%15,P512%256=32=4],m![L%16]>().collect::<m![L/16%15,P512%256=32=4],m![L%16]>()
        .transpose::<m![L%240],m![P512%256=32=4#16]>().commit_trim::<m![P512%256=32=4]>().commit_view(output);
}

fn geglu(device:&mut Device,up:&DmTensor<bf16,Chip,Cluster,UpRows,m![L % 240,P512%256=32]>,
    gate:&DmTensor<bf16,Chip,Cluster,UpRows,m![L % 240,P512%256=32]>,
    us:&HbmTensor<f32,Chip,m![1]>,gs:&HbmTensor<f32,Chip,m![1]>)
    ->DmTensor<bf16,Chip,Cluster,UpRows,m![L % 240,P512%256=32]> {
    let mut result:DmTensor<bf16,Chip,Cluster,UpRows,m![L % 240,P512%256=32]>=DmTensor::new();
    for chunk in 0..8 {
        geglu_tile_into(device,
            up.view().tile::<m![P512%256=32],4,m![L % 240,P512%256=32=4 # 32]>(4*chunk),
            gate.view().tile::<m![P512%256=32],4,m![L % 240,P512%256=32=4 # 32]>(4*chunk),us,gs,
            result.view_mut().tile::<m![P512%256=32],4,m![L % 240,P512%256=32=4 #{!} 32]>(4*chunk));
    }
    result
}

#[device(chip=1)]
pub fn project_upgate_geglu(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512H960>,
    xscale:&HbmTensor<f32,Chip,m![P512]>,up:&HbmTensor<f8e4m3,Chip,UpWeight>,gate:&HbmTensor<f8e4m3,Chip,UpWeight>,
    up_dequant:&HbmTensor<f32,Chip,m![L]>,gate_dequant:&HbmTensor<f32,Chip,m![L]>,
    up_global:&HbmTensor<f32,Chip,m![1]>,gate_global:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,P512L240>) {
    let up:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![L%240,H%960]>=up.to_dm(&mut device.tdma);
    let gate:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![L%240,H%960]>=gate.to_dm(&mut device.tdma);
    let us:DmTensor<f32,Chip,Cluster,UpRows,m![L%240]>=up_dequant.to_dm(&mut device.tdma);
    let gs:DmTensor<f32,Chip,Cluster,UpRows,m![L%240]>=gate_dequant.to_dm(&mut device.tdma);
    for chunk in 0..8 {
        let input:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![P512%256=32,H%960]>=input.view()
            .tile::<m![P512%256],32,m![P512/256,H/960,P512%256=32#256,H%960]>(32*chunk).to_dm(&mut device.tdma);
        let xscale:DmTensor<f32,Chip,Cluster,UpRows,m![P512%256=32]>=xscale.view()
            .tile::<m![P512%256],32,m![P512/256,P512%256=32#256]>(32*chunk).to_dm(&mut device.tdma);
        let x:TrfTensor<f8e4m3,Chip,Cluster,UpParts,m![P512%256=32%8],m![P512%256=32/8,H%960]>=device.sub.begin(input.view())
            .fetch::<m![P512%256=32%8,P512%256=32/8,H/32%30],m![H%32]>()
            .collect::<m![P512%256=32%8,P512%256=32/8,H/32%30],m![H%32]>().to_trf();
        let a=matrix_tile(device,&x,&up,&xscale,&us);
        let b=matrix_tile(device,&x,&gate,&xscale,&gs);
        let hidden=geglu(device,&a,&b,up_global,gate_global);
        let hidden:DmTensor<bf16,Chip,Cluster,UpRows,m![P512%256=32,L%240]>=device.main.begin(hidden.view())
            .fetch::<m![P512%256=32/16,L%240],m![P512%256=32%16]>().collect::<m![P512%256=32/16,L%240],m![P512%256=32%16]>()
            .transpose::<m![P512%256=32/16,L/4%60,P512%256=32%16],m![L%4#16]>().commit_trim::<m![L%4]>().commit();
        hidden.view().to_hbm_view(&mut device.tdma,output.view_mut().tile::<m![P512%256],32,m![P512/256,L/240,P512%256=32#{!}256,L%240]>(32*chunk));
    }
}
