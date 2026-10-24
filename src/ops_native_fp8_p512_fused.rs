//! Exact P512 up/gate input/TRF sharing with GeGLU before HBM output.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P512,H,L}};
pub use crate::ops_native_fp8_blocked::{UpWeight,DownWeight};
pub use crate::ops_native_fp8_p512::{quantize_hidden,project_down_scaled,quantize_ffn};
type Cluster=m![L/7680];
type UpParts=m![L/120%64,H/960];
type UpRows=m![L/120%64,1#4];
fn matrix_tile(device:&mut Device,
    input:&TrfTensor<f8e4m3,Chip,Cluster,UpParts,m![P512=64%8],m![P512=64/8,H%960]>,
    weight:&DmTensor<f8e4m3,Chip,Cluster,UpParts,m![L%120,H%960]>,
    xs:&DmTensor<f32,Chip,Cluster,UpRows,m![P512=64]>,
    ws:&DmTensor<f32,Chip,Cluster,UpRows,m![L%120]>)
    ->DmTensor<bf16,Chip,Cluster,UpRows,m![L%120,P512=64]> {
    let xs:VrfTensor<f32,Chip,Cluster,UpRows,m![P512=64]>=device.sub.begin(xs.view())
        .fetch::<m![P512=64/8],m![P512=64%8]>().collect::<m![P512=64/8],m![P512=64%8]>().to_vrf();
    let ws:VrfTensor<f32,Chip,Cluster,UpRows,m![L%120]>=device.sub.begin(ws.view())
        .fetch::<m![L/8%15],m![L%8]>().collect::<m![L/8%15],m![L%8]>().to_vrf();
    device.main.begin(weight.view())
        .fetch::<m![L%120,H/32%30],m![H%32]>().collect::<m![L%120,H/32%30],m![H%32]>()
        .contract_outer::<m![L%120,H/64%15,P512=64/8],m![H%64],_,_,_>(input)
        .contract_packet::<m![1]>().contract_time::<m![L%120,P512=64/8]>()
        .contract_lane::<m![L%120,P512=64/8],m![P512=64%8]>(LaneMode::Interleaved)
        .vector_init().vector_inter_slice_reduce::<UpRows,m![L%120,P512=64/8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L%120,P512=64/4],m![P512=64%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![L%120,P512=64/8],m![P512=64%8]>()
        .vector_final().cast::<bf16,m![P512=64%8#16]>().commit_trim::<m![P512=64%8]>().commit()
}

fn scale_tile(
    device: &mut Device,
    input: DmTensorView<'_,bf16,Chip,Cluster,UpRows,m![L % 120,P512=64=8 # 64]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpRows, m![L % 120, P512=64=8]> {
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
        .begin(input)
        .fetch::<m![L % 120], m![P512=64=8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 120, P512=64=8 / 8], m![P512=64=8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 120, P512=64=8 / 4], m![P512=64=8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![L % 120, P512=64=8 / 8], m![P512=64=8 % 8]>()
        .vector_final()
        .cast::<bf16, m![P512=64=8 % 8 # 16]>()
        .commit_trim::<m![P512=64=8 % 8]>()
        .commit()
}
fn geglu_tile_into(
    device: &mut Device,
    up: DmTensorView<'_,bf16,Chip,Cluster,UpRows,m![L % 120,P512=64=8 # 64]>,
    gate: DmTensorView<'_,bf16,Chip,Cluster,UpRows,m![L % 120,P512=64=8 # 64]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    output:DmTensorViewMut<'_,bf16,Chip,Cluster,UpRows,m![L % 120,P512=64=8 #{!} 64]>,
) {
    let up = scale_tile(device, up, up_global_scale);
    let gate = scale_tile(device, gate, gate_global_scale);
    let gelu: DmTensor<f32, Chip, Cluster, UpRows, m![L % 120, P512=64=8]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![L % 120, P512=64=8 / 8], m![P512=64=8 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 120, P512=64=8 / 8], m![P512=64=8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 120, P512=64=8 / 4], m![P512=64=8 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.70710678118f32)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L % 120, P512=64=8 / 8], m![P512=64=8 % 8]>()
        .vector_final()
        .commit_trim::<m![P512=64=8 % 8]>()
        .commit();
    let gelu: VrfTensor<f32, Chip, Cluster, UpRows, m![L % 120, P512=64=8]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![L % 120, P512=64=8 / 8], m![P512=64=8 % 8]>()
        .collect::<m![L % 120, P512=64=8 / 8], m![P512=64=8 % 8]>()
        .to_vrf();
    device
        .main
        .begin(up.view())
        .fetch::<m![L % 120, P512=64=8 / 8], m![P512=64=8 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 120, P512=64=8 / 8], m![P512=64=8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 120, P512=64=8 / 4], m![P512=64=8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L % 120, P512=64=8 / 8], m![P512=64=8 % 8]>()
        .vector_final()
        .cast::<bf16, m![P512=64=8 % 8 # 16]>()
        .commit_trim::<m![P512=64=8 % 8]>()
        .commit_view(output);
}

fn geglu(device:&mut Device,up:&DmTensor<bf16,Chip,Cluster,UpRows,m![L % 120,P512=64]>,
    gate:&DmTensor<bf16,Chip,Cluster,UpRows,m![L % 120,P512=64]>,
    us:&HbmTensor<f32,Chip,m![1]>,gs:&HbmTensor<f32,Chip,m![1]>)
    ->DmTensor<bf16,Chip,Cluster,UpRows,m![L % 120,P512=64]> {
    let mut result:DmTensor<bf16,Chip,Cluster,UpRows,m![L % 120,P512=64]>=DmTensor::new();
    for chunk in 0..8 {
        geglu_tile_into(device,
            up.view().tile::<m![P512=64],8,m![L % 120,P512=64=8 # 64]>(8*chunk),
            gate.view().tile::<m![P512=64],8,m![L % 120,P512=64=8 # 64]>(8*chunk),us,gs,
            result.view_mut().tile::<m![P512=64],8,m![L % 120,P512=64=8 #{!} 64]>(8*chunk));
    }
    result
}

#[device(chip=1)]
pub fn project_upgate_geglu(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![P512,H]>,
    xscale:&HbmTensor<f32,Chip,m![P512]>,up:&HbmTensor<f8e4m3,Chip,UpWeight>,gate:&HbmTensor<f8e4m3,Chip,UpWeight>,
    up_dequant:&HbmTensor<f32,Chip,m![L]>,gate_dequant:&HbmTensor<f32,Chip,m![L]>,
    up_global:&HbmTensor<f32,Chip,m![1]>,gate_global:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,m![P512,L]>) {
    let up:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![L%120,H%960]>=up.to_dm(&mut device.tdma);
    let gate:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![L%120,H%960]>=gate.to_dm(&mut device.tdma);
    let us:DmTensor<f32,Chip,Cluster,UpRows,m![L%120]>=up_dequant.to_dm(&mut device.tdma);
    let gs:DmTensor<f32,Chip,Cluster,UpRows,m![L%120]>=gate_dequant.to_dm(&mut device.tdma);
    for chunk in 0..8 {
        let input:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![P512=64,H%960]>=input.view()
            .tile::<m![P512],64,m![P512=64#512,H]>(64*chunk).to_dm(&mut device.tdma);
        let xscale:DmTensor<f32,Chip,Cluster,UpRows,m![P512=64]>=xscale.view()
            .tile::<m![P512],64,m![P512=64#512]>(64*chunk).to_dm(&mut device.tdma);
        let x:TrfTensor<f8e4m3,Chip,Cluster,UpParts,m![P512=64%8],m![P512=64/8,H%960]>=device.sub.begin(input.view())
            .fetch::<m![P512=64%8,P512=64/8,H/32%30],m![H%32]>()
            .collect::<m![P512=64%8,P512=64/8,H/32%30],m![H%32]>().to_trf();
        let a=matrix_tile(device,&x,&up,&xscale,&us);
        let b=matrix_tile(device,&x,&gate,&xscale,&gs);
        let hidden=geglu(device,&a,&b,up_global,gate_global);
        let hidden:DmTensor<bf16,Chip,Cluster,UpRows,m![P512=64,L%120]>=device.main.begin(hidden.view())
            .fetch::<m![P512=64/16,L%120],m![P512=64%16]>().collect::<m![P512=64/16,L%120],m![P512=64%16]>()
            .transpose::<m![P512=64/16,L/4%30,P512=64%16],m![L%4#16]>().commit_trim::<m![L%4]>().commit();
        hidden.view().to_hbm_view(&mut device.tdma,output.view_mut().tile::<m![P512],64,m![P512=64#{!}512,L]>(64*chunk));
    }
}
