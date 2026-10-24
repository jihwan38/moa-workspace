//! Same GeGLU arithmetic with direct B8 views and direct output commits.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,L}};
type Cluster=m![P128/64];
type UpReduced=m![L/240,1 # 4];
fn scale_tile(
    device: &mut Device,
    input: DmTensorView<'_,bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64 = 8 # 64]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpReduced, m![L % 240, P128 % 64 = 8]> {
    let scale: DmTensor<f32, Chip, Cluster, UpReduced, m![1 # 8]> =
        global_scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, UpReduced, m![1 # 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    device
        .main
        .begin(input)
        .fetch::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 240, P128 % 64 = 8 / 4], m![P128 % 64 = 8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128 % 64 = 8 % 8 # 16]>()
        .commit_trim::<m![P128 % 64 = 8 % 8]>()
        .commit()
}
fn geglu_tile_into(
    device: &mut Device,
    up: DmTensorView<'_,bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64 = 8 # 64]>,
    gate: DmTensorView<'_,bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64 = 8 # 64]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    output:DmTensorViewMut<'_,bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64 = 8 #{!} 64]>,
) {
    let up = scale_tile(device, up, up_global_scale);
    let gate = scale_tile(device, gate, gate_global_scale);
    let gelu: DmTensor<f32, Chip, Cluster, UpReduced, m![L % 240, P128 % 64 = 8]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 240, P128 % 64 = 8 / 4], m![P128 % 64 = 8 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.70710678118f32)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .vector_final()
        .commit_trim::<m![P128 % 64 = 8 % 8]>()
        .commit();
    let gelu: VrfTensor<f32, Chip, Cluster, UpReduced, m![L % 240, P128 % 64 = 8]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .collect::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .to_vrf();
    device
        .main
        .begin(up.view())
        .fetch::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 240, P128 % 64 = 8 / 4], m![P128 % 64 = 8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128 % 64 = 8 % 8 # 16]>()
        .commit_trim::<m![P128 % 64 = 8 % 8]>()
        .commit_view(output);
}

pub(crate) fn geglu(device:&mut Device,up:&DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64]>,
    gate:&DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64]>,
    us:&HbmTensor<f32,Chip,m![1]>,gs:&HbmTensor<f32,Chip,m![1]>)
    ->DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64]> {
    let mut result:DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        geglu_tile_into(device,
            up.view().tile::<m![P128 % 64],8,m![L % 240,P128 % 64 = 8 # 64]>(8*chunk),
            gate.view().tile::<m![P128 % 64],8,m![L % 240,P128 % 64 = 8 # 64]>(8*chunk),us,gs,
            result.view_mut().tile::<m![P128 % 64],8,m![L % 240,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    result
}
