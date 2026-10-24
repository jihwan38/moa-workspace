//! Direct P128 hidden normalization and residuals with the original H/480 sum tree.
use crate::axes::{P128, Dummy8, H};
type Cluster=m![P128/64];
use crate::{Chip, EPS};
use furiosa_opt_std::prelude::*;
const H_F32: f32 = H::SIZE as f32;
type Shards = m![P128/2%32,H/480];
fn normalize(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Shards, m![P128%2,H % 480]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Shards, m![P128%2,H % 480]> {
    type ReducingSlices = Shards;

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![P128%2,1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![P128%2,H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%2,H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%2,H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![P128%2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![P128/2%32,Dummy8], m![P128%2,1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![P128%2], m![1 # 8]>()
        .collect::<m![P128%2], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![P128/2%32,Dummy8], m![P128%2]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![P128/2%32,Dummy8], m![P128%2,1 # 8]> = device
        .main
        .begin(reduced_mean_square.view())
        .fetch::<m![P128%2], m![1 # 8]>()
        .collect::<m![P128%2], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![P128%2,1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> =
        rms_weight.to_dm(&mut device.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![P128%2,1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![P128%2], m![1 # 8]>()
        .collect::<m![P128%2], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<f32, Chip, Cluster, ReducingSlices, m![P128%2,H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![P128%2,H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%2,H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%2,H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![P128%2,H / 8 % 60], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();

    device
        .main
        .begin(normalized.view())
        .fetch::<m![P128%2,H / 8 % 60], m![H % 8]>()
        .collect::<m![P128%2,H / 8 % 60], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
fn residual_add(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Shards, m![P128%2,H % 480]>,
    previous: &HbmTensor<bf16, Chip, m![P128,H]>,
) -> DmTensor<bf16, Chip, Cluster, Shards, m![P128%2,H % 480]> {
    let previous: DmTensor<bf16, Chip, Cluster, Shards, m![P128%2,H % 480]> =
        previous.to_dm(&mut device.tdma);
    let previous: VrfTensor<f32, Chip, Cluster, Shards, m![P128%2,H % 480]> = device
        .sub
        .begin(previous.view())
        .fetch::<m![P128%2,H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%2,H / 8 % 60], m![H % 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![P128%2,H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%2,H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &previous)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
fn gate(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Shards, m![P128%2,H % 480]>,
    scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, Shards, m![P128%2,H % 480]> {
    let scalar: DmTensor<bf16, Chip, Cluster, Shards, m![1 # 8]> = scalar.to_dm(&mut device.tdma);
    let scalar: VrfTensor<f32, Chip, Cluster, Shards, m![1 # 8]> = device
        .sub
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![P128%2,H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%2,H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%2,H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scalar)
        .vector_widen_concat::<m![P128%2,H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
#[device(chip = 1)]
pub fn rmsnorm_p128(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![P128,H]>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    out: &mut HbmTensor<bf16, Chip, m![P128,H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Shards, m![P128%2,H % 480]> = x.to_dm(&mut device.tdma);
    let x = normalize(device, &x, weight);
    x.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
#[device(chip = 1)]
pub fn post_attention_p128(
    device: &mut Device,
    projected: &HbmTensor<bf16, Chip, m![P128,H]>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, m![P128,H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Shards, m![P128%2,H % 480]> = projected.to_dm(&mut device.tdma);
    let x = normalize(device, &x, weight);
    let x = residual_add(device, &x, residual);
    x.view().to_hbm_view(&mut device.tdma, residual.view_mut());
}
#[device(chip = 1)]
pub fn post_feedforward_p128(
    device: &mut Device,
    projected: &HbmTensor<bf16, Chip, m![P128,H]>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_gate: &HbmTensor<bf16, Chip, m![1 # 8]>,
    residual: &mut HbmTensor<bf16, Chip, m![P128,H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Shards, m![P128%2,H % 480]> = projected.to_dm(&mut device.tdma);
    let x = normalize(device, &x, weight);
    let x = residual_add(device, &x, residual);
    let x = gate(device, &x, layer_gate);
    x.view().to_hbm_view(&mut device.tdma, residual.view_mut());
}
