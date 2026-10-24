//! Native E4 projection cores sharing one materialized per-row FP8 input.
//! Original weights/scales and H960 F32 reduction are retained. No fused-DM quantizer.
pub use crate::ops_query_r64_native_fp8::{project_query_quantized, quantize_input};
use crate::{
    Chip,
    axes::{H, R64, Pf, Ps, Qf},
};
use furiosa_opt_std::prelude::*;
type Cluster = m![R64 / 32];

type PsParts = m![Ps / 32, H / 960];
type PsRows = m![Ps/32,1#4];
fn pack_sliding_kv_scales(
    device: &mut Device,
    xscale: &DmTensor<f32, Chip, Cluster, PsRows, m![R64 % 32,1 # 8]>,
) -> DmTensor<f32, Chip, Cluster, PsRows, m![R64 % 32]> {
    // Keep the 32-byte scalar slot for aligned inter-slice DMA, then pack by TU.
    let xscale: DmTensor<f32, Chip, Cluster, PsRows, m![R64 % 32]> = device
        .sub
        .begin(xscale.view())
        .fetch::<m![R64 % 32], m![1 # 8]>()
        .collect::<m![R64 % 32], m![1 # 8]>()
        .transpose::<m![R64 % 32 / 2], m![R64 % 32 % 2 # 8]>()
        .commit_trim::<m![R64 % 32 % 2]>()
        .commit();
    xscale
}
fn matrix_sliding_kv(
    device: &mut Device,
    input: &DmTensor<f8e4m3, Chip, Cluster, PsParts, m![R64 % 32, H % 960]>,
    xscale: &DmTensor<f32, Chip, Cluster, PsRows, m![R64 % 32]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> DmTensor<bf16, Chip, Cluster, PsRows, m![Ps % 32, R64 % 32]> {
    let weight: DmTensor<f8e4m3, Chip, Cluster, PsParts, m![Ps % 32, H % 960]> =
        weight.to_dm(&mut device.tdma);
    let xs: VrfTensor<f32, Chip, Cluster, PsRows, m![R64 % 32]> = device
        .sub
        .begin(xscale.view())
        .fetch::<m![R64 / 8 % 4], m![R64 % 8]>()
        .collect::<m![R64 / 8 % 4], m![R64 % 8]>()
        .to_vrf();
    let x: TrfTensor<f8e4m3, Chip, Cluster, PsParts, m![R64 % 8], m![R64 / 8 % 4, H % 960]> =
        device
            .sub
            .begin(input.view())
            .fetch::<m![R64 % 8, R64 / 8 % 4, H / 32 % 30], m![H % 32]>()
            .collect::<m![R64 % 8, R64 / 8 % 4, H / 32 % 30], m![H % 32]>()
            .to_trf();
    device
        .main
        .begin(weight.view())
        .fetch::<m![Ps % 32, H / 32 % 30], m![H % 32]>()
        .collect::<m![Ps % 32, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![Ps % 32, H / 64 % 15, R64 / 8 % 4], m![H % 64], _, _, _>(&x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 32, R64 / 8 % 4]>()
        .contract_lane::<m![Ps % 32, R64 / 8 % 4], m![R64 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<PsRows, m![Ps % 32, R64 / 8 % 4]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps % 32, R64 / 4 % 8], m![R64 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &xs)
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
    output: &mut HbmTensor<bf16, Chip, m![R64, Ps]>,
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

#[device(chip = 1)]
pub fn project_sliding_kv_quantized(
    device: &mut Device,
    input: &HbmTensor<f8e4m3, Chip, m![R64, H]>,
    xscale: &HbmTensor<f32, Chip, m![R64,1#8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    scale: &HbmTensor<bf16, Chip, m![Ps]>,
    output: &mut HbmTensor<bf16, Chip, m![R64, Ps]>,
) {
    let input: DmTensor<f8e4m3, Chip, Cluster, PsParts, m![R64 % 32, H % 960]> =
        input.to_dm(&mut device.tdma);
    let xscale: DmTensor<f32, Chip, Cluster, PsRows, m![R64 % 32,1#8]> =
        xscale.to_dm(&mut device.tdma);
    let xscale = pack_sliding_kv_scales(device, &xscale);
    let dot = matrix_sliding_kv(device, &input, &xscale, weight);
    finish_sliding_kv(device, &dot, scale, output);
}

type QfParts = m![Qf / 128, H / 960];
type QfRows = m![Qf/128,1#4];
fn pack_full_query_scales(
    device: &mut Device,
    xscale: &DmTensor<f32, Chip, Cluster, QfRows, m![R64 % 32,1 # 8]>,
) -> DmTensor<f32, Chip, Cluster, QfRows, m![R64 % 32]> {
    // Keep the 32-byte scalar slot for aligned inter-slice DMA, then pack by TU.
    let xscale: DmTensor<f32, Chip, Cluster, QfRows, m![R64 % 32]> = device
        .sub
        .begin(xscale.view())
        .fetch::<m![R64 % 32], m![1 # 8]>()
        .collect::<m![R64 % 32], m![1 # 8]>()
        .transpose::<m![R64 % 32 / 2], m![R64 % 32 % 2 # 8]>()
        .commit_trim::<m![R64 % 32 % 2]>()
        .commit();
    xscale
}
fn matrix_full_query(
    device: &mut Device,
    input: &DmTensor<f8e4m3, Chip, Cluster, QfParts, m![R64 % 32, H % 960]>,
    xscale: &DmTensor<f32, Chip, Cluster, QfRows, m![R64 % 32]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qf, H]>,
) -> DmTensor<bf16, Chip, Cluster, QfRows, m![Qf % 128, R64 % 32]> {
    let weight: DmTensor<f8e4m3, Chip, Cluster, QfParts, m![Qf % 128, H % 960]> =
        weight.to_dm(&mut device.tdma);
    let xs: VrfTensor<f32, Chip, Cluster, QfRows, m![R64 % 32]> = device
        .sub
        .begin(xscale.view())
        .fetch::<m![R64 / 8 % 4], m![R64 % 8]>()
        .collect::<m![R64 / 8 % 4], m![R64 % 8]>()
        .to_vrf();
    let x: TrfTensor<f8e4m3, Chip, Cluster, QfParts, m![R64 % 8], m![R64 / 8 % 4, H % 960]> =
        device
            .sub
            .begin(input.view())
            .fetch::<m![R64 % 8, R64 / 8 % 4, H / 32 % 30], m![H % 32]>()
            .collect::<m![R64 % 8, R64 / 8 % 4, H / 32 % 30], m![H % 32]>()
            .to_trf();
    device
        .main
        .begin(weight.view())
        .fetch::<m![Qf % 128, H / 32 % 30], m![H % 32]>()
        .collect::<m![Qf % 128, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![Qf % 128, H / 64 % 15, R64 / 8 % 4], m![H % 64], _, _, _>(&x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qf % 128, R64 / 8 % 4]>()
        .contract_lane::<m![Qf % 128, R64 / 8 % 4], m![R64 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<QfRows, m![Qf % 128, R64 / 8 % 4]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qf % 128, R64 / 4 % 8], m![R64 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &xs)
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
    output: &mut HbmTensor<bf16, Chip, m![R64, Qf]>,
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

#[device(chip = 1)]
pub fn project_full_query_quantized(
    device: &mut Device,
    input: &HbmTensor<f8e4m3, Chip, m![R64, H]>,
    xscale: &HbmTensor<f32, Chip, m![R64,1#8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    scale: &HbmTensor<bf16, Chip, m![Qf]>,
    output: &mut HbmTensor<bf16, Chip, m![R64, Qf]>,
) {
    let input: DmTensor<f8e4m3, Chip, Cluster, QfParts, m![R64 % 32, H % 960]> =
        input.to_dm(&mut device.tdma);
    let xscale: DmTensor<f32, Chip, Cluster, QfRows, m![R64 % 32,1#8]> =
        xscale.to_dm(&mut device.tdma);
    let xscale = pack_full_query_scales(device, &xscale);
    let dot = matrix_full_query(device, &input, &xscale, weight);
    finish_full_query(device, &dot, scale, output);
}

type PfParts = m![Pf / 8, H / 960];
type PfRows = m![Pf/8,1#4];
fn pack_full_key_scales(
    device: &mut Device,
    xscale: &DmTensor<f32, Chip, Cluster, PfRows, m![R64 % 32,1 # 8]>,
) -> DmTensor<f32, Chip, Cluster, PfRows, m![R64 % 32]> {
    // Keep the 32-byte scalar slot for aligned inter-slice DMA, then pack by TU.
    let xscale: DmTensor<f32, Chip, Cluster, PfRows, m![R64 % 32]> = device
        .sub
        .begin(xscale.view())
        .fetch::<m![R64 % 32], m![1 # 8]>()
        .collect::<m![R64 % 32], m![1 # 8]>()
        .transpose::<m![R64 % 32 / 2], m![R64 % 32 % 2 # 8]>()
        .commit_trim::<m![R64 % 32 % 2]>()
        .commit();
    xscale
}
fn matrix_full_key(
    device: &mut Device,
    input: &DmTensor<f8e4m3, Chip, Cluster, PfParts, m![R64 % 32, H % 960]>,
    xscale: &DmTensor<f32, Chip, Cluster, PfRows, m![R64 % 32]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Pf, H]>,
) -> DmTensor<bf16, Chip, Cluster, PfRows, m![Pf % 8, R64 % 32]> {
    let weight: DmTensor<f8e4m3, Chip, Cluster, PfParts, m![Pf % 8, H % 960]> =
        weight.to_dm(&mut device.tdma);
    let xs: VrfTensor<f32, Chip, Cluster, PfRows, m![R64 % 32]> = device
        .sub
        .begin(xscale.view())
        .fetch::<m![R64 / 8 % 4], m![R64 % 8]>()
        .collect::<m![R64 / 8 % 4], m![R64 % 8]>()
        .to_vrf();
    let x: TrfTensor<f8e4m3, Chip, Cluster, PfParts, m![R64 % 8], m![R64 / 8 % 4, H % 960]> =
        device
            .sub
            .begin(input.view())
            .fetch::<m![R64 % 8, R64 / 8 % 4, H / 32 % 30], m![H % 32]>()
            .collect::<m![R64 % 8, R64 / 8 % 4, H / 32 % 30], m![H % 32]>()
            .to_trf();
    device
        .main
        .begin(weight.view())
        .fetch::<m![Pf % 8, H / 32 % 30], m![H % 32]>()
        .collect::<m![Pf % 8, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![Pf % 8, H / 64 % 15, R64 / 8 % 4], m![H % 64], _, _, _>(&x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Pf % 8, R64 / 8 % 4]>()
        .contract_lane::<m![Pf % 8, R64 / 8 % 4], m![R64 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<PfRows, m![Pf % 8, R64 / 8 % 4]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Pf % 8, R64 / 4 % 8], m![R64 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &xs)
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
    output: &mut HbmTensor<bf16, Chip, m![R64, Pf]>,
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

#[device(chip = 1)]
pub fn project_full_key_quantized(
    device: &mut Device,
    input: &HbmTensor<f8e4m3, Chip, m![R64, H]>,
    xscale: &HbmTensor<f32, Chip, m![R64,1#8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    scale: &HbmTensor<bf16, Chip, m![Pf]>,
    output: &mut HbmTensor<bf16, Chip, m![R64, Pf]>,
) {
    let input: DmTensor<f8e4m3, Chip, Cluster, PfParts, m![R64 % 32, H % 960]> =
        input.to_dm(&mut device.tdma);
    let xscale: DmTensor<f32, Chip, Cluster, PfRows, m![R64 % 32,1#8]> =
        xscale.to_dm(&mut device.tdma);
    let xscale = pack_full_key_scales(device, &xscale);
    let dot = matrix_full_key(device, &input, &xscale, weight);
    finish_full_key(device, &dot, scale, output);
}
