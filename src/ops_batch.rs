//! Prompt-row matrix products. All rows use the provided model's actual weights.
//!
//! B16 is executed as two B8 TRF lane tiles: H3840 x 8 x BF16 fits the full
//! TRF, whereas H3840 x 16 does not. The decoded weight tile is shared by both.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{B, H, Pf, Ps, Qf, Qs};

type Cluster = m![1 # 2];
type QRows = m![Qs / 16];

/// Raw sliding-query projection, matching the baseline's BF16 dot boundary
/// followed by per-output-channel scaling. Input must already be RMS-normalized.
#[device(chip = 1)]
pub fn project_query_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    scale: &HbmTensor<bf16, Chip, m![Qs]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Qs]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, QRows, m![B, H]> = input.to_dm(&mut device.tdma);
    let weight: DmTensor<f8e4m3, Chip, Cluster, QRows, m![Qs % 16, H]> =
        weight.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, QRows, m![Qs % 16, H]> = device
        .main
        .begin(weight.view())
        .fetch::<m![Qs % 16, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 16, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    let scale: DmTensor<bf16, Chip, Cluster, QRows, m![Qs % 16]> = scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, QRows, m![Qs % 16]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 2], m![Qs % 8]>()
        .to_vrf();
    let mut result: DmTensor<bf16, Chip, Cluster, QRows, m![Qs % 16, B]> = DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, QRows, m![B = 8], m![H]> = device
            .sub
            .begin(input.view().tile::<m![B], 8, m![B = 8 # 16, H]>(chunk * 8))
            .fetch::<m![B = 8, H / 16], m![H % 16]>()
            .collect::<m![B = 8, H / 16], m![H % 16]>()
            .to_trf();
        let dot: DmTensor<bf16, Chip, Cluster, QRows, m![Qs % 16, B = 8]> = device
            .main
            .begin(weight.view())
            .fetch::<m![Qs % 16, H / 16], m![H % 16]>()
            .collect::<m![Qs % 16, H / 16], m![H % 16]>()
            .contract_outer::<m![Qs % 16, H / 32], m![H % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![Qs % 16]>()
            .contract_lane::<m![Qs % 16], m![B = 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B = 8 # 16]>()
            .commit_trim::<m![B = 8]>()
            .commit();
        device
            .main
            .begin(dot.view())
            .fetch::<m![Qs % 16], m![B = 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Qs % 16], m![B = 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Qs % 16, B = 8 / 4], m![B = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![Qs % 16], m![B = 8]>()
            .vector_final()
            .cast::<bf16, m![B = 8 # 16]>()
            .commit_trim::<m![B = 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![B], 8, m![Qs % 16, B = 8 #{!} 16]>(chunk * 8),
            );
    }
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

use crate::axes::{Df, Ds, Gf, Gs, L, Ns};
type UpRows = m![L / 60];
type DownColumns = m![H / 120, L / 1920];
type DownRows = m![H / 120, 1 # 8];

fn upgate_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, B]> {
    let input: DmTensor<bf16, Chip, Cluster, UpRows, m![B, H]> = input.to_dm(&mut device.tdma);
    let packed: DmTensor<f4e2m1, Chip, Cluster, UpRows, m![L % 60, H]> =
        weight.to_dm(&mut device.tdma);
    let weight: DmTensor<f8e4m3, Chip, Cluster, UpRows, m![L % 60, H]> = device
        .main
        .begin(packed.view())
        .fetch::<m![L % 60], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 60, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let scale: DmTensor<f8e4m3, Chip, Cluster, UpRows, m![L % 60, H / 16]> =
        scale.to_dm(&mut device.tdma);
    let mut result: DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, B]> = DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, UpRows, m![B = 8], m![H]> = device
            .sub
            .begin(input.view().tile::<m![B], 8, m![B = 8 # 16,H]>(8 * chunk))
            .fetch::<m![B = 8, H / 16], m![H % 16]>()
            .collect::<m![B = 8, H / 16], m![H % 16]>()
            .to_trf();
        let mut partial: DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, B = 8]> = DmTensor::new();
        for row in 0..15 {
            let scales: VrfTensor<f32, Chip, Cluster, UpRows, m![L % 60 = 4, H / 16]> = device
                .sub
                .begin(
                    scale
                        .view()
                        .tile::<m![L % 60], 4, m![L % 60 = 4 # 60,H / 16]>(4 * row),
                )
                .fetch::<m![L % 60 = 4], m![H / 16]>()
                .fetch_cast::<f32>()
                .collect::<m![L % 60 = 4, H / 16 / 8], m![H / 16 % 8]>()
                .to_vrf();
            let weights: DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60 = 4, H]> = device
                .main
                .begin(
                    weight
                        .view()
                        .tile::<m![L % 60], 4, m![L % 60 = 4 # 60,H]>(4 * row),
                )
                .fetch::<m![L % 60 = 4, H / 32], m![H % 32]>()
                .fetch_cast::<f32>()
                .collect::<m![L % 60 = 4, H / 8], m![H % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![L % 60 = 4, H / 4], m![H % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scales)
                .vector_widen_concat::<m![L % 60 = 4, H / 8], m![H % 8]>()
                .vector_final()
                .cast::<bf16, m![H % 8 # 16]>()
                .commit_trim::<m![H % 8]>()
                .commit();
            device
                .main
                .begin(weights.view())
                .fetch::<m![L % 60 = 4, H / 16], m![H % 16]>()
                .collect::<m![L % 60 = 4, H / 16], m![H % 16]>()
                .contract_outer::<m![L % 60 = 4, H / 32], m![H % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![L % 60 = 4]>()
                .contract_lane::<m![L % 60 = 4], m![B = 8]>(LaneMode::Interleaved)
                .cast::<bf16, m![B = 8 # 16]>()
                .commit_trim::<m![B = 8]>()
                .commit_view(
                    partial
                        .view_mut()
                        .tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60,B = 8]>(4 * row),
                );
        }
        partial.view().to_dm_view(
            &mut device.tdma,
            result
                .view_mut()
                .tile::<m![B], 8, m![L % 60,B = 8 #{!} 16]>(8 * chunk),
        );
    }
    result
}

fn scale_upgate_batch(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, B]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, B]> {
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
        .fetch::<m![L % 60, B / 8], m![B % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60, B / 8], m![B % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60, B / 4], m![B % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![L % 60, B / 8], m![B % 8]>()
        .vector_final()
        .cast::<bf16, m![B % 8 # 16]>()
        .commit_trim::<m![B % 8]>()
        .commit()
}

fn geglu_batch(
    device: &mut Device,
    up: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, B]>,
    gate: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, B]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, B]> {
    let up = scale_upgate_batch(device, up, up_global_scale);
    let gate = scale_upgate_batch(device, gate, gate_global_scale);
    let gelu: DmTensor<f32, Chip, Cluster, UpRows, m![L % 60, B]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![L % 60, B / 8], m![B % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60, B / 8], m![B % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60, B / 4], m![B % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.70710678118f32)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L % 60, B / 8], m![B % 8]>()
        .vector_final()
        .commit_trim::<m![B % 8]>()
        .commit();
    let gelu: VrfTensor<f32, Chip, Cluster, UpRows, m![L % 60, B]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![L % 60, B / 8], m![B % 8]>()
        .collect::<m![L % 60, B / 8], m![B % 8]>()
        .to_vrf();
    device
        .main
        .begin(up.view())
        .fetch::<m![L % 60, B / 8], m![B % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60, B / 8], m![B % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60, B / 4], m![B % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L % 60, B / 8], m![B % 8]>()
        .vector_final()
        .cast::<bf16, m![B % 8 # 16]>()
        .commit_trim::<m![B % 8]>()
        .commit()
}

fn down_batch(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, B]>,
    weight: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, B]> {
    let input: DmTensor<bf16, Chip, Cluster, UpRows, m![B, L % 60]> = device
        .main
        .begin(input.view())
        .fetch::<m![L % 60], m![B]>()
        .collect::<m![L % 60], m![B]>()
        .transpose::<m![L / 4 % 15, B], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();
    let input: DmTensor<bf16, Chip, Cluster, DownColumns, m![B, L % 1920]> =
        input.to_dm(&mut device.tdma);
    let packed: DmTensor<f4e2m1, Chip, Cluster, DownColumns, m![H % 120, L % 1920]> =
        weight.to_dm(&mut device.tdma);
    let weight: DmTensor<f8e4m3, Chip, Cluster, DownColumns, m![H % 120, L % 1920]> = device
        .main
        .begin(packed.view())
        .fetch::<m![H % 120], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 120, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit();
    let scale: DmTensor<f8e4m3, Chip, Cluster, DownColumns, m![H % 120, L / 16 % 120]> =
        scale.to_dm(&mut device.tdma);
    let mut result: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, B]> = DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, DownColumns, m![B = 8], m![L % 1920]> = device
            .sub
            .begin(
                input
                    .view()
                    .tile::<m![B], 8, m![B = 8 # 16,L % 1920]>(8 * chunk),
            )
            .fetch::<m![B = 8, L / 16 % 120], m![L % 16]>()
            .collect::<m![B = 8, L / 16 % 120], m![L % 16]>()
            .to_trf();
        let mut partial: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, B = 8]> =
            DmTensor::new();
        for row in 0..30 {
            let scales: VrfTensor<f32, Chip, Cluster, DownColumns, m![H % 120 = 4, L / 16 % 120]> =
                device
                    .sub
                    .begin(
                        scale
                            .view()
                            .tile::<m![H % 120], 4, m![H % 120 = 4 # 120,L / 16 % 120]>(4 * row),
                    )
                    .fetch::<m![H % 120 = 4], m![L / 16 % 120]>()
                    .fetch_cast::<f32>()
                    .collect::<m![H % 120 = 4, L / 128 % 15], m![L / 16 % 8]>()
                    .to_vrf();
            let weights: DmTensor<bf16, Chip, Cluster, DownColumns, m![H % 120 = 4, L % 1920]> =
                device
                    .main
                    .begin(
                        weight
                            .view()
                            .tile::<m![H % 120], 4, m![H % 120 = 4 # 120,L % 1920]>(4 * row),
                    )
                    .fetch::<m![H % 120 = 4, L / 32 % 60], m![L % 32]>()
                    .fetch_cast::<f32>()
                    .collect::<m![H % 120 = 4, L / 8 % 240], m![L % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![H % 120 = 4, L / 4 % 480], m![L % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scales)
                    .vector_widen_concat::<m![H % 120 = 4, L / 8 % 240], m![L % 8]>()
                    .vector_final()
                    .cast::<bf16, m![L % 8 # 16]>()
                    .commit_trim::<m![L % 8]>()
                    .commit();
            device
                .main
                .begin(weights.view())
                .fetch::<m![H % 120 = 4, L / 16 % 120], m![L % 16]>()
                .collect::<m![H % 120 = 4, L / 16 % 120], m![L % 16]>()
                .contract_outer::<m![H % 120 = 4, L / 32 % 60], m![L % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![H % 120 = 4]>()
                .contract_lane::<m![H % 120 = 4], m![B = 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<DownRows, m![H % 120 = 4]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![B = 8 # 16]>()
                .commit_trim::<m![B = 8]>()
                .commit_view(
                    partial
                        .view_mut()
                        .tile::<m![H % 120], 4, m![H % 120 = 4 #{!} 120,B = 8]>(4 * row),
                );
        }
        partial.view().to_dm_view(
            &mut device.tdma,
            result
                .view_mut()
                .tile::<m![B], 8, m![H % 120,B = 8 #{!} 16]>(8 * chunk),
        );
    }
    let global_scale: DmTensor<f32, Chip, Cluster, DownRows, m![1 # 8]> =
        global_scale.to_dm(&mut device.tdma);
    let global_scale: VrfTensor<f32, Chip, Cluster, DownRows, m![1 # 8]> = device
        .sub
        .begin(global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    device
        .main
        .begin(result.view())
        .fetch::<m![H % 120, B / 8], m![B % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120, B / 8], m![B % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120, B / 4], m![B % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &global_scale)
        .vector_widen_concat::<m![H % 120, B / 8], m![B % 8]>()
        .vector_final()
        .cast::<bf16, m![B % 8 # 16]>()
        .commit_trim::<m![B % 8]>()
        .commit()
}

/// Complete feedforward core for 16 already-normalized activation rows.
/// Norms, residual addition and layer gating are intentionally caller epilogues.
#[device(chip = 1)]
pub fn batch_ffn(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    up_weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![B, H]>,
) {
    let up = upgate_batch(device, input, up_weight, up_scale);
    let gate = upgate_batch(device, input, gate_weight, gate_scale);
    let hidden = geglu_batch(device, &up, &gate, up_global_scale, gate_global_scale);
    let out = down_batch(device, &hidden, down_weight, down_scale, down_global_scale);
    out.view().to_hbm_view(&mut device.tdma, output.view_mut());
}

/// Raw batched attention projection, matching the baseline's BF16 dot boundary
/// followed by per-output-channel scaling. Input must already be RMS-normalized.
#[device(chip = 1)]
pub fn project_sliding_kv_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    scale: &HbmTensor<bf16, Chip, m![Ps]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Ps]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, m![Ps / 8 # 256], m![B, H]> =
        input.to_dm(&mut device.tdma);
    let weight: DmTensor<f8e4m3, Chip, Cluster, m![Ps / 8 # 256], m![Ps % 8, H]> =
        weight.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, m![Ps / 8 # 256], m![Ps % 8, H]> = device
        .main
        .begin(weight.view())
        .fetch::<m![Ps % 8, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 8, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    let scale: DmTensor<bf16, Chip, Cluster, m![Ps / 8 # 256], m![Ps % 8]> =
        scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, m![Ps / 8 # 256], m![Ps % 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Ps / 8 % 1], m![Ps % 8]>()
        .to_vrf();
    let mut result: DmTensor<bf16, Chip, Cluster, m![Ps / 8 # 256], m![Ps % 8, B]> =
        DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, m![Ps / 8 # 256], m![B = 8], m![H]> = device
            .sub
            .begin(input.view().tile::<m![B], 8, m![B = 8 # 16, H]>(chunk * 8))
            .fetch::<m![B = 8, H / 16], m![H % 16]>()
            .collect::<m![B = 8, H / 16], m![H % 16]>()
            .to_trf();
        let dot: DmTensor<bf16, Chip, Cluster, m![Ps / 8 # 256], m![Ps % 8, B = 8]> = device
            .main
            .begin(weight.view())
            .fetch::<m![Ps % 8, H / 16], m![H % 16]>()
            .collect::<m![Ps % 8, H / 16], m![H % 16]>()
            .contract_outer::<m![Ps % 8, H / 32], m![H % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![Ps % 8]>()
            .contract_lane::<m![Ps % 8], m![B = 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B = 8 # 16]>()
            .commit_trim::<m![B = 8]>()
            .commit();
        device
            .main
            .begin(dot.view())
            .fetch::<m![Ps % 8], m![B = 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Ps % 8], m![B = 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Ps % 8, B = 8 / 4], m![B = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![Ps % 8], m![B = 8]>()
            .vector_final()
            .cast::<bf16, m![B = 8 # 16]>()
            .commit_trim::<m![B = 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![B], 8, m![Ps % 8, B = 8 #{!} 16]>(chunk * 8),
            );
    }
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

/// Raw batched attention projection, matching the baseline's BF16 dot boundary
/// followed by per-output-channel scaling. Input must already be RMS-normalized.
#[device(chip = 1)]
pub fn project_full_query_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    scale: &HbmTensor<bf16, Chip, m![Qf]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Qf]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, m![Qf / 32 # 256], m![B, H]> =
        input.to_dm(&mut device.tdma);
    let weight: DmTensor<f8e4m3, Chip, Cluster, m![Qf / 32 # 256], m![Qf % 32, H]> =
        weight.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, m![Qf / 32 # 256], m![Qf % 32, H]> = device
        .main
        .begin(weight.view())
        .fetch::<m![Qf % 32, H / 32], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qf % 32, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    let scale: DmTensor<bf16, Chip, Cluster, m![Qf / 32 # 256], m![Qf % 32]> =
        scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, m![Qf / 32 # 256], m![Qf % 32]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Qf % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Qf / 8 % 4], m![Qf % 8]>()
        .to_vrf();
    let mut result: DmTensor<bf16, Chip, Cluster, m![Qf / 32 # 256], m![Qf % 32, B]> =
        DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, m![Qf / 32 # 256], m![B = 8], m![H]> = device
            .sub
            .begin(input.view().tile::<m![B], 8, m![B = 8 # 16, H]>(chunk * 8))
            .fetch::<m![B = 8, H / 16], m![H % 16]>()
            .collect::<m![B = 8, H / 16], m![H % 16]>()
            .to_trf();
        let dot: DmTensor<bf16, Chip, Cluster, m![Qf / 32 # 256], m![Qf % 32, B = 8]> = device
            .main
            .begin(weight.view())
            .fetch::<m![Qf % 32, H / 16], m![H % 16]>()
            .collect::<m![Qf % 32, H / 16], m![H % 16]>()
            .contract_outer::<m![Qf % 32, H / 32], m![H % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![Qf % 32]>()
            .contract_lane::<m![Qf % 32], m![B = 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B = 8 # 16]>()
            .commit_trim::<m![B = 8]>()
            .commit();
        device
            .main
            .begin(dot.view())
            .fetch::<m![Qf % 32], m![B = 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Qf % 32], m![B = 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Qf % 32, B = 8 / 4], m![B = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![Qf % 32], m![B = 8]>()
            .vector_final()
            .cast::<bf16, m![B = 8 # 16]>()
            .commit_trim::<m![B = 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![B], 8, m![Qf % 32, B = 8 #{!} 16]>(chunk * 8),
            );
    }
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

/// Raw batched attention projection, matching the baseline's BF16 dot boundary
/// followed by per-output-channel scaling. Input must already be RMS-normalized.
#[device(chip = 1)]
pub fn project_full_key_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    scale: &HbmTensor<bf16, Chip, m![Pf]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Pf]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, m![Pf / 8 # 256], m![B, H]> =
        input.to_dm(&mut device.tdma);
    let weight: DmTensor<f8e4m3, Chip, Cluster, m![Pf / 8 # 256], m![Pf % 8, H]> =
        weight.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, m![Pf / 8 # 256], m![Pf % 8, H]> = device
        .main
        .begin(weight.view())
        .fetch::<m![Pf % 8, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Pf % 8, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    let scale: DmTensor<bf16, Chip, Cluster, m![Pf / 8 # 256], m![Pf % 8]> =
        scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, m![Pf / 8 # 256], m![Pf % 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Pf % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Pf / 8 % 1], m![Pf % 8]>()
        .to_vrf();
    let mut result: DmTensor<bf16, Chip, Cluster, m![Pf / 8 # 256], m![Pf % 8, B]> =
        DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, m![Pf / 8 # 256], m![B = 8], m![H]> = device
            .sub
            .begin(input.view().tile::<m![B], 8, m![B = 8 # 16, H]>(chunk * 8))
            .fetch::<m![B = 8, H / 16], m![H % 16]>()
            .collect::<m![B = 8, H / 16], m![H % 16]>()
            .to_trf();
        let dot: DmTensor<bf16, Chip, Cluster, m![Pf / 8 # 256], m![Pf % 8, B = 8]> = device
            .main
            .begin(weight.view())
            .fetch::<m![Pf % 8, H / 16], m![H % 16]>()
            .collect::<m![Pf % 8, H / 16], m![H % 16]>()
            .contract_outer::<m![Pf % 8, H / 32], m![H % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![Pf % 8]>()
            .contract_lane::<m![Pf % 8], m![B = 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B = 8 # 16]>()
            .commit_trim::<m![B = 8]>()
            .commit();
        device
            .main
            .begin(dot.view())
            .fetch::<m![Pf % 8], m![B = 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Pf % 8], m![B = 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Pf % 8, B = 8 / 4], m![B = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![Pf % 8], m![B = 8]>()
            .vector_final()
            .cast::<bf16, m![B = 8 # 16]>()
            .commit_trim::<m![B = 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![B], 8, m![Pf % 8, B = 8 #{!} 16]>(chunk * 8),
            );
    }
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

/// Test-only callable control: unmodified official serial feedforward core.
#[device(chip = 1)]
pub fn serial_ffn_control(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![H]>,
    up_weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, crate::device::layout::Replicated, m![H]> =
        input.to_dm(&mut device.tdma);
    let out = crate::device::shared::mlp::feedforward(
        device,
        input,
        up_weight,
        gate_weight,
        down_weight,
        up_scale,
        gate_scale,
        down_scale,
        up_global_scale,
        gate_global_scale,
        down_global_scale,
    );
    out.view().to_hbm_view(&mut device.tdma, output.view_mut());
}

fn add_output_partials_batch(
    device: &mut Device,
    left: &DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, B]>,
    right: &DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, B]>,
) -> DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, B]> {
    let left: VrfTensor<f32, Chip, Cluster, DownRows, m![H % 120, B]> = device
        .sub
        .begin(left.view())
        .fetch::<m![H % 120, B / 8], m![B % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120, B / 8], m![B % 8]>()
        .to_vrf();
    device
        .main
        .begin(right.view())
        .fetch::<m![H % 120, B / 8], m![B % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120, B / 8], m![B % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &left)
        .vector_final()
        .cast::<bf16, m![B % 8 # 16]>()
        .commit_trim::<m![B % 8]>()
        .commit()
}

fn scale_output_batch(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, B]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    output: &mut HbmTensor<bf16, Chip, m![B, H]>,
) {
    let scale: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120]> = scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, DownRows, m![H % 120]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();
    let scaled: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, B]> = device
        .main
        .begin(input.view())
        .fetch::<m![H % 120, B / 8], m![B % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120, B / 8], m![B % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120, B / 4], m![B % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![H % 120, B / 8], m![B % 8]>()
        .vector_final()
        .cast::<bf16, m![B % 8 # 16]>()
        .commit_trim::<m![B % 8]>()
        .commit();
    let scaled: DmTensor<bf16, Chip, Cluster, DownRows, m![B, H % 120]> = device
        .main
        .begin(scaled.view())
        .fetch::<m![H % 120], m![B]>()
        .collect::<m![H % 120], m![B]>()
        .transpose::<m![H / 4 % 30, B], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();
    scaled
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

fn qs_output_partial_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    offset: usize,
) -> DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, B]> {
    let input: HbmTensorView<'_, bf16, Chip, m![B, Qs]> = unsafe { input.view().reshape() };
    let input: DmTensor<bf16, Chip, Cluster, DownRows, m![B, Qs = 1024]> = input
        .tile::<m![Qs], 1024, m![B,Qs = 1024 # 4096]>(offset)
        .to_dm(&mut device.tdma);
    let packed: DmTensor<f8e4m3, Chip, Cluster, DownRows, m![H % 120, Qs = 1024]> = weight
        .view()
        .tile::<m![Qs], 1024, m![H,Qs = 1024 # 4096]>(offset)
        .to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, Qs = 1024]> = device
        .main
        .begin(packed.view())
        .fetch::<m![H % 120, Qs = 1024 / 32], m![Qs = 1024 % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 120, Qs = 1024 / 16], m![Qs = 1024 % 16]>()
        .commit_trim::<m![Qs = 1024 % 16]>()
        .commit();
    let mut result: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, B]> = DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, DownRows, m![B = 8], m![Qs = 1024]> = device
            .sub
            .begin(
                input
                    .view()
                    .tile::<m![B], 8, m![B = 8 # 16,Qs = 1024]>(chunk * 8),
            )
            .fetch::<m![B = 8, Qs = 1024 / 16], m![Qs = 1024 % 16]>()
            .collect::<m![B = 8, Qs = 1024 / 16], m![Qs = 1024 % 16]>()
            .to_trf();
        device
            .main
            .begin(weight.view())
            .fetch::<m![H % 120, Qs = 1024 / 16], m![Qs = 1024 % 16]>()
            .collect::<m![H % 120, Qs = 1024 / 16], m![Qs = 1024 % 16]>()
            .contract_outer::<m![H % 120, Qs = 1024 / 32], m![Qs = 1024 % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![H % 120]>()
            .contract_lane::<m![H % 120], m![B = 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B = 8 # 16]>()
            .commit_trim::<m![B = 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![B], 8, m![H % 120,B = 8 #{!} 16]>(8 * chunk),
            );
    }
    result
}

/// Batched attention output projection with the baseline1024-column BF16 sum tree.
#[device(chip = 1)]
pub fn project_sliding_output_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    output: &mut HbmTensor<bf16, Chip, m![B, H]>,
) {
    let p0 = qs_output_partial_batch(device, input, weight, 0);
    let p1 = qs_output_partial_batch(device, input, weight, 1024);
    let p2 = qs_output_partial_batch(device, input, weight, 2048);
    let p3 = qs_output_partial_batch(device, input, weight, 3072);
    let p0p1 = add_output_partials_batch(device, &p0, &p1);
    let p2p3 = add_output_partials_batch(device, &p2, &p3);
    let p0p1p2p3 = add_output_partials_batch(device, &p0p1, &p2p3);
    scale_output_batch(device, &p0p1p2p3, scale, output);
}

fn qf_output_partial_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    offset: usize,
) -> DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, B]> {
    let input: HbmTensorView<'_, bf16, Chip, m![B, Qf]> = unsafe { input.view().reshape() };
    let input: DmTensor<bf16, Chip, Cluster, DownRows, m![B, Qf = 1024]> = input
        .tile::<m![Qf], 1024, m![B,Qf = 1024 # 8192]>(offset)
        .to_dm(&mut device.tdma);
    let packed: DmTensor<f8e4m3, Chip, Cluster, DownRows, m![H % 120, Qf = 1024]> = weight
        .view()
        .tile::<m![Qf], 1024, m![H,Qf = 1024 # 8192]>(offset)
        .to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, Qf = 1024]> = device
        .main
        .begin(packed.view())
        .fetch::<m![H % 120, Qf = 1024 / 32], m![Qf = 1024 % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 120, Qf = 1024 / 16], m![Qf = 1024 % 16]>()
        .commit_trim::<m![Qf = 1024 % 16]>()
        .commit();
    let mut result: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, B]> = DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, DownRows, m![B = 8], m![Qf = 1024]> = device
            .sub
            .begin(
                input
                    .view()
                    .tile::<m![B], 8, m![B = 8 # 16,Qf = 1024]>(chunk * 8),
            )
            .fetch::<m![B = 8, Qf = 1024 / 16], m![Qf = 1024 % 16]>()
            .collect::<m![B = 8, Qf = 1024 / 16], m![Qf = 1024 % 16]>()
            .to_trf();
        device
            .main
            .begin(weight.view())
            .fetch::<m![H % 120, Qf = 1024 / 16], m![Qf = 1024 % 16]>()
            .collect::<m![H % 120, Qf = 1024 / 16], m![Qf = 1024 % 16]>()
            .contract_outer::<m![H % 120, Qf = 1024 / 32], m![Qf = 1024 % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![H % 120]>()
            .contract_lane::<m![H % 120], m![B = 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B = 8 # 16]>()
            .commit_trim::<m![B = 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![B], 8, m![H % 120,B = 8 #{!} 16]>(8 * chunk),
            );
    }
    result
}

/// Batched attention output projection with the baseline1024-column BF16 sum tree.
#[device(chip = 1)]
pub fn project_full_output_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    output: &mut HbmTensor<bf16, Chip, m![B, H]>,
) {
    let p0 = qf_output_partial_batch(device, input, weight, 0);
    let p1 = qf_output_partial_batch(device, input, weight, 1024);
    let p2 = qf_output_partial_batch(device, input, weight, 2048);
    let p3 = qf_output_partial_batch(device, input, weight, 3072);
    let p4 = qf_output_partial_batch(device, input, weight, 4096);
    let p5 = qf_output_partial_batch(device, input, weight, 5120);
    let p6 = qf_output_partial_batch(device, input, weight, 6144);
    let p7 = qf_output_partial_batch(device, input, weight, 7168);
    let p0p1 = add_output_partials_batch(device, &p0, &p1);
    let p2p3 = add_output_partials_batch(device, &p2, &p3);
    let p4p5 = add_output_partials_batch(device, &p4, &p5);
    let p6p7 = add_output_partials_batch(device, &p6, &p7);
    let p0p1p2p3 = add_output_partials_batch(device, &p0p1, &p2p3);
    let p4p5p6p7 = add_output_partials_batch(device, &p4p5, &p6p7);
    let p0p1p2p3p4p5p6p7 = add_output_partials_batch(device, &p0p1p2p3, &p4p5p6p7);
    scale_output_batch(device, &p0p1p2p3p4p5p6p7, scale, output);
}
