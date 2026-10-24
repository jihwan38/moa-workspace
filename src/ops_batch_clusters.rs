//! Distribute the two B8 halves across clusters instead of replicating both.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{B,H,Qs}};
type Cluster=m![B / 8];
type QRows=m![Qs/16];
#[device(chip = 1)]
pub fn project_query_clusters(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    scale: &HbmTensor<bf16, Chip, m![Qs]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Qs]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, QRows, m![B % 8, H]> = input.to_dm(&mut device.tdma);
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
    let mut result: DmTensor<bf16, Chip, Cluster, QRows, m![Qs % 16, B % 8]> = DmTensor::new();
    
        let x: TrfTensor<bf16, Chip, Cluster, QRows, m![B % 8], m![H]> = device
            .sub
            .begin(input.view())
            .fetch::<m![B % 8, H / 16], m![H % 16]>()
            .collect::<m![B % 8, H / 16], m![H % 16]>()
            .to_trf();
        let dot: DmTensor<bf16, Chip, Cluster, QRows, m![Qs % 16, B % 8]> = device
            .main
            .begin(weight.view())
            .fetch::<m![Qs % 16, H / 16], m![H % 16]>()
            .collect::<m![Qs % 16, H / 16], m![H % 16]>()
            .contract_outer::<m![Qs % 16, H / 32], m![H % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![Qs % 16]>()
            .contract_lane::<m![Qs % 16], m![B % 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B % 8 # 16]>()
            .commit_trim::<m![B % 8]>()
            .commit();
        device
            .main
            .begin(dot.view())
            .fetch::<m![Qs % 16], m![B % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Qs % 16], m![B % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Qs % 16, B % 8 / 4], m![B % 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![Qs % 16], m![B % 8]>()
            .vector_final()
            .cast::<bf16, m![B % 8 # 16]>()
            .commit_trim::<m![B % 8]>()
            .commit_view(result.view_mut());
    let result: DmTensor<bf16,Chip,Cluster,QRows,m![B % 8,Qs % 16]> = device.main.begin(result.view())
        .fetch::<m![Qs % 16],m![B % 8 # 16]>().collect::<m![Qs % 16],m![B % 8 # 16]>()
        .transpose::<m![Qs / 4 % 4,B % 8],m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

use crate::axes::L;
type Row=m![B % 8];
type UpRows=m![L/60];
type DownColumns=m![H/120,L/1920];
type DownRows=m![H/120,1 # 8];
fn upgate_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, {Row}]> {
    let input: DmTensor<bf16, Chip, Cluster, UpRows, m![{Row}, H]> = input.to_dm(&mut device.tdma);
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
    let mut result: DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, {Row}]> = DmTensor::new();
    {
        let x: TrfTensor<bf16, Chip, Cluster, UpRows, m![{Row} = 8], m![H]> = device
            .sub
            .begin(input.view().tile::<m![{Row}], 8, m![{Row} = 8,H]>(0))
            .fetch::<m![{Row} = 8, H / 16], m![H % 16]>()
            .collect::<m![{Row} = 8, H / 16], m![H % 16]>()
            .to_trf();
        let mut partial: DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, {Row} = 8]> = DmTensor::new();
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
                .contract_lane::<m![L % 60 = 4], m![{Row} = 8]>(LaneMode::Interleaved)
                .cast::<bf16, m![{Row} = 8 # 16]>()
                .commit_trim::<m![{Row} = 8]>()
                .commit_view(
                    partial
                        .view_mut()
                        .tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60,{Row} = 8]>(4 * row),
                );
        }
        partial.view().to_dm_view(
            &mut device.tdma,
            result
                .view_mut()
                .tile::<m![{Row}], 8, m![L % 60,{Row} = 8]>(0),
        );
    }
    result
}

fn scale_upgate_batch(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, {Row}]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, {Row}]> {
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
        .fetch::<m![L % 60, {Row} / 8], m![{Row} % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60, {Row} / 8], m![{Row} % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60, {Row} / 4], m![{Row} % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![L % 60, {Row} / 8], m![{Row} % 8]>()
        .vector_final()
        .cast::<bf16, m![{Row} % 8 # 16]>()
        .commit_trim::<m![{Row} % 8]>()
        .commit()
}

pub(crate) fn geglu_batch(
    device: &mut Device,
    up: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, {Row}]>,
    gate: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, {Row}]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, {Row}]> {
    let up = scale_upgate_batch(device, up, up_global_scale);
    let gate = scale_upgate_batch(device, gate, gate_global_scale);
    let gelu: DmTensor<f32, Chip, Cluster, UpRows, m![L % 60, {Row}]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![L % 60, {Row} / 8], m![{Row} % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60, {Row} / 8], m![{Row} % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60, {Row} / 4], m![{Row} % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.70710678118f32)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L % 60, {Row} / 8], m![{Row} % 8]>()
        .vector_final()
        .commit_trim::<m![{Row} % 8]>()
        .commit();
    let gelu: VrfTensor<f32, Chip, Cluster, UpRows, m![L % 60, {Row}]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![L % 60, {Row} / 8], m![{Row} % 8]>()
        .collect::<m![L % 60, {Row} / 8], m![{Row} % 8]>()
        .to_vrf();
    device
        .main
        .begin(up.view())
        .fetch::<m![L % 60, {Row} / 8], m![{Row} % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60, {Row} / 8], m![{Row} % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60, {Row} / 4], m![{Row} % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L % 60, {Row} / 8], m![{Row} % 8]>()
        .vector_final()
        .cast::<bf16, m![{Row} % 8 # 16]>()
        .commit_trim::<m![{Row} % 8]>()
        .commit()
}

fn down_batch(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, {Row}]>,
    weight: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, {Row}]> {
    let input: DmTensor<bf16, Chip, Cluster, UpRows, m![{Row}, L % 60]> = device
        .main
        .begin(input.view())
        .fetch::<m![L % 60], m![{Row}]>()
        .collect::<m![L % 60], m![{Row} # 16]>()
        .transpose::<m![L / 4 % 15, {Row}], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();
    let input: DmTensor<bf16, Chip, Cluster, DownColumns, m![{Row}, L % 1920]> =
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
    let mut result: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, {Row}]> = DmTensor::new();
    {
        let x: TrfTensor<bf16, Chip, Cluster, DownColumns, m![{Row} = 8], m![L % 1920]> = device
            .sub
            .begin(
                input
                    .view()
                    .tile::<m![{Row}], 8, m![{Row} = 8,L % 1920]>(0),
            )
            .fetch::<m![{Row} = 8, L / 16 % 120], m![L % 16]>()
            .collect::<m![{Row} = 8, L / 16 % 120], m![L % 16]>()
            .to_trf();
        let mut partial: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, {Row} = 8]> =
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
                .contract_lane::<m![H % 120 = 4], m![{Row} = 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<DownRows, m![H % 120 = 4]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![{Row} = 8 # 16]>()
                .commit_trim::<m![{Row} = 8]>()
                .commit_view(
                    partial
                        .view_mut()
                        .tile::<m![H % 120], 4, m![H % 120 = 4 #{!} 120,{Row} = 8]>(4 * row),
                );
        }
        partial.view().to_dm_view(
            &mut device.tdma,
            result
                .view_mut()
                .tile::<m![{Row}], 8, m![H % 120,{Row} = 8]>(0),
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
        .fetch::<m![H % 120, {Row} / 8], m![{Row} % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120, {Row} / 8], m![{Row} % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120, {Row} / 4], m![{Row} % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &global_scale)
        .vector_widen_concat::<m![H % 120, {Row} / 8], m![{Row} % 8]>()
        .vector_final()
        .cast::<bf16, m![{Row} % 8 # 16]>()
        .commit_trim::<m![{Row} % 8]>()
        .commit()
}

/// Complete feedforward core for 16 already-normalized activation rows.
/// Norms, residual addition and layer gating are intentionally caller epilogues.
#[device(chip = 1)]
pub fn batch_ffn_clusters(
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
    let out: DmTensor<bf16,Chip,Cluster,DownRows,m![{Row},H % 120]> = device.main.begin(out.view())
        .fetch::<m![H % 120],m![{Row} # 16]>().collect::<m![H % 120],m![{Row} # 16]>()
        .transpose::<m![H / 4 % 30,{Row}],m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>().commit();
    out.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
