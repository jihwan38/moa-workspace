//! Wide prompt linear kernels. P128 is split into64 rows per cluster; every
//! weight shard is reused across eight B8 activation tiles in each cluster.
use crate::{
    Chip,
    axes::{Df, Ds, Gf, Gs, H, L, Ns, P128, Pf, Ps, Qf, Qs},
};
use furiosa_opt_std::prelude::*;
type Cluster = m![P128 / 64];
type UpRows = m![L / 60];
type DownColumns = m![H / 120, L / 1920];
type DownRows = m![H / 120,1 # 8];

#[device(chip = 1)]
pub fn project_query_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P128, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    scale: &HbmTensor<bf16, Chip, m![Qs]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, Qs]>,
) {
    let weight: DmTensor<f8e4m3, Chip, Cluster, m![Qs / 16 # 256], m![Qs % 16, H]> =
        weight.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, m![Qs / 16 # 256], m![Qs % 16, H]> = device
        .main
        .begin(weight.view())
        .fetch::<m![Qs % 16, H / 32], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 16, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let scale: DmTensor<bf16, Chip, Cluster, m![Qs / 16 # 256], m![Qs % 16]> =
        scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, m![Qs / 16 # 256], m![Qs % 16]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 2], m![Qs % 8]>()
        .to_vrf();
    let mut result: DmTensor<bf16, Chip, Cluster, m![Qs / 16 # 256], m![Qs % 16, P128 % 64]> =
        DmTensor::new();
    for chunk in 0..8 {
        let input: DmTensor<bf16, Chip, Cluster, m![Qs / 16 # 256], m![P128 % 64 = 8, H]> = input
            .view()
            .tile::<m![P128 % 64], 8, m![P128 / 64,P128 % 64 = 8 # 64,H]>(8 * chunk)
            .to_dm(&mut device.tdma);
        let x: TrfTensor<bf16, Chip, Cluster, m![Qs / 16 # 256], m![P128 % 64 = 8], m![H]> = device
            .sub
            .begin(input.view())
            .fetch::<m![P128 % 64 = 8, H / 16], m![H % 16]>()
            .collect::<m![P128 % 64 = 8, H / 16], m![H % 16]>()
            .to_trf();
        let dot: DmTensor<bf16, Chip, Cluster, m![Qs / 16 # 256], m![Qs % 16, P128 % 64 = 8]> =
            device
                .main
                .begin(weight.view())
                .fetch::<m![Qs % 16, H / 16], m![H % 16]>()
                .collect::<m![Qs % 16, H / 16], m![H % 16]>()
                .contract_outer::<m![Qs % 16, H / 32], m![H % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![Qs % 16]>()
                .contract_lane::<m![Qs % 16], m![P128 % 64 = 8]>(LaneMode::Interleaved)
                .cast::<bf16, m![P128 % 64 = 8 # 16]>()
                .commit_trim::<m![P128 % 64 = 8]>()
                .commit();
        device
            .main
            .begin(dot.view())
            .fetch::<m![Qs % 16], m![P128 % 64 = 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Qs % 16], m![P128 % 64 = 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Qs % 16, P128 % 64 = 8 / 4], m![P128 % 64 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![Qs % 16], m![P128 % 64 = 8]>()
            .vector_final()
            .cast::<bf16, m![P128 % 64 = 8 # 16]>()
            .commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![P128 % 64], 8, m![Qs % 16,P128 % 64 = 8 #{!} 64]>(8 * chunk),
            );
    }
    let result: DmTensor<bf16, Chip, Cluster, m![Qs / 16 # 256], m![P128 % 64, Qs % 16]> = device
        .main
        .begin(result.view())
        .fetch::<m![P128 / 16 % 4, Qs % 16], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, Qs % 16], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, Qs / 4 % 4, P128 % 16], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn project_sliding_kv_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P128, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    scale: &HbmTensor<bf16, Chip, m![Ps]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, Ps]>,
) {
    let weight: DmTensor<f8e4m3, Chip, Cluster, m![Ps / 8 # 256], m![Ps % 8, H]> =
        weight.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, m![Ps / 8 # 256], m![Ps % 8, H]> = device
        .main
        .begin(weight.view())
        .fetch::<m![Ps % 8, H / 32], m![H % 32]>()
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
    let mut result: DmTensor<bf16, Chip, Cluster, m![Ps / 8 # 256], m![Ps % 8, P128 % 64]> =
        DmTensor::new();
    for chunk in 0..8 {
        let input: DmTensor<bf16, Chip, Cluster, m![Ps / 8 # 256], m![P128 % 64 = 8, H]> = input
            .view()
            .tile::<m![P128 % 64], 8, m![P128 / 64,P128 % 64 = 8 # 64,H]>(8 * chunk)
            .to_dm(&mut device.tdma);
        let x: TrfTensor<bf16, Chip, Cluster, m![Ps / 8 # 256], m![P128 % 64 = 8], m![H]> = device
            .sub
            .begin(input.view())
            .fetch::<m![P128 % 64 = 8, H / 16], m![H % 16]>()
            .collect::<m![P128 % 64 = 8, H / 16], m![H % 16]>()
            .to_trf();
        let dot: DmTensor<bf16, Chip, Cluster, m![Ps / 8 # 256], m![Ps % 8, P128 % 64 = 8]> =
            device
                .main
                .begin(weight.view())
                .fetch::<m![Ps % 8, H / 16], m![H % 16]>()
                .collect::<m![Ps % 8, H / 16], m![H % 16]>()
                .contract_outer::<m![Ps % 8, H / 32], m![H % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![Ps % 8]>()
                .contract_lane::<m![Ps % 8], m![P128 % 64 = 8]>(LaneMode::Interleaved)
                .cast::<bf16, m![P128 % 64 = 8 # 16]>()
                .commit_trim::<m![P128 % 64 = 8]>()
                .commit();
        device
            .main
            .begin(dot.view())
            .fetch::<m![Ps % 8], m![P128 % 64 = 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Ps % 8], m![P128 % 64 = 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Ps % 8, P128 % 64 = 8 / 4], m![P128 % 64 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![Ps % 8], m![P128 % 64 = 8]>()
            .vector_final()
            .cast::<bf16, m![P128 % 64 = 8 # 16]>()
            .commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![P128 % 64], 8, m![Ps % 8,P128 % 64 = 8 #{!} 64]>(8 * chunk),
            );
    }
    let result: DmTensor<bf16, Chip, Cluster, m![Ps / 8 # 256], m![P128 % 64, Ps % 8]> = device
        .main
        .begin(result.view())
        .fetch::<m![P128 / 16 % 4, Ps % 8], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, Ps % 8], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, Ps / 4 % 2, P128 % 16], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn project_full_query_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P128, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    scale: &HbmTensor<bf16, Chip, m![Qf]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, Qf]>,
) {
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
    let mut result: DmTensor<bf16, Chip, Cluster, m![Qf / 32 # 256], m![Qf % 32, P128 % 64]> =
        DmTensor::new();
    for chunk in 0..8 {
        let input: DmTensor<bf16, Chip, Cluster, m![Qf / 32 # 256], m![P128 % 64 = 8, H]> = input
            .view()
            .tile::<m![P128 % 64], 8, m![P128 / 64,P128 % 64 = 8 # 64,H]>(8 * chunk)
            .to_dm(&mut device.tdma);
        let x: TrfTensor<bf16, Chip, Cluster, m![Qf / 32 # 256], m![P128 % 64 = 8], m![H]> = device
            .sub
            .begin(input.view())
            .fetch::<m![P128 % 64 = 8, H / 16], m![H % 16]>()
            .collect::<m![P128 % 64 = 8, H / 16], m![H % 16]>()
            .to_trf();
        let dot: DmTensor<bf16, Chip, Cluster, m![Qf / 32 # 256], m![Qf % 32, P128 % 64 = 8]> =
            device
                .main
                .begin(weight.view())
                .fetch::<m![Qf % 32, H / 16], m![H % 16]>()
                .collect::<m![Qf % 32, H / 16], m![H % 16]>()
                .contract_outer::<m![Qf % 32, H / 32], m![H % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![Qf % 32]>()
                .contract_lane::<m![Qf % 32], m![P128 % 64 = 8]>(LaneMode::Interleaved)
                .cast::<bf16, m![P128 % 64 = 8 # 16]>()
                .commit_trim::<m![P128 % 64 = 8]>()
                .commit();
        device
            .main
            .begin(dot.view())
            .fetch::<m![Qf % 32], m![P128 % 64 = 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Qf % 32], m![P128 % 64 = 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Qf % 32, P128 % 64 = 8 / 4], m![P128 % 64 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![Qf % 32], m![P128 % 64 = 8]>()
            .vector_final()
            .cast::<bf16, m![P128 % 64 = 8 # 16]>()
            .commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![P128 % 64], 8, m![Qf % 32,P128 % 64 = 8 #{!} 64]>(8 * chunk),
            );
    }
    let result: DmTensor<bf16, Chip, Cluster, m![Qf / 32 # 256], m![P128 % 64, Qf % 32]> = device
        .main
        .begin(result.view())
        .fetch::<m![P128 / 16 % 4, Qf % 32], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, Qf % 32], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, Qf / 4 % 8, P128 % 16], m![Qf % 4 # 16]>()
        .commit_trim::<m![Qf % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn project_full_key_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P128, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    scale: &HbmTensor<bf16, Chip, m![Pf]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, Pf]>,
) {
    let weight: DmTensor<f8e4m3, Chip, Cluster, m![Pf / 8 # 256], m![Pf % 8, H]> =
        weight.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, m![Pf / 8 # 256], m![Pf % 8, H]> = device
        .main
        .begin(weight.view())
        .fetch::<m![Pf % 8, H / 32], m![H % 32]>()
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
    let mut result: DmTensor<bf16, Chip, Cluster, m![Pf / 8 # 256], m![Pf % 8, P128 % 64]> =
        DmTensor::new();
    for chunk in 0..8 {
        let input: DmTensor<bf16, Chip, Cluster, m![Pf / 8 # 256], m![P128 % 64 = 8, H]> = input
            .view()
            .tile::<m![P128 % 64], 8, m![P128 / 64,P128 % 64 = 8 # 64,H]>(8 * chunk)
            .to_dm(&mut device.tdma);
        let x: TrfTensor<bf16, Chip, Cluster, m![Pf / 8 # 256], m![P128 % 64 = 8], m![H]> = device
            .sub
            .begin(input.view())
            .fetch::<m![P128 % 64 = 8, H / 16], m![H % 16]>()
            .collect::<m![P128 % 64 = 8, H / 16], m![H % 16]>()
            .to_trf();
        let dot: DmTensor<bf16, Chip, Cluster, m![Pf / 8 # 256], m![Pf % 8, P128 % 64 = 8]> =
            device
                .main
                .begin(weight.view())
                .fetch::<m![Pf % 8, H / 16], m![H % 16]>()
                .collect::<m![Pf % 8, H / 16], m![H % 16]>()
                .contract_outer::<m![Pf % 8, H / 32], m![H % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![Pf % 8]>()
                .contract_lane::<m![Pf % 8], m![P128 % 64 = 8]>(LaneMode::Interleaved)
                .cast::<bf16, m![P128 % 64 = 8 # 16]>()
                .commit_trim::<m![P128 % 64 = 8]>()
                .commit();
        device
            .main
            .begin(dot.view())
            .fetch::<m![Pf % 8], m![P128 % 64 = 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Pf % 8], m![P128 % 64 = 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Pf % 8, P128 % 64 = 8 / 4], m![P128 % 64 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![Pf % 8], m![P128 % 64 = 8]>()
            .vector_final()
            .cast::<bf16, m![P128 % 64 = 8 # 16]>()
            .commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![P128 % 64], 8, m![Pf % 8,P128 % 64 = 8 #{!} 64]>(8 * chunk),
            );
    }
    let result: DmTensor<bf16, Chip, Cluster, m![Pf / 8 # 256], m![P128 % 64, Pf % 8]> = device
        .main
        .begin(result.view())
        .fetch::<m![P128 / 16 % 4, Pf % 8], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, Pf % 8], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, Pf / 4 % 2, P128 % 16], m![Pf % 4 # 16]>()
        .commit_trim::<m![Pf % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

fn upgate_wide(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P128, H]>,
    packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64]> {
    let packed: DmTensor<f4e2m1, Chip, Cluster, UpRows, m![L % 60, H]> =
        packed.to_dm(&mut device.tdma);
    let packed: DmTensor<f8e4m3, Chip, Cluster, UpRows, m![L % 60, H]> = device
        .main
        .begin(packed.view())
        .fetch::<m![L % 60], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 60, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let scale: DmTensor<f8e4m3, Chip, Cluster, UpRows, m![L % 60, H / 16]> =
        scale.to_dm(&mut device.tdma);
    let mut result: DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64]> = DmTensor::new();
    for pass in 0..5 {
        let mut weights: DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60 = 12, H]> =
            DmTensor::new();
        for part in 0..3 {
            let scales: VrfTensor<f32, Chip, Cluster, UpRows, m![L % 60 = 4, H / 16]> = device
                .sub
                .begin(
                    scale
                        .view()
                        .tile::<m![L % 60], 4, m![L % 60 = 4 # 60,H / 16]>(12 * pass + 4 * part),
                )
                .fetch::<m![L % 60 = 4], m![H / 16]>()
                .fetch_cast::<f32>()
                .collect::<m![L % 60 = 4, H / 128], m![H / 16 % 8]>()
                .to_vrf();
            device
                .main
                .begin(
                    packed
                        .view()
                        .tile::<m![L % 60], 4, m![L % 60 = 4 # 60,H]>(12 * pass + 4 * part),
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
                .commit_view(
                    weights
                        .view_mut()
                        .tile::<m![L % 60 = 12], 4, m![L % 60 = 12 = 4 #{!} 12,H]>(4 * part),
                );
        }
        let mut partial: DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60 = 12, P128 % 64]> =
            DmTensor::new();
        for chunk in 0..8 {
            let input: DmTensor<bf16, Chip, Cluster, UpRows, m![P128 % 64 = 8, H]> = input
                .view()
                .tile::<m![P128 % 64], 8, m![P128 / 64,P128 % 64 = 8 # 64,H]>(8 * chunk)
                .to_dm(&mut device.tdma);
            let x: TrfTensor<bf16, Chip, Cluster, UpRows, m![P128 % 64 = 8], m![H]> = device
                .sub
                .begin(input.view())
                .fetch::<m![P128 % 64 = 8, H / 16], m![H % 16]>()
                .collect::<m![P128 % 64 = 8, H / 16], m![H % 16]>()
                .to_trf();
            device
                .main
                .begin(weights.view())
                .fetch::<m![L % 60 = 12, H / 16], m![H % 16]>()
                .collect::<m![L % 60 = 12, H / 16], m![H % 16]>()
                .contract_outer::<m![L % 60 = 12, H / 32], m![H % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![L % 60 = 12]>()
                .contract_lane::<m![L % 60 = 12], m![P128 % 64 = 8]>(LaneMode::Interleaved)
                .cast::<bf16, m![P128 % 64 = 8 # 16]>()
                .commit_trim::<m![P128 % 64 = 8]>()
                .commit_view(
                    partial
                        .view_mut()
                        .tile::<m![P128 % 64], 8, m![L % 60 = 12,P128 % 64 = 8 #{!} 64]>(8 * chunk),
                );
        }
        partial.view().to_dm_view(
            &mut device.tdma,
            result
                .view_mut()
                .tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60,P128 % 64]>(12 * pass),
        );
    }
    result
}

fn scale_upgate_tile(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64 = 16]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64 = 16]> {
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
        .fetch::<m![L % 60, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60, P128 % 64 = 16 / 4], m![P128 % 64 = 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![L % 60, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128 % 64 = 16 % 8 # 16]>()
        .commit_trim::<m![P128 % 64 = 16 % 8]>()
        .commit()
}
fn geglu_tile(
    device: &mut Device,
    up: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64 = 16]>,
    gate: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64 = 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64 = 16]> {
    let up = scale_upgate_tile(device, up, up_global_scale);
    let gate = scale_upgate_tile(device, gate, gate_global_scale);
    let gelu: DmTensor<f32, Chip, Cluster, UpRows, m![L % 60, P128 % 64 = 16]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![L % 60, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60, P128 % 64 = 16 / 4], m![P128 % 64 = 16 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.70710678118f32)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L % 60, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
        .vector_final()
        .commit_trim::<m![P128 % 64 = 16 % 8]>()
        .commit();
    let gelu: VrfTensor<f32, Chip, Cluster, UpRows, m![L % 60, P128 % 64 = 16]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![L % 60, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
        .collect::<m![L % 60, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
        .to_vrf();
    device
        .main
        .begin(up.view())
        .fetch::<m![L % 60, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60, P128 % 64 = 16 / 4], m![P128 % 64 = 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L % 60, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128 % 64 = 16 % 8 # 16]>()
        .commit_trim::<m![P128 % 64 = 16 % 8]>()
        .commit()
}

fn geglu_wide(
    device: &mut Device,
    up: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64]>,
    gate: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64]>,
    us: &HbmTensor<f32, Chip, m![1]>,
    gs: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64]> {
    let mut out: DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64]> = DmTensor::new();
    for chunk in 0..4 {
        let mut u: DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64 = 16]> =
            DmTensor::new();
        up.view()
            .tile::<m![P128 % 64], 16, m![L % 60,P128 % 64 = 16 # 64]>(16 * chunk)
            .to_dm_view(&mut device.tdma, u.view_mut());
        let mut g: DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64 = 16]> =
            DmTensor::new();
        gate.view()
            .tile::<m![P128 % 64], 16, m![L % 60,P128 % 64 = 16 # 64]>(16 * chunk)
            .to_dm_view(&mut device.tdma, g.view_mut());
        let y = geglu_tile(device, &u, &g, us, gs);
        y.view().to_dm_view(
            &mut device.tdma,
            out.view_mut()
                .tile::<m![P128 % 64], 16, m![L % 60,P128 % 64 = 16 #{!} 64]>(16 * chunk),
        );
    }
    out
}

fn down_wide(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64]>,
    packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    global: &HbmTensor<f32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![P128, H]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, UpRows, m![P128 % 64, L % 60]> = device
        .main
        .begin(input.view())
        .fetch::<m![P128 / 16 % 4, L % 60], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, L % 60], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, L / 4 % 15, P128 % 16], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();
    let packed: DmTensor<f4e2m1, Chip, Cluster, DownColumns, m![H % 120, L % 1920]> =
        packed.to_dm(&mut device.tdma);
    let packed: DmTensor<f8e4m3, Chip, Cluster, DownColumns, m![H % 120, L % 1920]> = device
        .main
        .begin(packed.view())
        .fetch::<m![H % 120], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 120, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit();
    let scale: DmTensor<f8e4m3, Chip, Cluster, DownColumns, m![H % 120, L / 16 % 120]> =
        scale.to_dm(&mut device.tdma);
    let mut result: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, P128 % 64]> =
        DmTensor::new();
    for pass in 0..5 {
        let mut weights: DmTensor<bf16, Chip, Cluster, DownColumns, m![H % 120 = 24, L % 1920]> =
            DmTensor::new();
        for part in 0..6 {
            let scales: VrfTensor<f32, Chip, Cluster, DownColumns, m![H % 120 = 4, L / 16 % 120]> =
                device
                    .sub
                    .begin(
                        scale
                            .view()
                            .tile::<m![H % 120], 4, m![H % 120 = 4 # 120,L / 16 % 120]>(
                                24 * pass + 4 * part,
                            ),
                    )
                    .fetch::<m![H % 120 = 4], m![L / 16 % 120]>()
                    .fetch_cast::<f32>()
                    .collect::<m![H % 120 = 4, L / 128 % 15], m![L / 16 % 8]>()
                    .to_vrf();
            device
                .main
                .begin(
                    packed
                        .view()
                        .tile::<m![H % 120], 4, m![H % 120 = 4 # 120,L % 1920]>(
                            24 * pass + 4 * part,
                        ),
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
                .commit_view(
                    weights
                        .view_mut()
                        .tile::<m![H % 120 = 24], 4, m![H % 120 = 24 = 4 #{!} 24,L % 1920]>(
                            4 * part,
                        ),
                );
        }
        let mut partial: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120 = 24, P128 % 64]> =
            DmTensor::new();
        for chunk in 0..8 {
            let mut x: DmTensor<bf16, Chip, Cluster, DownColumns, m![P128 % 64 = 8, L % 1920]> =
                DmTensor::new();
            input
                .view()
                .tile::<m![P128 % 64], 8, m![P128 % 64 = 8 # 64,L % 60]>(8 * chunk)
                .to_dm_view(&mut device.tdma, x.view_mut());
            let x: TrfTensor<bf16, Chip, Cluster, DownColumns, m![P128 % 64 = 8], m![L % 1920]> =
                device
                    .sub
                    .begin(x.view())
                    .fetch::<m![P128 % 64 = 8, L / 16 % 120], m![L % 16]>()
                    .collect::<m![P128 % 64 = 8, L / 16 % 120], m![L % 16]>()
                    .to_trf();
            device
                .main
                .begin(weights.view())
                .fetch::<m![H % 120 = 24, L / 16 % 120], m![L % 16]>()
                .collect::<m![H % 120 = 24, L / 16 % 120], m![L % 16]>()
                .contract_outer::<m![H % 120 = 24, L / 32 % 60], m![L % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![H % 120 = 24]>()
                .contract_lane::<m![H % 120 = 24], m![P128 % 64 = 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<DownRows, m![H % 120 = 24]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![P128 % 64 = 8 # 16]>()
                .commit_trim::<m![P128 % 64 = 8]>()
                .commit_view(
                    partial
                        .view_mut()
                        .tile::<m![P128 % 64], 8, m![H % 120 = 24,P128 % 64 = 8 #{!} 64]>(
                            8 * chunk,
                        ),
                );
        }
        partial.view().to_dm_view(
            &mut device.tdma,
            result
                .view_mut()
                .tile::<m![H % 120], 24, m![H % 120 = 24 #{!} 120,P128 % 64]>(24 * pass),
        );
    }
    let global: DmTensor<f32, Chip, Cluster, DownRows, m![1 # 8]> = global.to_dm(&mut device.tdma);
    let global: VrfTensor<f32, Chip, Cluster, DownRows, m![1 # 8]> = device
        .sub
        .begin(global.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, P128 % 64]> = device
        .main
        .begin(result.view())
        .fetch::<m![H % 120, P128 / 8 % 8], m![P128 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120, P128 / 8 % 8], m![P128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &global)
        .vector_widen_concat::<m![H % 120, P128 / 8 % 8], m![P128 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128 % 8 # 16]>()
        .commit_trim::<m![P128 % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, DownRows, m![P128 % 64, H % 120]> = device
        .main
        .begin(result.view())
        .fetch::<m![P128 / 16 % 4, H % 120], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, H % 120], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, H / 4 % 30, P128 % 16], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn batch_ffn(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P128, H]>,
    up: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    us: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gs: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    ds: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    ug: &HbmTensor<f32, Chip, m![1]>,
    gg: &HbmTensor<f32, Chip, m![1]>,
    dg: &HbmTensor<f32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, H]>,
) {
    let up = upgate_wide(device, input, up, us);
    let gate = upgate_wide(device, input, gate, gs);
    let hidden = geglu_wide(device, &up, &gate, ug, gg);
    down_wide(device, &hidden, down, ds, dg, output);
}

fn add_partials_wide(device:&mut Device,
    left:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128 % 64]>,
    right:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128 % 64]>)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128 % 64]> {
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128 % 64]>=DmTensor::new();
    for chunk in 0..4 {
        let mut left_tile:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128 % 64 = 16]>=DmTensor::new();
        left.view().tile::<m![P128 % 64],16,m![H % 120,P128 % 64 = 16 # 64]>(16*chunk).to_dm_view(&mut device.tdma,left_tile.view_mut());
        let left:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 120,P128 % 64 = 16]>=device.sub
            .begin(left_tile.view())
            .fetch::<m![H % 120,P128 % 64 = 16 / 8],m![P128 % 64 = 16 % 8]>().fetch_cast::<f32>()
            .collect::<m![H % 120,P128 % 64 = 16 / 8],m![P128 % 64 = 16 % 8]>().to_vrf();
        device.main.begin(right.view().tile::<m![P128 % 64],16,m![H % 120,P128 % 64 = 16 # 64]>(16*chunk))
            .fetch::<m![H % 120,P128 % 64 = 16 / 8],m![P128 % 64 = 16 % 8]>().fetch_cast::<f32>()
            .collect::<m![H % 120,P128 % 64 = 16 / 8],m![P128 % 64 = 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add,&left).vector_final()
            .cast::<bf16,m![P128 % 64 = 16 % 8 # 16]>().commit_trim::<m![P128 % 64 = 16 % 8]>()
            .commit_view(result.view_mut().tile::<m![P128 % 64],16,m![H % 120,P128 % 64 = 16 #{!} 64]>(16*chunk));
    }
    result
}

fn scale_output_wide(device:&mut Device,input:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128 % 64]>,
    scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    let scale:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 120]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![H % 120]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 15],m![H % 8]>().to_vrf();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128 % 64]>=device.main.begin(input.view())
        .fetch::<m![H % 120,P128 / 8 % 8],m![P128 % 8]>().fetch_cast::<f32>()
        .collect::<m![H % 120,P128 / 8 % 8],m![P128 % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120,P128 / 4 % 16],m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
        .vector_widen_concat::<m![H % 120,P128 / 8 % 8],m![P128 % 8]>()
        .vector_final().cast::<bf16,m![P128 % 8 # 16]>().commit_trim::<m![P128 % 8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![P128 % 64,H % 120]>=device.main.begin(result.view())
        .fetch::<m![P128 / 16 % 4,H % 120],m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4,H % 120],m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4,H / 4 % 30,P128 % 16],m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

fn sliding_partial(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qs]>,offset:usize)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128 % 64]> {
    let packed:DmTensor<f8e4m3,Chip,Cluster,DownRows,m![H % 120,Qs = 1024]>=weight.view()
        .tile::<m![Qs],1024,m![H,Qs = 1024 # 4096]>(offset).to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,Qs = 1024]>=device.main.begin(packed.view())
        .fetch::<m![H % 120,Qs = 1024 / 32],m![Qs = 1024 % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H % 120,Qs = 1024 / 16],m![Qs = 1024 % 16]>()
        .commit_trim::<m![Qs = 1024 % 16]>().commit();
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let flat:HbmTensorView<'_,bf16,Chip,m![P128,Qs]>=unsafe{input.view().reshape()};
        let x:DmTensor<bf16,Chip,Cluster,DownRows,m![P128 % 64 = 8,Qs = 1024]>=flat
            .tile::<m![Qs],1024,m![P128,Qs = 1024 # 4096]>(offset)
            .tile::<m![P128 % 64],8,m![P128 / 64,P128 % 64 = 8 # 64,Qs = 1024 # 4096]>(8*chunk).to_dm(&mut device.tdma);
        let x:TrfTensor<bf16,Chip,Cluster,DownRows,m![P128 % 64 = 8],m![Qs = 1024]>=device.sub.begin(x.view())
            .fetch::<m![P128 % 64 = 8,Qs = 1024 / 16],m![Qs = 1024 % 16]>()
            .collect::<m![P128 % 64 = 8,Qs = 1024 / 16],m![Qs = 1024 % 16]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H % 120,Qs = 1024 / 16],m![Qs = 1024 % 16]>()
            .collect::<m![H % 120,Qs = 1024 / 16],m![Qs = 1024 % 16]>()
            .contract_outer::<m![H % 120,Qs = 1024 / 32],m![Qs = 1024 % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H % 120]>()
            .contract_lane::<m![H % 120],m![P128 % 64 = 8]>(LaneMode::Interleaved)
            .cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(result.view_mut().tile::<m![P128 % 64],8,m![H % 120,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    result
}

#[device(chip = 1)]
pub fn project_sliding_output_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qs]>,scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![P128,H]>){
    let p0=sliding_partial(device,input,weight,0);
    let p1=sliding_partial(device,input,weight,1024);
    let p2=sliding_partial(device,input,weight,2048);
    let p3=sliding_partial(device,input,weight,3072);
    let p0p1=add_partials_wide(device,&p0,&p1);
    let p2p3=add_partials_wide(device,&p2,&p3);
    let p0p1p2p3=add_partials_wide(device,&p0p1,&p2p3);
    scale_output_wide(device,&p0p1p2p3,scale,output);
}

fn full_partial(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,Gf,Df]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qf]>,offset:usize)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128 % 64]> {
    let packed:DmTensor<f8e4m3,Chip,Cluster,DownRows,m![H % 120,Qf = 1024]>=weight.view()
        .tile::<m![Qf],1024,m![H,Qf = 1024 # 8192]>(offset).to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,Qf = 1024]>=device.main.begin(packed.view())
        .fetch::<m![H % 120,Qf = 1024 / 32],m![Qf = 1024 % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H % 120,Qf = 1024 / 16],m![Qf = 1024 % 16]>()
        .commit_trim::<m![Qf = 1024 % 16]>().commit();
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let flat:HbmTensorView<'_,bf16,Chip,m![P128,Qf]>=unsafe{input.view().reshape()};
        let x:DmTensor<bf16,Chip,Cluster,DownRows,m![P128 % 64 = 8,Qf = 1024]>=flat
            .tile::<m![Qf],1024,m![P128,Qf = 1024 # 8192]>(offset)
            .tile::<m![P128 % 64],8,m![P128 / 64,P128 % 64 = 8 # 64,Qf = 1024 # 8192]>(8*chunk).to_dm(&mut device.tdma);
        let x:TrfTensor<bf16,Chip,Cluster,DownRows,m![P128 % 64 = 8],m![Qf = 1024]>=device.sub.begin(x.view())
            .fetch::<m![P128 % 64 = 8,Qf = 1024 / 16],m![Qf = 1024 % 16]>()
            .collect::<m![P128 % 64 = 8,Qf = 1024 / 16],m![Qf = 1024 % 16]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H % 120,Qf = 1024 / 16],m![Qf = 1024 % 16]>()
            .collect::<m![H % 120,Qf = 1024 / 16],m![Qf = 1024 % 16]>()
            .contract_outer::<m![H % 120,Qf = 1024 / 32],m![Qf = 1024 % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H % 120]>()
            .contract_lane::<m![H % 120],m![P128 % 64 = 8]>(LaneMode::Interleaved)
            .cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(result.view_mut().tile::<m![P128 % 64],8,m![H % 120,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    result
}

#[device(chip = 1)]
pub fn project_full_output_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,Gf,Df]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qf]>,scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![P128,H]>){
    let p0=full_partial(device,input,weight,0);
    let p1=full_partial(device,input,weight,1024);
    let p2=full_partial(device,input,weight,2048);
    let p3=full_partial(device,input,weight,3072);
    let p4=full_partial(device,input,weight,4096);
    let p5=full_partial(device,input,weight,5120);
    let p6=full_partial(device,input,weight,6144);
    let p7=full_partial(device,input,weight,7168);
    let p0p1=add_partials_wide(device,&p0,&p1);
    let p2p3=add_partials_wide(device,&p2,&p3);
    let p4p5=add_partials_wide(device,&p4,&p5);
    let p6p7=add_partials_wide(device,&p6,&p7);
    let p0p1p2p3=add_partials_wide(device,&p0p1,&p2p3);
    let p4p5p6p7=add_partials_wide(device,&p4p5,&p6p7);
    let p0p1p2p3p4p5p6p7=add_partials_wide(device,&p0p1p2p3,&p4p5p6p7);
    scale_output_wide(device,&p0p1p2p3p4p5p6p7,scale,output);
}
