use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, L};
use crate::device::layout::{Cluster, Replicated, Slice};

macro_rules! static_passes {
    ($index:ident, [$($value:expr),+], $body:block) => {
        $(
            {
                let $index: usize = $value;
                $body
            }
        )+
    };
}

const INVSQRT2: f32 = 0.70710678118f32;

pub(crate) type UpGateRows = m![L / 60];
pub(crate) type UpGateRowsPaired = m![L / 120, 1 # 2];

// Local block scales apply to 16-element FP32 dot products. FP4*FP8 weight
// products are BF16-exact, but this changes FP32 reduction order; validate on NPU.
fn project_four_rows_scaled_after_dot(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, Cluster, UpGateRows, m![1], m![H]>,
    weights: &DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H]>,
    scales: &DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H / 16]>,
    output: &mut DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    row_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRows, m![L % 60 = 4, H / 16]> = device
        .sub
        .begin(scales.view().tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H / 16]>(row_offset))
        .fetch::<m![L % 60 = 4], m![H / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 4, H / 16 / 8], m![H / 16 % 8]>()
        .to_vrf();

    device
        .main
        .begin(weights.view().tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H]>(row_offset))
        .fetch::<m![L % 60 = 4, H / 32], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![L % 60 = 4, H / 16], m![H % 16]>()
        // Keep each 16-element scale block separate through the DPE.
        .contract_outer::<m![L % 60 = 4, H / 16], m![H % 16], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 4, H / 16]>()
        .contract_lane::<m![L % 60 = 4, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 60 = 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![L % 60 = 4 # 16]>()
        .commit_trim::<m![L % 60 = 4]>()
        .commit_view(output.view_mut().tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60]>(row_offset));
}

pub(crate) fn project_up_and_gate(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, Cluster, UpGateRows, m![1], m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
) {
    let mut up: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> = DmTensor::new();
    let up_weight_packed: DmTensor<f4e2m1, Chip, Cluster, UpGateRows, m![L % 60, H]> =
        up_weight_packed.to_dm(&mut device.tdma);
    let up_weight_packed: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H]> = device
        .main
        .begin(up_weight_packed.view())
        .fetch::<m![L % 60], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 60, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let up_weight_scale: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H / 16]> =
        up_weight_scale.to_dm(&mut device.tdma);

    static_passes!(i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], {
        project_four_rows_scaled_after_dot(device, x_trf, &up_weight_packed, &up_weight_scale, &mut up, 4 * i);
    });

    let mut gate: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> = DmTensor::new();
    let gate_weight_packed: DmTensor<f4e2m1, Chip, Cluster, UpGateRows, m![L % 60, H]> =
        gate_weight_packed.to_dm(&mut device.tdma);
    let gate_weight_packed: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H]> = device
        .main
        .begin(gate_weight_packed.view())
        .fetch::<m![L % 60], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 60, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let gate_weight_scale: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H / 16]> =
        gate_weight_scale.to_dm(&mut device.tdma);

    static_passes!(i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], {
        project_four_rows_scaled_after_dot(device, x_trf, &gate_weight_packed, &gate_weight_scale, &mut gate, 4 * i);
    });
    (up, gate)
}

pub(crate) fn feedforward(
    device: &mut Device,
    x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x: DmTensor<bf16, Chip, Cluster, UpGateRows, m![H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<bf16, Chip, Cluster, UpGateRows, m![1], m![H]> = device
        .sub
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let (up, gate) = project_up_and_gate(
        device,
        &x_trf,
        up_weight_packed,
        gate_weight_packed,
        up_weight_scale,
        gate_weight_scale,
    );
    let x = geglu(device, up, gate, up_global_scale, gate_global_scale);
    let x = x.to_dm(&mut device.tdma);
    let down = project_down(device, &x, down_weight_packed, down_weight_scale);

    let down_global_scale: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = down_global_scale.to_dm(&mut device.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = device
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = device
        .main
        .begin(down.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

pub(crate) fn geglu(
    device: &mut Device,
    up: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    gate: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> {
    let up: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = device
        .main
        .begin(up.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 16]>()
        .switch::<UpGateRowsPaired, m![L / 4 % 15, L / 60 % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 1 })
        .collect::<m![L / 4 % 15, L / 60 % 2], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();

    let up_global_scale: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> =
        up_global_scale.to_dm(&mut device.tdma);
    let up_global_scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> = device
        .sub
        .begin(up_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let up: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = device
        .main
        .begin(up.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gate_global_scale: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> =
        gate_global_scale.to_dm(&mut device.tdma);
    let gate_global_scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> = device
        .sub
        .begin(gate_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let gate: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = device
        .main
        .begin(gate.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 16]>()
        .switch::<UpGateRowsPaired, m![L / 4 % 15, L / 60 % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 1 })
        .collect::<m![L / 4 % 15, L / 60 % 2], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();

    let gate: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = device
        .main
        .begin(gate.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gelu: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gelu_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .to_vrf();

    device
        .main
        .begin(up.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit()
}

pub(crate) type DownRows = m![H / 120, 1 # 8];
pub(crate) type DownRowsByColumns = m![H / 120, L / 1920];

// Preserve the local L%1920 column partition and its eight-slice FP32 sum.
fn project_down_four_rows_scaled_after_dot(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, Cluster, DownRowsByColumns, m![1], m![L % 1920]>,
    weights: &DmTensor<f8e4m3, Chip, Cluster, DownRowsByColumns, m![H % 120, L % 1920]>,
    scales: &DmTensor<f8e4m3, Chip, Cluster, DownRowsByColumns, m![H % 120, L / 16 % 120]>,
    output: &mut DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120]>,
    row_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, Cluster, DownRowsByColumns, m![H % 120 = 4, L / 16 % 120]> = device
        .sub
        .begin(scales.view().tile::<m![H % 120], 4, m![H % 120 = 4 # 120, L / 16 % 120]>(row_offset))
        .fetch::<m![H % 120 = 4], m![L / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 = 4, L / 16 / 8 % 15], m![L / 16 % 8]>()
        .to_vrf();

    device
        .main
        .begin(weights.view().tile::<m![H % 120], 4, m![H % 120 = 4 # 120, L % 1920]>(row_offset))
        .fetch::<m![H % 120 = 4, L / 32 % 60], m![L % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 120 = 4, L / 16 % 120], m![L % 16]>()
        .contract_outer::<m![H % 120 = 4, L / 16 % 120], m![L % 16], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 4, L / 16 % 120]>()
        .contract_lane::<m![H % 120 = 4, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        // 120 local blocks sum in FP32, then the eight L-column slices sum in FP32.
        .vector_intra_slice_reduce::<L, m![H % 120 = 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<DownRows, m![H % 120 = 4]>(InterSliceReduceOpF32::Add)
        .vector_final()
        // Keep the original BF16 boundary AFTER the eight-slice FP32 sum.
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![H % 120 = 4 # 16]>()
        .commit_trim::<m![H % 120 = 4]>()
        .commit_view(output.view_mut().tile::<m![H % 120], 4, m![H % 120 = 4 #{!} 120]>(row_offset));
}

pub(crate) fn project_down(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, DownRows, m![L]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x: DmTensor<bf16, Chip, Cluster, DownRowsByColumns, m![L % 1920]> = x.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, Cluster, DownRowsByColumns, m![1], m![L % 1920]> = device
        .sub
        .begin(x.view())
        .fetch::<m![L / 16 % 120], m![L % 16]>()
        .collect::<m![L / 16 % 120], m![L % 16]>()
        .to_trf();
    let mut down: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120]> = DmTensor::new();
    let down_weight_scale: DmTensor<f8e4m3, Chip, Cluster, DownRowsByColumns, m![H % 120, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut device.tdma);
    let down_weight_packed: DmTensor<f4e2m1, Chip, Cluster, DownRowsByColumns, m![H % 120, L % 1920]> =
        down_weight_packed.to_dm(&mut device.tdma);
    let down_weight_packed: DmTensor<f8e4m3, Chip, Cluster, DownRowsByColumns, m![H % 120, L % 1920]> = device
        .main
        .begin(down_weight_packed.view())
        .fetch::<m![H % 120], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 120, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit();

    static_passes!(i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], {
        project_down_four_rows_scaled_after_dot(device, &x_trf, &down_weight_packed, &down_weight_scale, &mut down, 4 * i);
    });
    down.to_dm(&mut device.tdma)
}
