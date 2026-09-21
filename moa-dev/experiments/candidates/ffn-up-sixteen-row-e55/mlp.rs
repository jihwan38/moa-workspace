use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Dummy2, Dummy256, H, L};
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

pub(crate) type UpGateCluster = m![Dummy2];
pub(crate) type UpGateSlices = m![L / 32 % 240 # 256];

// Local block scales apply to 16-element FP32 dot products. FP4*FP8 weight
// products are BF16-exact, but this changes FP32 reduction order; validate on NPU.
fn project_eight_rows_scaled_after_dot(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![1], m![H]>,
    weights: &DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]>,
    scales: &DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![L % 32, H / 16]>,
    output: &mut DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32]>,
    row_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![L % 32 = 8, H / 16]> = device
        .sub
        .begin(scales.view().tile::<m![L % 32], 8, m![L % 32 = 8 # 32, H / 16]>(row_offset))
        .fetch::<m![L % 32 = 8], m![H / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 32 = 8, H / 16 / 8], m![H / 16 % 8]>()
        .to_vrf();

    device
        .main
        .begin(weights.view().tile::<m![L % 32], 8, m![L % 32 = 8 # 32, H]>(row_offset))
        .fetch::<m![L % 32 = 8, H / 16], m![H % 16]>()
        .collect::<m![L % 32 = 8, H / 16], m![H % 16]>()
        // Keep each 16-element scale block separate through the DPE.
        .contract_outer::<m![L % 32 = 8, H / 16], m![H % 16], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 32 = 8, H / 16]>()
        .contract_lane::<m![L % 32 = 8, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 32 = 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 32 = 8 / 4], m![L % 32 = 8 % 4 # 16]>()
        .commit_trim::<m![L % 32 = 8 % 4]>()
        .commit_view(output.view_mut().tile::<m![L % 32], 8, m![L % 32 = 8 #{!} 32]>(row_offset));
}

fn project_eight_rows_from_sixteen(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![1], m![H]>,
    weights: &DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32 = 16, H]>,
    scales: &DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![L % 32, H / 16]>,
    output: &mut DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32]>,
    row_offset: usize,
    local_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![L % 32 = 8, H / 16]> = device
        .sub
        .begin(scales.view().tile::<m![L % 32], 8, m![L % 32 = 8 # 32, H / 16]>(row_offset))
        .fetch::<m![L % 32 = 8], m![H / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 32 = 8, H / 16 / 8], m![H / 16 % 8]>()
        .to_vrf();

    device
        .main
        .begin(weights.view().tile::<m![L % 32 = 16], 8, m![L % 32 = 16 = 8 # 16, H]>(local_offset))
        .fetch::<m![L % 32 = 8, H / 16], m![H % 16]>()
        .collect::<m![L % 32 = 8, H / 16], m![H % 16]>()
        // Keep each 16-element scale block separate through the DPE.
        .contract_outer::<m![L % 32 = 8, H / 16], m![H % 16], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 32 = 8, H / 16]>()
        .contract_lane::<m![L % 32 = 8, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 32 = 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 32 = 8 / 4], m![L % 32 = 8 % 4 # 16]>()
        .commit_trim::<m![L % 32 = 8 % 4]>()
        .commit_view(output.view_mut().tile::<m![L % 32], 8, m![L % 32 = 8 #{!} 32]>(row_offset));
}

// Tile the row-major HBM view before copying it to each slice.
fn load_up_sixteen_rows(
    device: &mut Device,
    source: &HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, L / 32 % 240, L % 32, H]>,
    row_offset: usize,
) -> DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![L % 32 = 16, H]> {
    source.tile::<m![L % 32], 16, m![Dummy2, L / 32 % 240, L % 32 = 16 # 32, H]>(row_offset)
        .to_dm(&mut device.tdma)
}

fn decode_up_sixteen_rows(
    device: &mut Device,
    packed: &DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![L % 32 = 16, H]>,
) -> DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32 = 16, H]> {
    device.main.begin(packed.view())
        .fetch::<m![L % 32 = 16], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 32 = 16, H / 8], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn project_up_and_gate(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![1], m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
) {
    let mut up: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32]> = DmTensor::new();

    let up_weight_packed: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, L / 32 % 240, L % 32, H]> =
        unsafe { up_weight_packed.view().reshape() };
    let up_weight_scale: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, L % 7680, H / 16]> =
        unsafe { up_weight_scale.view().reshape() };
    let up_weight_scale: DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![L % 32, H / 16]> =
        up_weight_scale.to_dm(&mut device.tdma);

    // These are independent immutable tensors. Allocation and overlap remain
    // compiler decisions; no fixed physical buffer count is assumed.
    let first_packed = load_up_sixteen_rows(device, &up_weight_packed, 0);
    let second_packed = load_up_sixteen_rows(device, &up_weight_packed, 16);
    let first = decode_up_sixteen_rows(device, &first_packed);
    project_eight_rows_from_sixteen(device, x_trf, &first, &up_weight_scale, &mut up, 0, 0);
    project_eight_rows_from_sixteen(device, x_trf, &first, &up_weight_scale, &mut up, 8, 8);
    let second = decode_up_sixteen_rows(device, &second_packed);
    project_eight_rows_from_sixteen(device, x_trf, &second, &up_weight_scale, &mut up, 16, 0);
    project_eight_rows_from_sixteen(device, x_trf, &second, &up_weight_scale, &mut up, 24, 8);

    let mut gate: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32]> = DmTensor::new();

    let gate_weight_packed: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, L % 7680, H]> = unsafe { gate_weight_packed.view().reshape() };
    let gate_weight_packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]> =
        gate_weight_packed.to_dm(&mut device.tdma);
    let gate_weight_packed: DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]> = device
        .main
        .begin(gate_weight_packed.view())
        .fetch::<m![L % 32], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 32, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();

    // Materialize each unscaled BF16 matrix once; later row tiles reuse it.
    let gate_weight_bf16: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]> = device
        .main
        .begin(gate_weight_packed.view())
        .fetch::<m![L % 32], m![H]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![L % 32, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    let gate_weight_scale: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, L % 7680, H / 16]> = unsafe { gate_weight_scale.view().reshape() };
    let gate_weight_scale: DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![L % 32, H / 16]> =
        gate_weight_scale.to_dm(&mut device.tdma);

    static_passes!(i, [0, 1, 2, 3], {
        project_eight_rows_scaled_after_dot(device, x_trf, &gate_weight_bf16, &gate_weight_scale, &mut gate, 8 * i);
    });

    // Remove padding with typed copies before reinterpreting the high row axis.
    let up: DmTensor<bf16, Chip, UpGateCluster, Slice, m![L % 7680]> = up.to_dm(&mut device.tdma);
    let gate: DmTensor<bf16, Chip, UpGateCluster, Slice, m![L % 7680]> = gate.to_dm(&mut device.tdma);
    let up: DmTensor<bf16, Chip, m![L / 7680], Slice, m![L % 7680]> = unsafe { up.reshape() };
    let gate: DmTensor<bf16, Chip, m![L / 7680], Slice, m![L % 7680]> = unsafe { gate.reshape() };
    (up.to_dm(&mut device.tdma), gate.to_dm(&mut device.tdma))
}

pub(crate) fn feedforward(
    device: &mut Device,
    x: DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
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
    let down = feedforward_sharded(device, x, up_weight_packed, gate_weight_packed, down_weight_packed, up_weight_scale, gate_weight_scale, down_weight_scale, up_global_scale, gate_global_scale, down_global_scale);
    down.to_dm(&mut device.tdma)
}

pub(crate) fn feedforward_sharded(
    device: &mut Device,
    x: DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, DownEpilogueSlices, m![H % 480]> {
    // Broadcast the normalized single-slice activation directly into
    // both row clusters before the up/gate projections.
    let x: DmTensor<bf16, Chip, UpGateCluster, m![Dummy256 / 16, 1 # 16], m![H]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, UpGateCluster, m![Dummy256 / 16, 16], m![H]> = device
        .main
        .begin(x.view())
        .fetch::<m![1], m![H]>()
        .switch::<m![Dummy256 / 16, 16], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 16 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let x: DmTensor<bf16, Chip, UpGateCluster, Replicated, m![H]> = unsafe { x.reshape() };
    // The last 16 replicated slices in each cluster are padding for row mapping.
    let x_two_cluster: DmTensorView<'_, bf16, Chip, UpGateCluster, UpGateSlices, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![1], m![H]> = device
        .sub
        .begin(x_two_cluster)
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
    let down = project_down_sharded(device, &x, down_weight_packed, down_weight_scale);

    let down_global_scale: DmTensor<f32, Chip, Cluster, DownEpilogueSlices, m![1 # 8]> = down_global_scale.to_dm(&mut device.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, DownEpilogueSlices, m![1 # 8]> = device
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, DownEpilogueSlices, m![H % 480]> = device
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
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

pub(crate) type DownEpilogueSlices = m![1 # 32, H / 480];
pub(crate) type DownCluster = m![Dummy2];
pub(crate) type DownRows = m![H / 60 % 32, 1 # 8];
pub(crate) type DownRowsByColumns = m![H / 60 % 32, L / 1920];

pub(crate) fn project_down(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let down = project_down_sharded(device, x, down_weight_packed, down_weight_scale);
    down.to_dm(&mut device.tdma)
}

pub(crate) fn project_down_sharded(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, DownEpilogueSlices, m![H % 480]> {
    let x: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![L % 1920]> = x.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![1], m![L % 1920]> = device
        .sub
        .begin(x.view())
        .fetch::<m![L / 16 % 120], m![L % 16]>()
        .collect::<m![L / 16 % 120], m![L % 16]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownCluster, DownRows, m![H % 60]> = DmTensor::new();

    let down_weight_scale: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, H % 1920, L / 16]> = unsafe { down_weight_scale.view().reshape() };
    let down_weight_scale: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut device.tdma);

    // The same full-matrix unpack pattern is used by the up/gate path.
    // Keep the column partition during both HBM loading and FP4 expansion.
    let down_weight_packed: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, H % 1920, L]> = unsafe { down_weight_packed.view().reshape() };
    let down_weight_packed: DmTensor<f4e2m1, Chip, DownCluster, DownRowsByColumns, m![H % 60, L % 1920]> =
        down_weight_packed.to_dm(&mut device.tdma);
    // Keep the two-pass decode topology validated for E12 up/gate on RNGD.
    let down_weight_fp8: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 60, L % 1920]> = device
        .main
        .begin(down_weight_packed.view())
        .fetch::<m![H % 60], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit();
    let down_weight_bf16: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![H % 60, L % 1920]> = device
        .main
        .begin(down_weight_fp8.view())
        .fetch::<m![H % 60], m![L % 1920]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 60, L / 16 % 120], m![L % 16]>()
        .commit_trim::<m![L % 16]>()
        .commit();

    static_passes!(i, [0, 1, 2, 3, 4], {
        let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = device
            .sub
            .begin(down_weight_scale.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(12 * i))
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 16 / 8 % 15], m![L / 16 % 8]>()
            .to_vrf();

        device
            .main
            .begin(down_weight_bf16.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(12 * i))
            .fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![H % 60 = 12, L / 16 % 120], m![L % 16], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            // Identity time contraction preserves 120 independent scale blocks.
            .contract_time::<m![H % 60 = 12, L / 16 % 120]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            // Preserve local FP32 sum -> eight column-slice FP32 sum -> BF16.
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12 * i));
    });

    // Only rename the existing high-row cluster axis. Typed DMA performs
    // the actual gather: h = 1920 * cluster + 60 * row_group + local_row.
    let down: DmTensor<bf16, Chip, m![H / 1920], DownRows, m![H % 60]> = unsafe { down.reshape() };
    down.to_dm(&mut device.tdma)
}
