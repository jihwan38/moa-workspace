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

axes![IntTerm = 2];

const INVSQRT2: f32 = 0.70710678118f32;

pub(crate) type UpGateRows = m![L / 60];
pub(crate) type UpGateRowsPaired = m![L / 120, 1 # 2];

pub(crate) type UpGateCluster = m![Dummy2];
pub(crate) type UpGateSlices = m![L / 32 % 240 # 256];

// Local block scales apply to 16-element FP32 dot products. FP4*FP8 weight
// products are BF16-exact, but this changes FP32 reduction order; validate on NPU.
fn reduce_eight_rows_f32(
    device: &mut Device,
    weights: &DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![L % 32, H / 16]>,
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

    device.main
        .begin(weights.view().tile::<m![L % 32], 8, m![L % 32 = 8 # 32, H / 16]>(row_offset))
        .fetch::<m![L % 32 = 8, H / 128], m![H / 16 % 8]>()
        .collect::<m![L % 32 = 8, H / 128], m![H / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 32 = 8, H / 64], m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 32 = 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 32 = 8 / 4], m![L % 32 = 8 % 4 # 16]>()
        .commit_trim::<m![L % 32 = 8 % 4]>()
        .commit_view(output.view_mut().tile::<m![L % 32], 8, m![L % 32 = 8 #{!} 32]>(row_offset));
}

// Load restoration scales inside each projection so their VRF lifetime ends
// before the 7,680-byte weight-scale tile is loaded by the row reducer.
fn integer_upgate_blocks(
    device: &mut Device,
    x_trf: &TrfTensor<i8, Chip, UpGateCluster, UpGateSlices, m![IntTerm], m![H]>,
    restore: &DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![H / 16]>,
    weights: &DmTensor<i8, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]>,
) -> DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![L % 32, H / 16]> {
    let restore_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![H / 16]> = device.sub
        .begin(restore.view())
        .fetch::<m![H / 128], m![H / 16 % 8]>()
        .collect::<m![H / 128], m![H / 16 % 8]>().to_vrf();
    device.main.begin(weights.view())
        .fetch::<m![L % 32, H / 32], m![H % 32]>()
        .collect::<m![L % 32, H / 32], m![H % 32]>()
        .contract_outer::<m![L % 32, H / 64], m![H % 64], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![L % 32, H / 64]>()
        .contract_lane::<m![L % 32, H / 64, IntTerm], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_unzip::<IntTerm, m![L % 32, H / 64]>()
        .vector_fxp(FxpBinaryOp::LeftShift, 7i32, ())
        .vector_fxp_zip(FxpBinaryOp::AddFxp)
        .vector_fxp_to_fp(31)
        .vector_narrow_trim::<m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &restore_vrf)
        .vector_widen_pad::<m![H / 16 % 4 # 8]>().vector_final()
        .commit_trim::<m![H / 16 % 4]>().commit()
}

pub(crate) fn project_up_and_gate(
    device: &mut Device,
    x_trf: &TrfTensor<i8, Chip, UpGateCluster, UpGateSlices, m![IntTerm], m![H]>,
    restore: &DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![H / 16]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
) {
    let mut up: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32]> = DmTensor::new();

    let up_weight_packed: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, L % 7680, H]> = unsafe { up_weight_packed.view().reshape() };
    let up_weight_packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]> =
        up_weight_packed.to_dm(&mut device.tdma);
    // Each INT8 weight stores exactly twice the original FP4 value.
    let up_weight_integer: DmTensor<i8, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]> = device.main
        .begin(up_weight_packed.view())
        .fetch::<m![L % 32], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 32, H / 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_fp_to_fxp(30).vector_final()
        .cast::<i8, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>().commit();

    let up_weight_scale: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, L % 7680, H / 16]> = unsafe { up_weight_scale.view().reshape() };
    let up_weight_scale: DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![L % 32, H / 16]> =
        up_weight_scale.to_dm(&mut device.tdma);

    let partials = integer_upgate_blocks(device, x_trf, restore, &up_weight_integer);

    static_passes!(i, [0, 1, 2, 3], {
        reduce_eight_rows_f32(device, &partials, &up_weight_scale, &mut up, 8 * i);
    });

    let mut gate: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32]> = DmTensor::new();

    let gate_weight_packed: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, L % 7680, H]> = unsafe { gate_weight_packed.view().reshape() };
    let gate_weight_packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]> =
        gate_weight_packed.to_dm(&mut device.tdma);
    // Each INT8 weight stores exactly twice the original FP4 value.
    let gate_weight_integer: DmTensor<i8, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]> = device.main
        .begin(gate_weight_packed.view())
        .fetch::<m![L % 32], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 32, H / 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_fp_to_fxp(30).vector_final()
        .cast::<i8, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>().commit();

    let gate_weight_scale: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, L % 7680, H / 16]> = unsafe { gate_weight_scale.view().reshape() };
    let gate_weight_scale: DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![L % 32, H / 16]> =
        gate_weight_scale.to_dm(&mut device.tdma);

    let partials = integer_upgate_blocks(device, x_trf, restore, &gate_weight_integer);

    static_passes!(i, [0, 1, 2, 3], {
        reduce_eight_rows_f32(device, &partials, &gate_weight_scale, &mut gate, 8 * i);
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
    let (digits, restore) = quantize_h(device, &x);
    let digits: DmTensor<i8, Chip, UpGateCluster, m![Dummy256 / 16, 1 # 16], m![IntTerm, H]> = digits.to_dm(&mut device.tdma);
    let digits: DmTensor<i8, Chip, UpGateCluster, m![Dummy256 / 16, 16], m![IntTerm, H]> = device.main
        .begin(digits.view())
        .fetch::<m![IntTerm], m![H]>()
        .switch::<m![Dummy256 / 16, 16], m![IntTerm]>(SwitchConfig::CustomBroadcast { ring_size: 16 })
        .collect::<m![IntTerm, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>().commit();
    let digits: DmTensor<i8, Chip, UpGateCluster, Replicated, m![IntTerm, H]> = unsafe { digits.reshape() };
    let digits: DmTensorView<'_, i8, Chip, UpGateCluster, UpGateSlices, m![IntTerm, H]> = unsafe { digits.view().reshape() };
    let x_trf: TrfTensor<i8, Chip, UpGateCluster, UpGateSlices, m![IntTerm], m![H]> = device.sub
        .begin(digits)
        .fetch::<m![IntTerm, H / 32], m![H % 32]>()
        .collect::<m![IntTerm, H / 32], m![H % 32]>().to_trf();
    let restore: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![H / 16]> = restore.to_dm(&mut device.tdma);

    let (up, gate) = project_up_and_gate(
        device,
        &x_trf,
        &restore,
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
    // Quantize the actual GeGLU output; compact before redistributing it.
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![L]> = x.to_dm(&mut device.tdma);
    let (digits, restore) = quantize_l(device, &x);
    let digits: DmTensor<i8, Chip, DownCluster, m![Dummy256 / 32, 1 # 4, L / 1920], m![IntTerm, L % 1920]> = digits.to_dm(&mut device.tdma);
    let digits: DmTensor<i8, Chip, DownCluster, m![Dummy256 / 32, 4, L / 1920], m![IntTerm, L % 1920]> = device.main
        .begin(digits.view())
        .fetch::<m![IntTerm], m![L % 1920]>()
        .switch::<m![Dummy256 / 32, 4, L / 1920], m![IntTerm]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![IntTerm, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>().commit();
    let digits: DmTensor<i8, Chip, DownCluster, DownRowsByColumns, m![IntTerm, L % 1920]> = unsafe { digits.reshape() };
    let x_trf: TrfTensor<i8, Chip, DownCluster, DownRowsByColumns, m![IntTerm], m![L % 1920]> = device.sub
        .begin(digits.view())
        .fetch::<m![IntTerm, L / 32 % 60], m![L % 32]>()
        .collect::<m![IntTerm, L / 32 % 60], m![L % 32]>().to_trf();
    let restore: DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![L / 16 % 120]> = restore.to_dm(&mut device.tdma);
    let restore_vrf: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![L / 16 % 120]> = device.sub
        .begin(restore.view())
        .fetch::<m![L / 128 % 15], m![L / 16 % 8]>()
        .collect::<m![L / 128 % 15], m![L / 16 % 8]>().to_vrf();

    let mut down: DmTensor<bf16, Chip, DownCluster, DownRows, m![H % 60]> = DmTensor::new();

    let down_weight_scale: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, H % 1920, L / 16]> = unsafe { down_weight_scale.view().reshape() };
    let down_weight_scale: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut device.tdma);

    // The same full-matrix unpack pattern is used by the up/gate path.
    // Keep the column partition during both HBM loading and FP4 expansion.
    let down_weight_packed: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, H % 1920, L]> = unsafe { down_weight_packed.view().reshape() };
    let down_weight_packed: DmTensor<f4e2m1, Chip, DownCluster, DownRowsByColumns, m![H % 60, L % 1920]> =
        down_weight_packed.to_dm(&mut device.tdma);
    let down_weight_integer: DmTensor<i8, Chip, DownCluster, DownRowsByColumns, m![H % 60, L % 1920]> = device.main
        .begin(down_weight_packed.view())
        .fetch::<m![H % 60], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
        .collect::<m![H % 60, L / 8 % 240], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_fp_to_fxp(30).vector_final()
        .cast::<i8, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>().commit();
    let partials: DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 60, L / 16 % 120]> = device.main
        .begin(down_weight_integer.view())
        .fetch::<m![H % 60, L / 32 % 60], m![L % 32]>()
        .collect::<m![H % 60, L / 32 % 60], m![L % 32]>()
        .contract_outer::<m![H % 60, L / 64 % 30], m![L % 64], _, _, _>(&x_trf)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 60, L / 64 % 30]>()
        .contract_lane::<m![H % 60, L / 64 % 30, IntTerm], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_unzip::<IntTerm, m![H % 60, L / 64 % 30]>()
        .vector_fxp(FxpBinaryOp::LeftShift, 7i32, ())
        .vector_fxp_zip(FxpBinaryOp::AddFxp)
        .vector_fxp_to_fp(31)
        .vector_narrow_trim::<m![L / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &restore_vrf)
        .vector_widen_pad::<m![L / 16 % 4 # 8]>().vector_final()
        .commit_trim::<m![L / 16 % 4]>().commit();

    static_passes!(i, [0, 1, 2, 3, 4], {
        let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = device
            .sub
            .begin(down_weight_scale.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(12 * i))
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 16 / 8 % 15], m![L / 16 % 8]>()
            .to_vrf();

        device.main
            .begin(partials.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(12 * i))
            .fetch::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 12, L / 64 % 30], m![L / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            // Preserve local FP32 sum -> 8 column-slice FP32 sum -> BF16.
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

// Block scale and signed radix-128 digits. This is run once on normalized x
// and again on the BF16 GeGLU output, without changing the input contract.
fn quantize_h(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> (
    DmTensor<i8, Chip, Cluster, Slice, m![IntTerm, H]>,
    DmTensor<f32, Chip, Cluster, Slice, m![H / 16]>,
) {
    type QuantSlices = m![H / 64 # 256];
    let x: DmTensor<bf16, Chip, Cluster, QuantSlices, m![H % 64]> = x.to_dm(&mut device.tdma);
    let amax: DmTensor<f32, Chip, Cluster, QuantSlices, m![H / 16 % 4, 1 # 8]> = device.main
        .begin(x.view())
        .fetch::<m![H / 16 % 4], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H / 4 % 16], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![H / 16 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 7.52316384526264e-37f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Scale stays normal for finite BF16 inputs (floor 2^-120, then /64).
    let scale: DmTensor<f32, Chip, Cluster, QuantSlices, m![H / 16 % 4, 1 # 8]> = device.main
        .begin(amax.view())
        .fetch::<m![H / 16 % 4], m![1 # 8]>()
        .collect::<m![H / 16 % 4], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7f80_0000i32)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.015625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    let scale_vrf: VrfTensor<f32, Chip, Cluster, QuantSlices, m![H / 16 % 4, 1 # 8]> = device.sub
        .begin(scale.view())
        .fetch::<m![H / 16 % 4], m![1 # 8]>()
        .collect::<m![H / 16 % 4], m![1 # 8]>().to_vrf();
    let hi: DmTensor<i8, Chip, Cluster, QuantSlices, m![H % 64]> = device.main
        .begin(x.view())
        .fetch::<m![H / 16 % 4], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 16], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![H / 8 % 8], m![H % 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::Min, 127i32)
        .vector_clip(ClipBinaryOpI32::Max, -127i32)
        .vector_final().cast::<i8, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>().commit();
    let hi_fp: DmTensor<f32, Chip, Cluster, QuantSlices, m![H % 64]> = device.sub
        .begin(hi.view())
        .fetch::<m![H / 32 % 2], m![H % 32]>().fetch_cast::<i32>()
        .collect::<m![H / 8 % 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(31).vector_final()
        .commit_trim::<m![H % 8]>().commit();
    let hi_vrf: VrfTensor<f32, Chip, Cluster, QuantSlices, m![H % 64]> = device.sub
        .begin(hi_fp.view())
        .fetch::<m![H / 8 % 8], m![H % 8]>()
        .collect::<m![H / 8 % 8], m![H % 8]>().to_vrf();
    let lo: DmTensor<i8, Chip, Cluster, QuantSlices, m![H % 64]> = device.main
        .begin(x.view())
        .fetch::<m![H / 16 % 4], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 16], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 128f32)
        .vector_widen_concat::<m![H / 8 % 8], m![H % 8]>()
        .vector_fp_to_fxp(31).vector_final()
        .cast::<i8, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>().commit();
    let mut digits: DmTensor<i8, Chip, Cluster, QuantSlices, m![IntTerm, H % 64]> = DmTensor::new();
    hi.view().to_dm_view(&mut device.tdma,
        digits.view_mut().tile::<m![IntTerm], 1, m![IntTerm = 1 #{!} 2, H % 64]>(0));
    lo.view().to_dm_view(&mut device.tdma,
        digits.view_mut().tile::<m![IntTerm], 1, m![IntTerm = 1 #{!} 2, H % 64]>(1));
    // Contracted weights are 2*w; the radix value is 128*hi+lo.
    let restore: DmTensor<f32, Chip, Cluster, QuantSlices, m![H / 16 % 4, 1 # 8]> = device.main
        .begin(scale.view())
        .fetch::<m![H / 16 % 4], m![1 # 8]>()
        .collect::<m![H / 16 % 4], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.00390625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    (digits.to_dm(&mut device.tdma), restore.to_dm(&mut device.tdma))
}

// Block scale and signed radix-128 digits. This is run once on normalized x
// and again on the BF16 GeGLU output, without changing the input contract.
fn quantize_l(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![L]>,
) -> (
    DmTensor<i8, Chip, Cluster, Slice, m![IntTerm, L]>,
    DmTensor<f32, Chip, Cluster, Slice, m![L / 16]>,
) {
    type QuantSlices = m![L / 64 # 256];
    let x: DmTensor<bf16, Chip, Cluster, QuantSlices, m![L % 64]> = x.to_dm(&mut device.tdma);
    let amax: DmTensor<f32, Chip, Cluster, QuantSlices, m![L / 16 % 4, 1 # 8]> = device.main
        .begin(x.view())
        .fetch::<m![L / 16 % 4], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 8], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![L / 4 % 16], m![L % 4]>()
        .vector_intra_slice_reduce::<L, m![L / 16 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 7.52316384526264e-37f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Scale stays normal for finite BF16 inputs (floor 2^-120, then /64).
    let scale: DmTensor<f32, Chip, Cluster, QuantSlices, m![L / 16 % 4, 1 # 8]> = device.main
        .begin(amax.view())
        .fetch::<m![L / 16 % 4], m![1 # 8]>()
        .collect::<m![L / 16 % 4], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7f80_0000i32)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.015625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    let scale_vrf: VrfTensor<f32, Chip, Cluster, QuantSlices, m![L / 16 % 4, 1 # 8]> = device.sub
        .begin(scale.view())
        .fetch::<m![L / 16 % 4], m![1 # 8]>()
        .collect::<m![L / 16 % 4], m![1 # 8]>().to_vrf();
    let hi: DmTensor<i8, Chip, Cluster, QuantSlices, m![L % 64]> = device.main
        .begin(x.view())
        .fetch::<m![L / 16 % 4], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 8], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 16], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![L / 8 % 8], m![L % 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::Min, 127i32)
        .vector_clip(ClipBinaryOpI32::Max, -127i32)
        .vector_final().cast::<i8, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>().commit();
    let hi_fp: DmTensor<f32, Chip, Cluster, QuantSlices, m![L % 64]> = device.sub
        .begin(hi.view())
        .fetch::<m![L / 32 % 2], m![L % 32]>().fetch_cast::<i32>()
        .collect::<m![L / 8 % 8], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(31).vector_final()
        .commit_trim::<m![L % 8]>().commit();
    let hi_vrf: VrfTensor<f32, Chip, Cluster, QuantSlices, m![L % 64]> = device.sub
        .begin(hi_fp.view())
        .fetch::<m![L / 8 % 8], m![L % 8]>()
        .collect::<m![L / 8 % 8], m![L % 8]>().to_vrf();
    let lo: DmTensor<i8, Chip, Cluster, QuantSlices, m![L % 64]> = device.main
        .begin(x.view())
        .fetch::<m![L / 16 % 4], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 8], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 16], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 128f32)
        .vector_widen_concat::<m![L / 8 % 8], m![L % 8]>()
        .vector_fp_to_fxp(31).vector_final()
        .cast::<i8, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>().commit();
    let mut digits: DmTensor<i8, Chip, Cluster, QuantSlices, m![IntTerm, L % 64]> = DmTensor::new();
    hi.view().to_dm_view(&mut device.tdma,
        digits.view_mut().tile::<m![IntTerm], 1, m![IntTerm = 1 #{!} 2, L % 64]>(0));
    lo.view().to_dm_view(&mut device.tdma,
        digits.view_mut().tile::<m![IntTerm], 1, m![IntTerm = 1 #{!} 2, L % 64]>(1));
    // Contracted weights are 2*w; the radix value is 128*hi+lo.
    let restore: DmTensor<f32, Chip, Cluster, QuantSlices, m![L / 16 % 4, 1 # 8]> = device.main
        .begin(scale.view())
        .fetch::<m![L / 16 % 4], m![1 # 8]>()
        .collect::<m![L / 16 % 4], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.00390625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    (digits.to_dm(&mut device.tdma), restore.to_dm(&mut device.tdma))
}
