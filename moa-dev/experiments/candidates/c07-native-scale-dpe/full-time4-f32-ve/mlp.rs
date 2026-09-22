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

axes![FfnTerm = 2, FfnSlice = 256, FfnRow = 30];

const INVSQRT2: f32 = 0.70710678118f32;

pub(crate) type UpGateRows = m![L / 60];
pub(crate) type UpGateRowsPaired = m![L / 120, 1 # 2];

pub(crate) type UpGateCluster = m![Dummy2];
pub(crate) type UpGateSlices = m![FfnSlice];

// Local block scales apply to 16-element FP32 dot products. FP4*FP8 weight
// products are BF16-exact, but this changes FP32 reduction order; validate on NPU.
fn project_rows_with_scale_contraction(
    device: &mut Device,
    x_trf: &TrfTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnTerm, H]>,
    x_scale_vrf: &VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]>,
    weights: &DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]>,
    scales: &DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]>,
    output: &mut DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow]>,
) {
    // Preserve F32 block partials; the following VE passes apply block scales
    // and reduce the block axis before the final projection BF16 cast.
    let partials: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> = device
        .main.begin(weights.view())
        .fetch::<m![FfnRow], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![FfnRow, H / 32], m![H % 32]>()
        .contract_outer::<m![FfnRow, H / 64, FfnTerm], m![H % 64], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![FfnRow, H / 64]>()
        .contract_lane::<m![FfnRow, H / 64], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), x_scale_vrf)
        .vector_widen_pad::<m![H / 16 % 4 # 8]>()
        .vector_final()
        .commit_trim::<m![H / 16 % 4]>()
        .commit();
    static_passes!(i, [0, 1, 2, 3, 4], {
        let scale_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow = 6, H / 16]> = device.sub
            .begin(scales.view().tile::<m![FfnRow], 6, m![FfnRow = 6 # 30, H / 16]>(6 * i))
            .fetch::<m![FfnRow = 6], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![FfnRow = 6, H / 16 / 8], m![H / 16 % 8]>().to_vrf();
        device.main
            .begin(partials.view().tile::<m![FfnRow], 6, m![FfnRow = 6 # 30, H / 16]>(6 * i))
            .fetch::<m![FfnRow = 6, H / 16 / 8], m![H / 16 % 8]>()
            .collect::<m![FfnRow = 6, H / 16 / 8], m![H / 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![FfnRow = 6, H / 16 / 4], m![H / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<H, m![FfnRow = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>().vector_final()
            .transpose::<m![FfnRow = 6 / 2], m![FfnRow = 6 % 2 # 8]>().commit_trim::<m![FfnRow = 6 % 2]>()
            .commit_view(output.view_mut().tile::<m![FfnRow], 6, m![FfnRow = 6 #{!} 30]>(6 * i));
    });
}

pub(crate) fn project_up_and_gate(
    device: &mut Device,
    x_trf: &TrfTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnTerm, H]>,
    x_scale_vrf: &VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
) {
    let mut up: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow]> = DmTensor::new();

    let up_weight_packed: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, FfnSlice, FfnRow, H]> = unsafe { up_weight_packed.view().reshape() };
    let up_weight_packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]> =
        up_weight_packed.to_dm(&mut device.tdma);
    let up_weight_scale: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, FfnSlice, FfnRow, H / 16]> = unsafe { up_weight_scale.view().reshape() };

    let up_weight_scale: DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> =
        up_weight_scale.to_dm(&mut device.tdma);

    project_rows_with_scale_contraction(device, x_trf, x_scale_vrf, &up_weight_packed, &up_weight_scale, &mut up);

    let mut gate: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow]> = DmTensor::new();

    let gate_weight_packed: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, FfnSlice, FfnRow, H]> = unsafe { gate_weight_packed.view().reshape() };
    let gate_weight_packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]> =
        gate_weight_packed.to_dm(&mut device.tdma);
    let gate_weight_scale: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, FfnSlice, FfnRow, H / 16]> = unsafe { gate_weight_scale.view().reshape() };

    let gate_weight_scale: DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> =
        gate_weight_scale.to_dm(&mut device.tdma);

    project_rows_with_scale_contraction(device, x_trf, x_scale_vrf, &gate_weight_packed, &gate_weight_scale, &mut gate);

    // F32 rows allow 8-byte commits for 30-row slices. Cast after the gather.
    let up: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30]> = unsafe { up.reshape() };
    let gate: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30]> = unsafe { gate.reshape() };
    let up: DmTensor<f32, Chip, Cluster, UpGateRows, m![L % 60]> = up.to_dm(&mut device.tdma);
    let gate: DmTensor<f32, Chip, Cluster, UpGateRows, m![L % 60]> = gate.to_dm(&mut device.tdma);
    (cast_gathered_rows(device, &up), cast_gathered_rows(device, &gate))
}

fn cast_gathered_rows(device: &mut Device, x: &DmTensor<f32, Chip, Cluster, UpGateRows, m![L % 60]>)
    -> DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> {
    device.main.begin(x.view()).fetch::<m![L / 4 % 15], m![L % 4 # 8]>()
        .collect::<m![L / 4 % 15], m![L % 4 # 8]>()
        .cast::<bf16, m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit()
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
    // Quantize the actual runtime BF16 activation; both projections reuse it.
    let x: DmTensor<bf16, Chip, m![2], Slice, m![H]> = x.to_dm(&mut device.tdma);
    let (x, x_scale) = prepare_ffn_fp8_split(device, &x);
    let x: DmTensorView<'_, f8e4m3, Chip, UpGateCluster, UpGateSlices, m![FfnTerm, H]> = unsafe { x.view().reshape() };
    let x_scale: DmTensorView<'_, f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> = unsafe { x_scale.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnTerm, H]> = device
        .sub.begin(x)
        .fetch::<m![FfnTerm], m![H]>()
        .collect::<m![FfnTerm, H / 32], m![H % 32]>().to_trf();
    let x_scale_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> = device
        .sub.begin(x_scale)
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    let (up, gate) = project_up_and_gate(
        device,
        &x_trf,
        &x_scale_vrf,
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
// Four columns x 64 row groups; all 512 physical slices hold 30 real rows.
pub(crate) type DownCluster = m![Dummy2];
pub(crate) type DownRows = m![H / 30 % 64, 1 # 4];
pub(crate) type DownRowsByColumns = m![H / 30 % 64, L / 3840];
axes![DownQuantK = 3840, DownQuantTile = 8, DownQuantPart = 480];

fn quantize_down(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![DownQuantK]>,
) -> (
    DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![FfnTerm, DownQuantK]>,
    DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![1 # 8]>,
) {
    let amax: DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![1 # 8]> = device.main
        .begin(x.view()).fetch::<m![DownQuantK / 16], m![DownQuantK % 16]>().fetch_cast::<f32>()
        .collect::<m![DownQuantK / 8], m![DownQuantK % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![DownQuantK / 4], m![DownQuantK % 4]>()
        .vector_intra_slice_reduce::<DownQuantK, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.504632769052528e-36f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale: DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![1 # 8]> = device.main
        .begin(amax.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, 0x7f80_0000i32)
        .vector_reinterpret::<f32>().vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0078125f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![1 # 8]> = device.sub
        .begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let hi: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![DownQuantK]> = device.main
        .begin(x.view()).fetch::<m![DownQuantK / 16], m![DownQuantK % 16]>().fetch_cast::<f32>()
        .collect::<m![DownQuantK / 8], m![DownQuantK % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![DownQuantK / 4], m![DownQuantK % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![DownQuantK / 8], m![DownQuantK % 8]>().vector_final()
        .cast::<f8e4m3, m![DownQuantK % 8 # 32]>().commit_trim::<m![DownQuantK % 8]>().commit();
    let mut lo: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![DownQuantTile, DownQuantPart]> = DmTensor::new();
    let x_tiled: DmTensorView<'_, bf16, Chip, DownCluster, DownRowsByColumns, m![DownQuantTile, DownQuantPart]> = unsafe { x.view().reshape() };
    let hi_tiled: DmTensorView<'_, f8e4m3, Chip, DownCluster, DownRowsByColumns, m![DownQuantTile, DownQuantPart]> = unsafe { hi.view().reshape() };
    // 480-element residual tiles keep the F32 high-term VRF below 8 KiB.
    static_passes!(i, [0, 1, 2, 3, 4, 5, 6, 7], {
        let hi_vrf: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![DownQuantPart]> = device.sub
            .begin(hi_tiled.tile::<m![DownQuantTile], 1, m![DownQuantTile = 1 # 8, DownQuantPart]>(i))
            .fetch::<m![DownQuantPart / 32], m![DownQuantPart % 32]>().fetch_cast::<f32>()
            .collect::<m![DownQuantPart / 8], m![DownQuantPart % 8]>().to_vrf();
        device.main
            .begin(x_tiled.tile::<m![DownQuantTile], 1, m![DownQuantTile = 1 # 8, DownQuantPart]>(i))
            .fetch::<m![DownQuantPart / 16], m![DownQuantPart % 16]>().fetch_cast::<f32>()
            .collect::<m![DownQuantPart / 8], m![DownQuantPart % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![DownQuantPart / 4], m![DownQuantPart % 4]>()
            .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
            .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
            .vector_widen_concat::<m![DownQuantPart / 8], m![DownQuantPart % 8]>().vector_final()
            .cast::<f8e4m3, m![DownQuantPart % 8 # 32]>().commit_trim::<m![DownQuantPart % 8]>()
            .commit_view(lo.view_mut().tile::<m![DownQuantTile], 1, m![DownQuantTile = 1 #{!} 8, DownQuantPart]>(i));
    });
    let lo: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![DownQuantK]> = unsafe { lo.reshape() };
    let mut packed: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![FfnTerm, DownQuantK]> = DmTensor::new();
    hi.view().to_dm_view(&mut device.tdma, packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 2, DownQuantK]>(0));
    lo.view().to_dm_view(&mut device.tdma, packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 2, DownQuantK]>(1));
    (packed, scale)
}

pub(crate) fn project_down(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    project_down_sharded(device, x, down_weight_packed, down_weight_scale).to_dm(&mut device.tdma)
}

pub(crate) fn project_down_sharded(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, DownEpilogueSlices, m![H % 480]> {
    let x: DmTensor<bf16, Chip, DownCluster, m![Dummy256 / 16, 1 # 4, L / 3840], m![L % 3840]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, DownCluster, m![Dummy256 / 16, 4, L / 3840], m![L % 3840]> = device.main
        .begin(x.view()).fetch::<m![1], m![L % 3840]>()
        .switch::<m![Dummy256 / 16, 4, L / 3840], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 16 })
        .collect::<m![L / 16 % 240], m![L % 16]>().commit_trim::<m![L % 16]>().commit();
    let x: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![DownQuantK]> = unsafe { x.reshape() };
    let (x, x_scale) = quantize_down(device, &x);
    let x: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![FfnTerm, L % 3840]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![1], m![FfnTerm, L % 3840]> = device.sub
        .begin(x.view()).fetch::<m![FfnTerm], m![L % 3840]>()
        .collect::<m![FfnTerm, L / 32 % 120], m![L % 32]>().to_trf();
    let x_scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![1 # 8]> = device.sub
        .begin(x_scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weights: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, H % 1920, L]> = unsafe { down_weight_packed.view().reshape() };
    let weights: DmTensor<f4e2m1, Chip, DownCluster, DownRowsByColumns, m![H % 30, L % 3840]> = weights.to_dm(&mut device.tdma);
    let scales: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, H % 1920, L / 16]> = unsafe { down_weight_scale.view().reshape() };
    let scales: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 30, L / 16 % 240]> = scales.to_dm(&mut device.tdma);
    let mut partial: DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 30, 1 # 2]> = DmTensor::new();
    project_native_down_rows(device, &x_trf, &x_scale_vrf, &weights, &scales, &mut partial);
    let down: DmTensor<f32, Chip, DownCluster, DownRows, m![H % 30]> = device.main
        .begin(partial.view()).fetch::<m![H % 30], m![1 # 8]>().collect::<m![H % 30], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>().vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<DownRows, m![H % 30]>(InterSliceReduceOpF32::Add)
        .vector_final().transpose::<m![H / 2 % 15], m![H % 2 # 8]>().commit_trim::<m![H % 2]>().commit();
    let down: DmTensor<f32, Chip, m![H / 1920], DownRows, m![H % 30]> = unsafe { down.reshape() };
    let down: DmTensor<f32, Chip, Cluster, DownEpilogueSlices, m![H % 480]> = down.to_dm(&mut device.tdma);
    device.main.begin(down.view()).fetch::<m![H / 8 % 60], m![H % 8]>().collect::<m![H / 8 % 60], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>().commit()
}

fn project_native_down_rows(
    device: &mut Device,
    x_trf: &TrfTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![1], m![FfnTerm, L % 3840]>,
    x_scale_vrf: &VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![1 # 8]>,
    weights: &DmTensor<f4e2m1, Chip, DownCluster, DownRowsByColumns, m![H % 30, L % 3840]>,
    scales: &DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 30, L / 16 % 240]>,
    output: &mut DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 30, 1 # 2]>,
) {
    // Preserve F32 block partials; the following VE passes apply block scales
    // and reduce the block axis before the final projection BF16 cast.
    let partials: DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 30, L / 16 % 240]> = device
        .main.begin(weights.view())
        .fetch::<m![H % 30], m![L % 3840]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 30, L / 32 % 120], m![L % 32]>()
        .contract_outer::<m![H % 30, L / 64 % 60, FfnTerm], m![L % 64], _, _, _>(x_trf)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 30, L / 64 % 60]>()
        .contract_lane::<m![H % 30, L / 64 % 60], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![L / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), x_scale_vrf)
        .vector_widen_pad::<m![L / 16 % 4 # 8]>().vector_final()
        .commit_trim::<m![L / 16 % 4]>()
        .commit();
    static_passes!(i, [0, 1, 2, 3, 4], {
        let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 30 = 6, L / 16 % 240]> = device.sub
            .begin(scales.view().tile::<m![H % 30], 6, m![H % 30 = 6 # 30, L / 16 % 240]>(6 * i))
            .fetch::<m![H % 30 = 6], m![L / 16 % 240]>().fetch_cast::<f32>()
            .collect::<m![H % 30 = 6, L / 16 % 240 / 8], m![L / 16 % 240 % 8]>().to_vrf();
        device.main
            .begin(partials.view().tile::<m![H % 30], 6, m![H % 30 = 6 # 30, L / 16 % 240]>(6 * i))
            .fetch::<m![H % 30 = 6, L / 16 % 240 / 8], m![L / 16 % 240 % 8]>()
            .collect::<m![H % 30 = 6, L / 16 % 240 / 8], m![L / 16 % 240 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 30 = 6, L / 16 % 240 / 4], m![L / 16 % 240 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>().vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit_view(output.view_mut().tile::<m![H % 30], 6, m![H % 30 = 6 #{!} 30, 1 # 2]>(6 * i));
    });
}

type FfnQuantSlices = m![1 # 32, H / 480];

fn broadcast_ffn_quantized(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, m![2], FfnQuantSlices, m![FfnTerm, H % 480]>,
) -> DmTensor<f8e4m3, Chip, m![2], Replicated, m![FfnTerm, H]> {
    let x: DmTensor<f8e4m3, Chip, m![2], m![Dummy256 / 32, 1 # 32], m![FfnTerm, H]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<f8e4m3, Chip, m![2], m![Dummy256 / 32, 32], m![FfnTerm, H]> = device
        .main.begin(x.view())
        .fetch::<m![FfnTerm], m![H]>()
        .switch::<m![Dummy256 / 32, 32], m![FfnTerm]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![FfnTerm, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>().commit();
    // All 512 physical slices now contain the same original H-vector order.
    unsafe { x.reshape() }
}

pub(crate) fn prepare_ffn_fp8_split(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], Slice, m![H]>,
) -> (
    DmTensor<f8e4m3, Chip, m![2], Replicated, m![FfnTerm, H]>,
    DmTensor<f32, Chip, m![2], Replicated, m![1 # 8]>,
) {
    // Amax on the canonical input avoids an expensive inter-slice reduction.
    // i32 reinterpret makes the absolute-value mask a non-NaN immediate.
    let amax: DmTensor<f32, Chip, m![2], Slice, m![1 # 8]> = device
        .main.begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.504632769052528e-36f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Exponent mask produces 2^floor(log2(amax)); floor 2^-119 guarantees
    // scale >= 2^-126, and finite BF16 inputs imply scale <= 2^120.
    let scale: DmTensor<f32, Chip, m![2], Slice, m![1 # 8]> = device
        .main.begin(amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7f80_0000i32)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0078125f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    let quant_scale: DmTensor<f32, Chip, m![2], FfnQuantSlices, m![1 # 8]> = scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, m![2], FfnQuantSlices, m![1 # 8]> = device
        .sub.begin(quant_scale.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let x: DmTensor<bf16, Chip, m![2], FfnQuantSlices, m![H % 480]> = x.to_dm(&mut device.tdma);
    let hi: DmTensor<f8e4m3, Chip, m![2], FfnQuantSlices, m![H % 480]> = device
        .main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>().commit();
    let hi_vrf: VrfTensor<f32, Chip, m![2], FfnQuantSlices, m![H % 480]> = device
        .sub.begin(hi.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>().to_vrf();
    let lo: DmTensor<f8e4m3, Chip, m![2], FfnQuantSlices, m![H % 480]> = device
        .main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>().commit();
    let mut packed: DmTensor<f8e4m3, Chip, m![2], FfnQuantSlices, m![FfnTerm, H % 480]> = DmTensor::new();
    hi.view().to_dm_view(&mut device.tdma,
        packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 2, H % 480]>(0));
    lo.view().to_dm_view(&mut device.tdma,
        packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 2, H % 480]>(1));
    (broadcast_ffn_quantized(device, &packed), scale.to_dm(&mut device.tdma))
}
