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

axes![FfnPartial = 2, FfnTerm = 4, FfnSlice = 256, FfnRow = 30];

const INVSQRT2: f32 = 0.70710678118f32;

pub(crate) type UpGateRows = m![L / 60];
pub(crate) type UpGateRowsPaired = m![L / 120, 1 # 2];

pub(crate) type UpGateCluster = m![Dummy2];
pub(crate) type UpGateSlices = m![FfnSlice];

// Local block scales apply to 16-element FP32 dot products. FP4*FP8 weight
// products are BF16-exact, but this changes FP32 reduction order; validate on NPU.
fn project_rows_with_scale_contraction(
    device: &mut Device,
    x_trf: &TrfTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnTerm, H]>,
    x_scale_vrf: &VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]>,
    weights: &DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]>,
    scales: &DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]>,
    output: &mut DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow]>,
) {
    // This deliberately adds BF16 rounding of each 16-element block dot.
    // The second contraction multiplies block scales and reduces the block axis.
    let partials: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> = device
        .main.begin(weights.view())
        .fetch::<m![FfnRow], m![H]>()
        .fetch_table_lookup::<f8e5m2>()
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
    let hi: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> = device.sub
        .begin(partials.view()).fetch::<m![FfnRow, H / 128], m![H / 16 % 8]>()
        .collect::<m![FfnRow, H / 128], m![H / 16 % 8]>()
        .cast::<bf16, m![H / 16 % 8 # 16]>().commit_trim::<m![H / 16 % 8]>().commit();
    let mut lo: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> = DmTensor::new();
    static_passes!(i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], {
        let hi_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow = 3, H / 16]> = device.sub
            .begin(hi.view().tile::<m![FfnRow], 3, m![FfnRow = 3 # 30, H / 16]>(3 * i))
            .fetch::<m![FfnRow = 3, H / 256], m![H / 16 % 16]>().fetch_cast::<f32>()
            .collect::<m![FfnRow = 3, H / 128], m![H / 16 % 8]>().to_vrf();
        device.main
            .begin(partials.view().tile::<m![FfnRow], 3, m![FfnRow = 3 # 30, H / 16]>(3 * i))
            .fetch::<m![FfnRow = 3, H / 128], m![H / 16 % 8]>()
            .collect::<m![FfnRow = 3, H / 128], m![H / 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![FfnRow = 3, H / 64], m![H / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
            .vector_widen_concat::<m![FfnRow = 3, H / 128], m![H / 16 % 8]>().vector_final()
            .cast::<bf16, m![H / 16 % 8 # 16]>().commit_trim::<m![H / 16 % 8]>()
            .commit_view(lo.view_mut().tile::<m![FfnRow], 3, m![FfnRow = 3 #{!} 30, H / 16]>(3 * i));
    });
    let mut compensated: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![FfnPartial, FfnRow, H / 16]> = DmTensor::new();
    hi.view().to_dm_view(&mut device.tdma, compensated.view_mut().tile::<m![FfnPartial], 1, m![FfnPartial = 1 #{!} 2, FfnRow, H / 16]>(0));
    lo.view().to_dm_view(&mut device.tdma, compensated.view_mut().tile::<m![FfnPartial], 1, m![FfnPartial = 1 #{!} 2, FfnRow, H / 16]>(1));
    let scales: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> = device
        .sub.begin(scales.view())
        .fetch::<m![FfnRow, H / 128], m![H / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![FfnRow, H / 128], m![H / 16 % 8]>()
        .cast::<bf16, m![H / 16 % 8 # 16]>()
        .commit_trim::<m![H / 16 % 8]>()
        .commit();
    let scales_trf: TrfTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnRow, H / 16]> = device
        .sub.begin(scales.view())
        .fetch::<m![FfnRow, H / 16 / 16], m![H / 16 % 16]>()
        .collect::<m![FfnRow, H / 16 / 16], m![H / 16 % 16]>()
        .to_trf();
    device.main.begin(compensated.view())
        .fetch::<m![FfnRow, FfnPartial, H / 256], m![H / 16 % 16]>()
        .collect::<m![FfnRow, FfnPartial, H / 256], m![H / 16 % 16]>()
        .contract_outer::<m![FfnRow, FfnPartial, H / 256], m![H / 16 % 16], _, _, _>(&scales_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![FfnRow]>()
        .contract_lane::<m![FfnRow], m![1 # 8]>(LaneMode::Interleaved)
        .transpose::<m![FfnRow / 2], m![FfnRow % 2 # 8]>()
        .commit_trim::<m![FfnRow % 2]>()
        .commit_view(output.view_mut());
}

pub(crate) fn project_up_and_gate(
    device: &mut Device,
    x_trf: &TrfTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnTerm, H]>,
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
    let x: DmTensorView<'_, f8e5m2, Chip, UpGateCluster, UpGateSlices, m![FfnTerm, H]> = unsafe { x.view().reshape() };
    let x_scale: DmTensorView<'_, f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> = unsafe { x_scale.view().reshape() };
    let x_trf: TrfTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnTerm, H]> = device
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
    // Each ring receives one segment per column; its row replicas use Switch.
    // The anonymous replica axis denotes equal values, not distinct H rows.
    let x: DmTensor<bf16, Chip, DownCluster, m![Dummy256 / 32, 1 # 4, L / 1920], m![L % 1920]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, DownCluster, m![Dummy256 / 32, 4, L / 1920], m![L % 1920]> = device.main
        .begin(x.view())
        .fetch::<m![1], m![L % 1920]>()
        .switch::<m![Dummy256 / 32, 4, L / 1920], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![L / 16 % 120], m![L % 16]>()
        .commit_trim::<m![L % 16]>()
        .commit();
    // Physical slice s holds column s % 8; row replicas share x.
    // Padded row groups are excluded by the typed consumer mapping below.
    let x: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![L % 1920]> = unsafe { x.reshape() };
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
    // Decode only into the DPE stream. The stored tensor has one F32 per block.
    let partials: DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 60, L / 16 % 120]> = device.main
        .begin(down_weight_fp8.view())
        .fetch::<m![H % 60], m![L % 1920]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 60, L / 16 % 120], m![L % 16]>()
        .contract_outer::<m![H % 60, L / 16 % 120], m![L % 16], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60, L / 16 % 120]>()
        .contract_lane::<m![H % 60, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
        .transpose::<m![H % 60, L / 32 % 60], m![L / 16 % 2 # 8]>()
        .commit_trim::<m![L / 16 % 2]>()
        .commit();

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

type FfnQuantSlices = m![1 # 32, H / 480];

fn broadcast_ffn_quantized(
    device: &mut Device,
    x: &DmTensor<f8e5m2, Chip, m![2], FfnQuantSlices, m![FfnTerm, H % 480]>,
) -> DmTensor<f8e5m2, Chip, m![2], Replicated, m![FfnTerm, H]> {
    let x: DmTensor<f8e5m2, Chip, m![2], m![Dummy256 / 32, 1 # 32], m![FfnTerm, H]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<f8e5m2, Chip, m![2], m![Dummy256 / 32, 32], m![FfnTerm, H]> = device
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
    DmTensor<f8e5m2, Chip, m![2], Replicated, m![FfnTerm, H]>,
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
        .vector_clip(ClipBinaryOpF32::Max, 1.925929944387236e-34f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Exponent mask produces 2^floor(log2(amax)); floor 2^-112 guarantees
    // scale >= 2^-126, and finite BF16 inputs imply scale <= 2^113.
    let scale: DmTensor<f32, Chip, m![2], Slice, m![1 # 8]> = device
        .main.begin(amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7f80_0000i32)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.00006103515625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    let quant_scale: DmTensor<f32, Chip, m![2], FfnQuantSlices, m![1 # 8]> = scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, m![2], FfnQuantSlices, m![1 # 8]> = device
        .sub.begin(quant_scale.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let x: DmTensor<bf16, Chip, m![2], FfnQuantSlices, m![H % 480]> = x.to_dm(&mut device.tdma);
    let hi: DmTensor<f8e5m2, Chip, m![2], FfnQuantSlices, m![H % 480]> = device
        .main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e5m2, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>().commit();
    let hi_vrf: VrfTensor<f32, Chip, m![2], FfnQuantSlices, m![H % 480]> = device
        .sub.begin(hi.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>().to_vrf();
    let lo: DmTensor<f8e5m2, Chip, m![2], FfnQuantSlices, m![H % 480]> = device
        .main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e5m2, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>().commit();
    let sum_terms: DmTensor<f32, Chip, m![2], FfnQuantSlices, m![H % 480]> = device.sub
        .begin(lo.view()).fetch::<m![H / 32 % 15], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &hi_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final().commit_trim::<m![H % 8]>().commit();
    let sum_vrf: VrfTensor<f32, Chip, m![2], FfnQuantSlices, m![H % 480]> = device.sub
        .begin(sum_terms.view()).fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>().to_vrf();
    let third: DmTensor<f8e5m2, Chip, m![2], FfnQuantSlices, m![H % 480]> = device
        .main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e5m2, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>().commit();
    let mut packed: DmTensor<f8e5m2, Chip, m![2], FfnQuantSlices, m![FfnTerm, H % 480]> = DmTensor::new();
    let mut zero: DmTensor<f32, Chip, m![2], FfnQuantSlices, m![H % 480]> = DmTensor::new();
    zero.view_mut().memset(0.0f32, &mut device.sub);
    device.sub.begin(zero.view()).fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>().cast::<f8e5m2, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 4, H % 480]>(3));
    hi.view().to_dm_view(&mut device.tdma,
        packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 4, H % 480]>(0));
    lo.view().to_dm_view(&mut device.tdma,
        packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 4, H % 480]>(1));
    third.view().to_dm_view(&mut device.tdma,
        packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 4, H % 480]>(2));
    (broadcast_ffn_quantized(device, &packed), scale.to_dm(&mut device.tdma))
}
