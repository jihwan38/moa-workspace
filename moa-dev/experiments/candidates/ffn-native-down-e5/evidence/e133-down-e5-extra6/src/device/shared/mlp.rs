// E128: native dot/scale/reduce fusion with E127 M03 ownership and materialized weights.
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

axes![FfnTerm = 3, FfnSlice = 256, FfnRow = 30, TileL = 7680];

const INVSQRT2: f32 = 0.70710678118f32;

// Four adjacent native slices compact into one 120-value GeGLU packet.
// Dummy2 continues to own the same L half throughout GeGLU and down.
type LocalRows = m![TileL / 120, 1 # 4];
type DownRows = m![H / 60, 1 # 4];
type DownColumns = m![H / 60, TileL / 1920];
pub(crate) type DownEpilogueSlices = m![1 # 32, H / 480];
type DownPartial = DmTensor<f32, Chip, Cluster, DownEpilogueSlices, m![Dummy2, H % 480]>;

pub(crate) type UpGateCluster = m![Dummy2];
pub(crate) type UpGateSlices = m![FfnSlice];

// Local block scales apply to 16-element FP32 dot products. FP4*FP8 weight
// products are BF16-exact, but this changes FP32 reduction order; validate on NPU.
fn project_rows_with_scale_contraction(
    device: &mut Device,
    x_trf: &TrfTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnTerm, H]>,
    x_scale_vrf: &VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]>,
    weights: &DmTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]>,
    scales: &DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]>,
    output: &mut DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow]>,
) {
    // E128: keep block dots in the native stream; only the six final rows reach DM.
    // Preserve ((dot * input_scale) * block_scale), followed by F32 H reduction.
    static_passes!(i, [0, 1, 2, 3, 4], {
        let scale_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow = 6, H / 16]> = device.sub
            .begin(scales.view().tile::<m![FfnRow], 6, m![FfnRow = 6 # 30, H / 16]>(6 * i))
            .fetch::<m![FfnRow = 6], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![FfnRow = 6, H / 16 / 8], m![H / 16 % 8]>().to_vrf();
        device.main
            .begin(weights.view().tile::<m![FfnRow], 6, m![FfnRow = 6 # 30, H]>(6 * i))
            .fetch::<m![FfnRow = 6], m![H]>()
            .collect::<m![FfnRow = 6, H / 32], m![H % 32]>()
            .contract_outer::<m![FfnRow = 6, H / 64, FfnTerm], m![H % 64], _, _, _>(x_trf)
            .contract_packet::<m![H / 16 % 4]>()
            .contract_time::<m![FfnRow = 6, H / 64]>()
            .contract_lane::<m![FfnRow = 6, H / 64], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![H / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), x_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
            .vector_intra_slice_reduce::<H, m![FfnRow = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>().vector_final()
            .transpose::<m![FfnRow = 6 / 2], m![FfnRow = 6 % 2 # 8]>().commit_trim::<m![FfnRow = 6 % 2]>()
            .commit_view(output.view_mut().tile::<m![FfnRow], 6, m![FfnRow = 6 #{!} 30]>(6 * i));
    });
}

pub(crate) fn project_up_and_gate(
    device: &mut Device,
    x: &DmTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![FfnTerm, H]>,
    x_scale: &DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<bf16, Chip, UpGateCluster, LocalRows, m![TileL % 120]>,
    DmTensor<bf16, Chip, UpGateCluster, LocalRows, m![TileL % 120]>,
) {
    let mut up: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow]> = DmTensor::new();

    let up_weight_packed: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, FfnSlice, FfnRow, H]> = unsafe { up_weight_packed.view().reshape() };
    let up_weight_packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]> =
        up_weight_packed.to_dm(&mut device.tdma);
    let up_weight_packed: DmTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]> = device.main
        .begin(up_weight_packed.view()).fetch::<m![FfnRow], m![H]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![FfnRow, H / 32], m![H % 32]>().commit_trim::<m![H % 32]>().commit();
    let up_weight_scale: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, FfnSlice, FfnRow, H / 16]> = unsafe { up_weight_scale.view().reshape() };

    let up_weight_scale: DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> =
        up_weight_scale.to_dm(&mut device.tdma);

    let x_trf: TrfTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnTerm, H]> = device
        .sub.begin(x.view())
        .fetch::<m![FfnTerm], m![H]>()
        .collect::<m![FfnTerm, H / 32], m![H % 32]>().to_trf();
    let x_scale_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> = device
        .sub.begin(x_scale.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    project_rows_with_scale_contraction(device, &x_trf, &x_scale_vrf, &up_weight_packed, &up_weight_scale, &mut up);

    let mut gate: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow]> = DmTensor::new();

    let gate_weight_packed: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, FfnSlice, FfnRow, H]> = unsafe { gate_weight_packed.view().reshape() };
    let gate_weight_packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]> =
        gate_weight_packed.to_dm(&mut device.tdma);
    let gate_weight_packed: DmTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]> = device.main
        .begin(gate_weight_packed.view()).fetch::<m![FfnRow], m![H]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![FfnRow, H / 32], m![H % 32]>().commit_trim::<m![H % 32]>().commit();
    let gate_weight_scale: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, FfnSlice, FfnRow, H / 16]> = unsafe { gate_weight_scale.view().reshape() };

    let gate_weight_scale: DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> =
        gate_weight_scale.to_dm(&mut device.tdma);

    let x_trf: TrfTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnTerm, H]> = device
        .sub.begin(x.view())
        .fetch::<m![FfnTerm], m![H]>()
        .collect::<m![FfnTerm, H / 32], m![H % 32]>().to_trf();
    let x_scale_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> = device
        .sub.begin(x_scale.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    project_rows_with_scale_contraction(device, &x_trf, &x_scale_vrf, &gate_weight_packed, &gate_weight_scale, &mut gate);

    // Compact four adjacent 30-row F32 producers within the same cluster.
    // Only then cast; a 30-row BF16 producer would violate 8-byte commit alignment.
    let up: DmTensor<f32, Chip, UpGateCluster, m![TileL / 30], m![TileL % 30]> = unsafe { up.reshape() };
    let gate: DmTensor<f32, Chip, UpGateCluster, m![TileL / 30], m![TileL % 30]> = unsafe { gate.reshape() };
    let up: DmTensor<f32, Chip, UpGateCluster, LocalRows, m![TileL % 120]> = up.to_dm(&mut device.tdma);
    let gate: DmTensor<f32, Chip, UpGateCluster, LocalRows, m![TileL % 120]> = gate.to_dm(&mut device.tdma);
    (cast_gathered_rows(device, &up), cast_gathered_rows(device, &gate))
}

fn cast_gathered_rows(device: &mut Device, x: &DmTensor<f32, Chip, UpGateCluster, LocalRows, m![TileL % 120]>)
    -> DmTensor<bf16, Chip, UpGateCluster, LocalRows, m![TileL % 120]> {
    device.main.begin(x.view()).fetch::<m![TileL / 4 % 30], m![TileL % 4 # 8]>()
        .collect::<m![TileL / 4 % 30], m![TileL % 4 # 8]>()
        .cast::<bf16, m![TileL % 4 # 16]>().commit_trim::<m![TileL % 4]>().commit()
}

fn broadcast_geglu_scale(
    device: &mut Device,
    scale: &HbmTensor<f32, Chip, m![1]>,
) -> VrfTensor<f32, Chip, UpGateCluster, LocalRows, m![1 # 8]> {
    let scale: DmTensor<f32, Chip, UpGateCluster, m![Dummy256 / 16, 1 # 16], m![1 # 8]> = scale.to_dm(&mut device.tdma);
    let scale: DmTensor<f32, Chip, UpGateCluster, m![Dummy256 / 16, 16], m![1 # 8]> = device
        .main
        .begin(scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Dummy256 / 16, 16], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 16 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let scale: DmTensor<f32, Chip, UpGateCluster, Replicated, m![1 # 8]> = unsafe { scale.reshape() };
    let scale: DmTensorView<'_, f32, Chip, UpGateCluster, LocalRows, m![1 # 8]> = unsafe { scale.view().reshape() };
    device
        .sub
        .begin(scale)
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

pub(crate) fn geglu(
    device: &mut Device,
    up: DmTensor<bf16, Chip, UpGateCluster, LocalRows, m![TileL % 120]>,
    gate: DmTensor<bf16, Chip, UpGateCluster, LocalRows, m![TileL % 120]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, UpGateCluster, LocalRows, m![TileL % 120]> {

    let up_global_scale_vrf = broadcast_geglu_scale(device, up_global_scale);

    let up: DmTensor<bf16, Chip, UpGateCluster, LocalRows, m![TileL % 120]> = device
        .main
        .begin(up.view())
        .fetch::<m![1], m![TileL % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 15], m![TileL % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL / 4 % 30], m![TileL % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_global_scale_vrf)
        .vector_widen_concat::<m![TileL / 8 % 15], m![TileL % 8]>()
        .vector_final()
        .cast::<bf16, m![TileL % 8 # 16]>()
        .commit_trim::<m![TileL % 8]>()
        .commit();

    let gate_global_scale_vrf = broadcast_geglu_scale(device, gate_global_scale);


    let gate: DmTensor<bf16, Chip, UpGateCluster, LocalRows, m![TileL % 120]> = device
        .main
        .begin(gate.view())
        .fetch::<m![1], m![TileL % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 15], m![TileL % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL / 4 % 30], m![TileL % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_global_scale_vrf)
        .vector_widen_concat::<m![TileL / 8 % 15], m![TileL % 8]>()
        .vector_final()
        .cast::<bf16, m![TileL % 8 # 16]>()
        .commit_trim::<m![TileL % 8]>()
        .commit();

    let gelu: DmTensor<f32, Chip, UpGateCluster, LocalRows, m![TileL % 120]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![1], m![TileL % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 15], m![TileL % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL / 4 % 30], m![TileL % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![TileL / 8 % 15], m![TileL % 8]>()
        .vector_final()
        .commit_trim::<m![TileL % 8]>()
        .commit();

    let gelu_vrf: VrfTensor<f32, Chip, UpGateCluster, LocalRows, m![TileL % 120]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![TileL / 8 % 15], m![TileL % 8]>()
        .collect::<m![TileL / 8 % 15], m![TileL % 8]>()
        .to_vrf();

    device
        .main
        .begin(up.view())
        .fetch::<m![TileL / 8 % 15], m![TileL % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 15], m![TileL % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL / 4 % 30], m![TileL % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![TileL / 8 % 15], m![TileL % 8]>()
        .vector_final()
        .cast::<bf16, m![TileL % 8 # 16]>()
        .commit_trim::<m![TileL % 8]>()
        .commit()
}

// E133: quantize one compact 7680-value L half per cluster, before replication.
// Dummy2 remains a real owner axis; the two clusters have independent scales.
type DownQuantSlices = m![1 # 16, TileL / 480];

fn broadcast_down_quantized(
    device: &mut Device,
    packed: &DmTensor<f8e5m2, Chip, UpGateCluster, DownQuantSlices, m![FfnTerm, TileL % 480]>,
) -> DmTensor<f8e5m2, Chip, UpGateCluster, DownColumns, m![FfnTerm, TileL % 1920]> {
    let x: DmTensor<f8e5m2, Chip, UpGateCluster, Slice, m![FfnTerm, TileL]> = packed.to_dm(&mut device.tdma);
    let x: DmTensor<f8e5m2, Chip, UpGateCluster, m![Dummy256 / 32, 1 # 8, TileL / 1920], m![FfnTerm, TileL % 1920]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<f8e5m2, Chip, UpGateCluster, m![Dummy256 / 32, 8, TileL / 1920], m![FfnTerm, TileL % 1920]> = device.main
        .begin(x.view())
        .fetch::<m![FfnTerm], m![TileL % 1920]>()
        .switch::<m![Dummy256 / 32, 8, TileL / 1920], m![FfnTerm]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![FfnTerm, TileL / 32 % 60], m![TileL % 32]>()
        .commit_trim::<m![TileL % 32]>()
        .commit();
    unsafe { x.reshape() }
}

fn prepare_down_fp8_split(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, UpGateCluster, Slice, m![TileL]>,
) -> (
    DmTensor<f8e5m2, Chip, UpGateCluster, DownColumns, m![FfnTerm, TileL % 1920]>,
    DmTensor<f32, Chip, UpGateCluster, DownColumns, m![1 # 8]>,
) {
    // Amax on the canonical input avoids an expensive inter-slice reduction.
    // i32 reinterpret makes the absolute-value mask a non-NaN immediate.
    let amax: DmTensor<f32, Chip, UpGateCluster, Slice, m![1 # 8]> = device
        .main.begin(x.view())
        .fetch::<m![TileL / 16], m![TileL % 16]>().fetch_cast::<f32>()
        .collect::<m![TileL / 8], m![TileL % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![TileL / 4], m![TileL % 4]>()
        .vector_intra_slice_reduce::<TileL, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.925929944387236e-34f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Exponent mask produces 2^floor(log2(amax)); floor 2^-112 guarantees
    // scale >= 2^-126, and finite BF16 inputs imply scale <= 2^113.
    let scale: DmTensor<f32, Chip, UpGateCluster, Slice, m![1 # 8]> = device
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
    let quant_scale: DmTensor<f32, Chip, UpGateCluster, DownQuantSlices, m![1 # 8]> = scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, UpGateCluster, DownQuantSlices, m![1 # 8]> = device
        .sub.begin(quant_scale.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let x: DmTensor<bf16, Chip, UpGateCluster, DownQuantSlices, m![TileL % 480]> = x.to_dm(&mut device.tdma);
    let hi: DmTensor<f8e5m2, Chip, UpGateCluster, DownQuantSlices, m![TileL % 480]> = device
        .main.begin(x.view())
        .fetch::<m![TileL / 16 % 30], m![TileL % 16]>().fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 60], m![TileL % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL / 4 % 120], m![TileL % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![TileL / 8 % 60], m![TileL % 8]>().vector_final()
        .cast::<f8e5m2, m![TileL % 8 # 32]>()
        .commit_trim::<m![TileL % 8]>().commit();
    let hi_vrf: VrfTensor<f32, Chip, UpGateCluster, DownQuantSlices, m![TileL % 480]> = device
        .sub.begin(hi.view())
        .fetch::<m![TileL / 32 % 15], m![TileL % 32]>().fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 60], m![TileL % 8]>().to_vrf();
    let lo: DmTensor<f8e5m2, Chip, UpGateCluster, DownQuantSlices, m![TileL % 480]> = device
        .main.begin(x.view())
        .fetch::<m![TileL / 16 % 30], m![TileL % 16]>().fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 60], m![TileL % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL / 4 % 120], m![TileL % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![TileL / 8 % 60], m![TileL % 8]>().vector_final()
        .cast::<f8e5m2, m![TileL % 8 # 32]>()
        .commit_trim::<m![TileL % 8]>().commit();
    let sum_terms: DmTensor<f32, Chip, UpGateCluster, DownQuantSlices, m![TileL % 480]> = device.sub
        .begin(lo.view()).fetch::<m![TileL / 32 % 15], m![TileL % 32]>().fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 60], m![TileL % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL / 4 % 120], m![TileL % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &hi_vrf)
        .vector_widen_concat::<m![TileL / 8 % 60], m![TileL % 8]>().vector_final().commit_trim::<m![TileL % 8]>().commit();
    let sum_vrf: VrfTensor<f32, Chip, UpGateCluster, DownQuantSlices, m![TileL % 480]> = device.sub
        .begin(sum_terms.view()).fetch::<m![TileL / 8 % 60], m![TileL % 8]>()
        .collect::<m![TileL / 8 % 60], m![TileL % 8]>().to_vrf();
    let third: DmTensor<f8e5m2, Chip, UpGateCluster, DownQuantSlices, m![TileL % 480]> = device
        .main.begin(x.view())
        .fetch::<m![TileL / 16 % 30], m![TileL % 16]>().fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 60], m![TileL % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL / 4 % 120], m![TileL % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_widen_concat::<m![TileL / 8 % 60], m![TileL % 8]>().vector_final()
        .cast::<f8e5m2, m![TileL % 8 # 32]>()
        .commit_trim::<m![TileL % 8]>().commit();
    let mut packed: DmTensor<f8e5m2, Chip, UpGateCluster, DownQuantSlices, m![FfnTerm, TileL % 480]> = DmTensor::new();
    // Exact 480B stream copy: typed DMA to a padded term view expands its source extent.
    device.sub.begin(hi.view())
        .fetch::<m![TileL / 32 % 15], m![TileL % 32]>()
        .collect::<m![TileL / 32 % 15], m![TileL % 32]>()
        .commit_trim::<m![TileL % 32]>()
        .commit_view(packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 3, TileL % 480]>(0));
    // Exact 480B stream copy: typed DMA to a padded term view expands its source extent.
    device.sub.begin(lo.view())
        .fetch::<m![TileL / 32 % 15], m![TileL % 32]>()
        .collect::<m![TileL / 32 % 15], m![TileL % 32]>()
        .commit_trim::<m![TileL % 32]>()
        .commit_view(packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 3, TileL % 480]>(1));
    // Exact 480B stream copy: typed DMA to a padded term view expands its source extent.
    device.sub.begin(third.view())
        .fetch::<m![TileL / 32 % 15], m![TileL % 32]>()
        .collect::<m![TileL / 32 % 15], m![TileL % 32]>()
        .commit_trim::<m![TileL % 32]>()
        .commit_view(packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 3, TileL % 480]>(2));
    (broadcast_down_quantized(device, &packed), scale.to_dm(&mut device.tdma))
}

fn project_down_tile(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, UpGateCluster, LocalRows, m![TileL % 120]>,
    weight: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DownPartial {
    // Compact only this cluster's L half; no full-L or cross-cluster gather.
    let x: DmTensor<bf16, Chip, UpGateCluster, Slice, m![TileL]> = x.to_dm(&mut device.tdma);
    let (x, x_scale) = prepare_down_fp8_split(device, &x);
    let x_scale_vrf: VrfTensor<f32, Chip, UpGateCluster, DownColumns, m![1 # 8]> = device.sub
        .begin(x_scale.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let x_trf: TrfTensor<f8e5m2, Chip, UpGateCluster, DownColumns, m![1], m![FfnTerm, TileL % 1920]> = device.sub
        .begin(x.view())
        .fetch::<m![FfnTerm, TileL / 32 % 60], m![TileL % 32]>()
        .collect::<m![FfnTerm, TileL / 32 % 60], m![TileL % 32]>()
        .to_trf();

    // HBM remains [H,L], with contiguous L halves owned by separate clusters.
    // Packed row pitch is 7680 bytes; every local K segment is 960 bytes.
    let weight: HbmTensorView<'_, f4e2m1, Chip, m![H, Dummy2, TileL]> = unsafe { weight.view().reshape() };
    let packed: DmTensor<f4e2m1, Chip, UpGateCluster, DownColumns, m![H % 60, TileL % 1920]> = weight.to_dm(&mut device.tdma);
    let scale: HbmTensorView<'_, f8e4m3, Chip, m![H, Dummy2, TileL / 16]> = unsafe { scale.view().reshape() };
    let scales: DmTensor<f8e4m3, Chip, UpGateCluster, DownColumns, m![H % 60, TileL / 16 % 120]> = scale.to_dm(&mut device.tdma);
    // Direct FP4 lookup feeds native E5; no full decoded-weight DM tensor.
    // Sum the three same-scale terms, restore the cluster's input scale in F32,
    // then preserve the existing F32 block-scale/reduction path below.
    let partials: DmTensor<f32, Chip, UpGateCluster, DownColumns, m![H % 60, TileL / 16 % 120]> = device.main
        .begin(packed.view())
        .fetch::<m![H % 60], m![TileL % 1920]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![H % 60, TileL / 32 % 60], m![TileL % 32]>()
        .contract_outer::<m![H % 60, TileL / 64 % 30, FfnTerm], m![TileL % 64], _, _, _>(&x_trf)
        .contract_packet::<m![TileL / 16 % 4]>()
        .contract_time::<m![H % 60, TileL / 64 % 30]>()
        .contract_lane::<m![H % 60, TileL / 64 % 30], m![TileL / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![TileL / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_scale_vrf)
        .vector_widen_pad::<m![TileL / 16 % 4 # 8]>().vector_final()
        .commit_trim::<m![TileL / 16 % 4]>().commit();
    let mut down: DmTensor<f32, Chip, UpGateCluster, DownRows, m![H % 60]> = DmTensor::new();
    static_passes!(i, [0, 1, 2, 3, 4], {
        let scale_vrf: VrfTensor<f32, Chip, UpGateCluster, DownColumns, m![H % 60 = 12, TileL / 16 % 120]> = device.sub
            .begin(scales.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, TileL / 16 % 120]>(12 * i))
            .fetch::<m![H % 60 = 12], m![TileL / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, TileL / 128 % 15], m![TileL / 16 % 8]>()
            .to_vrf();
        device.main
            .begin(partials.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, TileL / 16 % 120]>(12 * i))
            .fetch::<m![H % 60 = 12, TileL / 128 % 15], m![TileL / 16 % 8]>()
            .collect::<m![H % 60 = 12, TileL / 128 % 15], m![TileL / 16 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 12, TileL / 64 % 30], m![TileL / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<TileL, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .transpose::<m![H % 60 = 12 / 2], m![H % 60 = 12 % 2 # 8]>()
            .commit_trim::<m![H % 60 = 12 % 2]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12 * i));
    });
    // Only H partials cross clusters. Typed DMA preserves the Dummy2 (logical L-high) axis
    // as a two-element reduction axis in the epilogue's local element mapping.
    down.to_dm(&mut device.tdma)
}

fn finish_down(device: &mut Device, partial: &DownPartial)
    -> DmTensor<bf16, Chip, Cluster, DownEpilogueSlices, m![H % 480]>
{
    device.main
        .begin(partial.view())
        .fetch::<m![H / 8 % 60, Dummy2], m![H % 8]>()
        .collect::<m![H / 8 % 60, Dummy2], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120, Dummy2], m![H % 4]>()
        .vector_intra_slice_reduce::<Dummy2, m![H / 4 % 120], m![H % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
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
    let x: DmTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![FfnTerm, H]> = unsafe { x.reshape() };
    let x_scale: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> = unsafe { x_scale.reshape() };
    let (up, gate) = project_up_and_gate(
        device,
        &x,
        &x_scale,
        up_weight_packed,
        gate_weight_packed,
        up_weight_scale,
        gate_weight_scale,
    );
    let x = geglu(device, up, gate, up_global_scale, gate_global_scale);
    let partial = project_down_tile(device, &x, down_weight_packed, down_weight_scale);
    let down = finish_down(device, &partial);

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
    hi.view().to_dm_view(&mut device.tdma,
        packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 3, H % 480]>(0));
    lo.view().to_dm_view(&mut device.tdma,
        packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 3, H % 480]>(1));
    third.view().to_dm_view(&mut device.tdma,
        packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 3, H % 480]>(2));
    (broadcast_ffn_quantized(device, &packed), scale.to_dm(&mut device.tdma))
}
