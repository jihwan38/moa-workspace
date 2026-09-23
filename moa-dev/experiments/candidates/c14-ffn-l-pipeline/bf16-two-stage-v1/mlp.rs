// C14-A: E74 BF16 arithmetic composed with producer-local GeGLU and L-owned down.
// This is an experimental whole-FFN graph, not the adopted runtime.
use furiosa_opt_std::prelude::*;
use crate::Chip;
use crate::axes::{Dummy2, Dummy256, H, L};
use crate::device::layout::{Cluster, Replicated, Slice};

axes![TileL = 7680];
macro_rules! static_passes {
    ($index:ident, [$($value:expr),+], $body:block) => {
        $({ let $index: usize = $value; $body })+
    };
}
const INVSQRT2: f32 = 0.70710678118f32;
type UpGateCluster = m![TileL / 3840];
type UpGateSlices = m![TileL / 16 % 240 # 256];
type DownRows = m![H / 30, 1 # 2];
type DownColumns = m![H / 30, TileL / 1920 % 2];
pub(crate) type DownEpilogueSlices = m![1 # 32, H / 480];
type DownPartial = DmTensor<f32, Chip, Cluster, DownEpilogueSlices, m![TileL / 3840, H % 480]>;

fn reduce_eight_rows_f32(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![1], m![H]>,
    weights: &DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![TileL % 16, H / 16]>,
    scales: &DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![TileL % 16, H / 16]>,
    output: &mut DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![TileL % 16]>,
    row_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![TileL % 16 = 8, H / 16]> = device
        .sub
        .begin(scales.view().tile::<m![TileL % 16], 8, m![TileL % 16 = 8 # 16, H / 16]>(row_offset))
        .fetch::<m![TileL % 16 = 8], m![H / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![TileL % 16 = 8, H / 16 / 8], m![H / 16 % 8]>()
        .to_vrf();

    device.main
        .begin(weights.view().tile::<m![TileL % 16], 8, m![TileL % 16 = 8 # 16, H / 16]>(row_offset))
        .fetch::<m![TileL % 16 = 8, H / 128], m![H / 16 % 8]>()
        .collect::<m![TileL % 16 = 8, H / 128], m![H / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL % 16 = 8, H / 64], m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![TileL % 16 = 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![TileL % 16 = 8 / 4], m![TileL % 16 = 8 % 4 # 16]>()
        .commit_trim::<m![TileL % 16 = 8 % 4]>()
        .commit_view(output.view_mut().tile::<m![TileL % 16], 8, m![TileL % 16 = 8 #{!} 16]>(row_offset));
}

fn project_tile_rows(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![1], m![H]>,
    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    offset: usize,
) -> DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![TileL % 16]> {
    // Rename a contiguous row tile only; preserve the original outer extent.
    let weight = weight.view().tile::<m![L], 7680, m![L = 7680 # 15360, H]>(offset);
    let weight: HbmTensorView<'_, f4e2m1, Chip, m![TileL # 15360, H]> = unsafe { weight.reshape() };
    let packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![TileL % 16, H]> = weight.to_dm(&mut device.tdma);
    let fp8: DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![TileL % 16, H]> = device.main
        .begin(packed.view())
        .fetch::<m![TileL % 16], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![TileL % 16, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let scale = scale.view().tile::<m![L], 7680, m![L = 7680 # 15360, H / 16]>(offset);
    let scale: HbmTensorView<'_, f8e4m3, Chip, m![TileL # 15360, H / 16]> = unsafe { scale.reshape() };
    let scale: DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![TileL % 16, H / 16]> = scale.to_dm(&mut device.tdma);
    let partials: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![TileL % 16, H / 16]> = device.main
        .begin(fp8.view())
        .fetch::<m![TileL % 16], m![H]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![TileL % 16, H / 16], m![H % 16]>()
        .contract_outer::<m![TileL % 16, H / 16], m![H % 16], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![TileL % 16, H / 16]>()
        .contract_lane::<m![TileL % 16, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .transpose::<m![TileL % 16, H / 32], m![H / 16 % 2 # 8]>()
        .commit_trim::<m![H / 16 % 2]>()
        .commit();
    let mut output = DmTensor::new();
    static_passes!(i, [0, 1], {
        reduce_eight_rows_f32(device, x_trf, &partials, &scale, &mut output, 8 * i);
    });
    output
}

fn broadcast_geglu_scale(
    device: &mut Device,
    scale: &HbmTensor<f32, Chip, m![1]>,
) -> VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> {
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
    let scale: DmTensorView<'_, f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> = unsafe { scale.view().reshape() };
    device
        .sub
        .begin(scale)
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

pub(crate) fn geglu(
    device: &mut Device,
    up: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![TileL % 16]>,
    gate: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![TileL % 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![TileL % 16]> {

    let up_global_scale_vrf = broadcast_geglu_scale(device, up_global_scale);

    let up: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![TileL % 16]> = device
        .main
        .begin(up.view())
        .fetch::<m![1], m![TileL % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 2], m![TileL % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL / 4 % 4], m![TileL % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_global_scale_vrf)
        .vector_widen_concat::<m![TileL / 8 % 2], m![TileL % 8]>()
        .vector_final()
        .cast::<bf16, m![TileL % 8 # 16]>()
        .commit_trim::<m![TileL % 8]>()
        .commit();

    let gate_global_scale_vrf = broadcast_geglu_scale(device, gate_global_scale);


    let gate: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![TileL % 16]> = device
        .main
        .begin(gate.view())
        .fetch::<m![1], m![TileL % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 2], m![TileL % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL / 4 % 4], m![TileL % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_global_scale_vrf)
        .vector_widen_concat::<m![TileL / 8 % 2], m![TileL % 8]>()
        .vector_final()
        .cast::<bf16, m![TileL % 8 # 16]>()
        .commit_trim::<m![TileL % 8]>()
        .commit();

    let gelu: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![TileL % 16]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![1], m![TileL % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 2], m![TileL % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL / 4 % 4], m![TileL % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![TileL / 8 % 2], m![TileL % 8]>()
        .vector_final()
        .commit_trim::<m![TileL % 8]>()
        .commit();

    let gelu_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![TileL % 16]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![TileL / 8 % 2], m![TileL % 8]>()
        .collect::<m![TileL / 8 % 2], m![TileL % 8]>()
        .to_vrf();

    device
        .main
        .begin(up.view())
        .fetch::<m![TileL / 8 % 2], m![TileL % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![TileL / 8 % 2], m![TileL % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![TileL / 4 % 4], m![TileL % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![TileL / 8 % 2], m![TileL % 8]>()
        .vector_final()
        .cast::<bf16, m![TileL % 8 # 16]>()
        .commit_trim::<m![TileL % 8]>()
        .commit()
}

fn project_down_tile(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![TileL % 16]>,
    weight: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    offset: usize,
) -> DownPartial {
    // Each cluster keeps its own L half. Gather directly into ring leaders,
    // then replicate along H row groups; there is no full-L canonical tensor.
    let x: DmTensor<bf16, Chip, UpGateCluster, m![Dummy256 / 32, 1 # 16, TileL / 1920 % 2], m![TileL % 1920]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, UpGateCluster, m![Dummy256 / 32, 16, TileL / 1920 % 2], m![TileL % 1920]> = device.main
        .begin(x.view())
        .fetch::<m![1], m![TileL % 1920]>()
        .switch::<m![Dummy256 / 32, 16, TileL / 1920 % 2], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![TileL / 16 % 120], m![TileL % 16]>()
        .commit_trim::<m![TileL % 16]>()
        .commit();
    // Replica coordinate 0..127 becomes the H/30 coordinate. Column and cluster
    // keep the same L ownership. No padded producer slices are reinterpreted.
    let x: DmTensor<bf16, Chip, UpGateCluster, DownColumns, m![TileL % 1920]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<bf16, Chip, UpGateCluster, DownColumns, m![1], m![TileL % 1920]> = device.sub
        .begin(x.view())
        .fetch::<m![TileL / 16 % 120], m![TileL % 16]>()
        .collect::<m![TileL / 16 % 120], m![TileL % 16]>()
        .to_trf();

    // The row pitch remains L/2=7680 packed bytes (and L/16=960 scale bytes).
    // Removing #15360 or #960 here would silently select the wrong HBM rows.
    let weight = weight.view().tile::<m![L], 7680, m![H, L = 7680 # 15360]>(offset);
    let weight: HbmTensorView<'_, f4e2m1, Chip, m![H, TileL # 15360]> = unsafe { weight.reshape() };
    let packed: DmTensor<f4e2m1, Chip, UpGateCluster, DownColumns, m![H % 30, TileL % 1920]> = weight.to_dm(&mut device.tdma);
    let fp8: DmTensor<f8e4m3, Chip, UpGateCluster, DownColumns, m![H % 30, TileL % 1920]> = device.main
        .begin(packed.view())
        .fetch::<m![H % 30], m![TileL % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 30, TileL / 32 % 60], m![TileL % 32]>()
        .commit_trim::<m![TileL % 32]>()
        .commit();
    let scale = scale.view().tile::<m![L / 16], 480, m![H, L / 16 = 480 # 960]>(offset / 16);
    let scale: HbmTensorView<'_, f8e4m3, Chip, m![H, TileL / 16 # 960]> = unsafe { scale.reshape() };
    let scales: DmTensor<f8e4m3, Chip, UpGateCluster, DownColumns, m![H % 30, TileL / 16 % 120]> = scale.to_dm(&mut device.tdma);
    let partials: DmTensor<f32, Chip, UpGateCluster, DownColumns, m![H % 30, TileL / 16 % 120]> = device.main
        .begin(fp8.view())
        .fetch::<m![H % 30], m![TileL % 1920]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 30, TileL / 16 % 120], m![TileL % 16]>()
        .contract_outer::<m![H % 30, TileL / 16 % 120], m![TileL % 16], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 30, TileL / 16 % 120]>()
        .contract_lane::<m![H % 30, TileL / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
        .transpose::<m![H % 30, TileL / 32 % 60], m![TileL / 16 % 2 # 8]>()
        .commit_trim::<m![TileL / 16 % 2]>()
        .commit();
    let mut down: DmTensor<f32, Chip, UpGateCluster, DownRows, m![H % 30]> = DmTensor::new();
    static_passes!(i, [0, 1, 2, 3, 4], {
        let scale_vrf: VrfTensor<f32, Chip, UpGateCluster, DownColumns, m![H % 30 = 6, TileL / 16 % 120]> = device.sub
            .begin(scales.view().tile::<m![H % 30], 6, m![H % 30 = 6 # 30, TileL / 16 % 120]>(6 * i))
            .fetch::<m![H % 30 = 6], m![TileL / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 30 = 6, TileL / 128 % 15], m![TileL / 16 % 8]>()
            .to_vrf();
        device.main
            .begin(partials.view().tile::<m![H % 30], 6, m![H % 30 = 6 # 30, TileL / 16 % 120]>(6 * i))
            .fetch::<m![H % 30 = 6, TileL / 128 % 15], m![TileL / 16 % 8]>()
            .collect::<m![H % 30 = 6, TileL / 128 % 15], m![TileL / 16 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 30 = 6, TileL / 64 % 30], m![TileL / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<TileL, m![H % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 30 = 6]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .transpose::<m![H % 30 = 6 / 2], m![H % 30 = 6 % 2 # 8]>()
            .commit_trim::<m![H % 30 = 6 % 2]>()
            .commit_view(down.view_mut().tile::<m![H % 30], 6, m![H % 30 = 6 #{!} 30]>(6 * i));
    });
    // Only H partials cross clusters. Typed DMA preserves the TileL/3840 axis
    // as a two-element reduction axis in the epilogue's local element mapping.
    down.to_dm(&mut device.tdma)
}

fn finish_down(device: &mut Device, first: &DownPartial, second: &DownPartial)
    -> DmTensor<bf16, Chip, Cluster, DownEpilogueSlices, m![H % 480]>
{
    let second: VrfTensor<f32, Chip, Cluster, DownEpilogueSlices, m![H / 8 % 60, TileL / 3840, H % 8]> = device.sub
        .begin(second.view())
        .fetch::<m![H / 8 % 60, TileL / 3840], m![H % 8]>()
        .collect::<m![H / 8 % 60, TileL / 3840], m![H % 8]>()
        .to_vrf();
    device.main
        .begin(first.view())
        .fetch::<m![H / 8 % 60, TileL / 3840], m![H % 8]>()
        .collect::<m![H / 8 % 60, TileL / 3840], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120, TileL / 3840], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &second)
        .vector_intra_slice_reduce::<TileL, m![H / 4 % 120], m![H % 4]>(IntraSliceReduceOpF32::Add)
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
    // Replicated values may safely acquire the producer's row coordinates.
    let x: DmTensor<bf16, Chip, m![Dummy2], m![Dummy256 / 16, 1 # 16], m![H]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, m![Dummy2], m![Dummy256 / 16, 16], m![H]> = device.main
        .begin(x.view())
        .fetch::<m![1], m![H]>()
        .switch::<m![Dummy256 / 16, 16], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 16 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let x: DmTensorView<'_, bf16, Chip, UpGateCluster, UpGateSlices, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![1], m![H]> = device.sub
        .begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let up0 = project_tile_rows(device, &x_trf, up_weight_packed, up_weight_scale, 0);
    let gate0 = project_tile_rows(device, &x_trf, gate_weight_packed, gate_weight_scale, 0);
    let activation0 = geglu(device, up0, gate0, up_global_scale, gate_global_scale);
    let down0 = project_down_tile(device, &activation0, down_weight_packed, down_weight_scale, 0);

    // The second producer is not an input dependency of down0. Both large stages
    // are visible in one static graph, so DMA/compute overlap can be scheduled.
    let up1 = project_tile_rows(device, &x_trf, up_weight_packed, up_weight_scale, 7680);
    let gate1 = project_tile_rows(device, &x_trf, gate_weight_packed, gate_weight_scale, 7680);
    let activation1 = geglu(device, up1, gate1, up_global_scale, gate_global_scale);
    let down1 = project_down_tile(device, &activation1, down_weight_packed, down_weight_scale, 7680);
    let down = finish_down(device, &down0, &down1);
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

