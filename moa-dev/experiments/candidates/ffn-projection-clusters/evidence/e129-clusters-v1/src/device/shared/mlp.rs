// E129: projection-owned clusters, compact FP4 direct native streams, actual M03 consumer redistribution.
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

axes![FfnTerm = 3, FfnSlice = 256, FfnRow = 60, TileL = 7680, Projection = 2];

const INVSQRT2: f32 = 0.70710678118f32;

// GeGLU/down retain E128 L-half ownership; Projection names the DIFFERENT
// upstream physical-cluster ownership and is converted by actual TDMA below.
type LocalRows = m![TileL / 120, 1 # 4];
type DownRows = m![H / 60, 1 # 4];
type DownColumns = m![H / 60, TileL / 1920];
pub(crate) type DownEpilogueSlices = m![1 # 32, H / 480];
type DownPartial = DmTensor<f32, Chip, Cluster, DownEpilogueSlices, m![Dummy2, H % 480]>;

pub(crate) type UpGateCluster = m![Dummy2];
pub(crate) type UpGateSlices = m![FfnSlice];

// The public/consumer contract keeps Dummy2=L-half. Only the direct producer
// below uses Projection=physical cluster (0=up,1=gate), l=60*slice+row.
type ProjectionClusters = m![Projection];

fn project_cluster_rows(
    device: &mut Device,
    x_trf: &TrfTensor<f8e5m2, Chip, ProjectionClusters, UpGateSlices, m![1], m![FfnTerm, H]>,
    x_scale_vrf: &VrfTensor<f32, Chip, ProjectionClusters, UpGateSlices, m![1 # 8]>,
    weights: &DmTensor<f4e2m1, Chip, ProjectionClusters, UpGateSlices, m![FfnRow, H]>,
    scales: &DmTensor<f8e4m3, Chip, ProjectionClusters, UpGateSlices, m![FfnRow, H / 16]>,
    output: &mut DmTensor<f32, Chip, ProjectionClusters, UpGateSlices, m![FfnRow]>,
) {
    // No full FP8 weight or block-partial DM exists in this path. Each of ten
    // Main streams decodes six compact rows and consumes them immediately.
    // Preserve ((dot * x_scale) * block_scale), F32 H reduction, then row commit.
    static_passes!(i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], {
        let scale_vrf: VrfTensor<f32, Chip, ProjectionClusters, UpGateSlices, m![FfnRow = 6, H / 16]> = device.sub
            .begin(scales.view().tile::<m![FfnRow], 6, m![FfnRow = 6 # 60, H / 16]>(6 * i))
            .fetch::<m![FfnRow = 6], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![FfnRow = 6, H / 16 / 8], m![H / 16 % 8]>().to_vrf();
        device.main
            .begin(weights.view().tile::<m![FfnRow], 6, m![FfnRow = 6 # 60, H]>(6 * i))
            .fetch::<m![FfnRow = 6], m![H]>()
            .fetch_table_lookup::<f8e5m2>()
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
            .commit_view(output.view_mut().tile::<m![FfnRow], 6, m![FfnRow = 6 #{!} 60]>(6 * i));
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
    let mut weights: DmTensor<f4e2m1, Chip, ProjectionClusters, UpGateSlices, m![FfnRow, H]> = DmTensor::new();
    let up_hbm: HbmTensorView<'_, f4e2m1, Chip, m![FfnSlice, FfnRow, H]> = unsafe { up_weight_packed.view().reshape() };
    let gate_hbm: HbmTensorView<'_, f4e2m1, Chip, m![FfnSlice, FfnRow, H]> = unsafe { gate_weight_packed.view().reshape() };
    // Cluster tiling changes the hardware destination mask/offset, not the HBM
    // shape. Each source has exactly L*H/2 bytes; no padded Element destination.
    up_hbm.to_dm_view(&mut device.tdma,
        weights.view_mut().cluster_tile::<m![Projection], 1, m![Projection = 1 #{!} 2]>(0));
    gate_hbm.to_dm_view(&mut device.tdma,
        weights.view_mut().cluster_tile::<m![Projection], 1, m![Projection = 1 #{!} 2]>(1));
    let mut scales: DmTensor<f8e4m3, Chip, ProjectionClusters, UpGateSlices, m![FfnRow, H / 16]> = DmTensor::new();
    let up_scale_hbm: HbmTensorView<'_, f8e4m3, Chip, m![FfnSlice, FfnRow, H / 16]> = unsafe { up_weight_scale.view().reshape() };
    let gate_scale_hbm: HbmTensorView<'_, f8e4m3, Chip, m![FfnSlice, FfnRow, H / 16]> = unsafe { gate_weight_scale.view().reshape() };
    up_scale_hbm.to_dm_view(&mut device.tdma,
        scales.view_mut().cluster_tile::<m![Projection], 1, m![Projection = 1 #{!} 2]>(0));
    gate_scale_hbm.to_dm_view(&mut device.tdma,
        scales.view_mut().cluster_tile::<m![Projection], 1, m![Projection = 1 #{!} 2]>(1));

    // Every physical slice already has the same actual quantized input. Relabel
    // the two cluster positions without moving bytes; no projection-specific x.
    let x: DmTensorView<'_, f8e5m2, Chip, ProjectionClusters, UpGateSlices, m![FfnTerm, H]> = unsafe { x.view().reshape() };
    let x_scale: DmTensorView<'_, f32, Chip, ProjectionClusters, UpGateSlices, m![1 # 8]> = unsafe { x_scale.view().reshape() };
    let x_trf: TrfTensor<f8e5m2, Chip, ProjectionClusters, UpGateSlices, m![1], m![FfnTerm, H]> = device.sub
        .begin(x).fetch::<m![FfnTerm], m![H]>()
        .collect::<m![FfnTerm, H / 32], m![H % 32]>().to_trf();
    let x_scale_vrf: VrfTensor<f32, Chip, ProjectionClusters, UpGateSlices, m![1 # 8]> = device.sub
        .begin(x_scale).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let mut rows: DmTensor<f32, Chip, ProjectionClusters, UpGateSlices, m![FfnRow]> = DmTensor::new();
    project_cluster_rows(device, &x_trf, &x_scale_vrf, &weights, &scales, &mut rows);

    // Sixty F32 rows=240B and sixty BF16 rows=120B per slice. Each cast commit
    // writes four BF16=8B. Do not attempt a six-BF16=12B direct row commit.
    let rows: DmTensor<bf16, Chip, ProjectionClusters, UpGateSlices, m![FfnRow]> = device.main
        .begin(rows.view()).fetch::<m![FfnRow / 4], m![FfnRow % 4 # 8]>()
        .collect::<m![FfnRow / 4], m![FfnRow % 4 # 8]>()
        .cast::<bf16, m![FfnRow % 4 # 16]>().commit_trim::<m![FfnRow % 4]>().commit();

    // Select the REAL source cluster, relabel its in-cluster row coordinate as
    // l=60*slice+row, then TDMA redistributes L halves to the consumer clusters.
    // The Projection axis is never reinterpreted as the L-half axis.
    // Rename the FULL owned tensor before selecting either cluster, so the
    // tiled view retains its original offset/base_map without a later reshape.
    let rows: DmTensor<bf16, Chip, ProjectionClusters, m![L / 60], m![L % 60]> = unsafe { rows.reshape() };
    let up_source = rows.view().cluster_tile::<m![Projection], 1, m![1 # 2]>(0);
    let gate_source = rows.view().cluster_tile::<m![Projection], 1, m![1 # 2]>(1);
    let mut up: DmTensor<bf16, Chip, m![L / 7680], m![L / 120 % 64, 1 # 4], m![L % 120]> = DmTensor::new();
    let mut gate: DmTensor<bf16, Chip, m![L / 7680], m![L / 120 % 64, 1 # 4], m![L % 120]> = DmTensor::new();
    up_source.to_dm_view(&mut device.tdma, up.view_mut());
    gate_source.to_dm_view(&mut device.tdma, gate.view_mut());
    // After the actual transfer, l=7680*cluster+120*group+row is precisely the
    // existing Dummy2/TileL consumer coordinate; these final reshapes move no data.
    (unsafe { up.reshape() }, unsafe { gate.reshape() })
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

fn project_down_tile(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, UpGateCluster, LocalRows, m![TileL % 120]>,
    weight: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DownPartial {
    // Compact only this cluster's L half; no full-L or cross-cluster gather.
    let x: DmTensor<bf16, Chip, UpGateCluster, Slice, m![TileL]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, UpGateCluster, m![Dummy256 / 32, 1 # 8, TileL / 1920], m![TileL % 1920]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, UpGateCluster, m![Dummy256 / 32, 8, TileL / 1920], m![TileL % 1920]> = device.main
        .begin(x.view())
        .fetch::<m![1], m![TileL % 1920]>()
        .switch::<m![Dummy256 / 32, 8, TileL / 1920], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![TileL / 16 % 120], m![TileL % 16]>()
        .commit_trim::<m![TileL % 16]>()
        .commit();
    // s = 4*r + k: row group r=0..63 and L-quarter k=0..3.
    let x: DmTensor<bf16, Chip, UpGateCluster, DownColumns, m![TileL % 1920]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<bf16, Chip, UpGateCluster, DownColumns, m![1], m![TileL % 1920]> = device.sub
        .begin(x.view())
        .fetch::<m![TileL / 16 % 120], m![TileL % 16]>()
        .collect::<m![TileL / 16 % 120], m![TileL % 16]>()
        .to_trf();

    // HBM remains [H,L], with contiguous L halves owned by separate clusters.
    // Packed row pitch is 7680 bytes; every local K segment is 960 bytes.
    let weight: HbmTensorView<'_, f4e2m1, Chip, m![H, Dummy2, TileL]> = unsafe { weight.view().reshape() };
    let packed: DmTensor<f4e2m1, Chip, UpGateCluster, DownColumns, m![H % 60, TileL % 1920]> = weight.to_dm(&mut device.tdma);
    let fp8: DmTensor<f8e4m3, Chip, UpGateCluster, DownColumns, m![H % 60, TileL % 1920]> = device.main
        .begin(packed.view())
        .fetch::<m![H % 60], m![TileL % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60, TileL / 32 % 60], m![TileL % 32]>()
        .commit_trim::<m![TileL % 32]>()
        .commit();
    let scale: HbmTensorView<'_, f8e4m3, Chip, m![H, Dummy2, TileL / 16]> = unsafe { scale.view().reshape() };
    let scales: DmTensor<f8e4m3, Chip, UpGateCluster, DownColumns, m![H % 60, TileL / 16 % 120]> = scale.to_dm(&mut device.tdma);
    let partials: DmTensor<f32, Chip, UpGateCluster, DownColumns, m![H % 60, TileL / 16 % 120]> = device.main
        .begin(fp8.view())
        .fetch::<m![H % 60], m![TileL % 1920]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 60, TileL / 16 % 120], m![TileL % 16]>()
        .contract_outer::<m![H % 60, TileL / 16 % 120], m![TileL % 16], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60, TileL / 16 % 120]>()
        .contract_lane::<m![H % 60, TileL / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
        .transpose::<m![H % 60, TileL / 32 % 60], m![TileL / 16 % 2 # 8]>()
        .commit_trim::<m![TileL / 16 % 2]>()
        .commit();
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
