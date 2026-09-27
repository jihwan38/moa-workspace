use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Dummy2, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{Cluster, Replicated};

pub(crate) fn project_query(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], Replicated, m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]> {
    type QueryCluster = m![Qs / 2048];
    type QueryRows = m![Qs / 64 % 32, Qs % 8];


    let x: DmTensorView<'_, bf16, Chip, QueryCluster, QueryRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, QueryCluster, QueryRows, m![1], m![H]> = device
        .sub
        .begin(x)
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, QueryCluster, QueryRows, m![Qs / 8 % 8, H]> = weight.to_dm(&mut device.tdma);
    // Canonical contiguous scale DMA satisfies TDMA's 8-byte alignment.
    // Inverse row Switch routes the matching scale into each raw dot slice.
    // VRF stores one valid scalar in each 32-byte Time slot; padding is unread.
    let weight_scale: DmTensor<bf16, Chip, QueryCluster, m![Qs / 8 % 256], m![Qs % 8]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, QueryCluster, QueryRows, m![Qs / 8 % 8, 1 # 8]> = device
        .sub.begin(weight_scale.view())
        .fetch::<m![Qs % 8], m![1 # 4]>()
        .fetch_cast::<f32>()
        .switch::<QueryRows, m![Qs / 8 % 8]>(SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![Qs / 8 % 8], m![1 # 8]>()
        .to_vrf();
    let contraction: DmTensor<bf16, Chip, QueryCluster, QueryRows, m![Qs / 8 % 8]> = device
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs / 8 % 8, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs / 8 % 8, H / 16], m![H % 16]>()
        .contract_outer::<m![Qs / 8 % 8, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 8 % 8]>()
        .contract_lane::<m![Qs / 8 % 8], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 32 % 2], m![Qs / 8 % 4 # 16]>()
        .commit_trim::<m![Qs / 8 % 4]>()
        .commit();

    // The complete F32 dot is scaled before its first BF16 cast; Switch only moves BF16 values.
    // Switch swaps the low slice coordinate with the local row coordinate.
    let scaled: DmTensor<bf16, Chip, QueryCluster, m![Qs / 8 % 256], m![Qs % 8]> = device
        .main.begin(contraction.view())
        .fetch::<m![Qs / 8 % 8], m![1 # 4]>()
        .switch::<m![Qs / 8 % 256], m![Qs % 8]>(SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![Qs % 8], m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    let output: DmTensor<bf16, Chip, m![Ns / 4], m![Ns % 4, Gs, Ds / 8], m![Ds % 8]> =
        unsafe { scaled.reshape() };
    output.to_dm(&mut device.tdma)
}

type KvCluster = m![Ps / 1024];
type KvRows = m![Ps / 32 % 32, Ps % 8];

fn project_one_kv_matrix(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, KvCluster, KvRows, m![1], m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> {
    let weight_f8: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps / 8 % 4, H]> = weight.to_dm(&mut device.tdma);
    // Canonical contiguous scale DMA satisfies TDMA's 8-byte alignment.
    // Inverse row Switch routes the matching scale into each raw dot slice.
    // VRF stores one valid scalar in each 32-byte Time slot; padding is unread.
    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, KvCluster, KvRows, m![Ps / 8 % 4 # 8, 1 # 8]> = device
        .sub.begin(weight_scale.view())
        .fetch::<m![Ps % 8], m![1 # 4]>()
        .fetch_cast::<f32>()
        .switch::<KvRows, m![Ps / 8 % 4 # 8]>(SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![Ps / 8 % 4 # 8], m![1 # 8]>()
        .to_vrf();
    let contraction: DmTensor<bf16, Chip, KvCluster, KvRows, m![Ps / 8 % 4]> = device
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps / 8 % 4, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps / 8 % 4, H / 16], m![H % 16]>()
        .contract_outer::<m![Ps / 8 % 4, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps / 8 % 4]>()
        .contract_lane::<m![Ps / 8 % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps / 8 % 4 # 16]>()
        .commit_trim::<m![Ps / 8 % 4]>()
        .commit();

    // Scale the complete F32 dot before BF16 cast, then preserve the scaled value across Switch.
    // Four valid local rows are padded to an eight-way Switch group. Invalid
    // row slots stay padded in Slice until the final typed head gather.
    let scaled: DmTensor<bf16, Chip, KvCluster, m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]> = device
        .main.begin(contraction.view())
        .fetch::<m![Ps / 8 % 4 # 8], m![1 # 4]>()
        .switch::<m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]>(SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![Ps % 8], m![1 # 16]>()
        .transpose::<m![Ps / 4 % 2], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let output: DmTensor<bf16, Chip, m![Ns / 4], m![Ns % 4, Ds / 32, Ds / 8 % 4 # 8], m![Ds % 8]> =
        unsafe { scaled.reshape() };
    output.to_dm(&mut device.tdma)
}

pub(crate) fn project_key_value(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], Replicated, m![H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]>,
    DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]>,
) {
    let x: DmTensorView<'_, bf16, Chip, KvCluster, KvRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, KvCluster, KvRows, m![1], m![H]> = device
        .sub
        .begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let k: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> =
        project_one_kv_matrix(device, &x_trf, k_weight, k_weight_scale);
    let v: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> =
        project_one_kv_matrix(device, &x_trf, v_weight, v_weight_scale);

    (k, v)
}

// E137 uses E134 QKV unchanged; O changes activation representation and native reduction.
use crate::device::layout::Slice;
axes![OutputTerm = 4];
type OutputQuantSlices = m![H / 120 % 16, Qs / 256];
type OutputQuantRows = m![H / 120 % 16, 1 # 16];

// Every consumer row group holds the same sixteen K256 input shards.
// Quantize locally: no packed-term redistribution between row groups.
pub(crate) fn prepare_output_terms(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], OutputQuantSlices, m![Qs % 256]>,
) -> (DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![OutputTerm, Qs % 256]>,
      DmTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![1 # 8]>,
      DmTensor<f32, Chip, OutputCluster, OutputColumns, m![OutputTerm]>) {
    // Each (cluster,row-group) receives identical complete K4096 BF16 input.
    // Its sixteen-column maximum is the same global amax in all32 groups.
    // Local floor commutes with max across the sixteen valid 256-value shards.
    let local_amax: DmTensor<f32, Chip, m![2], OutputQuantSlices, m![1 # 8]> = device
        .main.begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_intra_slice_reduce::<Qs, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.504632769052528e-36f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Preserve H/120 row-group ownership and reduce only the sixteen K columns.
    // Exactly q=0 is initialized in each group; padded q slots stay invalid.
    let amax: DmTensor<f32, Chip, m![2], OutputQuantRows, m![1 # 8]> = device
        .main.begin(local_amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<OutputQuantRows, m![1]>(InterSliceReduceOpF32::Max)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Exponent mask produces 2^floor(log2(amax)); floor 2^-119 guarantees
    // scale >= 2^-126, and finite BF16 inputs imply scale <= 2^120.
    let scale: DmTensor<f32, Chip, m![2], OutputQuantRows, m![1 # 8]> = device
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
    let quant_scale: DmTensor<f32, Chip, m![2], OutputQuantSlices, m![1 # 8]> = scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, m![2], OutputQuantSlices, m![1 # 8]> = device
        .sub.begin(quant_scale.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let mut packed: DmTensor<f8e4m3, Chip, m![2], OutputQuantSlices, m![OutputTerm, Qs % 256]> = DmTensor::new();
    device.main.begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(packed.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, Qs % 256]>(0));
    let sum_vrf: VrfTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.sub
        .begin(packed.view().tile::<m![OutputTerm], 1, m![OutputTerm = 1 # 4, Qs % 256]>(0))
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>().to_vrf();
    // Term 1: residual is x/s minus the F32 cumulative reconstruction.
    device.main.begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16.0f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(packed.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, Qs % 256]>(1));
    let sum: DmTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.main
        .begin(packed.view().tile::<m![OutputTerm], 1, m![OutputTerm = 1 # 4, Qs % 256]>(1))
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0625f32)
        .vector_fp_binary(FpBinaryOp::AddF, &sum_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .commit_trim::<m![Qs % 8]>().commit();
    let sum_vrf: VrfTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.sub
        .begin(sum.view()).fetch::<m![Qs / 8 % 32], m![Qs % 8]>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>().to_vrf();
    // Term 2: residual is x/s minus the F32 cumulative reconstruction.
    device.main.begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 4194304.0f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(packed.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, Qs % 256]>(2));
    let sum: DmTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.main
        .begin(packed.view().tile::<m![OutputTerm], 1, m![OutputTerm = 1 # 4, Qs % 256]>(2))
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0000002384185791015625f32)
        .vector_fp_binary(FpBinaryOp::AddF, &sum_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .commit_trim::<m![Qs % 8]>().commit();
    let sum_vrf: VrfTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.sub
        .begin(sum.view()).fetch::<m![Qs / 8 % 32], m![Qs % 8]>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>().to_vrf();
    // Term 3: residual is x/s minus the F32 cumulative reconstruction.
    device.main.begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 134217728.0f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(packed.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, Qs % 256]>(3));
    // Constant normalized coefficients: common s is deliberately restored after Term sum.
    // Zero*finite(scale)+constant builds exact bits without an extra public input.
    let mut coefficients: DmTensor<f32, Chip, m![2], OutputQuantRows, m![OutputTerm, 1 # 8]> = DmTensor::new();
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, 1 # 8]>(0));
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.0625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, 1 # 8]>(1));
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.0000002384185791015625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, 1 # 8]>(2));
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.000000007450580596923828125f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, 1 # 8]>(3));
    // Spatial Term lanes need consecutive VRF cells. Pack the four scalar
    // time slots with two supported 8B commits, before replication.
    let coefficients: DmTensor<f32, Chip, m![2], OutputQuantRows, m![OutputTerm]> = device.sub
        .begin(coefficients.view()).fetch::<m![OutputTerm], m![1 # 8]>()
        .collect::<m![OutputTerm], m![1 # 8]>()
        .transpose::<m![OutputTerm / 2], m![OutputTerm % 2 # 8]>()
        .commit_trim::<m![OutputTerm % 2]>().commit();
    // Input DMA gives each (c,g) the identical4096 finite BF16 values x[256q+u].
    // K16 Max/floor/exponentmask/pow2scale are deterministic, so all32 scalars
    // are equal. Relabel only the full allocation Hgroup axis as anonymous16;
    // keep physical256 slices and q=0 live mask unchanged. Never shrink to1#256.
    let shared_scale: DmTensor<f32, Chip, m![2], m![16, 1 # 16], m![1 # 8]> = unsafe { scale.reshape() };
    // Packed data is already in the consuming row/K slices. The cluster relabel
    // m![2] -> Dummy2 changes no physical position or byte, and makes no DMA.
    let packed: DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![OutputTerm, Qs % 256]> = unsafe { packed.reshape() };
    (packed, shared_scale.to_dm(&mut device.tdma), coefficients.to_dm(&mut device.tdma))
}

pub(crate) fn project_output(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> {
    type OutputRows = m![H / 120 % 16, 1 # 16];
    let x: DmTensor<bf16, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = x.to_dm(&mut device.tdma);
    let (x, x_scale, coefficients) = prepare_output_terms(device, &x);
    let x_trf: TrfTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![OutputTerm], m![Qs % 256]> = device
        .sub.begin(x.view())
        .fetch::<m![OutputTerm, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![OutputTerm, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();
    let coefficients: VrfTensor<f32, Chip, OutputCluster, OutputColumns, m![OutputTerm # 8]> = device
        .sub.begin(coefficients.view()).fetch::<m![1], m![OutputTerm]>()
        .collect::<m![1], m![OutputTerm # 8]>().to_vrf();
    let weight: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, H / 120 % 16, H % 120, Qs]> = unsafe { weight.view().reshape() };
    // Distinct DM allocations expose the second DMA while the first row half computes.
    let first: DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 120 = 60, Qs % 256]> =
        weight.tile::<m![H % 120], 60, m![Dummy2, H / 120 % 16, H % 120 = 60 # 120, Qs]>(0)
            .to_dm(&mut device.tdma);
    let second: DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 120 = 60, Qs % 256]> =
        weight.tile::<m![H % 120], 60, m![Dummy2, H / 120 % 16, H % 120 = 60 # 120, Qs]>(60)
            .to_dm(&mut device.tdma);
    let mut result: DmTensor<f32, Chip, OutputCluster, OutputRows, m![H % 120]> = DmTensor::new();
    project_output_stage(device, &x_trf, &coefficients, &first, &mut result, 0);
    project_output_stage(device, &x_trf, &coefficients, &second, &mut result, 60);
    let result: DmTensor<f32, Chip, m![H / 1920], OutputRows, m![H % 120]> = unsafe { result.reshape() };
    let result: DmTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> = result.to_dm(&mut device.tdma);
    apply_output_channel_scale_f32(device, &result, &x_scale, weight_scale)
}

fn project_output_stage(
    device: &mut Device,
    x: &TrfTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![OutputTerm], m![Qs % 256]>,
    coefficients: &VrfTensor<f32, Chip, OutputCluster, OutputColumns, m![OutputTerm # 8]>,
    weight: &DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 120 = 60, Qs % 256]>,
    output: &mut DmTensor<f32, Chip, OutputCluster, m![H / 120 % 16, 1 # 16], m![H % 120]>,
    offset: usize,
) {
    device.main.begin(weight.view())
        .fetch::<m![H % 120 = 60, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![H % 120 = 60, Qs / 32 % 8], m![Qs % 32]>()
        // Four native lanes consume the same streamed original E4 weight packet.
        .contract_outer::<m![H % 120 = 60, Qs / 64 % 4], m![Qs % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 60]>()
        .contract_lane::<m![H % 120 = 60], m![OutputTerm # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![OutputTerm]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), coefficients)
        .vector_intra_slice_reduce::<OutputTerm, m![H % 120 = 60], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![H / 120 % 16, 1 # 16], m![H % 120 = 60]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .transpose::<m![H % 120 = 60 / 2], m![H % 120 = 60 % 2 # 8]>()
        .commit_trim::<m![H % 120 = 60 % 2]>()
        .commit_view(output.view_mut().tile::<m![H % 120], 60, m![H % 120 = 60 #{!} 120]>(offset));
}

pub(crate) type OutputEpilogueSlices = m![1 # 32, H / 480];

fn apply_output_channel_scale_f32(
    device: &mut Device,
    x: &DmTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![H % 480]>,
    x_scale: &DmTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![1 # 8]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> {
    let x_scale_vrf: VrfTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![1 # 8]> = device
        .sub.begin(x_scale.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let weight_scale: DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> = device
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    device
        .main
        .begin(x.view())
        .fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn add_output_residual(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]>,
    residual: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> {
    type ResidualSlices = m![1 # 32, H / 480];
    let residual: DmTensor<bf16, Chip, Cluster, ResidualSlices, m![H % 480]> = residual.to_dm(&mut device.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ResidualSlices, m![H % 480]> = device
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, ResidualSlices, m![H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    result
}

// O projection partitions 4096 BF16 inputs into sixteen 256-column groups.
type OutputCluster = m![Dummy2];
type OutputColumns = m![H / 120 % 16, Qs / 256];
