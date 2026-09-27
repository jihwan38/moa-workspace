// E136: canonical output-row lanes, original E4 weight TRF, weighted activation terms.
use furiosa_opt_std::prelude::*;
use crate::Chip;
use crate::axes::{Ds, Dummy2, Dummy256, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{Cluster, Replicated, Slice};
axes![QkvTerm = 4];

type QuantSlices = m![1 # 32, H / 480];

pub(crate) fn project_query(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, m![2], Replicated, m![QkvTerm, H]>,
    x_scale: &DmTensor<f32, Chip, m![2], Replicated, m![1 # 8]>,
    coefficients: &DmTensor<f32, Chip, m![2], Replicated, m![QkvTerm, 1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]> {
    type Rows = m![Qs / 8 % 256];
    type Owner = m![Qs / 2048];
    let x: DmTensorView<'_, f8e4m3, Chip, Owner, Rows, m![QkvTerm, H]> = unsafe { x.view().reshape() };
    let weight: DmTensor<f8e4m3, Chip, Owner, Rows, m![Qs % 8, H]> = weight.to_dm(&mut device.tdma);
    // Original E4 codes go through plain Sub fetch/collect. No lookup or BF16 weight DM.
    let weights: TrfTensor<f8e4m3, Chip, Owner, Rows, m![Qs % 8], m![H]> = device.sub
        .begin(weight.view()).fetch::<m![Qs % 8, H / 32], m![H % 32]>()
        .collect::<m![Qs % 8, H / 32], m![H % 32]>().to_trf();
    let coefficient_view: DmTensorView<'_, f32, Chip, Owner, Rows, m![QkvTerm, 1 # 8]> = unsafe { coefficients.view().reshape() };
    let coefficients: VrfTensor<f32, Chip, Owner, Rows, m![QkvTerm, 1 # 8]> = device.sub
        .begin(coefficient_view).fetch::<m![QkvTerm], m![1 # 8]>()
        .collect::<m![QkvTerm], m![1 # 8]>().to_vrf();
    let raw: DmTensor<f32, Chip, Owner, Rows, m![Qs % 8]> = device.main
        .begin(x).fetch::<m![QkvTerm, H / 32], m![H % 32]>()
        .collect::<m![QkvTerm, H / 32], m![H % 32]>()
        .contract_outer::<m![QkvTerm, H / 64], m![H % 64], _, _, _>(&weights)
        .contract_packet::<m![1]>().contract_time::<m![QkvTerm]>()
        .contract_lane::<m![QkvTerm], m![Qs % 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![QkvTerm, Qs / 4 % 2], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &coefficients)
        .vector_intra_slice_reduce::<QkvTerm, m![Qs / 4 % 2], m![Qs % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_concat::<m![1], m![Qs % 8]>().vector_final()
        .commit_trim::<m![Qs % 8]>().commit();
    let x_scale: DmTensorView<'_, f32, Chip, Owner, Rows, m![1 # 8]> = unsafe { x_scale.view().reshape() };
    let x_scale: VrfTensor<f32, Chip, Owner, Rows, m![1 # 8]> = device.sub
        .begin(x_scale).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weight_scale: DmTensor<bf16, Chip, Owner, Rows, m![Qs % 8]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale: VrfTensor<f32, Chip, Owner, Rows, m![Qs % 8]> = device.sub
        .begin(weight_scale.view()).fetch::<m![1], m![Qs % 8]>().fetch_cast::<f32>()
        .collect::<m![1], m![Qs % 8]>().to_vrf();
    // Restore the common input scale AFTER the weighted term sum, then channel scale.
    // Preserve F32 until the public scaled_projection rounding boundary.
    let scaled: DmTensor<bf16, Chip, Owner, Rows, m![Qs % 8]> = device.main
        .begin(raw.view()).fetch::<m![1], m![Qs % 8]>()
        .collect::<m![1], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 2], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_scale)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_scale)
        .vector_widen_concat::<m![1], m![Qs % 8]>().vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>().commit_trim::<m![Qs % 8]>().commit();
    let output: DmTensor<bf16, Chip, m![Ns / 4], m![Ns % 4, Gs, Ds / 8], m![Ds % 8]> = unsafe { scaled.reshape() };
    output.to_dm(&mut device.tdma)
}

fn project_one_kv_matrix(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, m![2], Replicated, m![QkvTerm, H]>,
    x_scale: &DmTensor<f32, Chip, m![2], Replicated, m![1 # 8]>,
    coefficients: &DmTensor<f32, Chip, m![2], Replicated, m![QkvTerm, 1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> {
    type Rows = m![Ps / 4 % 256];
    type Owner = m![Ps / 1024];
    let x: DmTensorView<'_, f8e4m3, Chip, Owner, Rows, m![QkvTerm, H]> = unsafe { x.view().reshape() };
    let weight: DmTensor<f8e4m3, Chip, Owner, Rows, m![Ps % 4, H]> = weight.to_dm(&mut device.tdma);
    // Original E4 codes go through plain Sub fetch/collect. No lookup or BF16 weight DM.
    let weights: TrfTensor<f8e4m3, Chip, Owner, Rows, m![Ps % 4], m![H]> = device.sub
        .begin(weight.view()).fetch::<m![Ps % 4, H / 32], m![H % 32]>()
        .collect::<m![Ps % 4, H / 32], m![H % 32]>().to_trf();
    let coefficient_view: DmTensorView<'_, f32, Chip, Owner, Rows, m![QkvTerm, 1 # 8]> = unsafe { coefficients.view().reshape() };
    let coefficients: VrfTensor<f32, Chip, Owner, Rows, m![QkvTerm, 1 # 8]> = device.sub
        .begin(coefficient_view).fetch::<m![QkvTerm], m![1 # 8]>()
        .collect::<m![QkvTerm], m![1 # 8]>().to_vrf();
    let raw: DmTensor<f32, Chip, Owner, Rows, m![Ps % 4]> = device.main
        .begin(x).fetch::<m![QkvTerm, H / 32], m![H % 32]>()
        .collect::<m![QkvTerm, H / 32], m![H % 32]>()
        .contract_outer::<m![QkvTerm, H / 64], m![H % 64], _, _, _>(&weights)
        .contract_packet::<m![1]>().contract_time::<m![QkvTerm]>()
        .contract_lane::<m![QkvTerm], m![Ps % 4 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &coefficients)
        .vector_intra_slice_reduce::<QkvTerm, m![1], m![Ps % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![Ps % 4 # 8]>().vector_final()
        .commit_trim::<m![Ps % 4]>().commit();
    let x_scale: DmTensorView<'_, f32, Chip, Owner, Rows, m![1 # 8]> = unsafe { x_scale.view().reshape() };
    let x_scale: VrfTensor<f32, Chip, Owner, Rows, m![1 # 8]> = device.sub
        .begin(x_scale).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weight_scale: DmTensor<bf16, Chip, Owner, Rows, m![Ps % 4]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale: VrfTensor<f32, Chip, Owner, Rows, m![Ps % 4 # 8]> = device.sub
        .begin(weight_scale.view()).fetch::<m![1], m![Ps % 4]>().fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 4 # 8]>().to_vrf();
    // Restore the common input scale AFTER the weighted term sum, then channel scale.
    // Preserve F32 until the public scaled_projection rounding boundary.
    let scaled: DmTensor<bf16, Chip, Owner, Rows, m![Ps % 4]> = device.main
        .begin(raw.view()).fetch::<m![1], m![Ps % 4]>()
        .collect::<m![1], m![Ps % 4 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_scale)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_scale)
        .vector_widen_pad::<m![Ps % 4 # 8]>().vector_final()
        .cast::<bf16, m![Ps % 4 # 16]>().commit_trim::<m![Ps % 4]>().commit();
    let output: DmTensor<bf16, Chip, m![Ns / 4], m![Ns % 4, Ds / 4], m![Ds % 4]> = unsafe { scaled.reshape() };
    output.to_dm(&mut device.tdma)
}

pub(crate) fn project_key_value(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, m![2], Replicated, m![QkvTerm, H]>,
    scale: &DmTensor<f32, Chip, m![2], Replicated, m![1 # 8]>,
    coefficients: &DmTensor<f32, Chip, m![2], Replicated, m![QkvTerm, 1 # 8]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]>,
      DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]>) {
    let k = project_one_kv_matrix(device, x, scale, coefficients, k_weight, k_weight_scale);
    let v = project_one_kv_matrix(device, x, scale, coefficients, v_weight, v_weight_scale);
    (k, v)
}

// Eight valid H480 shards per cluster; quantize before full-vector replication.
pub(crate) fn prepare_native_terms(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], QuantSlices, m![H % 480]>,
) -> (DmTensor<f8e4m3, Chip, m![2], Replicated, m![QkvTerm, H]>,
      DmTensor<f32, Chip, m![2], Replicated, m![1 # 8]>,
      DmTensor<f32, Chip, m![2], Replicated, m![QkvTerm, 1 # 8]>) {
    // Preserve the whole-H global amax while keeping normalized input sharded.
    // Local floor commutes with max across the eight valid 480-value shards.
    let local_amax: DmTensor<f32, Chip, m![2], QuantSlices, m![1 # 8]> = device
        .main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.504632769052528e-36f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Output has exactly one logical scalar per cluster. No padded element
    // is reinterpreted as valid input; this typed reduction must compile.
    let amax: DmTensor<f32, Chip, m![2], Slice, m![1 # 8]> = device
        .main.begin(local_amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<Slice, m![1]>(InterSliceReduceOpF32::Max)
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
    let quant_scale: DmTensor<f32, Chip, m![2], QuantSlices, m![1 # 8]> = scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, m![2], QuantSlices, m![1 # 8]> = device
        .sub.begin(quant_scale.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let mut packed: DmTensor<f8e4m3, Chip, m![2], QuantSlices, m![QkvTerm, H % 480]> = DmTensor::new();
    device.main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
        .commit_view(packed.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, H % 480]>(0));
    let sum_vrf: VrfTensor<f32, Chip, m![2], QuantSlices, m![H % 480]> = device.sub
        .begin(packed.view().tile::<m![QkvTerm], 1, m![QkvTerm = 1 # 4, H % 480]>(0))
        .fetch::<m![H / 32 % 15], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>().to_vrf();
    // Term 1: residual is x/s minus the F32 cumulative reconstruction.
    device.main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16.0f32)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
        .commit_view(packed.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, H % 480]>(1));
    let sum: DmTensor<f32, Chip, m![2], QuantSlices, m![H % 480]> = device.main
        .begin(packed.view().tile::<m![QkvTerm], 1, m![QkvTerm = 1 # 4, H % 480]>(1))
        .fetch::<m![H / 32 % 15], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0625f32)
        .vector_fp_binary(FpBinaryOp::AddF, &sum_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .commit_trim::<m![H % 8]>().commit();
    let sum_vrf: VrfTensor<f32, Chip, m![2], QuantSlices, m![H % 480]> = device.sub
        .begin(sum.view()).fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>().to_vrf();
    // Term 2: residual is x/s minus the F32 cumulative reconstruction.
    device.main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 4194304.0f32)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
        .commit_view(packed.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, H % 480]>(2));
    let sum: DmTensor<f32, Chip, m![2], QuantSlices, m![H % 480]> = device.main
        .begin(packed.view().tile::<m![QkvTerm], 1, m![QkvTerm = 1 # 4, H % 480]>(2))
        .fetch::<m![H / 32 % 15], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0000002384185791015625f32)
        .vector_fp_binary(FpBinaryOp::AddF, &sum_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .commit_trim::<m![H % 8]>().commit();
    let sum_vrf: VrfTensor<f32, Chip, m![2], QuantSlices, m![H % 480]> = device.sub
        .begin(sum.view()).fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>().to_vrf();
    // Term 3: residual is x/s minus the F32 cumulative reconstruction.
    device.main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 134217728.0f32)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
        .commit_view(packed.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, H % 480]>(3));
    // Constant normalized coefficients: common s is deliberately restored after Term sum.
    // Zero*finite(scale)+constant builds exact bits without an extra public input.
    let mut coefficients: DmTensor<f32, Chip, m![2], Slice, m![QkvTerm, 1 # 8]> = DmTensor::new();
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, 1 # 8]>(0));
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.0625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, 1 # 8]>(1));
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.0000002384185791015625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, 1 # 8]>(2));
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.000000007450580596923828125f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, 1 # 8]>(3));
    let packed: DmTensor<f8e4m3, Chip, m![2], m![Dummy256 / 32, 1 # 32], m![QkvTerm, H]> = packed.to_dm(&mut device.tdma);
    let packed: DmTensor<f8e4m3, Chip, m![2], m![Dummy256 / 32, 32], m![QkvTerm, H]> = device.main
        .begin(packed.view()).fetch::<m![QkvTerm], m![H]>()
        .switch::<m![Dummy256 / 32, 32], m![QkvTerm]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![QkvTerm, H / 32], m![H % 32]>().commit_trim::<m![H % 32]>().commit();
    let packed: DmTensor<f8e4m3, Chip, m![2], Replicated, m![QkvTerm, H]> = unsafe { packed.reshape() };
    (packed, scale.to_dm(&mut device.tdma), coefficients.to_dm(&mut device.tdma))
}

pub(crate) fn project_output(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> {
    type OutputRows = m![H / 120 % 16, 1 # 16];
    // Preserve every stored BF16 activation; no amax, FP8 hi/lo, or input scale.
    let x: DmTensor<bf16, Chip, OutputCluster, OutputColumns, m![Qs % 256]> = x.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, OutputCluster, OutputColumns, m![1], m![Qs % 256]> = device
        .sub.begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .collect::<m![Qs / 16 % 16], m![Qs % 16]>()
        .to_trf();
    let weight: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, H / 120 % 16, H % 120, Qs]> = unsafe { weight.view().reshape() };
    // Distinct DM allocations expose the second DMA while the first row half computes.
    let first: DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 120 = 60, Qs % 256]> =
        weight.tile::<m![H % 120], 60, m![Dummy2, H / 120 % 16, H % 120 = 60 # 120, Qs]>(0)
            .to_dm(&mut device.tdma);
    let second: DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 120 = 60, Qs % 256]> =
        weight.tile::<m![H % 120], 60, m![Dummy2, H / 120 % 16, H % 120 = 60 # 120, Qs]>(60)
            .to_dm(&mut device.tdma);
    let mut result: DmTensor<f32, Chip, OutputCluster, OutputRows, m![H % 120]> = DmTensor::new();
    project_output_stage(device, &x_trf, &first, &mut result, 0);
    project_output_stage(device, &x_trf, &second, &mut result, 60);
    let result: DmTensor<f32, Chip, m![H / 1920], OutputRows, m![H % 120]> = unsafe { result.reshape() };
    let result: DmTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> = result.to_dm(&mut device.tdma);
    apply_output_channel_scale_f32(device, &result, weight_scale)
}

fn project_output_stage(
    device: &mut Device,
    x: &TrfTensor<bf16, Chip, OutputCluster, OutputColumns, m![1], m![Qs % 256]>,
    weight: &DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 120 = 60, Qs % 256]>,
    output: &mut DmTensor<f32, Chip, OutputCluster, m![H / 120 % 16, 1 # 16], m![H % 120]>,
    offset: usize,
) {
    device.main.begin(weight.view())
        .fetch::<m![H % 120 = 60, Qs / 16 % 16], m![Qs % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 120 = 60, Qs / 16 % 16], m![Qs % 16]>()
        .contract_outer::<m![H % 120 = 60, Qs / 32 % 8], m![Qs % 32], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 60]>()
        .contract_lane::<m![H % 120 = 60], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
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
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> {
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
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
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
