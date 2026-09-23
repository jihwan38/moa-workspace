//! Diagnostic gate, not a competition kernel. Input digits are supplied by the
//! diagnostic caller; activation quantization belongs to the full C13 design.
use furiosa_opt_std::prelude::*;
use crate::Chip;
use crate::axes::{Rows, K, Term, H, L, IntTerm, IntBlockElt, QuantCopies};
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

#[device(chip = 1)]
pub fn integer_blocks(
    device: &mut Device,
    weight: &HbmTensor<f4e2m1, Chip, m![Rows, K]>,
    digits: &HbmTensor<i8, Chip, m![Term, K]>,
    decoded: &mut HbmTensor<i8, Chip, m![Rows, K]>,
    sums: &mut HbmTensor<f32, Chip, m![Rows, K / 16]>,
) {
    let packed: DmTensor<f4e2m1, Chip, Cluster, Slice, m![Rows, K]> = weight.to_dm(&mut device.tdma);
    // The fixed-point converter's fraction width is 31-int_width. Width 30
    // therefore computes round(2*w), exact for all sixteen FP4 E2M1 codes.
    let integer: DmTensor<i8, Chip, Cluster, Slice, m![Rows, K]> = device.main
        .begin(packed.view())
        .fetch::<m![Rows], m![K]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![Rows, K / 8], m![K % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fp_to_fxp(30)
        .vector_final()
        .cast::<i8, m![K % 8 # 32]>()
        .commit_trim::<m![K % 8]>()
        .commit();
    integer.view().to_hbm_view(&mut device.tdma, decoded.view_mut());
    let digits: DmTensor<i8, Chip, Cluster, Slice, m![Term, K]> = digits.to_dm(&mut device.tdma);
    let trf: TrfTensor<i8, Chip, Cluster, Slice, m![Term], m![K]> = device.sub
        .begin(digits.view())
        .fetch::<m![Term, K / 32], m![K % 32]>()
        .collect::<m![Term, K / 32], m![K % 32]>()
        .to_trf();
    let combined: DmTensor<f32, Chip, Cluster, Slice, m![Rows, K / 16]> = device.main
        .begin(integer.view())
        .fetch::<m![Rows, K / 32], m![K % 32]>()
        .collect::<m![Rows, K / 32], m![K % 32]>()
        .contract_outer::<m![Rows, K / 64], m![K % 64], _, _, _>(&trf)
        .contract_packet::<m![K / 16 % 4]>()
        .contract_time::<m![Rows, K / 64]>()
        .contract_lane::<m![Rows, K / 64, Term], m![K / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Rows, K / 64]>()
        .vector_fxp(FxpBinaryOp::LeftShift, 7i32, ())
        .vector_fxp_zip(FxpBinaryOp::AddFxp)
        .vector_fxp_to_fp(31)
        .vector_final()
        .commit_trim::<m![K / 16 % 4]>()
        .commit();
    combined.view().to_hbm_view(&mut device.tdma, sums.view_mut());
}

fn quantize_h(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> (
    DmTensor<i8, Chip, Cluster, Slice, m![IntTerm, H]>,
    DmTensor<f32, Chip, Cluster, Slice, m![H / 16]>,
) {
    // The fetch sequencer normalizes a 60-way partition to 64, dropping
    // anonymous high padding. Use four explicit identical copies instead.
    // QuantCopies participates in every quantizer producer/consumer mapping.
    type QuantSlices = m![QuantCopies, H / 64 # 64];
    let x: DmTensor<bf16, Chip, Cluster, QuantSlices, m![H % 64]> = x.to_dm(&mut device.tdma);
    // The reduction API reduces the named axis, including all of its time
    // factors. Separate the 16 values from the four retained block indices.
    let blocks: DmTensorView<'_, bf16, Chip, Cluster, QuantSlices, m![H / 16 % 4, IntBlockElt]> = unsafe { x.view().reshape() };
    let amax: DmTensor<f32, Chip, Cluster, QuantSlices, m![H / 16 % 4, 1 # 8]> = device.main
        .begin(blocks)
        .fetch::<m![H / 16 % 4], m![IntBlockElt]>().fetch_cast::<f32>()
        .collect::<m![H / 16 % 4, IntBlockElt / 8], m![IntBlockElt % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H / 16 % 4, IntBlockElt / 4], m![IntBlockElt % 4]>()
        .vector_intra_slice_reduce::<IntBlockElt, m![H / 16 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Max)
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
    let restore: DmTensor<f32, Chip, Cluster, QuantSlices, m![H / 16 % 4]> = device.main
        .begin(scale.view())
        .fetch::<m![H / 16 % 4], m![1 # 8]>()
        .collect::<m![H / 16 % 4], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.00390625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        // Pack scalar time steps in pairs: F32 transpose supports two rows.
        // The following DMA then transfers H/L scales in contiguous 8B records;
        // its consumer still indexes one F32 scale per 16 input elements.
        .transpose::<m![H / 32 % 2], m![H / 16 % 2 # 8]>()
        .commit_trim::<m![H / 16 % 2]>().commit();
    // Each copy was broadcast from the same BF16 input and executes the same
    // quantizer. Retain physical copy 0; high groups become inactive padding.
    let digits: DmTensor<i8, Chip, Cluster, m![1 # 4, H / 64 # 64], m![IntTerm, H % 64]> = unsafe { digits.reshape() };
    let restore: DmTensor<f32, Chip, Cluster, m![1 # 4, H / 64 # 64], m![H / 16 % 4]> = unsafe { restore.reshape() };
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
    // The reduction API reduces the named axis, including all of its time
    // factors. Separate the 16 values from the four retained block indices.
    let blocks: DmTensorView<'_, bf16, Chip, Cluster, QuantSlices, m![L / 16 % 4, IntBlockElt]> = unsafe { x.view().reshape() };
    let amax: DmTensor<f32, Chip, Cluster, QuantSlices, m![L / 16 % 4, 1 # 8]> = device.main
        .begin(blocks)
        .fetch::<m![L / 16 % 4], m![IntBlockElt]>().fetch_cast::<f32>()
        .collect::<m![L / 16 % 4, IntBlockElt / 8], m![IntBlockElt % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![L / 16 % 4, IntBlockElt / 4], m![IntBlockElt % 4]>()
        .vector_intra_slice_reduce::<IntBlockElt, m![L / 16 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Max)
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
    let restore: DmTensor<f32, Chip, Cluster, QuantSlices, m![L / 16 % 4]> = device.main
        .begin(scale.view())
        .fetch::<m![L / 16 % 4], m![1 # 8]>()
        .collect::<m![L / 16 % 4], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.00390625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        // Pack scalar time steps in pairs: F32 transpose supports two rows.
        // The following DMA then transfers H/L scales in contiguous 8B records;
        // its consumer still indexes one F32 scale per 16 input elements.
        .transpose::<m![L / 32 % 2], m![L / 16 % 2 # 8]>()
        .commit_trim::<m![L / 16 % 2]>().commit();
    (digits.to_dm(&mut device.tdma), restore.to_dm(&mut device.tdma))
}

#[device(chip = 1)]
pub fn quant_h(device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![H]>,
    digits_out: &mut HbmTensor<i8, Chip, m![IntTerm, H]>,
    restore_out: &mut HbmTensor<f32, Chip, m![H / 16]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = input.to_dm(&mut device.tdma);
    let (digits, restore) = quantize_h(device, &input);
    digits.view().to_hbm_view(&mut device.tdma, digits_out.view_mut());
    restore.view().to_hbm_view(&mut device.tdma, restore_out.view_mut());
}

#[device(chip = 1)]
pub fn quant_l(device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![L]>,
    digits_out: &mut HbmTensor<i8, Chip, m![IntTerm, L]>,
    restore_out: &mut HbmTensor<f32, Chip, m![L / 16]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, Slice, m![L]> = input.to_dm(&mut device.tdma);
    let (digits, restore) = quantize_l(device, &input);
    digits.view().to_hbm_view(&mut device.tdma, digits_out.view_mut());
    restore.view().to_hbm_view(&mut device.tdma, restore_out.view_mut());
}
