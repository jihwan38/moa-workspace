//! Independent requests attend their own dynamically indexed pooled KV. Masks use1/0.
use crate::axes::{B, Df, Ds, Gf, Gs, Ns, Tf, Ts};
use crate::device::layout::{Cluster, Slice};
use crate::{Chip, EPS};
use furiosa_opt_std::prelude::*;
axes![Rc = 16, Fc = 256, Sc = 16384, Fp = 131072];

type Rows = m![B, Gf];
type HeadRows = m![B, 1 # 16];
type Scalar = DmTensor<f32, Chip, Cluster, Rows, m![1 # 8]>;

fn scalar_vrf(device: &mut Device, x: &Scalar) -> VrfTensor<f32, Chip, Cluster, Rows, m![1 # 8]> {
    device
        .sub
        .begin(x.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

fn full_scores(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    page_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Tf]>,
) -> DmTensor<f32, Chip, Cluster, Rows, m![Tf]> {
    type KeyRows = m![B, Tf / 32];
    let q: DmTensor<bf16, Chip, Cluster, KeyRows, m![Gf, Df]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KeyRows, m![Tf % 32, Df]> = k
        .gather::<m![Fc], m![Tf, Df]>()
        .by_byte_offsets(page_offsets)
        .to_dm(&mut device.tdma);
    let kt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Tf % 8], m![Tf / 8 % 4, Df]> = device
        .sub
        .begin(k.view())
        .fetch::<m![Tf % 8, Tf / 8 % 4, Df / 16], m![Df % 16]>()
        .collect::<m![Tf % 8, Tf / 8 % 4, Df / 16], m![Df % 16]>()
        .to_trf();
    let scores: DmTensor<bf16, Chip, Cluster, KeyRows, m![Gf, Tf % 32]> = device
        .main
        .begin(q.view())
        .fetch::<m![Gf, Df / 16], m![Df % 16]>()
        .collect::<m![Gf, Df / 16], m![Df % 16]>()
        .contract_outer::<m![Gf, Df / 32, Tf / 8 % 4], m![Df % 32], _, _, _>(&kt)
        .contract_packet::<m![1]>()
        .contract_time::<m![Gf, Tf / 8 % 4]>()
        .contract_lane::<m![Gf, Tf / 8 % 4], m![Tf % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Tf % 8 # 16]>()
        .commit_trim::<m![Tf % 8]>()
        .commit();
    let scores: DmTensor<bf16, Chip, Cluster, Rows, m![Tf]> = scores.to_dm(&mut device.tdma);
    let mask: DmTensor<f32, Chip, Cluster, Rows, m![Tf]> = mask.to_dm(&mut device.tdma);
    let mask: DmTensor<f32, Chip, Cluster, Rows, m![Tf]> = device
        .main
        .begin(mask.view())
        .fetch::<m![Tf / 8], m![Tf % 8]>()
        .collect::<m![Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tf / 4], m![Tf % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, 1.0f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1e30f32)
        .vector_widen_concat::<m![Tf / 8], m![Tf % 8]>()
        .vector_final()
        .commit_trim::<m![Tf % 8]>()
        .commit();
    let mask: VrfTensor<f32, Chip, Cluster, Rows, m![Tf]> = device
        .sub
        .begin(mask.view())
        .fetch::<m![Tf / 8], m![Tf % 8]>()
        .collect::<m![Tf / 8], m![Tf % 8]>()
        .to_vrf();
    device
        .main
        .begin(scores.view())
        .fetch::<m![Tf / 16], m![Tf % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &mask)
        .vector_final()
        .commit_trim::<m![Tf % 8]>()
        .commit()
}

fn row_max(device: &mut Device, scores: &DmTensor<f32, Chip, Cluster, Rows, m![Tf]>) -> Scalar {
    device
        .main
        .begin(scores.view())
        .fetch::<m![Tf / 8], m![Tf % 8]>()
        .collect::<m![Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tf / 4], m![Tf % 4]>()
        .vector_intra_slice_reduce::<Tf, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn exp_scores(
    device: &mut Device,
    scores: &DmTensor<f32, Chip, Cluster, Rows, m![Tf]>,
    max: &Scalar,
    mask: &HbmTensor<f32, Chip, m![B, Tf]>,
) -> DmTensor<f32, Chip, Cluster, Rows, m![Tf]> {
    let max = scalar_vrf(device, max);
    let exp: DmTensor<f32, Chip, Cluster, Rows, m![Tf]> = device
        .main
        .begin(scores.view())
        .fetch::<m![Tf / 8], m![Tf % 8]>()
        .collect::<m![Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tf / 4], m![Tf % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &max)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Tf / 8], m![Tf % 8]>()
        .vector_final()
        .commit_trim::<m![Tf % 8]>()
        .commit();
    let mask: DmTensor<f32, Chip, Cluster, Rows, m![Tf]> = mask.to_dm(&mut device.tdma);
    let mask: VrfTensor<f32, Chip, Cluster, Rows, m![Tf]> = device
        .sub
        .begin(mask.view())
        .fetch::<m![Tf / 8], m![Tf % 8]>()
        .collect::<m![Tf / 8], m![Tf % 8]>()
        .to_vrf();
    device
        .main
        .begin(exp.view())
        .fetch::<m![Tf / 8], m![Tf % 8]>()
        .collect::<m![Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Min, &mask)
        .vector_final()
        .commit_trim::<m![Tf % 8]>()
        .commit()
}

fn row_sum(
    device: &mut Device,
    exp: &DmTensor<f32, Chip, Cluster, Rows, m![Tf]>,
    active: &HbmTensor<f32, Chip, m![B,1 # 8]>,
) -> Scalar {
    let sum: Scalar = device
        .main
        .begin(exp.view())
        .fetch::<m![Tf / 8], m![Tf % 8]>()
        .collect::<m![Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tf / 4], m![Tf % 4]>()
        .vector_intra_slice_reduce::<Tf, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let active: Scalar = active.to_dm(&mut device.tdma);
    let active = scalar_vrf(device, &active);
    device
        .main
        .begin(sum.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &active)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn weighted_values(
    device: &mut Device,
    exp: &DmTensor<f32, Chip, Cluster, Rows, m![Tf]>,
    page_offsets: &HbmTensor<i32, Chip, m![B]>,
    v: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) -> DmTensor<bf16, Chip, Cluster, HeadRows, m![Gf, Df]> {
    type KeyRows = m![B, Tf / 32];
    let weights: DmTensor<bf16, Chip, Cluster, Rows, m![Tf]> = device
        .main
        .begin(exp.view())
        .fetch::<m![Tf / 8], m![Tf % 8]>()
        .collect::<m![Tf / 8], m![Tf % 8]>()
        .cast::<bf16, m![Tf % 8 # 16]>()
        .commit_trim::<m![Tf % 8]>()
        .commit();
    let weights: DmTensor<bf16, Chip, Cluster, KeyRows, m![Gf, Tf % 32]> =
        weights.to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![Tf % 32, Df]> = v
        .gather::<m![Fc], m![Tf, Df]>()
        .by_byte_offsets(page_offsets)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![Df, Tf % 32]> = device
        .main
        .begin(v.view())
        .fetch::<m![Df / 8, Tf % 32], m![Df % 8 # 16]>()
        .collect::<m![Df / 8, Tf % 32], m![Df % 8 # 16]>()
        .transpose::<m![Df / 8, Tf / 4 % 8, Df % 8], m![Tf % 4 # 16]>()
        .commit_trim::<m![Tf % 4]>()
        .commit();
    let vt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Df % 8], m![Df / 8, Tf % 32]> = device
        .sub
        .begin(v.view())
        .fetch::<m![Df % 8, Df / 8, Tf / 16 % 2], m![Tf % 16]>()
        .collect::<m![Df % 8, Df / 8, Tf / 16 % 2], m![Tf % 16]>()
        .to_trf();
    let partial: DmTensor<bf16, Chip, Cluster, KeyRows, m![Gf, Df]> = device
        .main
        .begin(weights.view())
        .fetch::<m![Gf, Tf / 16 % 2], m![Tf % 16]>()
        .collect::<m![Gf, Tf / 16 % 2], m![Tf % 16]>()
        .contract_outer::<m![Gf, Df / 8], m![Tf % 32], _, _, _>(&vt)
        .contract_packet::<m![1]>()
        .contract_time::<m![Gf, Df / 8]>()
        .contract_lane::<m![Gf, Df / 8], m![Df % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();
    device
        .main
        .begin(partial.view())
        .fetch::<m![Gf, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<HeadRows, m![Gf, Df / 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

#[device(chip = 1)]
pub fn full_attention_first_page_b16(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    page_offsets: &HbmTensor<i32, Chip, m![B]>,
    visibility_mask: &HbmTensor<f32, Chip, m![B, Tf]>,
    page_active: &HbmTensor<f32, Chip, m![B,1 # 8]>,
    running_max: &mut HbmTensor<f32, Chip, m![B, Gf, 1 # 8]>,
    running_sum: &mut HbmTensor<f32, Chip, m![B, Gf, 1 # 8]>,
    out: &mut HbmTensor<bf16, Chip, m![B, Gf, Df]>,
) {
    let scores = full_scores(device, q, k, page_offsets, visibility_mask);
    let max = row_max(device, &scores);
    let exp = exp_scores(device, &scores, &max, visibility_mask);
    let sum = row_sum(device, &exp, page_active);
    let result = weighted_values(device, &exp, page_offsets, v);
    max.view()
        .to_hbm_view(&mut device.tdma, running_max.view_mut());
    sum.view()
        .to_hbm_view(&mut device.tdma, running_sum.view_mut());
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn full_attention_next_page_b16(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    page_offsets: &HbmTensor<i32, Chip, m![B]>,
    visibility_mask: &HbmTensor<f32, Chip, m![B, Tf]>,
    page_active: &HbmTensor<f32, Chip, m![B,1 # 8]>,
    running_max: &mut HbmTensor<f32, Chip, m![B, Gf, 1 # 8]>,
    running_sum: &mut HbmTensor<f32, Chip, m![B, Gf, 1 # 8]>,
    out: &mut HbmTensor<bf16, Chip, m![B, Gf, Df]>,
) {
    let scores = full_scores(device, q, k, page_offsets, visibility_mask);
    let page_max = row_max(device, &scores);
    let old_max: Scalar = running_max.to_dm(&mut device.tdma);
    let old_max_vrf = scalar_vrf(device, &old_max);
    let max: Scalar = device
        .main
        .begin(page_max.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Max, &old_max_vrf)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rescale: Scalar = device
        .main
        .begin(max.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1], m![1 # 4]>()
        .vector_fp_binary_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode10, &old_max_vrf)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![1], m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rescale_vrf = scalar_vrf(device, &rescale);
    let exp = exp_scores(device, &scores, &max, visibility_mask);
    let page_sum = row_sum(device, &exp, page_active);
    let page_sum_vrf = scalar_vrf(device, &page_sum);
    let old_sum: Scalar = running_sum.to_dm(&mut device.tdma);
    let sum: Scalar = device
        .main
        .begin(old_sum.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1], m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rescale_vrf)
        .vector_widen_concat::<m![1], m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &page_sum_vrf)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let page_result = weighted_values(device, &exp, page_offsets, v);
    let page_result: DmTensor<bf16, Chip, Cluster, Rows, m![Df]> =
        page_result.to_dm(&mut device.tdma);
    let page_result_vrf: VrfTensor<f32, Chip, Cluster, Rows, m![Df]> = device
        .sub
        .begin(page_result.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .to_vrf();
    let previous: DmTensor<bf16, Chip, Cluster, Rows, m![Df]> = out.to_dm(&mut device.tdma);
    // Preserve the baseline's BF16 rounding between rescaling and addition.
    let scaled: DmTensor<bf16, Chip, Cluster, Rows, m![Df]> = device
        .main
        .begin(previous.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rescale_vrf)
        .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, Rows, m![Df]> = device
        .main
        .begin(scaled.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &page_result_vrf)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();
    max.view()
        .to_hbm_view(&mut device.tdma, running_max.view_mut());
    sum.view()
        .to_hbm_view(&mut device.tdma, running_sum.view_mut());
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn full_attention_divide_b16(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    running_sum: &HbmTensor<f32, Chip, m![B, Gf, 1 # 8]>,
    out: &mut HbmTensor<bf16, Chip, m![B, Gf, Df]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Rows, m![Df]> = input.to_dm(&mut device.tdma);
    let sum: Scalar = running_sum.to_dm(&mut device.tdma);
    let sum: Scalar = device
        .main
        .begin(sum.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Max, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let sum = scalar_vrf(device, &sum);
    let result: DmTensor<bf16, Chip, Cluster, Rows, m![Df]> = device
        .main
        .begin(x.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_fp_div(&sum)
        .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// Migration bridge: copy one complete authoritative legacy cache into its pool slot.
#[device(chip = 1)]
pub fn copy_sliding_cache_to_pool(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    slot_byte_offset: &HbmTensor<i32, Chip, m![1]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    type Source = m![Ts/8,1 # 2];
    let k: DmTensor<bf16, Chip, Cluster, Source, m![Ts % 8, Ns, Ds]> = k.to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, Source, m![Ts % 8, Ns, Ds]> = v.to_dm(&mut device.tdma);
    pool_k
        .view_mut()
        .scatter::<m![Rc], m![Ts, Ns, Ds]>()
        .by_byte_offsets(slot_byte_offset.view())
        .from_dm(&mut device.tdma, k);
    pool_v
        .view_mut()
        .scatter::<m![Rc], m![Ts, Ns, Ds]>()
        .by_byte_offsets(slot_byte_offset.view())
        .from_dm(&mut device.tdma, v);
}
#[device(chip = 1)]
pub fn copy_full_page_to_pool(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    page_byte_offset: &HbmTensor<i32, Chip, m![1]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    type Source = m![Tf/8,1 # 4];
    let k: DmTensor<bf16, Chip, Cluster, Source, m![Tf % 8, Df]> = k.to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, Source, m![Tf % 8, Df]> = v.to_dm(&mut device.tdma);
    pool_k
        .view_mut()
        .scatter::<m![Fc], m![Tf, Df]>()
        .by_byte_offsets(page_byte_offset.view())
        .from_dm(&mut device.tdma, k);
    pool_v
        .view_mut()
        .scatter::<m![Fc], m![Tf, Df]>()
        .by_byte_offsets(page_byte_offset.view())
        .from_dm(&mut device.tdma, v);
}
fn commit_sliding_pool<const VALID: usize>(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    byte_offsets: &HbmTensor<i32, Chip, m![B = { VALID }]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![B = { VALID }, Ns, Ds]> = k
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![B = { VALID }, Ns, Ds]> = v
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    pool_k
        .view_mut()
        .scatter::<m![Rc, Ts], m![Ns, Ds]>()
        .by_byte_offsets(byte_offsets.view())
        .from_dm(&mut device.tdma, k);
    pool_v
        .view_mut()
        .scatter::<m![Rc, Ts], m![Ns, Ds]>()
        .by_byte_offsets(byte_offsets.view())
        .from_dm(&mut device.tdma, v);
}
fn commit_full_pool<const VALID: usize>(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    byte_offsets: &HbmTensor<i32, Chip, m![B = { VALID }]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![B = { VALID }, Df]> = k
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Df]>(0)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![B = { VALID }, Df]> = v
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Df]>(0)
        .to_dm(&mut device.tdma);
    pool_k
        .view_mut()
        .scatter::<m![Fc, Tf], m![Df]>()
        .by_byte_offsets(byte_offsets.view())
        .from_dm(&mut device.tdma, k);
    pool_v
        .view_mut()
        .scatter::<m![Fc, Tf], m![Df]>()
        .by_byte_offsets(byte_offsets.view())
        .from_dm(&mut device.tdma, v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_1(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 1]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<1>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_1(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 1]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<1>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_2(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 2]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<2>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_2(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 2]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<2>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_3(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 3]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<3>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_3(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 3]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<3>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_4(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 4]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<4>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_4(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 4]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<4>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_5(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 5]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<5>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_5(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 5]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<5>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_6(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 6]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<6>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_6(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 6]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<6>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_7(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 7]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<7>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_7(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 7]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<7>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_8(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 8]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<8>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_8(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 8]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<8>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_9(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 9]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<9>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_9(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 9]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<9>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_10(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 10]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<10>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_10(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 10]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<10>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_11(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 11]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<11>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_11(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 11]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<11>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_12(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 12]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<12>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_12(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 12]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<12>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_13(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 13]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<13>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_13(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 13]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<13>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_14(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 14]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<14>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_14(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 14]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<14>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_15(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 15]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<15>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_15(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 15]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<15>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn commit_sliding_pool_16(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 16]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
) {
    commit_sliding_pool::<16>(device, k, v, offsets, pool_k, pool_v);
}
#[device(chip = 1)]
pub fn commit_full_pool_16(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 16]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    commit_full_pool::<16>(device, k, v, offsets, pool_k, pool_v);
}

#[device(chip = 1)]
pub fn copy_sliding_pool_to_cache(
    device: &mut Device,
    pool_k: &HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    slot_byte_offset: &HbmTensor<i32, Chip, m![1]>,
    k: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    type Source = m![Ts/8,1 # 2];
    let rk: DmTensor<bf16, Chip, Cluster, Source, m![Ts % 8, Ns, Ds]> = pool_k
        .gather::<m![Rc], m![Ts, Ns, Ds]>()
        .by_byte_offsets(slot_byte_offset)
        .to_dm(&mut device.tdma);
    let rv: DmTensor<bf16, Chip, Cluster, Source, m![Ts % 8, Ns, Ds]> = pool_v
        .gather::<m![Rc], m![Ts, Ns, Ds]>()
        .by_byte_offsets(slot_byte_offset)
        .to_dm(&mut device.tdma);
    rk.view().to_hbm_view(&mut device.tdma, k.view_mut());
    rv.view().to_hbm_view(&mut device.tdma, v.view_mut());
}
#[device(chip = 1)]
pub fn copy_full_pool_to_page(
    device: &mut Device,
    pool_k: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    pool_v: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    page_byte_offset: &HbmTensor<i32, Chip, m![1]>,
    k: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    type Source = m![Tf/8,1 # 4];
    let rk: DmTensor<bf16, Chip, Cluster, Source, m![Tf % 8, Df]> = pool_k
        .gather::<m![Fc], m![Tf, Df]>()
        .by_byte_offsets(page_byte_offset)
        .to_dm(&mut device.tdma);
    let rv: DmTensor<bf16, Chip, Cluster, Source, m![Tf % 8, Df]> = pool_v
        .gather::<m![Fc], m![Tf, Df]>()
        .by_byte_offsets(page_byte_offset)
        .to_dm(&mut device.tdma);
    rk.view().to_hbm_view(&mut device.tdma, k.view_mut());
    rv.view().to_hbm_view(&mut device.tdma, v.view_mut());
}

// Four requests per tile preserve the scalar32-key BF16 value partials and
//32-way inter-slice sum tree while keeping each K/V TRF tile at64KiB.
type SlidingRows = m![B=4,Ns,Gs,1 # 4];
type SlidingScalar = DmTensor<f32, Chip, Cluster, SlidingRows, m![1 # 8]>;
fn sliding_scalar(
    device: &mut Device,
    x: &SlidingScalar,
) -> VrfTensor<f32, Chip, Cluster, SlidingRows, m![1 # 8]> {
    device
        .sub
        .begin(x.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}
fn sliding_premax(
    device: &mut Device,
    scores: &DmTensor<bf16, Chip, Cluster, SlidingRows, m![Ts]>,
    mask: &DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]>,
) -> DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> {
    let additive: DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> = device
        .main
        .begin(mask.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ts / 4], m![Ts % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, 1.0f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1e30f32)
        .vector_widen_concat::<m![Ts / 8], m![Ts % 8]>()
        .vector_final()
        .commit_trim::<m![Ts % 8]>()
        .commit();
    let additive: VrfTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> = device
        .sub
        .begin(additive.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .to_vrf();
    device
        .main
        .begin(scores.view())
        .fetch::<m![Ts / 16], m![Ts % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &additive)
        .vector_final()
        .commit_trim::<m![Ts % 8]>()
        .commit()
}
fn sliding_max(
    device: &mut Device,
    scores: &DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]>,
) -> SlidingScalar {
    device
        .main
        .begin(scores.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ts / 4], m![Ts % 4]>()
        .vector_intra_slice_reduce::<Ts, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}
fn sliding_exp(
    device: &mut Device,
    scores: &DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]>,
    max: &SlidingScalar,
    mask: &DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]>,
) -> DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> {
    let max = sliding_scalar(device, max);
    let exp: DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> = device
        .main
        .begin(scores.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ts / 4], m![Ts % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &max)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Ts / 8], m![Ts % 8]>()
        .vector_final()
        .commit_trim::<m![Ts % 8]>()
        .commit();
    let mask: VrfTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> = device
        .sub
        .begin(mask.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .to_vrf();
    device
        .main
        .begin(exp.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Min, &mask)
        .vector_final()
        .commit_trim::<m![Ts % 8]>()
        .commit()
}
fn sliding_sum(
    device: &mut Device,
    exp: &DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]>,
) -> SlidingScalar {
    device
        .main
        .begin(exp.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ts / 4], m![Ts % 4]>()
        .vector_intra_slice_reduce::<Ts, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}
#[device(chip = 1)]
pub fn sliding_attention_b16(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    pool_k: &HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    kv_byte_offsets: &HbmTensor<i32, Chip, m![B, Ts]>,
    visibility_mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    out: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
) {
    type KeyRows = m![Ns, Ts / 32];
    type Reduced = m![Ns,1 # 32];
    for group in 0..4 {
        let q: DmTensor<bf16, Chip, Cluster, KeyRows, m![B = 4, Gs, Ds]> = q
            .view()
            .tile::<m![B], 4, m![B=4 # 16,Ns,Gs,Ds]>(group * 4)
            .to_dm(&mut device.tdma);
        let index: DmTensor<i32, Chip, Cluster, Slice, m![B = 4, Ts]> = kv_byte_offsets
            .view()
            .tile::<m![B], 4, m![B=4 # 16,Ts]>(group * 4)
            .to_dm(&mut device.tdma);
        let index: HbmTensor<i32, Chip, m![B = 4, Ts]> = index.to_hbm(&mut device.tdma);
        let k: DmTensor<bf16, Chip, Cluster, KeyRows, m![B = 4, Ts % 32, Ds]> = pool_k
            .gather::<m![Rc, Ts], m![Ns, Ds]>()
            .by_byte_offsets(&index)
            .to_dm(&mut device.tdma);
        let kt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Ts % 8], m![B = 4, Ts / 8 % 4, Ds]> =
            device
                .sub
                .begin(k.view())
                .fetch::<m![Ts % 8, B = 4, Ts / 8 % 4, Ds / 16], m![Ds % 16]>()
                .collect::<m![Ts % 8, B = 4, Ts / 8 % 4, Ds / 16], m![Ds % 16]>()
                .to_trf();
        let scores: DmTensor<bf16, Chip, Cluster, KeyRows, m![B = 4, Gs, Ts % 32]> = device
            .main
            .begin(q.view())
            .fetch::<m![B = 4, Gs, Ds / 16], m![Ds % 16]>()
            .collect::<m![B = 4, Gs, Ds / 16], m![Ds % 16]>()
            .contract_outer::<m![B = 4, Gs, Ds / 32, Ts / 8 % 4], m![Ds % 32], _, _, _>(&kt)
            .contract_packet::<m![1]>()
            .contract_time::<m![B = 4, Gs, Ts / 8 % 4]>()
            .contract_lane::<m![B = 4, Gs, Ts / 8 % 4], m![Ts % 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![Ts%8 # 16]>()
            .commit_trim::<m![Ts % 8]>()
            .commit();
        let scores: DmTensor<bf16, Chip, Cluster, SlidingRows, m![Ts]> =
            scores.to_dm(&mut device.tdma);
        let mask: DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> = visibility_mask
            .view()
            .tile::<m![B], 4, m![B=4 # 16,Ts]>(group * 4)
            .to_dm(&mut device.tdma);
        let scores = sliding_premax(device, &scores, &mask);
        let max = sliding_max(device, &scores);
        let exp = sliding_exp(device, &scores, &max, &mask);
        let sum = sliding_sum(device, &exp);
        let sum = sliding_scalar(device, &sum);
        let weights: DmTensor<bf16, Chip, Cluster, SlidingRows, m![Ts]> = device
            .main
            .begin(exp.view())
            .fetch::<m![Ts / 8], m![Ts % 8]>()
            .collect::<m![Ts / 8], m![Ts % 8]>()
            .cast::<bf16, m![Ts%8 # 16]>()
            .commit_trim::<m![Ts % 8]>()
            .commit();
        let weights: DmTensor<bf16, Chip, Cluster, KeyRows, m![B = 4, Gs, Ts % 32]> =
            weights.to_dm(&mut device.tdma);

        let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![B = 4, Ts % 32, Ds]> = pool_v
            .gather::<m![Rc, Ts], m![Ns, Ds]>()
            .by_byte_offsets(&index)
            .to_dm(&mut device.tdma);
        let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![B = 4, Ds, Ts % 32]> = device
            .main
            .begin(v.view())
            .fetch::<m![B = 4, Ds / 8, Ts % 32], m![Ds%8 # 16]>()
            .collect::<m![B = 4, Ds / 8, Ts % 32], m![Ds%8 # 16]>()
            .transpose::<m![B = 4, Ds / 8, Ts / 4 % 8, Ds % 8], m![Ts%4 # 16]>()
            .commit_trim::<m![Ts % 4]>()
            .commit();
        let vt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Ds % 8], m![B = 4, Ds / 8, Ts % 32]> =
            device
                .sub
                .begin(v.view())
                .fetch::<m![Ds % 8, B = 4, Ds / 8, Ts / 16 % 2], m![Ts % 16]>()
                .collect::<m![Ds % 8, B = 4, Ds / 8, Ts / 16 % 2], m![Ts % 16]>()
                .to_trf();
        let partial: DmTensor<bf16, Chip, Cluster, KeyRows, m![B = 4, Gs, Ds]> = device
            .main
            .begin(weights.view())
            .fetch::<m![B = 4, Gs, Ts / 16 % 2], m![Ts % 16]>()
            .collect::<m![B = 4, Gs, Ts / 16 % 2], m![Ts % 16]>()
            .contract_outer::<m![B = 4, Gs, Ds / 8], m![Ts % 32], _, _, _>(&vt)
            .contract_packet::<m![1]>()
            .contract_time::<m![B = 4, Gs, Ds / 8]>()
            .contract_lane::<m![B = 4, Gs, Ds / 8], m![Ds % 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![Ds%8 # 16]>()
            .commit_trim::<m![Ds % 8]>()
            .commit();
        let combined: DmTensor<bf16, Chip, Cluster, Reduced, m![B = 4, Gs, Ds]> = device
            .main
            .begin(partial.view())
            .fetch::<m![B = 4, Gs, Ds / 16], m![Ds % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![B = 4, Gs, Ds / 8], m![Ds % 8]>()
            .vector_init()
            .vector_inter_slice_reduce::<Reduced, m![B = 4, Gs, Ds / 8]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![Ds%8 # 16]>()
            .commit_trim::<m![Ds % 8]>()
            .commit();
        let combined: DmTensor<bf16, Chip, Cluster, SlidingRows, m![Ds]> =
            combined.to_dm(&mut device.tdma);
        let result: DmTensor<bf16, Chip, Cluster, SlidingRows, m![Ds]> = device
            .main
            .begin(combined.view())
            .fetch::<m![Ds / 16], m![Ds % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Ds / 8], m![Ds % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
            .vector_fp_div(&sum)
            .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
            .vector_final()
            .cast::<bf16, m![Ds%8 # 16]>()
            .commit_trim::<m![Ds % 8]>()
            .commit();
        result.view().to_hbm_view(
            &mut device.tdma,
            out.view_mut()
                .tile::<m![B], 4, m![B=4 #{!} 16,Ns,Gs,Ds]>(group * 4),
        );
    }
}
