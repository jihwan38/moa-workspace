//! Prompt-row attention. Additive masks contain 0 for visible keys and -1e30 otherwise.
//! A batch must not straddle a full-attention cache page boundary. Invalid padded rows
//! may produce arbitrary output; callers must not consume or commit those rows.

use crate::axes::{B, Df, Gf, Tf};
use crate::device::layout::Cluster;
use crate::{Chip, EPS};
use crate::ops_request_attention::Fc;
use furiosa_opt_std::prelude::*;

type Rows = m![B, Gf];
type HeadRows = m![Gf, 1 # 16];
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
    k: &HbmTensor<bf16, Chip, m![Fc,Tf,Df]>,
    base: &HbmTensor<i32,Chip,m![1]>,
    mask: &HbmTensor<f32, Chip, m![B, Tf]>,
) -> DmTensor<f32, Chip, Cluster, Rows, m![Tf]> {
    type KeyRows = m![Tf / 8, 1 # 4];
    let q: DmTensor<bf16, Chip, Cluster, KeyRows, m![B, Gf, Df]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16,Chip,Cluster,KeyRows,m![Tf%8,Df]>=k.gather::<m![Fc],m![Tf,Df]>().by_byte_offsets(base).to_dm(&mut device.tdma);
    let kt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Tf % 8], m![Df]> = device
        .sub
        .begin(k.view())
        .fetch::<m![Tf % 8, Df / 16], m![Df % 16]>()
        .collect::<m![Tf % 8, Df / 16], m![Df % 16]>()
        .to_trf();
    let scores: DmTensor<bf16, Chip, Cluster, KeyRows, m![B, Gf, Tf % 8]> = device
        .main
        .begin(q.view())
        .fetch::<m![B, Gf, Df / 16], m![Df % 16]>()
        .collect::<m![B, Gf, Df / 16], m![Df % 16]>()
        .contract_outer::<m![B, Gf, Df / 32], m![Df % 32], _, _, _>(&kt)
        .contract_packet::<m![1]>()
        .contract_time::<m![B, Gf]>()
        .contract_lane::<m![B, Gf], m![Tf % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Tf % 8 # 16]>()
        .commit_trim::<m![Tf % 8]>()
        .commit();
    let scores: DmTensor<bf16, Chip, Cluster, Rows, m![Tf]> = scores.to_dm(&mut device.tdma);
    let mask: DmTensor<f32, Chip, Cluster, Rows, m![Tf]> = mask.to_dm(&mut device.tdma);
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
) -> DmTensor<f32, Chip, Cluster, Rows, m![Tf]> {
    let max = scalar_vrf(device, max);
    device
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
        .commit()
}

fn row_sum(device: &mut Device, exp: &DmTensor<f32, Chip, Cluster, Rows, m![Tf]>) -> Scalar {
    device
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
        .commit()
}

fn weighted_values(
    device: &mut Device,
    exp: &DmTensor<f32, Chip, Cluster, Rows, m![Tf]>,
    v: &HbmTensor<bf16, Chip, m![Fc,Tf,Df]>,
    base: &HbmTensor<i32,Chip,m![1]>,
) -> DmTensor<bf16, Chip, Cluster, HeadRows, m![B, Df]> {
    type KeyRows = m![Gf, Tf / 32];
    let weights: DmTensor<bf16, Chip, Cluster, Rows, m![Tf]> = device
        .main
        .begin(exp.view())
        .fetch::<m![Tf / 8], m![Tf % 8]>()
        .collect::<m![Tf / 8], m![Tf % 8]>()
        .cast::<bf16, m![Tf % 8 # 16]>()
        .commit_trim::<m![Tf % 8]>()
        .commit();
    let weights: DmTensor<bf16, Chip, Cluster, KeyRows, m![B, Tf % 32]> =
        weights.to_dm(&mut device.tdma);
    let v:DmTensor<bf16,Chip,Cluster,m![Tf/2],m![Tf%2,Df]>=v.gather::<m![Fc],m![Tf,Df]>().by_byte_offsets(base).to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![Tf % 32, Df]> = v.to_dm(&mut device.tdma);
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
    let partial: DmTensor<bf16, Chip, Cluster, KeyRows, m![B, Df]> = device
        .main
        .begin(weights.view())
        .fetch::<m![B, Tf / 16 % 2], m![Tf % 16]>()
        .collect::<m![B, Tf / 16 % 2], m![Tf % 16]>()
        .contract_outer::<m![B, Df / 8], m![Tf % 32], _, _, _>(&vt)
        .contract_packet::<m![1]>()
        .contract_time::<m![B, Df / 8]>()
        .contract_lane::<m![B, Df / 8], m![Df % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();
    device
        .main
        .begin(partial.view())
        .fetch::<m![B, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![B, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<HeadRows, m![B, Df / 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

#[device(chip = 1)]
pub fn full_attention_first_page_b16(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Fc,Tf,Df]>,
    v: &HbmTensor<bf16, Chip, m![Fc,Tf,Df]>,
    base: &HbmTensor<i32,Chip,m![1]>,
    additive_mask: &HbmTensor<f32, Chip, m![B, Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![B, Gf, 1 # 8]>,
    running_sum: &mut HbmTensor<f32, Chip, m![B, Gf, 1 # 8]>,
    out: &mut HbmTensor<bf16, Chip, m![B, Gf, Df]>,
) {
    let scores = full_scores(device, q, k, base, additive_mask);
    let max = row_max(device, &scores);
    let exp = exp_scores(device, &scores, &max);
    let sum = row_sum(device, &exp);
    let result = weighted_values(device, &exp, v, base);
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
    k: &HbmTensor<bf16, Chip, m![Fc,Tf,Df]>,
    v: &HbmTensor<bf16, Chip, m![Fc,Tf,Df]>,
    base: &HbmTensor<i32,Chip,m![1]>,
    additive_mask: &HbmTensor<f32, Chip, m![B, Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![B, Gf, 1 # 8]>,
    running_sum: &mut HbmTensor<f32, Chip, m![B, Gf, 1 # 8]>,
    out: &mut HbmTensor<bf16, Chip, m![B, Gf, Df]>,
) {
    let scores = full_scores(device, q, k, base, additive_mask);
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
    let exp = exp_scores(device, &scores, &max);
    let page_sum = row_sum(device, &exp);
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
    let page_result = weighted_values(device, &exp, v, base);
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
