//! Private head-major page statistics avoid strided tiny HBM writes. Arithmetic unchanged.
//! Exact per-page experiment: stage Q once per head and directly gather replicated page K/V.
//! Host supplies32 identical absolute page-byte offsets, one per query-cluster/head pair.
//! Prompt-row attention. Additive masks contain 0 for visible keys and -1e30 otherwise.
//! A batch must not straddle a full-attention cache page boundary. Invalid padded rows
//! may produce arbitrary output; callers must not consume or commit those rows.

use crate::axes::{Df, Gf, P128, Tf};
type Cluster = m![P128 / 64];
use crate::ops_request_attention::Fc;
use crate::{Chip, EPS};
use furiosa_opt_std::prelude::*;

pub type Stats = m![P128 / 64, Gf, P128 % 64];

type Rows = m![Gf, P128 / 4 % 16];
type HeadRows = m![Gf, 1 # 16];
type Scalar = DmTensor<f32, Chip, Cluster, Rows, m![P128%4,1 # 8]>;

type Compact = DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4]>;
type CompactHead = DmTensor<f32, Chip, Cluster, HeadRows, m![P128 % 64]>;
fn store_scalar(device: &mut Device, x: &Scalar, out: &mut HbmTensor<f32, Chip, Stats>) {
    let compact: Compact = device
        .main
        .begin(x.view())
        .fetch::<m![P128 % 4], m![1#8]>()
        .collect::<m![P128 % 4], m![1#8]>()
        .transpose::<m![P128 / 2 % 2], m![P128%2#8]>()
        .commit_trim::<m![P128 % 2]>()
        .commit();
    let compact: CompactHead = compact.to_dm(&mut device.tdma);
    compact.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
fn load_scalar(device: &mut Device, input: &HbmTensor<f32, Chip, Stats>) -> Scalar {
    let compact: CompactHead = input.to_dm(&mut device.tdma);
    let compact: Compact = compact.to_dm(&mut device.tdma);
    device
        .main
        .begin(compact.view())
        .fetch::<m![P128 / 2 % 2], m![P128%2#8]>()
        .collect::<m![P128 / 2 % 2], m![P128%2#8]>()
        .transpose::<m![P128 % 4], m![1#8]>()
        .commit_trim::<m![1#8]>()
        .commit()
}

fn scalar_vrf(
    device: &mut Device,
    x: &Scalar,
) -> VrfTensor<f32, Chip, Cluster, Rows, m![P128%4,1 # 8]> {
    device
        .sub
        .begin(x.view())
        .fetch::<m![P128 % 4], m![1 # 8]>()
        .collect::<m![P128 % 4], m![1 # 8]>()
        .to_vrf()
}

fn full_scores(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    base: &HbmTensor<i32, Chip, m![P128 / 64, Gf]>,
    mask: &HbmTensor<f32, Chip, m![P128, Tf]>,
) -> DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Tf]> {
    type KeyRows = m![Gf, Tf / 32];
    let q: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Df]> = q.to_dm(&mut device.tdma);
    let q: DmTensor<bf16, Chip, Cluster, KeyRows, m![P128 % 4, P128 / 4 % 16, Df]> = device
        .main
        .begin(q.view())
        .fetch::<m![P128 % 4], m![Df]>()
        .switch::<KeyRows, m![P128 % 4, P128 / 4 % 16]>(SwitchConfig::CustomBroadcast {
            ring_size: 16,
        })
        .collect::<m![P128 % 4, P128 / 4 % 16, Df / 16], m![Df % 16]>()
        .commit_trim::<m![Df % 16]>()
        .commit();
    let k: DmTensor<bf16, Chip, Cluster, KeyRows, m![Tf % 32, Df]> = k
        .gather::<m![Fc], m![Tf, Df]>()
        .by_byte_offsets(base)
        .to_dm(&mut device.tdma);
    let kt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Tf % 8], m![Tf / 8 % 4, Df]> = device
        .sub
        .begin(k.view())
        .fetch::<m![Tf % 8, Tf / 8 % 4, Df / 16], m![Df % 16]>()
        .collect::<m![Tf % 8, Tf / 8 % 4, Df / 16], m![Df % 16]>()
        .to_trf();
    let scores: DmTensor<bf16, Chip, Cluster, KeyRows, m![P128 % 64, Tf % 32]> = device
        .main
        .begin(q.view())
        .fetch::<m![P128 % 64, Df / 16], m![Df % 16]>()
        .collect::<m![P128 % 64, Df / 16], m![Df % 16]>()
        .contract_outer::<m![P128 % 64, Df / 32, Tf / 8 % 4], m![Df % 32], _, _, _>(&kt)
        .contract_packet::<m![1]>()
        .contract_time::<m![P128 % 64, Tf / 8 % 4]>()
        .contract_lane::<m![P128 % 64, Tf / 8 % 4], m![Tf % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Tf%8#16]>()
        .commit_trim::<m![Tf % 8]>()
        .commit();
    let scores: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Tf]> =
        scores.to_dm(&mut device.tdma);
    let mask: DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Tf]> = mask.to_dm(&mut device.tdma);
    let mask: VrfTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Tf]> = device
        .sub
        .begin(mask.view())
        .fetch::<m![P128 % 4, Tf / 8], m![Tf % 8]>()
        .collect::<m![P128 % 4, Tf / 8], m![Tf % 8]>()
        .to_vrf();
    device
        .main
        .begin(scores.view())
        .fetch::<m![P128 % 4, Tf / 16], m![Tf % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 4, Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &mask)
        .vector_final()
        .commit_trim::<m![Tf % 8]>()
        .commit()
}

fn row_max(
    device: &mut Device,
    scores: &DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Tf]>,
) -> Scalar {
    device
        .main
        .begin(scores.view())
        .fetch::<m![P128 % 4, Tf / 8], m![Tf % 8]>()
        .collect::<m![P128 % 4, Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4, Tf / 4], m![Tf % 4]>()
        .vector_intra_slice_reduce::<Tf, m![P128 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn exp_scores(
    device: &mut Device,
    scores: &DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Tf]>,
    max: &Scalar,
) -> DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Tf]> {
    let max = scalar_vrf(device, max);
    device
        .main
        .begin(scores.view())
        .fetch::<m![P128 % 4, Tf / 8], m![Tf % 8]>()
        .collect::<m![P128 % 4, Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4, Tf / 4], m![Tf % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &max)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![P128 % 4, Tf / 8], m![Tf % 8]>()
        .vector_final()
        .commit_trim::<m![Tf % 8]>()
        .commit()
}

fn row_sum(
    device: &mut Device,
    exp: &DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Tf]>,
) -> Scalar {
    device
        .main
        .begin(exp.view())
        .fetch::<m![P128 % 4, Tf / 8], m![Tf % 8]>()
        .collect::<m![P128 % 4, Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4, Tf / 4], m![Tf % 4]>()
        .vector_intra_slice_reduce::<Tf, m![P128 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn weighted_values(
    device: &mut Device,
    exp: &DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Tf]>,
    v: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    base_byte_offset: &HbmTensor<i32, Chip, m![P128 / 64, Gf]>,
) -> DmTensor<bf16, Chip, Cluster, HeadRows, m![P128 % 64, Df]> {
    type KeyRows = m![Gf, Tf / 32];
    let weights: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Tf]> = device
        .main
        .begin(exp.view())
        .fetch::<m![P128 % 4, Tf / 8], m![Tf % 8]>()
        .collect::<m![P128 % 4, Tf / 8], m![Tf % 8]>()
        .cast::<bf16, m![Tf % 8 # 16]>()
        .commit_trim::<m![Tf % 8]>()
        .commit();
    let weights: DmTensor<bf16, Chip, Cluster, KeyRows, m![P128 % 64, Tf % 32]> =
        weights.to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![Tf % 32, Df]> = v
        .gather::<m![Fc], m![Tf, Df]>()
        .by_byte_offsets(base_byte_offset)
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
    let partial: DmTensor<bf16, Chip, Cluster, KeyRows, m![P128 % 64, Df]> = device
        .main
        .begin(weights.view())
        .fetch::<m![P128 % 64, Tf / 16 % 2], m![Tf % 16]>()
        .collect::<m![P128 % 64, Tf / 16 % 2], m![Tf % 16]>()
        .contract_outer::<m![P128 % 64, Df / 8], m![Tf % 32], _, _, _>(&vt)
        .contract_packet::<m![1]>()
        .contract_time::<m![P128 % 64, Df / 8]>()
        .contract_lane::<m![P128 % 64, Df / 8], m![Df % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();
    device
        .main
        .begin(partial.view())
        .fetch::<m![P128 % 64, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 64, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<HeadRows, m![P128 % 64, Df / 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

#[device(chip = 1)]
pub fn full_attention_first_page_p128(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    base_byte_offset: &HbmTensor<i32, Chip, m![P128 / 64, Gf]>,
    additive_mask: &HbmTensor<f32, Chip, m![P128, Tf]>,
    running_max: &mut HbmTensor<f32, Chip, Stats>,
    running_sum: &mut HbmTensor<f32, Chip, Stats>,
    out: &mut HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
) {
    let scores = full_scores(device, q, k, base_byte_offset, additive_mask);
    let max = row_max(device, &scores);
    let exp = exp_scores(device, &scores, &max);
    let sum = row_sum(device, &exp);
    let result = weighted_values(device, &exp, v, base_byte_offset);
    store_scalar(device, &max, running_max);
    store_scalar(device, &sum, running_sum);
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn full_attention_next_page_p128(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    base_byte_offset: &HbmTensor<i32, Chip, m![P128 / 64, Gf]>,
    additive_mask: &HbmTensor<f32, Chip, m![P128, Tf]>,
    running_max: &mut HbmTensor<f32, Chip, Stats>,
    running_sum: &mut HbmTensor<f32, Chip, Stats>,
    out: &mut HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
) {
    let scores = full_scores(device, q, k, base_byte_offset, additive_mask);
    let page_max = row_max(device, &scores);
    let old_max: Scalar = load_scalar(device, running_max);
    let old_max_vrf = scalar_vrf(device, &old_max);
    let max: Scalar = device
        .main
        .begin(page_max.view())
        .fetch::<m![P128 % 4], m![1 # 8]>()
        .collect::<m![P128 % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Max, &old_max_vrf)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rescale: Scalar = device
        .main
        .begin(max.view())
        .fetch::<m![P128 % 4], m![1 # 8]>()
        .collect::<m![P128 % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4], m![1 # 4]>()
        .vector_fp_binary_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode10, &old_max_vrf)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![P128 % 4], m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rescale_vrf = scalar_vrf(device, &rescale);
    let exp = exp_scores(device, &scores, &max);
    let page_sum = row_sum(device, &exp);
    let page_sum_vrf = scalar_vrf(device, &page_sum);
    let old_sum: Scalar = load_scalar(device, running_sum);
    let sum: Scalar = device
        .main
        .begin(old_sum.view())
        .fetch::<m![P128 % 4], m![1 # 8]>()
        .collect::<m![P128 % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4], m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rescale_vrf)
        .vector_widen_concat::<m![P128 % 4], m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &page_sum_vrf)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let page_result = weighted_values(device, &exp, v, base_byte_offset);
    let page_result: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Df]> =
        page_result.to_dm(&mut device.tdma);
    let page_result_vrf: VrfTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Df]> = device
        .sub
        .begin(page_result.view())
        .fetch::<m![P128 % 4, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 4, Df / 8], m![Df % 8]>()
        .to_vrf();
    let previous: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Df]> =
        out.to_dm(&mut device.tdma);
    // Preserve the baseline's BF16 rounding between rescaling and addition.
    let scaled: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Df]> = device
        .main
        .begin(previous.view())
        .fetch::<m![P128 % 4, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 4, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4, Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rescale_vrf)
        .vector_widen_concat::<m![P128 % 4, Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Df]> = device
        .main
        .begin(scaled.view())
        .fetch::<m![P128 % 4, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 4, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &page_result_vrf)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();
    store_scalar(device, &max, running_max);
    store_scalar(device, &sum, running_sum);
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn full_attention_divide_p128(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    running_sum: &HbmTensor<f32, Chip, Stats>,
    out: &mut HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Df]> = input.to_dm(&mut device.tdma);
    let sum: Scalar = load_scalar(device, running_sum);
    let sum = scalar_vrf(device, &sum);
    let result: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Df]> = device
        .main
        .begin(x.view())
        .fetch::<m![P128 % 4, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 4, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4, Df / 4], m![Df % 4]>()
        .vector_fp_div(&sum)
        .vector_widen_concat::<m![P128 % 4, Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
