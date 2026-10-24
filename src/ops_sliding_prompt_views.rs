//! Cache-view variant; scalar arithmetic and row ordering match the source kernels.
//! Prompt-row attention. Additive masks contain 0 for visible keys and -1e30 otherwise.
//! A batch must not straddle a full-attention cache page boundary. Invalid padded rows
//! may produce arbitrary output; callers must not consume or commit those rows.

use crate::axes::{B, Ds, Gs, Ns, Ts};
use furiosa_opt_std::prelude::*;
pub use crate::ops_sliding_batch::Bk;
type Cluster=m![B/8];
use crate::{Chip, EPS};

type Rows = m![B%8, Ns, Gs,1#2];
type HeadRows = m![Ns, 1 # 32];
type Scalar = DmTensor<f32, Chip, Cluster, Rows, m![1 # 8]>;

fn scalar_vrf(device: &mut Device, x: &Scalar) -> VrfTensor<f32, Chip, Cluster, Rows, m![1 # 8]> {
    device
        .sub
        .begin(x.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

fn old_scores(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: HbmTensorView<'_, bf16, Chip, m![Ts # 65536, Ns, Ds]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
) -> DmTensor<f32, Chip, Cluster, Rows, m![Ts]> {
    type KeyRows = m![Ts / 8, 1 # 2];
    let q: DmTensor<bf16, Chip, Cluster, KeyRows, m![B%8, Ns, Gs, Ds]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KeyRows, m![Ts % 8, Ns, Ds]> = k.to_dm(&mut device.tdma);
    let kt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Ts % 8], m![Ns, Ds]> = device
        .sub
        .begin(k.view())
        .fetch::<m![Ts % 8, Ns, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ts % 8, Ns, Ds / 16], m![Ds % 16]>()
        .to_trf();
    let scores: DmTensor<bf16, Chip, Cluster, KeyRows, m![B%8, Ns, Gs, Ts % 8]> = device
        .main
        .begin(q.view())
        .fetch::<m![B%8, Ns, Gs, Ds / 16], m![Ds % 16]>()
        .collect::<m![B%8, Ns, Gs, Ds / 16], m![Ds % 16]>()
        .contract_outer::<m![B%8, Ns, Gs, Ds / 32], m![Ds % 32], _, _, _>(&kt)
        .contract_packet::<m![1]>()
        .contract_time::<m![B%8, Ns, Gs]>()
        .contract_lane::<m![B%8, Ns, Gs], m![Ts % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Ts % 8 # 16]>()
        .commit_trim::<m![Ts % 8]>()
        .commit();
    let scores: DmTensor<bf16, Chip, Cluster, Rows, m![Ts]> = scores.to_dm(&mut device.tdma);
    let mask: DmTensor<f32, Chip, Cluster, Rows, m![Ts]> = mask.to_dm(&mut device.tdma);
    let mask: VrfTensor<f32, Chip, Cluster, Rows, m![Ts]> = device
        .sub
        .begin(mask.view())
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
        .vector_clip(ClipBinaryOpF32::Add, &mask)
        .vector_final()
        .commit_trim::<m![Ts % 8]>()
        .commit()
}

fn row_max(device: &mut Device, scores: &DmTensor<f32, Chip, Cluster, Rows, m![Ts]>) -> Scalar {
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

fn exp_scores(
    device: &mut Device,
    scores: &DmTensor<f32, Chip, Cluster, Rows, m![Ts]>,
    max: &Scalar,
) -> DmTensor<f32, Chip, Cluster, Rows, m![Ts]> {
    let max = scalar_vrf(device, max);
    device
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
        .commit()
}

fn row_sum(device: &mut Device, exp: &DmTensor<f32, Chip, Cluster, Rows, m![Ts]>) -> Scalar {
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

fn weighted_values(
    device: &mut Device,
    exp: &DmTensor<f32, Chip, Cluster, Rows, m![Ts]>,
    v: HbmTensorView<'_, bf16, Chip, m![Ts # 65536, Ns, Ds]>,
) -> DmTensor<bf16, Chip, Cluster, HeadRows, m![B%8, Gs, Ds]> {
    type KeyRows = m![Ns, Ts / 32];
    let weights: DmTensor<bf16, Chip, Cluster, Rows, m![Ts]> = device
        .main
        .begin(exp.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .cast::<bf16, m![Ts % 8 # 16]>()
        .commit_trim::<m![Ts % 8]>()
        .commit();
    let weights: DmTensor<bf16, Chip, Cluster, KeyRows, m![B%8, Gs, Ts % 32]> =
        weights.to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![Ts % 32, Ds]> = v.to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![Ds, Ts % 32]> = device
        .main
        .begin(v.view())
        .fetch::<m![Ds / 8, Ts % 32], m![Ds % 8 # 16]>()
        .collect::<m![Ds / 8, Ts % 32], m![Ds % 8 # 16]>()
        .transpose::<m![Ds / 8, Ts / 4 % 8, Ds % 8], m![Ts % 4 # 16]>()
        .commit_trim::<m![Ts % 4]>()
        .commit();
    let vt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Ds % 8], m![Ds / 8, Ts % 32]> = device
        .sub
        .begin(v.view())
        .fetch::<m![Ds % 8, Ds / 8, Ts / 16 % 2], m![Ts % 16]>()
        .collect::<m![Ds % 8, Ds / 8, Ts / 16 % 2], m![Ts % 16]>()
        .to_trf();
    let partial: DmTensor<bf16, Chip, Cluster, KeyRows, m![B%8, Gs, Ds]> = device
        .main
        .begin(weights.view())
        .fetch::<m![B%8, Gs, Ts / 16 % 2], m![Ts % 16]>()
        .collect::<m![B%8, Gs, Ts / 16 % 2], m![Ts % 16]>()
        .contract_outer::<m![B%8, Gs, Ds / 8], m![Ts % 32], _, _, _>(&vt)
        .contract_packet::<m![1]>()
        .contract_time::<m![B%8, Gs, Ds / 8]>()
        .contract_lane::<m![B%8, Gs, Ds / 8], m![Ds % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    device
        .main
        .begin(partial.view())
        .fetch::<m![B%8, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![B%8, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<HeadRows, m![B%8, Gs, Ds / 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn chunk_scores(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    mask: &HbmTensor<f32, Chip, m![B, Bk]>,
) -> DmTensor<f32, Chip, Cluster, Rows, m![Bk]> {
    type KeyRows = m![Ns, Bk / 8, 1 # 16];
    // B and Bk have identical extent/order; only the key axis label changes.
    let k = unsafe { k.view().reshape::<Chip, m![Bk, Ns, Ds]>() };
    let q: DmTensor<bf16, Chip, Cluster, KeyRows, m![B%8, Gs, Ds]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KeyRows, m![Bk % 8, Ds]> = k.to_dm(&mut device.tdma);
    let kt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Bk % 8], m![Ds]> = device
        .sub
        .begin(k.view())
        .fetch::<m![Bk % 8, Ds / 16], m![Ds % 16]>()
        .collect::<m![Bk % 8, Ds / 16], m![Ds % 16]>()
        .to_trf();
    let scores: DmTensor<bf16, Chip, Cluster, KeyRows, m![B%8, Gs, Bk % 8]> = device
        .main
        .begin(q.view())
        .fetch::<m![B%8, Gs, Ds / 16], m![Ds % 16]>()
        .collect::<m![B%8, Gs, Ds / 16], m![Ds % 16]>()
        .contract_outer::<m![B%8, Gs, Ds / 32], m![Ds % 32], _, _, _>(&kt)
        .contract_packet::<m![1]>()
        .contract_time::<m![B%8, Gs]>()
        .contract_lane::<m![B%8, Gs], m![Bk % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Bk % 8 # 16]>()
        .commit_trim::<m![Bk % 8]>()
        .commit();
    let scores: DmTensor<bf16, Chip, Cluster, Rows, m![Bk]> = scores.to_dm(&mut device.tdma);
    let mask: DmTensor<f32, Chip, Cluster, Rows, m![Bk]> = mask.to_dm(&mut device.tdma);
    let mask: VrfTensor<f32, Chip, Cluster, Rows, m![Bk]> = device
        .sub
        .begin(mask.view())
        .fetch::<m![Bk / 8], m![Bk % 8]>()
        .collect::<m![Bk / 8], m![Bk % 8]>()
        .to_vrf();
    device
        .main
        .begin(scores.view())
        .fetch::<m![Bk / 16], m![Bk % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Bk / 8], m![Bk % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &mask)
        .vector_final()
        .commit_trim::<m![Bk % 8]>()
        .commit()
}

fn chunk_max(device: &mut Device, scores: &DmTensor<f32, Chip, Cluster, Rows, m![Bk]>) -> Scalar {
    device
        .main
        .begin(scores.view())
        .fetch::<m![Bk / 8], m![Bk % 8]>()
        .collect::<m![Bk / 8], m![Bk % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Bk / 4], m![Bk % 4]>()
        .vector_intra_slice_reduce::<Bk, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn chunk_exp(
    device: &mut Device,
    scores: &DmTensor<f32, Chip, Cluster, Rows, m![Bk]>,
    max: &Scalar,
) -> DmTensor<f32, Chip, Cluster, Rows, m![Bk]> {
    let max = scalar_vrf(device, max);
    device
        .main
        .begin(scores.view())
        .fetch::<m![Bk / 8], m![Bk % 8]>()
        .collect::<m![Bk / 8], m![Bk % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Bk / 4], m![Bk % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &max)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Bk / 8], m![Bk % 8]>()
        .vector_final()
        .commit_trim::<m![Bk % 8]>()
        .commit()
}

fn chunk_sum(device: &mut Device, exp: &DmTensor<f32, Chip, Cluster, Rows, m![Bk]>) -> Scalar {
    device
        .main
        .begin(exp.view())
        .fetch::<m![Bk / 8], m![Bk % 8]>()
        .collect::<m![Bk / 8], m![Bk % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Bk / 4], m![Bk % 4]>()
        .vector_intra_slice_reduce::<Bk, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn chunk_values(
    device: &mut Device,
    exp: &DmTensor<f32, Chip, Cluster, Rows, m![Bk]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Rows, m![Ds]> {
    let weights: DmTensor<bf16, Chip, Cluster, Rows, m![Bk]> = device
        .main
        .begin(exp.view())
        .fetch::<m![Bk / 8], m![Bk % 8]>()
        .collect::<m![Bk / 8], m![Bk % 8]>()
        .cast::<bf16, m![Bk % 8 # 16]>()
        .commit_trim::<m![Bk % 8]>()
        .commit();
    let weights: VrfTensor<f32, Chip, Cluster, Rows, m![Bk]> = device
        .sub
        .begin(weights.view())
        .fetch::<m![Bk / 16], m![Bk % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Bk / 8], m![Bk % 8]>()
        .to_vrf();
    let v = unsafe { v.view().reshape::<Chip, m![Bk, Ns, Ds]>() };
    let v: DmTensor<bf16, Chip, Cluster, Rows, m![Bk, Ds]> = v.to_dm(&mut device.tdma);
    device
        .main
        .begin(v.view())
        .fetch::<m![Ds / 16, Bk], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 16, Bk, Ds / 8 % 2], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 16, Bk, Ds / 4 % 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weights)
        .vector_intra_slice_reduce::<Bk, m![Ds / 4], m![Ds % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// Attend before committing this chunk to the old ring. The masks jointly describe
/// [max(0, query_position - 1023), query_position], with old keys restricted to
/// positions before chunk_start and current keys restricted to valid chunk rows.
#[device(chip = 1)]
pub fn sliding_attention_b16(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    old_k: HbmTensorView<'_, bf16, Chip, m![Ts # 65536, Ns, Ds]>,
    old_v: HbmTensorView<'_, bf16, Chip, m![Ts # 65536, Ns, Ds]>,
    chunk_k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    chunk_v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    old_mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    chunk_mask: &HbmTensor<f32, Chip, m![B, Bk]>,
    out: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
) {
    let old_scores = old_scores(device, q, old_k, old_mask);
    let new_scores = chunk_scores(device, q, chunk_k, chunk_mask);
    let old_max = row_max(device, &old_scores);
    let new_max = chunk_max(device, &new_scores);
    let new_max = scalar_vrf(device, &new_max);
    let max: Scalar = device
        .main
        .begin(old_max.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Max, &new_max)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let old_exp = exp_scores(device, &old_scores, &max);
    let new_exp = chunk_exp(device, &new_scores, &max);
    let old_sum = row_sum(device, &old_exp);
    let new_sum = chunk_sum(device, &new_exp);
    let new_sum = scalar_vrf(device, &new_sum);
    let sum: Scalar = device
        .main
        .begin(old_sum.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &new_sum)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let sum = scalar_vrf(device, &sum);
    let old_values = weighted_values(device, &old_exp, old_v);
    let old_values: DmTensor<bf16, Chip, Cluster, Rows, m![Ds]> =
        old_values.to_dm(&mut device.tdma);
    let new_values = chunk_values(device, &new_exp, chunk_v);
    let new_values: VrfTensor<f32, Chip, Cluster, Rows, m![Ds]> = device
        .sub
        .begin(new_values.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let total: DmTensor<bf16, Chip, Cluster, Rows, m![Ds]> = device
        .main
        .begin(old_values.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &new_values)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, Rows, m![Ds]> = device
        .main
        .begin(total.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_div(&sum)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
