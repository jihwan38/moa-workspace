//! Fixed unscaled E4 storage; exact widening into original BF16 attention arithmetic.
use crate::device::layout::Slice;
use crate::ops_pool_views::Rc64;
use crate::{
    Chip, EPS,
    axes::{B, Ds, Gs, Ns, Ts},
};
use furiosa_opt_std::prelude::*;
type Cluster = m![B / 4 % 2];
// Two B8 tiles split each group across clusters, preserving four local rows,
// scalar32-key BF16 partials and the original32-way interslice reduction.
type SlidingRows = m![B%4,Ns,Gs,1 # 4];
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
pub(crate) fn sliding_attention_b16_stage(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    pool_k: &HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    ring_byte_offsets: &HbmTensor<i32, Chip, m![B]>,
    visibility_mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    out: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
) {
    type KeyRows = m![Ns, Ts / 32];
    type Reduced = m![Ns,1 # 32];
    for group in 0..2 {
        let q: DmTensor<bf16, Chip, Cluster, KeyRows, m![B % 4, Gs, Ds]> = q
            .view()
            .tile::<m![B], 8, m![B=8 # 16,Ns,Gs,Ds]>(group * 8)
            .to_dm(&mut device.tdma);
        let index: DmTensor<i32, Chip, Cluster, Slice, m![B % 4]> = ring_byte_offsets
            .view()
            .tile::<m![B], 8, m![B=8 # 16]>(group * 8)
            .to_dm(&mut device.tdma);
        let index: HbmTensor<i32, Chip, m![B % 8]> = index.to_hbm(&mut device.tdma);
        let k: DmTensor<f8e4m3, Chip, Cluster, KeyRows, m![B % 4, Ts % 32, Ds]> = pool_k
            .gather::<m![Rc64], m![Ts, Ns, Ds]>()
            .by_byte_offsets(&index)
            .to_dm(&mut device.tdma);
        // SDK0.9 lookup exists only on Main. Materialize exact widened codes
        // in DM, then keep the original plain Sub→TRF ingestion.
        let k: DmTensor<bf16, Chip, Cluster, KeyRows, m![B % 4, Ts % 32, Ds]> = device.main.begin(k.view())
            .fetch::<m![B % 4, Ts % 32, Ds / 32], m![Ds % 32]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![B % 4, Ts % 32, Ds / 16], m![Ds % 16]>()
            .commit_trim::<m![Ds % 16]>()
            .commit();
        let kt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Ts % 8], m![B % 4, Ts / 8 % 4, Ds]> =
            device
                .sub
                .begin(k.view())
                .fetch::<m![Ts % 8, B % 4, Ts / 8 % 4, Ds / 16], m![Ds % 16]>()
                .collect::<m![Ts % 8, B % 4, Ts / 8 % 4, Ds / 16], m![Ds % 16]>()
                .to_trf();
        let scores: DmTensor<bf16, Chip, Cluster, KeyRows, m![B % 4, Gs, Ts % 32]> = device
            .main
            .begin(q.view())
            .fetch::<m![B % 4, Gs, Ds / 16], m![Ds % 16]>()
            .collect::<m![B % 4, Gs, Ds / 16], m![Ds % 16]>()
            .contract_outer::<m![B % 4, Gs, Ds / 32, Ts / 8 % 4], m![Ds % 32], _, _, _>(&kt)
            .contract_packet::<m![1]>()
            .contract_time::<m![B % 4, Gs, Ts / 8 % 4]>()
            .contract_lane::<m![B % 4, Gs, Ts / 8 % 4], m![Ts % 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![Ts%8 # 16]>()
            .commit_trim::<m![Ts % 8]>()
            .commit();
        let scores: DmTensor<bf16, Chip, Cluster, SlidingRows, m![Ts]> =
            scores.to_dm(&mut device.tdma);
        let mask: DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> = visibility_mask
            .view()
            .tile::<m![B], 8, m![B=8 # 16,Ts]>(group * 8)
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
        let weights: DmTensor<bf16, Chip, Cluster, KeyRows, m![B % 4, Gs, Ts % 32]> =
            weights.to_dm(&mut device.tdma);

        let v: DmTensor<f8e4m3, Chip, Cluster, KeyRows, m![B % 4, Ts % 32, Ds]> = pool_v
            .gather::<m![Rc64], m![Ts, Ns, Ds]>()
            .by_byte_offsets(&index)
            .to_dm(&mut device.tdma);
        // Preserve the original transpose packet exactly. Main E4 lookup has
        // a full32-code packet and therefore needs this explicit BF16 DM boundary.
        let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![B % 4, Ts % 32, Ds]> = device.main.begin(v.view())
            .fetch::<m![B % 4, Ts % 32, Ds / 32], m![Ds % 32]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![B % 4, Ts % 32, Ds / 16], m![Ds % 16]>()
            .commit_trim::<m![Ds % 16]>()
            .commit();
        let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![B % 4, Ds, Ts % 32]> = device
            .main
            .begin(v.view())
            .fetch::<m![B % 4, Ds / 8, Ts % 32], m![Ds%8 # 16]>()
            .collect::<m![B % 4, Ds / 8, Ts % 32], m![Ds%8 # 16]>()
            .transpose::<m![B % 4, Ds / 8, Ts / 4 % 8, Ds % 8], m![Ts%4 # 16]>()
            .commit_trim::<m![Ts % 4]>()
            .commit();
        let vt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Ds % 8], m![B % 4, Ds / 8, Ts % 32]> =
            device
                .sub
                .begin(v.view())
                .fetch::<m![Ds % 8, B % 4, Ds / 8, Ts / 16 % 2], m![Ts % 16]>()
                .collect::<m![Ds % 8, B % 4, Ds / 8, Ts / 16 % 2], m![Ts % 16]>()
                .to_trf();
        let partial: DmTensor<bf16, Chip, Cluster, KeyRows, m![B % 4, Gs, Ds]> = device
            .main
            .begin(weights.view())
            .fetch::<m![B % 4, Gs, Ts / 16 % 2], m![Ts % 16]>()
            .collect::<m![B % 4, Gs, Ts / 16 % 2], m![Ts % 16]>()
            .contract_outer::<m![B % 4, Gs, Ds / 8], m![Ts % 32], _, _, _>(&vt)
            .contract_packet::<m![1]>()
            .contract_time::<m![B % 4, Gs, Ds / 8]>()
            .contract_lane::<m![B % 4, Gs, Ds / 8], m![Ds % 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![Ds%8 # 16]>()
            .commit_trim::<m![Ds % 8]>()
            .commit();
        let combined: DmTensor<bf16, Chip, Cluster, Reduced, m![B % 4, Gs, Ds]> = device
            .main
            .begin(partial.view())
            .fetch::<m![B % 4, Gs, Ds / 16], m![Ds % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![B % 4, Gs, Ds / 8], m![Ds % 8]>()
            .vector_init()
            .vector_inter_slice_reduce::<Reduced, m![B % 4, Gs, Ds / 8]>(InterSliceReduceOpF32::Add)
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
                .tile::<m![B], 8, m![B=8 #{!} 16,Ns,Gs,Ds]>(group * 8),
        );
    }
}

#[device(chip = 1)]
pub fn sliding_attention_b16(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    pool_k: &HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    ring_byte_offsets: &HbmTensor<i32, Chip, m![B]>,
    visibility_mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    out: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
) {
    sliding_attention_b16_stage(device,q,pool_k,pool_v,ring_byte_offsets,visibility_mask,out);
}

use crate::axes::{H,Qs};
fn sliding<const VALID: usize>(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = { VALID }]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    crate::ops_fixed_e4_commit::commit_sliding_pool::<VALID>(
        device,
        k,
        v,
        commit_offsets,
        pool_k,
        pool_v,
    );
    sliding_attention_b16_stage(
        device,
        q,
        &*pool_k,
        &*pool_v,
        ring_offsets,
        mask,
        attention,
    );
    crate::ops_output_b16_hsplit::project_sliding_output_batch_stage(
        device, attention, weight, scale, output,
    );
}

#[device(chip = 1)]
pub fn sliding_1(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 1]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<1>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_2(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 2]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<2>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_3(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 3]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<3>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_4(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 4]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<4>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_5(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 5]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<5>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_6(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 6]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<6>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_7(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 7]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<7>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_8(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 8]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<8>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_9(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 9]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<9>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_10(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 10]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<10>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_11(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 11]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<11>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_12(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 12]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<12>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_13(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 13]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<13>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_14(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 14]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<14>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_15(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 15]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<15>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

#[device(chip = 1)]
pub fn sliding_16(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 16]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    sliding::<16>(
        device,
        q,
        k,
        v,
        commit_offsets,
        ring_offsets,
        mask,
        pool_k,
        pool_v,
        weight,
        scale,
        attention,
        output,
    );
}

