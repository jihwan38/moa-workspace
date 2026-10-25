//! Exact original commit, attention and output projection in one graph.
//! All HBM boundaries are retained; all1..16 live-prefix gates are instantiated after the14/16 complete win.
use crate::{
    Chip,
    axes::{B, Ds, Gs, H, Ns, Qs, Ts},
    ops_pool_views::Rc64,
};
use furiosa_opt_std::prelude::*;
fn sliding<const VALID: usize>(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = { VALID }]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    crate::ops_commit_r64::commit_sliding_pool::<VALID>(
        device,
        k,
        v,
        commit_offsets,
        pool_k,
        pool_v,
    );
    crate::ops_request_sliding_b16_rc64::sliding_attention_b16_stage(
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
pub fn sliding_14(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 14]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
pub fn sliding_16(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 16]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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

#[device(chip = 1)]
pub fn sliding_1(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 1]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
pub fn sliding_15(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    commit_offsets: &HbmTensor<i32, Chip, m![B = 15]>,
    ring_offsets: &HbmTensor<i32, Chip, m![B]>,
    mask: &HbmTensor<f32, Chip, m![B, Ts]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
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
