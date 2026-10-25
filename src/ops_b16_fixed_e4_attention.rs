//! Approximate fixed E4 append and stored history, original BF16 attention and O.
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
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, crate::ops_output_b16_hsplit::SlidingOutput>,
) {
    crate::ops_b16_fixed_e4_commit::commit_sliding_pool::<VALID>(
        device,
        k,
        v,
        commit_offsets,
        pool_k,
        pool_v,
    );
    crate::ops_b16_fixed_e4_reader::sliding_attention_b16_stage(
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
