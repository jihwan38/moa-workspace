//! Scalar QKV/attention entrypoints over canonical cache windows.
//! Arithmetic matches ops.rs; cache tensors borrow a pool allocation without copying.
use furiosa_opt_std::prelude::*;
use crate::axes::*;
use crate::device::layout::{self, Cluster, Replicated, Slice};
use crate::device::{full, shared, sliding};
use crate::Chip;
#[path = "cache_views/sliding_attention.rs"]
mod sliding_view;
#[path = "cache_views/full_attention.rs"]
mod full_view;
#[device(chip = 1)]
pub fn sliding_project_qkv(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x.to_dm(&mut device.tdma);
    let x = shared::rmsnorm::normalize(device, &x, input_rms_weight);

    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]> = layout::broadcast_hidden(device, &x);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> =
        sliding::projection::project_query(device, &x, q_weight, q_weight_scale);
    let (k, v) = sliding::projection::project_key_value(device, &x, k_weight, v_weight, k_weight_scale, v_weight_scale);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> =
        sliding::rmsnorm::normalize_query(device, &q, q_rms_weight);
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> = sliding::rmsnorm::normalize_key(device, &k, k_rms_weight);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> = sliding::rmsnorm::normalize_value(device, &v);

    let (q, k) = sliding::rope::apply_rope(device, &q, &k, rope_offset, cos, sin);

    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k_cache
        .scatter::<m![Ts], m![Ns, Ds]>()
        .by_byte_offsets(kv_offset.view())
        .from_dm(&mut device.tdma, k);
    v_cache
        .scatter::<m![Ts], m![Ns, Ds]>()
        .by_byte_offsets(kv_offset.view())
        .from_dm(&mut device.tdma, v);
}

#[device(chip = 1)]
pub fn full_project_qkv(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qf]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Pf]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x.to_dm(&mut device.tdma);
    let x = shared::rmsnorm::normalize(device, &x, input_rms_weight);

    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]> = layout::broadcast_hidden(device, &x);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> =
        full::projection::project_query(device, &x, q_weight, q_weight_scale);
    let k_raw: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> =
        full::projection::project_key(device, &x, k_weight, k_weight_scale);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = full::rmsnorm::normalize_query(device, &q, q_rms_weight);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = full::rmsnorm::normalize_value(device, &k_raw);
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = full::rmsnorm::normalize_key(device, &k_raw, k_rms_weight);

    let (q, k) = full::rope::apply_rope(device, &q, &k, rope_offset, cos, sin);

    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k_cache
        .scatter::<m![Tf], m![Df]>()
        .by_byte_offsets(kv_offset.view())
        .from_dm(&mut device.tdma, k);
    v_cache
        .scatter::<m![Tf], m![Df]>()
        .by_byte_offsets(kv_offset.view())
        .from_dm(&mut device.tdma, v);
}

#[device(chip = 1)]
pub fn sliding_attention(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    k: HbmTensorView<'_, bf16, Chip, m![Ts # 32768, Ns, Ds]>,
    v: HbmTensorView<'_, bf16, Chip, m![Ts # 32768, Ns, Ds]>,
    mask: &HbmTensor<f32, Chip, m![Ts]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    sliding_view::attend(device, q, k, v, mask, out_hbm);
}

#[device(chip = 1)]
pub fn full_attention_first_page(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: HbmTensorView<'_, bf16, Chip, m![Tf # 131072, Df]>,
    v: HbmTensorView<'_, bf16, Chip, m![Tf # 131072, Df]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![Gf]>,
    running_sum: &mut HbmTensor<f32, Chip, m![Gf]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    full_view::attend_first_page(device, q, k, v, mask, running_max, running_sum, out_hbm);
}

#[device(chip = 1)]
pub fn full_attention_page(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: HbmTensorView<'_, bf16, Chip, m![Tf # 131072, Df]>,
    v: HbmTensorView<'_, bf16, Chip, m![Tf # 131072, Df]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![Gf]>,
    running_sum: &mut HbmTensor<f32, Chip, m![Gf]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    full_view::attend_next_page(device, q, k, v, mask, running_max, running_sum, out_hbm);
}

