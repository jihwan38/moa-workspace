//! Scalar QKV/attention over canonical whole pools and explicit byte addresses.
//! Numerical operations match the scalar control; no cache copy or host window is used.
use crate::Chip;
use crate::axes::*;
use crate::device_e154::layout::{self, Cluster, Replicated, Slice};
use crate::device_e154::{full, shared, sliding};
use crate::ops_pool_views::Rc64;
use crate::ops_request_attention::Fc;
use furiosa_opt_std::prelude::*;
#[path = "cache_pools/full_attention.rs"]
mod full_pool;
#[path = "cache_pools/sliding_attention.rs"]
mod sliding_pool;
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
    k_cache: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    let x: DmTensor<bf16, Chip, m![2], m![1 # 32, H / 480], m![H % 480]> = x.to_dm(&mut device.tdma);
    let x = shared::rmsnorm::normalize_sharded_inter_slice(device, &x, input_rms_weight);
    let (q, k, v) = sliding::projection::project_qkv_native(
        device, &x, q_weight, k_weight, v_weight,
        q_weight_scale, k_weight_scale, v_weight_scale,
    );

    let q: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]> =
        sliding::rmsnorm::normalize_query(device, &q, q_rms_weight);
    let k: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> = sliding::rmsnorm::normalize_key(device, &k, k_rms_weight);
    let v: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> = sliding::rmsnorm::normalize_value(device, &v);

    let (q, k) = sliding::rope::apply_rope(device, &q, &k, rope_offset, cos, sin);
    let k = crate::ops_fixed_e4_quality::maybe_ds(device, k);
    let v = crate::ops_fixed_e4_quality::maybe_ds(device, v);

    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k_cache
        .view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(kv_offset.view())
        .from_dm(&mut device.tdma, k);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> = v.to_dm(&mut device.tdma);
    v_cache
        .view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
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
    k_cache: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
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
        .view_mut()
        .scatter::<m![Fc, Tf], m![Df]>()
        .by_byte_offsets(kv_offset.view())
        .from_dm(&mut device.tdma, k);
    v_cache
        .view_mut()
        .scatter::<m![Fc, Tf], m![Df]>()
        .by_byte_offsets(kv_offset.view())
        .from_dm(&mut device.tdma, v);
}

#[device(chip = 1)]
pub fn sliding_attention(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    base: &HbmTensor<i32, Chip, m![1]>,
    mask: &HbmTensor<f32, Chip, m![Ts]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    sliding_pool::attend(device, q, k, v, base, mask, out_hbm);
}

#[device(chip = 1)]
pub fn full_attention_first_page(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    base: &HbmTensor<i32, Chip, m![1]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![Gf]>,
    running_sum: &mut HbmTensor<f32, Chip, m![Gf]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    full_pool::attend_first_page(
        device,
        q,
        k,
        v,
        base,
        mask,
        running_max,
        running_sum,
        out_hbm,
    );
}

#[device(chip = 1)]
pub fn full_attention_page(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    base: &HbmTensor<i32, Chip, m![1]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![Gf]>,
    running_sum: &mut HbmTensor<f32, Chip, m![Gf]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    full_pool::attend_next_page(
        device,
        q,
        k,
        v,
        base,
        mask,
        running_max,
        running_sum,
        out_hbm,
    );
}
