//! B16 non-matmul stages. The row helpers retain scalar decode's arithmetic
//! boundaries while exposing one device launch for a complete prompt chunk.

use crate::device::layout::{Cluster, Slice};
use crate::{
    Chip,
    axes::*,
    device::{full, shared, sliding},
};
use furiosa_opt_std::prelude::*;

#[device(chip = 1)]
pub fn rmsnorm_b16(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    out: &mut HbmTensor<bf16, Chip, m![B, H]>,
) {
    for row in 0..B::SIZE {
        let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x
            .view()
            .tile::<m![B], 1, m![B=1 # 16,H]>(row)
            .to_dm(&mut device.tdma);
        let x = shared::rmsnorm::normalize(device, &x, weight);
        x.view().pad::<m![B=1 # 16,H]>().to_hbm_view(
            &mut device.tdma,
            out.view_mut().tile::<m![B], 1, m![B=1 #{!} 16,H]>(row),
        );
    }
}

#[device(chip = 1)]
pub fn post_attention_b16(
    device: &mut Device,
    projected: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, m![B, H]>,
) {
    for row in 0..B::SIZE {
        let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = projected
            .view()
            .tile::<m![B], 1, m![B=1 # 16,H]>(row)
            .to_dm(&mut device.tdma);
        let x = shared::rmsnorm::normalize(device, &x, weight);
        let previous: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = residual
            .view()
            .tile::<m![B], 1, m![B=1 # 16,H]>(row)
            .to_dm(&mut device.tdma);
        let x = shared::residual::add(device, &x, &previous);
        x.view().pad::<m![B=1 # 16,H]>().to_hbm_view(
            &mut device.tdma,
            residual.view_mut().tile::<m![B], 1, m![B=1 #{!} 16,H]>(row),
        );
    }
}

#[device(chip = 1)]
pub fn post_feedforward_b16(
    device: &mut Device,
    projected: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_gate: &HbmTensor<bf16, Chip, m![1 # 8]>,
    residual: &mut HbmTensor<bf16, Chip, m![B, H]>,
) {
    for row in 0..B::SIZE {
        let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = projected
            .view()
            .tile::<m![B], 1, m![B=1 # 16,H]>(row)
            .to_dm(&mut device.tdma);
        let x = shared::rmsnorm::normalize(device, &x, weight);
        let previous: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = residual
            .view()
            .tile::<m![B], 1, m![B=1 # 16,H]>(row)
            .to_dm(&mut device.tdma);
        let x = shared::residual::add(device, &x, &previous);
        let x = shared::residual::scale_by_layer_gate(device, &x, layer_gate);
        x.view().pad::<m![B=1 # 16,H]>().to_hbm_view(
            &mut device.tdma,
            residual.view_mut().tile::<m![B], 1, m![B=1 #{!} 16,H]>(row),
        );
    }
}

/// Copy one independently encoded media embedding into a prompt row. The caller
/// supplies an HBM byte offset, preserving ownership and original encoder scale.
#[device(chip = 1)]
pub fn insert_hidden_row(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![H]>,
    byte_offset: &HbmTensor<i32, Chip, m![1]>,
    batch: &mut HbmTensor<bf16, Chip, m![B, H]>,
) {
    let row: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = source.to_dm(&mut device.tdma);
    batch
        .view_mut()
        .scatter::<m![B], m![H]>()
        .by_byte_offsets(byte_offset.view())
        .from_dm(&mut device.tdma, row);
}

#[device(chip = 1)]
pub fn extract_hidden_row(
    device: &mut Device,
    batch: &HbmTensor<bf16, Chip, m![B, H]>,
    byte_offset: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let row: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = batch
        .gather::<m![B], m![H]>()
        .by_byte_offsets(byte_offset)
        .to_dm(&mut device.tdma);
    row.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// Normalize projected heads and rotate each position without touching shared KV.
#[device(chip = 1)]
pub fn sliding_heads_b16(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![B, Qs]>,
    k_raw: &HbmTensor<bf16, Chip, m![B, Ps]>,
    v_raw: &HbmTensor<bf16, Chip, m![B, Ps]>,
    q_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![B]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
) {
    let cos: DmTensor<bf16, Chip, Cluster, Slice, m![B, Ds]> = cos
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, Slice, m![B, Ds]> = sin
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    for row in 0..B::SIZE {
        let q: DmTensor<bf16, Chip, Cluster, Slice, m![Qs]> = q_raw
            .view()
            .tile::<m![B], 1, m![B=1 # 16,Qs]>(row)
            .to_dm(&mut device.tdma);
        let k: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = k_raw
            .view()
            .tile::<m![B], 1, m![B=1 # 16,Ps]>(row)
            .to_dm(&mut device.tdma);
        let q = unsafe { q.reshape::<Chip, Cluster, Slice, m![Ns, Gs, Ds]>() };
        let k = unsafe { k.reshape::<Chip, Cluster, Slice, m![Ns, Ds]>() };
        let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = v_raw
            .view()
            .tile::<m![B], 1, m![B=1 # 16,Ps]>(row)
            .to_dm(&mut device.tdma);
        let v = unsafe { v.reshape::<Chip, Cluster, Slice, m![Ns, Ds]>() };
        let v = sliding::rmsnorm::normalize_value(device, &v);
        let q = sliding::rmsnorm::normalize_query(device, &q, q_weight);
        let k = sliding::rmsnorm::normalize_key(device, &k, k_weight);
        let mut cos_row: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = DmTensor::new();
        cos.view()
            .tile::<m![B], 1, m![B=1 # 16,Ds]>(row)
            .to_dm_view(&mut device.tdma, cos_row.view_mut());
        let mut sin_row: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = DmTensor::new();
        sin.view()
            .tile::<m![B], 1, m![B=1 # 16,Ds]>(row)
            .to_dm_view(&mut device.tdma, sin_row.view_mut());
        let (q, k) = sliding::rope::apply_rope_values(device, &q, &k, &cos_row, &sin_row);
        q.view().pad::<m![B=1 # 16,Ns,Gs,Ds]>().to_hbm_view(
            &mut device.tdma,
            q_out
                .view_mut()
                .tile::<m![B], 1, m![B=1 #{!} 16,Ns,Gs,Ds]>(row),
        );
        k.view().pad::<m![B=1 # 16,Ns,Ds]>().to_hbm_view(
            &mut device.tdma,
            k_out
                .view_mut()
                .tile::<m![B], 1, m![B=1 #{!} 16,Ns,Ds]>(row),
        );
        v.view().pad::<m![B=1 # 16,Ns,Ds]>().to_hbm_view(
            &mut device.tdma,
            v_out
                .view_mut()
                .tile::<m![B], 1, m![B=1 #{!} 16,Ns,Ds]>(row),
        );
    }
}

/// Normalize projected heads and rotate each position without touching shared KV.
#[device(chip = 1)]
pub fn full_heads_b16(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![B, Qf]>,
    k_raw: &HbmTensor<bf16, Chip, m![B, Pf]>,
    q_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_weight: &HbmTensor<bf16, Chip, m![Df]>,
    rope_offsets: &HbmTensor<i32, Chip, m![B]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![B, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![B, Df]>,
) {
    let cos: DmTensor<bf16, Chip, Cluster, Slice, m![B, Df]> = cos
        .gather::<m![E], m![Df]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, Slice, m![B, Df]> = sin
        .gather::<m![E], m![Df]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    for row in 0..B::SIZE {
        let q: DmTensor<bf16, Chip, Cluster, Slice, m![Qf]> = q_raw
            .view()
            .tile::<m![B], 1, m![B=1 # 16,Qf]>(row)
            .to_dm(&mut device.tdma);
        let k: DmTensor<bf16, Chip, Cluster, Slice, m![Pf]> = k_raw
            .view()
            .tile::<m![B], 1, m![B=1 # 16,Pf]>(row)
            .to_dm(&mut device.tdma);
        let q = unsafe { q.reshape::<Chip, Cluster, Slice, m![Gf, Df]>() };
        let k = unsafe { k.reshape::<Chip, Cluster, Slice, m![Df]>() };
        let v = full::rmsnorm::normalize_value(device, &k);
        let q = full::rmsnorm::normalize_query(device, &q, q_weight);
        let k = full::rmsnorm::normalize_key(device, &k, k_weight);
        let mut cos_row: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = DmTensor::new();
        cos.view()
            .tile::<m![B], 1, m![B=1 # 16,Df]>(row)
            .to_dm_view(&mut device.tdma, cos_row.view_mut());
        let mut sin_row: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = DmTensor::new();
        sin.view()
            .tile::<m![B], 1, m![B=1 # 16,Df]>(row)
            .to_dm_view(&mut device.tdma, sin_row.view_mut());
        let (q, k) = full::rope::apply_rope_values(device, &q, &k, &cos_row, &sin_row);
        q.view().pad::<m![B=1 # 16,Gf,Df]>().to_hbm_view(
            &mut device.tdma,
            q_out
                .view_mut()
                .tile::<m![B], 1, m![B=1 #{!} 16,Gf,Df]>(row),
        );
        k.view().pad::<m![B=1 # 16,Df]>().to_hbm_view(
            &mut device.tdma,
            k_out.view_mut().tile::<m![B], 1, m![B=1 #{!} 16,Df]>(row),
        );
        v.view().pad::<m![B=1 # 16,Df]>().to_hbm_view(
            &mut device.tdma,
            v_out.view_mut().tile::<m![B], 1, m![B=1 #{!} 16,Df]>(row),
        );
    }
}

/// Commit only the live prefix, with an unpadded index vector required by scatter.
fn commit_sliding_kv<const VALID: usize>(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = { VALID }]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![B = { VALID }, Ns, Ds]> = k
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![B = { VALID }, Ns, Ds]> = v
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    k_cache
        .view_mut()
        .scatter::<m![Ts], m![Ns, Ds]>()
        .by_byte_offsets(offsets.view())
        .from_dm(&mut device.tdma, k);
    v_cache
        .view_mut()
        .scatter::<m![Ts], m![Ns, Ds]>()
        .by_byte_offsets(offsets.view())
        .from_dm(&mut device.tdma, v);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_1(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 1]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<1>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_2(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 2]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<2>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_3(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 3]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<3>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_4(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 4]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<4>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_5(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 5]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<5>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_6(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 6]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<6>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_7(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 7]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<7>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_8(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 8]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<8>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_9(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 9]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<9>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_10(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 10]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<10>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_11(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 11]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<11>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_12(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 12]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<12>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_13(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 13]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<13>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_14(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 14]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<14>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_15(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 15]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<15>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_16(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 16]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    commit_sliding_kv::<16>(device, k, v, offsets, k_cache, v_cache);
}

/// Commit only the live prefix, with an unpadded index vector required by scatter.
fn commit_full_kv<const VALID: usize>(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = { VALID }]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![B = { VALID }, Df]> = k
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Df]>(0)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![B = { VALID }, Df]> = v
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Df]>(0)
        .to_dm(&mut device.tdma);
    k_cache
        .view_mut()
        .scatter::<m![Tf], m![Df]>()
        .by_byte_offsets(offsets.view())
        .from_dm(&mut device.tdma, k);
    v_cache
        .view_mut()
        .scatter::<m![Tf], m![Df]>()
        .by_byte_offsets(offsets.view())
        .from_dm(&mut device.tdma, v);
}

#[device(chip = 1)]
pub fn commit_full_kv_1(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 1]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<1>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_2(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 2]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<2>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_3(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 3]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<3>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_4(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 4]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<4>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_5(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 5]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<5>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_6(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 6]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<6>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_7(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 7]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<7>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_8(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 8]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<8>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_9(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 9]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<9>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_10(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 10]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<10>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_11(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 11]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<11>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_12(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 12]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<12>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_13(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 13]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<13>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_14(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 14]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<14>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_15(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 15]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<15>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_16(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 16]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    commit_full_kv::<16>(device, k, v, offsets, k_cache, v_cache);
}
