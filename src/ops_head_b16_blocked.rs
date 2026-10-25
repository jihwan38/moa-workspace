//! Original B head arithmetic with producer-blocked raw inputs.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{B,Ns,Gs,Ds,Gf,Df,E},
 ops_qkv_blocked_b16::{QueryOutput,KeyOutput,FullQueryOutput,FullKeyOutput},
 ops_head_batch::{rms_ds,weight_ds,norm_ds,value_ds,rope_ds,rms_df,weight_df,norm_df,value_df,rope_df}};
use crate::device::layout::{Cluster,Slice};
#[device(chip = 1)]
pub fn sliding_heads_b16(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, QueryOutput>,
    k_raw: &HbmTensor<bf16, Chip, KeyOutput>,
    v_raw: &HbmTensor<bf16, Chip, KeyOutput>,
    q_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![B]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
) {
    type QRows = m![B, Ns, Gs];
    type KRows = m![B,Ns,1 # 2];
    let q = unsafe { q_raw.view().reshape::<Chip, m![B/8,Ns,Gs,Ds/64,B%8,Ds%64]>() };
    let k = unsafe { k_raw.view().reshape::<Chip, m![B/8,Ns,Ds/32,B%8,Ds%32]>() };
    let q: DmTensor<bf16, Chip, Cluster, QRows, m![Ds]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KRows, m![Ds]> = k.to_dm(&mut device.tdma);
    let qr = rms_ds(device, &q);
    let kr = rms_ds(device, &k);
    let v = unsafe { v_raw.view().reshape::<Chip, m![B/8,Ns,Ds/32,B%8,Ds%32]>() };
    let v: DmTensor<bf16, Chip, Cluster, KRows, m![Ds]> = v.to_dm(&mut device.tdma);
    let vr = rms_ds(device, &v);
    let v = value_ds(device, &v, &vr);
    let qw = weight_ds(device, q_weight);
    let kw = weight_ds(device, k_weight);
    let q = norm_ds(device, &q, &qr, &qw);
    let k = norm_ds(device, &k, &kr, &kw);
    let cos: DmTensor<bf16, Chip, Cluster, Slice, m![B, Ds]> = cos
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let cos: DmTensor<bf16, Chip, Cluster, QRows, m![Ds]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, Slice, m![B, Ds]> = sin
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, QRows, m![Ds]> = sin.to_dm(&mut device.tdma);
    let q = rope_ds(device, &q, &cos, &sin);
    let cos: DmTensor<bf16, Chip, Cluster, KRows, m![Ds]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KRows, m![Ds]> = sin.to_dm(&mut device.tdma);
    let k = rope_ds(device, &k, &cos, &sin);
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.view().to_hbm_view(&mut device.tdma, k_out.view_mut());
    v.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}

#[device(chip = 1)]
pub fn full_heads_b16(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, FullQueryOutput>,
    k_raw: &HbmTensor<bf16, Chip, FullKeyOutput>,
    q_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_weight: &HbmTensor<bf16, Chip, m![Df]>,
    rope_offsets: &HbmTensor<i32, Chip, m![B]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![B, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![B, Df]>,
) {
    type QRows = m![B, Gf];
    type KRows = m![B,1 # 16];
    let q = unsafe { q_raw.view().reshape::<Chip, m![B/8,Gf,Df/128,B%8,Df%128]>() };
    let k = unsafe { k_raw.view().reshape::<Chip, m![B/8,Df/8,B%8,Df%8]>() };
    let q: DmTensor<bf16, Chip, Cluster, QRows, m![Df]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KRows, m![Df]> = k.to_dm(&mut device.tdma);
    let qr = rms_df(device, &q);
    let kr = rms_df(device, &k);
    let v = value_df(device, &k, &kr);
    let qw = weight_df(device, q_weight);
    let kw = weight_df(device, k_weight);
    let q = norm_df(device, &q, &qr, &qw);
    let k = norm_df(device, &k, &kr, &kw);
    let cos: DmTensor<bf16, Chip, Cluster, Slice, m![B, Df]> = cos
        .gather::<m![E], m![Df]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let cos: DmTensor<bf16, Chip, Cluster, QRows, m![Df]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, Slice, m![B, Df]> = sin
        .gather::<m![E], m![Df]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, QRows, m![Df]> = sin.to_dm(&mut device.tdma);
    let q = rope_df(device, &q, &cos, &sin);
    let cos: DmTensor<bf16, Chip, Cluster, KRows, m![Df]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KRows, m![Df]> = sin.to_dm(&mut device.tdma);
    let k = rope_df(device, &k, &cos, &sin);
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.view().to_hbm_view(&mut device.tdma, k_out.view_mut());
    v.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}


