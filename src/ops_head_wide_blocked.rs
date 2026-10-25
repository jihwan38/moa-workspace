//! Unchanged P128 head arithmetic consuming producer-blocked Q/K/V directly.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,Ns,Gs,Ds,Gf,Df,E},
 ops_qkv_shared_blocked::{QueryOutput,KeyOutput},
 ops_qk_shared_blocked::{QueryOutput as FullQueryOutput,KeyOutput as FullKeyOutput},
 ops_head_wide::{rms_ds,weight_ds,norm_ds,value_ds,rope_ds,rms_df,weight_df,norm_df,value_df,rope_df}};
type Cluster=m![P128/64];
type RopeRows=m![P128/4%16,1#16];
#[device(chip = 1)]
pub fn sliding_heads_p128(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, QueryOutput>,
    k_raw: &HbmTensor<bf16, Chip, KeyOutput>,
    v_raw: &HbmTensor<bf16, Chip, KeyOutput>,
    q_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![P128]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![P128, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
) {
    type QRows = m![P128/4%16,Ns,Gs];
    type KRows = m![P128/4%16,Ns,1#2];
    let q = unsafe { q_raw.view().reshape::<Chip, m![P128/64,Ns,Gs,Ds/64,P128%64,Ds%64]>() };
    let k = unsafe { k_raw.view().reshape::<Chip, m![P128/64,Ns,Ds/32,P128%64,Ds%32]>() };
    let q: DmTensor<bf16, Chip, Cluster, QRows, m![P128%4,Ds]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Ds]> = k.to_dm(&mut device.tdma);
    let qr = rms_ds(device, &q);
    let kr = rms_ds(device, &k);
    let v = unsafe { v_raw.view().reshape::<Chip, m![P128/64,Ns,Ds/32,P128%64,Ds%32]>() };
    let v: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Ds]> = v.to_dm(&mut device.tdma);
    let vr = rms_ds(device, &v);
    let v = value_ds(device, &v, &vr);
    let qw = weight_ds(device, q_weight);
    let kw = weight_ds(device, k_weight);
    let q = norm_ds(device, &q, &qr, &qw);
    let k = norm_ds(device, &k, &kr, &kw);
    let cos: DmTensor<bf16, Chip, Cluster, RopeRows, m![P128%4,Ds]> = cos
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let cos: DmTensor<bf16, Chip, Cluster, QRows, m![P128%4,Ds]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, RopeRows, m![P128%4,Ds]> = sin
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, QRows, m![P128%4,Ds]> = sin.to_dm(&mut device.tdma);
    let q = rope_ds(device, &q, &cos, &sin);
    let cos: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Ds]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Ds]> = sin.to_dm(&mut device.tdma);
    let k = rope_ds(device, &k, &cos, &sin);
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.view().to_hbm_view(&mut device.tdma, k_out.view_mut());
    v.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}


#[device(chip = 1)]
pub fn full_heads_p128(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, FullQueryOutput>,
    k_raw: &HbmTensor<bf16, Chip, FullKeyOutput>,
    q_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_weight: &HbmTensor<bf16, Chip, m![Df]>,
    rope_offsets: &HbmTensor<i32, Chip, m![P128]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![P128, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![P128, Df]>,
) {
    type QRows = m![P128/4%16,Gf];
    type KRows = m![P128/4%16,1#16];
    let q = unsafe { q_raw.view().reshape::<Chip, m![P128/64,Gf,Df/128,P128%64,Df%128]>() };
    let k = unsafe { k_raw.view().reshape::<Chip, m![P128/64,Df/8,P128%64,Df%8]>() };
    let q: DmTensor<bf16, Chip, Cluster, QRows, m![P128%4,Df]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Df]> = k.to_dm(&mut device.tdma);
    let qr = rms_df(device, &q);
    let kr = rms_df(device, &k);
    let v = value_df(device, &k, &kr);
    let qw = weight_df(device, q_weight);
    let kw = weight_df(device, k_weight);
    let q = norm_df(device, &q, &qr, &qw);
    let k = norm_df(device, &k, &kr, &kw);
    let cos: DmTensor<bf16, Chip, Cluster, RopeRows, m![P128%4,Df]> = cos
        .gather::<m![E], m![Df]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let cos: DmTensor<bf16, Chip, Cluster, QRows, m![P128%4,Df]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, RopeRows, m![P128%4,Df]> = sin
        .gather::<m![E], m![Df]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, QRows, m![P128%4,Df]> = sin.to_dm(&mut device.tdma);
    let q = rope_df(device, &q, &cos, &sin);
    let cos: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Df]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Df]> = sin.to_dm(&mut device.tdma);
    let k = rope_df(device, &k, &cos, &sin);
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.view().to_hbm_view(&mut device.tdma, k_out.view_mut());
    v.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}

