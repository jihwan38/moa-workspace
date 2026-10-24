//! Read one B16 raw-head tile directly from a P128 projection buffer.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::*};
use crate::device::layout::{Cluster,Slice};
use crate::ops_head_batch::{rms_ds,weight_ds,norm_ds,value_ds,rope_ds,rms_df,weight_df,norm_df,value_df,rope_df};
#[device(chip = 1)]
pub fn sliding_heads_from_wide_tile(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![P128, Qs]>,
    k_raw: &HbmTensor<bf16, Chip, m![P128, Ps]>,
    v_raw: &HbmTensor<bf16, Chip, m![P128, Ps]>,
    tile_index: &HbmTensor<i32,Chip,m![1]>,
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
    let index:DmTensor<i32,Chip,Cluster,Slice,m![1#8]>=tile_index.to_dm(&mut device.tdma);
    let index:SpmTensor<i32,Chip,Cluster,m![1#4],m![1]>=index.to_spm(&mut device.tdma);
    let q:DmTensor<bf16,Chip,Cluster,m![P128%16,Qs/256],m![Qs%256]>=q_raw.gather::<m![P128/16],m![P128%16,Qs]>().by_positions::<m![1]>(&index).to_dm(&mut device.tdma);
    let k:DmTensor<bf16,Chip,Cluster,m![P128%16,Ps/256,1#2],m![Ps%256]>=k_raw.gather::<m![P128/16],m![P128%16,Ps]>().by_positions::<m![1]>(&index).to_dm(&mut device.tdma);
    let q:DmTensor<bf16,Chip,Cluster,QRows,m![Ds]>=unsafe{q.reshape()};
    let k:DmTensor<bf16,Chip,Cluster,KRows,m![Ds]>=unsafe{k.reshape()};
    let qr = rms_ds(device, &q);
    let kr = rms_ds(device, &k);
    let v:DmTensor<bf16,Chip,Cluster,m![P128%16,Ps/256,1#2],m![Ps%256]>=v_raw.gather::<m![P128/16],m![P128%16,Ps]>().by_positions::<m![1]>(&index).to_dm(&mut device.tdma);
    let v:DmTensor<bf16,Chip,Cluster,KRows,m![Ds]>=unsafe{v.reshape()};
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
pub fn full_heads_from_wide_tile(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![P128, Qf]>,
    k_raw: &HbmTensor<bf16, Chip, m![P128, Pf]>,
    tile_index: &HbmTensor<i32,Chip,m![1]>,
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
    let index:DmTensor<i32,Chip,Cluster,Slice,m![1#8]>=tile_index.to_dm(&mut device.tdma);
    let index:SpmTensor<i32,Chip,Cluster,m![1#4],m![1]>=index.to_spm(&mut device.tdma);
    let q:DmTensor<bf16,Chip,Cluster,m![P128%16,Qf/512],m![Qf%512]>=q_raw.gather::<m![P128/16],m![P128%16,Qf]>().by_positions::<m![1]>(&index).to_dm(&mut device.tdma);
    let k:DmTensor<bf16,Chip,Cluster,m![P128%16,1#16],m![Pf]>=k_raw.gather::<m![P128/16],m![P128%16,Pf]>().by_positions::<m![1]>(&index).to_dm(&mut device.tdma);
    let q:DmTensor<bf16,Chip,Cluster,QRows,m![Df]>=unsafe{q.reshape()};
    let k:DmTensor<bf16,Chip,Cluster,KRows,m![Df]>=unsafe{k.reshape()};
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

