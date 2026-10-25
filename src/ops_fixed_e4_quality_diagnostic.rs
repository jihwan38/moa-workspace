//! Diagnostic-only original scalar producer bodies from clean physical32 base.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::*,ops_pool_views::Rc64};
use crate::device_e154::layout::{Cluster,Slice};
use crate::device_e154::{shared,sliding};
#[device(chip = 1)]
pub fn scalar_canonical(
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
pub fn scalar_legacy(
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
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
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

    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k_cache
        .view_mut()
        .scatter::<m![Ts], m![Ns, Ds]>()
        .by_byte_offsets(kv_offset.view())
        .from_dm(&mut device.tdma, k);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> = v.to_dm(&mut device.tdma);
    v_cache
        .view_mut()
        .scatter::<m![Ts], m![Ns, Ds]>()
        .by_byte_offsets(kv_offset.view())
        .from_dm(&mut device.tdma, v);
}

#[device(chip=1)]
pub fn codec_b16(device:&mut Device,input:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,out:&mut HbmTensor<bf16,Chip,m![B,Ns,Ds]>){
 let x:DmTensor<bf16,Chip,m![1#2],m![B,Ns,1#2],m![Ds]>=input.to_dm(&mut device.tdma);
 let x=crate::ops_fixed_e4_quality::roundtrip_ds(device,&x);
 x.view().to_hbm_view(&mut device.tdma,out.view_mut());
}

#[device(chip=1)]
pub fn codec_r64(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,Ns,Ds]>,out:&mut HbmTensor<bf16,Chip,m![R64,Ns,Ds]>){
 let x:DmTensor<bf16,Chip,m![R64/32],m![R64/2%16,Ns,1#2],m![{m![R64%2]},Ds]>=input.to_dm(&mut device.tdma);
 let x=crate::ops_fixed_e4_quality::roundtrip_rows::<m![R64/32],m![R64/2%16,Ns,1#2],m![R64%2]>(device,&x);
 x.view().to_hbm_view(&mut device.tdma,out.view_mut());
}

#[device(chip=1)]
pub fn codec_p128(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,Ns,Ds]>,out:&mut HbmTensor<bf16,Chip,m![P128,Ns,Ds]>){
 let x:DmTensor<bf16,Chip,m![P128/64],m![P128/4%16,Ns,1#2],m![{m![P128%4]},Ds]>=input.to_dm(&mut device.tdma);
 let x=crate::ops_fixed_e4_quality::roundtrip_rows::<m![P128/64],m![P128/4%16,Ns,1#2],m![P128%4]>(device,&x);
 x.view().to_hbm_view(&mut device.tdma,out.view_mut());
}
