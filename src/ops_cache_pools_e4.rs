//! Scalar QKV/attention over canonical whole pools and explicit byte addresses.
//! Numerical operations match the scalar control; no cache copy or host window is used.
use crate::Chip;
use crate::axes::*;
use crate::device_e154::layout::{self, Cluster, Replicated, Slice};
use crate::device_e154::{full, shared, sliding};
use crate::ops_pool_views::Rc64;
use crate::ops_request_sliding_r64_fp8_scales::Scale8;
use furiosa_opt_std::prelude::*;
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
    scale_offsets: &HbmTensor<i32, Chip, m![Ns]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    k_cache: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    k_scales: &mut HbmTensor<f32, Chip, m![Rc64, Ns, Ts, Scale8]>,
    v_scales: &mut HbmTensor<f32, Chip, m![Rc64, Ns, Ts, Scale8]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    let x: DmTensor<bf16, Chip, m![2], m![1 # 32, H / 480], m![H % 480]> =
        x.to_dm(&mut device.tdma);
    let x = shared::rmsnorm::normalize_sharded_inter_slice(device, &x, input_rms_weight);
    let (q, k, v) = sliding::projection::project_qkv_native(
        device,
        &x,
        q_weight,
        k_weight,
        v_weight,
        q_weight_scale,
        k_weight_scale,
        v_weight_scale,
    );

    let q: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]> =
        sliding::rmsnorm::normalize_query(device, &q, q_rms_weight);
    let k: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> =
        sliding::rmsnorm::normalize_key(device, &k, k_rms_weight);
    let v: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> =
        sliding::rmsnorm::normalize_value(device, &v);

    let (q, k) = sliding::rope::apply_rope(device, &q, &k, rope_offset, cos, sin);

    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    type CodecRows = m![Ns,1#32];
    let x: DmTensor<bf16, Chip, Cluster, CodecRows, m![1, Ds]> = k.to_dm(&mut device.tdma);
    let (q, scale) = crate::ops_kv_e4::quantize(device, &x);
    k_cache
        .view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(kv_offset.view())
        .from_dm(&mut device.tdma, q);
    let scale = crate::ops_kv_e4_pool::repeat(device, &scale);
    k_scales
        .view_mut()
        .scatter::<m![Rc64, Ns, Ts], m![Scale8]>()
        .by_byte_offsets(scale_offsets.view())
        .from_dm(&mut device.tdma, scale);
    let x: DmTensor<bf16, Chip, Cluster, CodecRows, m![1, Ds]> = v.to_dm(&mut device.tdma);
    let (q, scale) = crate::ops_kv_e4::quantize(device, &x);
    v_cache
        .view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(kv_offset.view())
        .from_dm(&mut device.tdma, q);
    let scale = crate::ops_kv_e4_pool::repeat(device, &scale);
    v_scales
        .view_mut()
        .scatter::<m![Rc64, Ns, Ts], m![Scale8]>()
        .by_byte_offsets(scale_offsets.view())
        .from_dm(&mut device.tdma, scale);
}
