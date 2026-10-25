//! Exact post-attention + residual + pre-FFN normalization + blocked FP8 ingress.
//! Retains the original H480 sum tree and each BF16 boundary; only intermediate
//! HBM tensors and host launches disappear. Every residual remains independently
//! owned, including four chronological chunks of one request.
use crate::{Chip, axes::{H, P128, P512}};
use crate::ops_batch_layouts::P512H960;
use crate::ops_norm_wide::{normalize, residual_add};
use furiosa_opt_std::prelude::*;

type Cluster = m![P128 / 64];
type Shards = m![P128 / 2 % 32, H / 480];
type QuantRows = m![P128 % 64 # 256];

fn one(
    device: &mut Device,
    projected: &HbmTensor<bf16, Chip, m![P128, H]>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    group: usize,
    quantized: &mut HbmTensor<f8e4m3, Chip, P512H960>,
    scales: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    let projected: DmTensor<bf16, Chip, Cluster, Shards, m![P128 % 2, H % 480]> =
        projected.to_dm(&mut device.tdma);
    let post = normalize(device, &projected, post);
    let residual_dm = residual_add(device, &post, residual);
    // This rounded value is needed by the later FFN residual, and must survive
    // a layer yield independently for each group member.
    residual_dm.view().to_hbm_view(&mut device.tdma, residual.view_mut());
    let pre = normalize(device, &residual_dm, pre);
    // Redistribution changes placement only. Quantization sees the same BF16
    // row values as the original normalized-HBM -> pack -> quantize chain.
    let pre: DmTensor<bf16, Chip, Cluster, QuantRows, m![H]> =
        pre.to_dm(&mut device.tdma);
    let (q, scale) = crate::ops_query_native_fp8::quantize_hidden(device, &pre);
    let q: DmTensor<f8e4m3, Chip, m![P512%256=128/64],
        m![P512%256=128%64#256], m![H]> = unsafe { q.reshape() };
    q.view().to_hbm_view(&mut device.tdma,
        quantized.view_mut()
            .tile::<m![P512/256], 1,
                m![P512/256=1#{!}2, H/960, P512%256, H%960]>(group / 2)
            .tile::<m![P512%256], 128,
                m![P512/256=1#{!}2, H/960, P512%256=128#{!}256, H%960]>(128 * (group % 2)));
    let scale: DmTensor<f32, Chip, m![P512=128/64],
        m![P512=128%64#256], m![1#8]> = unsafe { scale.reshape() };
    scale.view().to_hbm_view(&mut device.tdma,
        scales.view_mut().tile::<m![P512], 128, m![P512=128#{!}512]>(128 * group));
}

#[device(chip = 1)]
pub fn post_attention_ffn_input_four(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![P128, H]>,
    b: &HbmTensor<bf16, Chip, m![P128, H]>,
    c: &HbmTensor<bf16, Chip, m![P128, H]>,
    d: &HbmTensor<bf16, Chip, m![P128, H]>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    ra: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    rb: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    rc: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    rd: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    quantized: &mut HbmTensor<f8e4m3, Chip, P512H960>,
    scales: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    one(device, a, post, pre, ra, 0, quantized, scales);
    one(device, b, post, pre, rb, 1, quantized, scales);
    one(device, c, post, pre, rc, 2, quantized, scales);
    one(device, d, post, pre, rd, 3, quantized, scales);
}

fn one_blocked(
    device: &mut Device,
    projected: &HbmTensor<bf16, Chip, crate::ops_output_p128_h64_blocked::Output>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    group: usize,
    quantized: &mut HbmTensor<f8e4m3, Chip, P512H960>,
    scales: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    let projected: DmTensor<bf16, Chip, Cluster, m![P128/2%32,H/960,1#2], m![P128%2,H%960]> = projected.to_dm(&mut device.tdma);
    let projected: DmTensor<bf16, Chip, Cluster, Shards, m![P128 % 2, H % 480]> = projected.to_dm(&mut device.tdma);
    let post = normalize(device, &projected, post);
    let residual_dm = residual_add(device, &post, residual);
    // This rounded value is needed by the later FFN residual, and must survive
    // a layer yield independently for each group member.
    residual_dm.view().to_hbm_view(&mut device.tdma, residual.view_mut());
    let pre = normalize(device, &residual_dm, pre);
    // Redistribution changes placement only. Quantization sees the same BF16
    // row values as the original normalized-HBM -> pack -> quantize chain.
    let pre: DmTensor<bf16, Chip, Cluster, QuantRows, m![H]> =
        pre.to_dm(&mut device.tdma);
    let (q, scale) = crate::ops_query_native_fp8::quantize_hidden(device, &pre);
    let q: DmTensor<f8e4m3, Chip, m![P512%256=128/64],
        m![P512%256=128%64#256], m![H]> = unsafe { q.reshape() };
    q.view().to_hbm_view(&mut device.tdma,
        quantized.view_mut()
            .tile::<m![P512/256], 1,
                m![P512/256=1#{!}2, H/960, P512%256, H%960]>(group / 2)
            .tile::<m![P512%256], 128,
                m![P512/256=1#{!}2, H/960, P512%256=128#{!}256, H%960]>(128 * (group % 2)));
    let scale: DmTensor<f32, Chip, m![P512=128/64],
        m![P512=128%64#256], m![1#8]> = unsafe { scale.reshape() };
    scale.view().to_hbm_view(&mut device.tdma,
        scales.view_mut().tile::<m![P512], 128, m![P512=128#{!}512]>(128 * group));
}

#[device(chip = 1)]
pub fn post_attention_ffn_input_four_blocked(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, crate::ops_output_p128_h64_blocked::Output>,
    b: &HbmTensor<bf16, Chip, crate::ops_output_p128_h64_blocked::Output>,
    c: &HbmTensor<bf16, Chip, crate::ops_output_p128_h64_blocked::Output>,
    d: &HbmTensor<bf16, Chip, crate::ops_output_p128_h64_blocked::Output>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    ra: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    rb: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    rc: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    rd: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    quantized: &mut HbmTensor<f8e4m3, Chip, P512H960>,
    scales: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    one_blocked(device, a, post, pre, ra, 0, quantized, scales);
    one_blocked(device, b, post, pre, rb, 1, quantized, scales);
    one_blocked(device, c, post, pre, rc, 2, quantized, scales);
    one_blocked(device, d, post, pre, rd, 3, quantized, scales);
}
