//! Exact same-cluster store grouping after unchanged Q4/H480 norm/quantization.
//! Canonical projection/QKV/FFN interfaces are retained; only hidden state moves.
use crate::ops_batch_layouts::P512H960;
use crate::ops_norm_blocked_residual::{Residual, load, store};
use crate::ops_norm_quarters::{
    Q4, load_projected, load_residual, normalize, quantize_shards, residual_add,
};
use crate::ops_output_p128_h64_blocked::Output;
use crate::{
    Chip,
    axes::{H, P128, P512},
};
use furiosa_opt_std::prelude::*;
type Cluster = m![P128 / 64];
type Shards = m![P128 / 2 % 32, H / 480];
type Local = m![Q4, P128 % 2, H % 480];
fn quantize_quarter(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, Shards, Local>,
    group: usize,
    output: &mut HbmTensor<f8e4m3, Chip, m![P128, H]>,
    scale: &mut HbmTensor<f32, Chip, m![P128,1#8]>,
) {
    let mut quarter: DmTensor<bf16, Chip, Cluster, Shards, m![Q4 = 1, P128 % 2, H % 480]> =
        DmTensor::new();
    input
        .view()
        .tile::<m![Q4], 1, m![Q4=1#4,P128%2,H%480]>(group)
        .to_dm_view(&mut device.tdma, quarter.view_mut());
    let input: DmTensor<bf16, Chip, Cluster, Shards, m![P128 % 2, H % 480]> =
        unsafe { quarter.reshape() };
    let input: DmTensor<bf16, Chip, Cluster, m![P128%64#256], m![H]> =
        input.to_dm(&mut device.tdma);
    let (q, s) = crate::ops_query_native_fp8::quantize_hidden(device, &input);
    q.view().to_hbm_view(&mut device.tdma, output.view_mut());
    s.view().to_hbm_view(&mut device.tdma, scale.view_mut());
}
#[device(chip = 1)]
pub fn normalize_qkv_four(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, Residual>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    qa: &mut HbmTensor<f8e4m3, Chip, m![P128, H]>,
    sa: &mut HbmTensor<f32, Chip, m![P128,1#8]>,
    qb: &mut HbmTensor<f8e4m3, Chip, m![P128, H]>,
    sb: &mut HbmTensor<f32, Chip, m![P128,1#8]>,
    qc: &mut HbmTensor<f8e4m3, Chip, m![P128, H]>,
    sc: &mut HbmTensor<f32, Chip, m![P128,1#8]>,
    qd: &mut HbmTensor<f8e4m3, Chip, m![P128, H]>,
    sd: &mut HbmTensor<f32, Chip, m![P128,1#8]>,
) {
    let input = load(device, input);
    let input = normalize(device, &input, weight);
    quantize_quarter(device, &input, 0, qa, sa);
    quantize_quarter(device, &input, 1, qb, sb);
    quantize_quarter(device, &input, 2, qc, sc);
    quantize_quarter(device, &input, 3, qd, sd);
}
fn post_attention(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, Shards, Local>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, Residual>,
    q: &mut HbmTensor<f8e4m3, Chip, P512H960>,
    scale: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    let input = normalize(device, input, post);
    let previous = load(device, residual);
    let updated = residual_add(device, &input, &previous);
    let normalized = normalize(device, &updated, pre);
    let (quantized, scales) = quantize_shards(device, &normalized);
    let quantized: DmTensor<
        f8e4m3,
        Chip,
        m![P512 / 64 % 2],
        m![P512 / 2 % 32, H / 480],
        m![P512 / 128, P512 % 2, H % 480],
    > = unsafe { quantized.reshape() };
    // Real same-cluster placement: join eight P2 rows and two H480 halves.
    // Each live destination writes four contiguous P16*H960 stripes.
    let quantized: DmTensor<f8e4m3, Chip, m![P512/64%2],
        m![H/960,P512/16%4,1#16], m![P512/128,P512%16,H%960]> =
        quantized.to_dm(&mut device.tdma);
    quantized.view().to_hbm_view(&mut device.tdma, q.view_mut());
    let scales: DmTensor<f32, Chip, m![P512 / 64 % 2], m![P512/2%32,1#8], m![P512/128,P512%2,1#8]> =
        unsafe { scales.reshape() };
    // Same supported scalar transpose as the current hidden-scale producer;
    // compact the eight-lane scalar packet before the real leader DM gather.
    let scales: DmTensor<f32, Chip, m![P512/64%2], m![P512/2%32,1#8],
        m![P512/128,P512%2]> = device.main.begin(scales.view())
        .fetch::<m![P512/128,P512%2],m![1#8]>()
        .collect::<m![P512/128,P512%2],m![1#8]>()
        .transpose::<m![P512/128],m![P512%2#8]>()
        .commit_trim::<m![P512%2]>().commit();
    // 1#256 has one live destination per cluster, not 256 replicas.
    let scales: DmTensor<f32, Chip, m![P512/64%2],m![1#256],
        m![P512/128,P512%64]> = scales.to_dm(&mut device.tdma);
    scales.view().to_hbm_view(&mut device.tdma, scale.view_mut());
    // Same rounded state feeds quantization and the persisted FFN residual.
    store(device, updated, residual);
}
pub(crate) fn post_attention_sliding_stage(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, Output>,
    b: &HbmTensor<bf16, Chip, Output>,
    c: &HbmTensor<bf16, Chip, Output>,
    d: &HbmTensor<bf16, Chip, Output>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, Residual>,
    q: &mut HbmTensor<f8e4m3, Chip, P512H960>,
    scale: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    let mut input: DmTensor<
        bf16,
        Chip,
        Cluster,
        m![P128/2%32,H/960,1#2],
        m![Q4, P128 % 2, H % 960],
    > = DmTensor::new();
    load_projected(device, a, 0, &mut input);
    load_projected(device, b, 1, &mut input);
    load_projected(device, c, 2, &mut input);
    load_projected(device, d, 3, &mut input);
    let input: DmTensor<bf16, Chip, Cluster, Shards, Local> = input.to_dm(&mut device.tdma);
    post_attention(device, &input, post, pre, residual, q, scale);
}

#[device(chip = 1)]
pub fn post_attention_sliding(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, Output>,
    b: &HbmTensor<bf16, Chip, Output>,
    c: &HbmTensor<bf16, Chip, Output>,
    d: &HbmTensor<bf16, Chip, Output>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, Residual>,
    q: &mut HbmTensor<f8e4m3, Chip, P512H960>,
    scale: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    post_attention_sliding_stage(device, a, b, c, d, post, pre, residual, q, scale);
}
pub(crate) fn post_attention_full_stage(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![P128, H]>,
    b: &HbmTensor<bf16, Chip, m![P128, H]>,
    c: &HbmTensor<bf16, Chip, m![P128, H]>,
    d: &HbmTensor<bf16, Chip, m![P128, H]>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, Residual>,
    q: &mut HbmTensor<f8e4m3, Chip, P512H960>,
    scale: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    let mut input: DmTensor<bf16, Chip, Cluster, Shards, Local> = DmTensor::new();
    load_residual(device, a, 0, &mut input);
    load_residual(device, b, 1, &mut input);
    load_residual(device, c, 2, &mut input);
    load_residual(device, d, 3, &mut input);
    post_attention(device, &input, post, pre, residual, q, scale);
}

#[device(chip = 1)]
pub fn post_attention_full(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![P128, H]>,
    b: &HbmTensor<bf16, Chip, m![P128, H]>,
    c: &HbmTensor<bf16, Chip, m![P128, H]>,
    d: &HbmTensor<bf16, Chip, m![P128, H]>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, Residual>,
    q: &mut HbmTensor<f8e4m3, Chip, P512H960>,
    scale: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    post_attention_full_stage(device, a, b, c, d, post, pre, residual, q, scale);
}
