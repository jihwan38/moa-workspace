//! Exact H480-local quantization after the unchanged grouped normalization chain.
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
    let (q, scale) = quantize_shards(device, &pre);
    let scale: DmTensor<f32, Chip, Cluster, QuantRows, m![1#8]> = scale.to_dm(&mut device.tdma);
    let q: DmTensor<f8e4m3, Chip, m![P512%256=128/64],
        m![P512%256=128/2%32,H/480], m![P512%256=128%2,H%480]> = unsafe { q.reshape() };
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
    let (q, scale) = quantize_shards(device, &pre);
    let scale: DmTensor<f32, Chip, Cluster, QuantRows, m![1#8]> = scale.to_dm(&mut device.tdma);
    let q: DmTensor<f8e4m3, Chip, m![P512%256=128/64],
        m![P512%256=128/2%32,H/480], m![P512%256=128%2,H%480]> = unsafe { q.reshape() };
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

type ScalarRows = m![P128/2%32,1#8];
fn quantize_shards(device:&mut Device,
    x:&DmTensor<bf16,Chip,Cluster,Shards,m![P128%2,H%480]>)
    -> (DmTensor<f8e4m3,Chip,Cluster,Shards,m![P128%2,H%480]>,
        DmTensor<f32,Chip,Cluster,ScalarRows,m![P128%2,1#8]>) {
    let amax:DmTensor<f32,Chip,Cluster,ScalarRows,m![P128%2,1#8]>=device.main.begin(x.view())
        .fetch::<m![P128%2,H/16%30],m![H%16]>().fetch_cast::<f32>()
        .collect::<m![P128%2,H/8%60],m![H%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32).vector_reinterpret::<f32>()
        .vector_narrow_split::<m![P128%2,H/4%120],m![H%4]>()
        .vector_intra_slice_reduce::<H,m![P128%2],m![1#4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1#8]>()
        .vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
        .vector_inter_slice_reduce::<ScalarRows,m![P128%2]>(InterSliceReduceOpF32::Max)
        .vector_final().commit_trim::<m![1#8]>().commit();
    let scale:DmTensor<f32,Chip,Cluster,ScalarRows,m![P128%2,1#8]>=device.main.begin(amax.view())
        .fetch::<m![P128%2],m![1#8]>().collect::<m![P128%2],m![1#8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7f80_0000i32).vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1#4]>().vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),0.0078125f32)
        .vector_widen_pad::<m![1#8]>().vector_final().commit_trim::<m![1#8]>().commit();
    let broadcast:DmTensor<f32,Chip,Cluster,Shards,m![P128%2,1#8]>=scale.to_dm(&mut device.tdma);
    let sv:VrfTensor<f32,Chip,Cluster,Shards,m![P128%2,1#8]>=device.sub.begin(broadcast.view())
        .fetch::<m![P128%2],m![1#8]>().collect::<m![P128%2],m![1#8]>().to_vrf();
    let q:DmTensor<f8e4m3,Chip,Cluster,Shards,m![P128%2,H%480]>=device.main.begin(x.view())
        .fetch::<m![P128%2,H/16%30],m![H%16]>().fetch_cast::<f32>()
        .collect::<m![P128%2,H/8%60],m![H%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%2,H/4%120],m![H%4]>()
        .vector_fp_binary(FpBinaryOp::DivF,&sv)
        .vector_widen_concat::<m![P128%2,H/8%60],m![H%8]>()
        .vector_final().cast::<f8e4m3,m![H%8#32]>().commit_trim::<m![H%8]>().commit();
    (q,scale)
}
