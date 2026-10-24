//! Isolated scaled-E4 KV cache experiment. Exact BF16 attention remains the production path.
//! Reconstruction to BF16 occurs before both original contraction boundaries.
use crate::device::layout::Slice;
use crate::ops_pool_views::Rc64;
use crate::{
    Chip, EPS,
    axes::{Ds, Gs, Ns, R64, Ts},
};
use furiosa_opt_std::prelude::*;
axes![Scale8 = 8];
type Cluster = m![R64 / 4 % 2];
// Eight R8 tiles split each group across clusters, preserving four local rows,
// scalar32-key BF16 partials and the original32-way interslice reduction.
type SlidingRows = m![R64%4,Ns,Gs,1 # 4];
type SlidingScalar = DmTensor<f32, Chip, Cluster, SlidingRows, m![1 # 8]>;
fn sliding_scalar(
    device: &mut Device,
    x: &SlidingScalar,
) -> VrfTensor<f32, Chip, Cluster, SlidingRows, m![1 # 8]> {
    device
        .sub
        .begin(x.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}
fn sliding_premax(
    device: &mut Device,
    scores: &DmTensor<bf16, Chip, Cluster, SlidingRows, m![Ts]>,
    mask: &DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]>,
) -> DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> {
    let additive: DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> = device
        .main
        .begin(mask.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ts / 4], m![Ts % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, 1.0f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1e30f32)
        .vector_widen_concat::<m![Ts / 8], m![Ts % 8]>()
        .vector_final()
        .commit_trim::<m![Ts % 8]>()
        .commit();
    let additive: VrfTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> = device
        .sub
        .begin(additive.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .to_vrf();
    device
        .main
        .begin(scores.view())
        .fetch::<m![Ts / 16], m![Ts % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &additive)
        .vector_final()
        .commit_trim::<m![Ts % 8]>()
        .commit()
}
fn sliding_max(
    device: &mut Device,
    scores: &DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]>,
) -> SlidingScalar {
    device
        .main
        .begin(scores.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ts / 4], m![Ts % 4]>()
        .vector_intra_slice_reduce::<Ts, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}
fn sliding_exp(
    device: &mut Device,
    scores: &DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]>,
    max: &SlidingScalar,
    mask: &DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]>,
) -> DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> {
    let max = sliding_scalar(device, max);
    let exp: DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> = device
        .main
        .begin(scores.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ts / 4], m![Ts % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &max)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Ts / 8], m![Ts % 8]>()
        .vector_final()
        .commit_trim::<m![Ts % 8]>()
        .commit();
    let mask: VrfTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> = device
        .sub
        .begin(mask.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .to_vrf();
    device
        .main
        .begin(exp.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Min, &mask)
        .vector_final()
        .commit_trim::<m![Ts % 8]>()
        .commit()
}
fn sliding_sum(
    device: &mut Device,
    exp: &DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]>,
) -> SlidingScalar {
    device
        .main
        .begin(exp.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ts / 4], m![Ts % 4]>()
        .vector_intra_slice_reduce::<Ts, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}
#[device(chip = 1)]
pub fn sliding_attention_r64(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![R64, Ns, Gs, Ds]>,
    pool_k: &HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    scale_k: &HbmTensor<f32, Chip, m![Rc64, Ts, Ns, Scale8]>,
    pool_v: &HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    scale_v: &HbmTensor<f32, Chip, m![Rc64, Ts, Ns, Scale8]>,
    ring_byte_offsets: &HbmTensor<i32, Chip, m![R64]>,
    scale_byte_offsets: &HbmTensor<i32, Chip, m![R64]>,
    visibility_mask: &HbmTensor<f32, Chip, m![R64, Ts]>,
    out: &mut HbmTensor<bf16, Chip, m![R64, Ns, Gs, Ds]>,
) {
    type KeyRows = m![Ns, Ts / 32];
    type Reduced = m![Ns,1 # 32];
    for group in 0..8 {
        let q: DmTensor<bf16, Chip, Cluster, KeyRows, m![R64 % 4, Gs, Ds]> = q
            .view()
            .tile::<m![R64], 8, m![R64=8 # 64,Ns,Gs,Ds]>(group * 8)
            .to_dm(&mut device.tdma);
        let index: DmTensor<i32, Chip, Cluster, Slice, m![R64 % 4]> = ring_byte_offsets
            .view()
            .tile::<m![R64], 8, m![R64=8 # 64]>(group * 8)
            .to_dm(&mut device.tdma);
        let index: HbmTensor<i32, Chip, m![R64 % 8]> = index.to_hbm(&mut device.tdma);
        let scale_index: DmTensor<i32, Chip, Cluster, Slice, m![R64 % 4]> = scale_byte_offsets
            .view()
            .tile::<m![R64], 8, m![R64=8 # 64]>(group * 8)
            .to_dm(&mut device.tdma);
        let scale_index: HbmTensor<i32, Chip, m![R64 % 8]> = scale_index.to_hbm(&mut device.tdma);
        let k: DmTensor<f8e4m3, Chip, Cluster, KeyRows, m![R64 % 4, Ts % 32, Ds]> = pool_k
            .gather::<m![Rc64], m![Ts, Ns, Ds]>()
            .by_byte_offsets(&index)
            .to_dm(&mut device.tdma);
        let ks: DmTensor<f32, Chip, Cluster, KeyRows, m![R64 % 4, Ts % 32, Scale8]> = scale_k
            .gather::<m![Rc64], m![Ts, Ns, Scale8]>()
            .by_byte_offsets(&scale_index)
            .to_dm(&mut device.tdma);
        let k = dequantize(device, &k, &ks);
        let kt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Ts % 8], m![R64 % 4, Ts / 8 % 4, Ds]> =
            device
                .sub
                .begin(k.view())
                .fetch::<m![Ts % 8, R64 % 4, Ts / 8 % 4, Ds / 16], m![Ds % 16]>()
                .collect::<m![Ts % 8, R64 % 4, Ts / 8 % 4, Ds / 16], m![Ds % 16]>()
                .to_trf();
        let scores: DmTensor<bf16, Chip, Cluster, KeyRows, m![R64 % 4, Gs, Ts % 32]> = device
            .main
            .begin(q.view())
            .fetch::<m![R64 % 4, Gs, Ds / 16], m![Ds % 16]>()
            .collect::<m![R64 % 4, Gs, Ds / 16], m![Ds % 16]>()
            .contract_outer::<m![R64 % 4, Gs, Ds / 32, Ts / 8 % 4], m![Ds % 32], _, _, _>(&kt)
            .contract_packet::<m![1]>()
            .contract_time::<m![R64 % 4, Gs, Ts / 8 % 4]>()
            .contract_lane::<m![R64 % 4, Gs, Ts / 8 % 4], m![Ts % 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![Ts%8 # 16]>()
            .commit_trim::<m![Ts % 8]>()
            .commit();
        let scores: DmTensor<bf16, Chip, Cluster, SlidingRows, m![Ts]> =
            scores.to_dm(&mut device.tdma);
        let mask: DmTensor<f32, Chip, Cluster, SlidingRows, m![Ts]> = visibility_mask
            .view()
            .tile::<m![R64], 8, m![R64=8 # 64,Ts]>(group * 8)
            .to_dm(&mut device.tdma);
        let scores = sliding_premax(device, &scores, &mask);
        let max = sliding_max(device, &scores);
        let exp = sliding_exp(device, &scores, &max, &mask);
        let sum = sliding_sum(device, &exp);
        let sum = sliding_scalar(device, &sum);
        let weights: DmTensor<bf16, Chip, Cluster, SlidingRows, m![Ts]> = device
            .main
            .begin(exp.view())
            .fetch::<m![Ts / 8], m![Ts % 8]>()
            .collect::<m![Ts / 8], m![Ts % 8]>()
            .cast::<bf16, m![Ts%8 # 16]>()
            .commit_trim::<m![Ts % 8]>()
            .commit();
        let weights: DmTensor<bf16, Chip, Cluster, KeyRows, m![R64 % 4, Gs, Ts % 32]> =
            weights.to_dm(&mut device.tdma);

        let v: DmTensor<f8e4m3, Chip, Cluster, KeyRows, m![R64 % 4, Ts % 32, Ds]> = pool_v
            .gather::<m![Rc64], m![Ts, Ns, Ds]>()
            .by_byte_offsets(&index)
            .to_dm(&mut device.tdma);
        let vs: DmTensor<f32, Chip, Cluster, KeyRows, m![R64 % 4, Ts % 32, Scale8]> = scale_v
            .gather::<m![Rc64], m![Ts, Ns, Scale8]>()
            .by_byte_offsets(&scale_index)
            .to_dm(&mut device.tdma);
        let v = dequantize(device, &v, &vs);
        let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![R64 % 4, Ds, Ts % 32]> = device
            .main
            .begin(v.view())
            .fetch::<m![R64 % 4, Ds / 8, Ts % 32], m![Ds%8 # 16]>()
            .collect::<m![R64 % 4, Ds / 8, Ts % 32], m![Ds%8 # 16]>()
            .transpose::<m![R64 % 4, Ds / 8, Ts / 4 % 8, Ds % 8], m![Ts%4 # 16]>()
            .commit_trim::<m![Ts % 4]>()
            .commit();
        let vt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Ds % 8], m![R64 % 4, Ds / 8, Ts % 32]> =
            device
                .sub
                .begin(v.view())
                .fetch::<m![Ds % 8, R64 % 4, Ds / 8, Ts / 16 % 2], m![Ts % 16]>()
                .collect::<m![Ds % 8, R64 % 4, Ds / 8, Ts / 16 % 2], m![Ts % 16]>()
                .to_trf();
        let partial: DmTensor<bf16, Chip, Cluster, KeyRows, m![R64 % 4, Gs, Ds]> = device
            .main
            .begin(weights.view())
            .fetch::<m![R64 % 4, Gs, Ts / 16 % 2], m![Ts % 16]>()
            .collect::<m![R64 % 4, Gs, Ts / 16 % 2], m![Ts % 16]>()
            .contract_outer::<m![R64 % 4, Gs, Ds / 8], m![Ts % 32], _, _, _>(&vt)
            .contract_packet::<m![1]>()
            .contract_time::<m![R64 % 4, Gs, Ds / 8]>()
            .contract_lane::<m![R64 % 4, Gs, Ds / 8], m![Ds % 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![Ds%8 # 16]>()
            .commit_trim::<m![Ds % 8]>()
            .commit();
        let combined: DmTensor<bf16, Chip, Cluster, Reduced, m![R64 % 4, Gs, Ds]> = device
            .main
            .begin(partial.view())
            .fetch::<m![R64 % 4, Gs, Ds / 16], m![Ds % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![R64 % 4, Gs, Ds / 8], m![Ds % 8]>()
            .vector_init()
            .vector_inter_slice_reduce::<Reduced, m![R64 % 4, Gs, Ds / 8]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .cast::<bf16, m![Ds%8 # 16]>()
            .commit_trim::<m![Ds % 8]>()
            .commit();
        let combined: DmTensor<bf16, Chip, Cluster, SlidingRows, m![Ds]> =
            combined.to_dm(&mut device.tdma);
        let result: DmTensor<bf16, Chip, Cluster, SlidingRows, m![Ds]> = device
            .main
            .begin(combined.view())
            .fetch::<m![Ds / 16], m![Ds % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Ds / 8], m![Ds % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
            .vector_fp_div(&sum)
            .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
            .vector_final()
            .cast::<bf16, m![Ds%8 # 16]>()
            .commit_trim::<m![Ds % 8]>()
            .commit();
        result.view().to_hbm_view(
            &mut device.tdma,
            out.view_mut()
                .tile::<m![R64], 8, m![R64=8 #{!} 64,Ns,Gs,Ds]>(group * 8),
        );
    }
}

fn dequantize(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, Cluster, m![Ns, Ts / 32], m![R64 % 4, Ts % 32, Ds]>,
    scale: &DmTensor<f32, Chip, Cluster, m![Ns, Ts / 32], m![R64 % 4, Ts % 32, Scale8]>,
) -> DmTensor<bf16, Chip, Cluster, m![Ns, Ts / 32], m![R64 % 4, Ts % 32, Ds]> {
    let scale: VrfTensor<f32, Chip, Cluster, m![Ns, Ts / 32], m![R64%4,Ts%32,Scale8=1 # 8]> =
        device
            .sub
            .begin(
                scale
                    .view()
                    .tile::<m![Scale8], 1, m![R64%4,Ts%32,Scale8=1 # 8]>(0),
            )
            .fetch::<m![R64 % 4, Ts % 32], m![Scale8=1 # 8]>()
            .collect::<m![R64 % 4, Ts % 32], m![Scale8=1 # 8]>()
            .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![R64 % 4, Ts % 32, Ds / 32], m![Ds % 32]>()
        .fetch_table_lookup::<bf16>()
        .fetch_cast::<f32>()
        .collect::<m![R64 % 4, Ts % 32, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R64 % 4, Ts % 32, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![R64 % 4, Ts % 32, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds%8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// Setup/validation helper. A production cache append would quantize only new rows.
#[device(chip = 1)]
pub fn quantize_ring(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    output: &mut HbmTensor<f8e4m3, Chip, m![Ts, Ns, Ds]>,
    scales: &mut HbmTensor<f32, Chip, m![Ts, Ns, Scale8]>,
) {
    type QCluster = m![Ts / 512];
    type QRows = m![Ts / 16 % 32, Ns];
    let x: DmTensor<bf16, Chip, QCluster, QRows, m![Ts % 16, Ds]> = input.to_dm(&mut device.tdma);
    let amax: DmTensor<f32, Chip, QCluster, QRows, m![Ts%16,1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ts % 16, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ts % 16, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Ts % 16, Ds / 4], m![Ds % 4]>()
        .vector_intra_slice_reduce::<Ds, m![Ts % 16], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.925929944387236e-34f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let scale: DmTensor<f32, Chip, QCluster, QRows, m![Ts%16,1 # 8]> = device
        .main
        .begin(amax.view())
        .fetch::<m![Ts % 16], m![1 # 8]>()
        .collect::<m![Ts % 16], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7f80_0000i32)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0078125f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let sv: VrfTensor<f32, Chip, QCluster, QRows, m![Ts%16,1 # 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![Ts % 16], m![1 # 8]>()
        .collect::<m![Ts % 16], m![1 # 8]>()
        .to_vrf();
    let q: DmTensor<f8e4m3, Chip, QCluster, QRows, m![Ts % 16, Ds]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ts % 16, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ts % 16, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ts % 16, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &sv)
        .vector_widen_concat::<m![Ts % 16, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Ds%8 # 32]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    q.view().to_hbm_view(&mut device.tdma, output.view_mut());
    scale
        .view()
        .to_hbm_view(&mut device.tdma, scales.view_mut());
}

#[device(chip = 1)]
pub fn copy_quantized_ring_to_pool(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![Ts, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![Ts, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![Ts, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![Ts, Ns, Scale8]>,
    data_offset: &HbmTensor<i32, Chip, m![1]>,
    scale_offset: &HbmTensor<i32, Chip, m![1]>,
    pk: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pv: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pks: &mut HbmTensor<f32, Chip, m![Rc64, Ts, Ns, Scale8]>,
    pvs: &mut HbmTensor<f32, Chip, m![Rc64, Ts, Ns, Scale8]>,
) {
    type C = m![Ts / 512];
    type S = m![Ts / 16 % 32, Ns];
    let k: DmTensor<f8e4m3, Chip, C, S, m![Ts % 16, Ds]> = k.to_dm(&mut device.tdma);
    let v: DmTensor<f8e4m3, Chip, C, S, m![Ts % 16, Ds]> = v.to_dm(&mut device.tdma);
    let ks: DmTensor<f32, Chip, C, S, m![Ts % 16, Scale8]> = ks.to_dm(&mut device.tdma);
    let vs: DmTensor<f32, Chip, C, S, m![Ts % 16, Scale8]> = vs.to_dm(&mut device.tdma);
    pk.view_mut()
        .scatter::<m![Rc64], m![Ts, Ns, Ds]>()
        .by_byte_offsets(data_offset.view())
        .from_dm(&mut device.tdma, k);
    pv.view_mut()
        .scatter::<m![Rc64], m![Ts, Ns, Ds]>()
        .by_byte_offsets(data_offset.view())
        .from_dm(&mut device.tdma, v);
    pks.view_mut()
        .scatter::<m![Rc64], m![Ts, Ns, Scale8]>()
        .by_byte_offsets(scale_offset.view())
        .from_dm(&mut device.tdma, ks);
    pvs.view_mut()
        .scatter::<m![Rc64], m![Ts, Ns, Scale8]>()
        .by_byte_offsets(scale_offset.view())
        .from_dm(&mut device.tdma, vs);
}
