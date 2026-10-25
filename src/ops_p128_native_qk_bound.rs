//! RESOURCE SCREEN ONLY: unscaled E4 old-history QK; not a numerical candidate.
//! Exact blocked P128 heads, attention and cache commit in one launch.
//! Headed Q remains in DM; K/V retain original HBM scratch and old-ring ordering.
//! Place query row slices within each Ns head; ring32 broadcasts stay within their own head.
//! Key-sharded scores/exp with one F32 exp redistribution to the ORIGINAL row-local denominator tree.
//! Causal experiment: preserve every original summation boundary; bit-exact gates required.
//! Prompt-row attention. Additive masks contain 0 for visible keys and -1e30 otherwise.
//! A batch must not straddle a full-attention cache page boundary. Invalid padded rows
//! may produce arbitrary output; callers must not consume or commit those rows.

use crate::axes::{Ds, E, Gs, Ns, P128, Ts};
use crate::ops_head_wide::{norm_ds, rms_ds, rope_ds, value_ds, weight_ds};
use crate::ops_pool_views::Rc64;
use crate::ops_qkv_shared_blocked::{KeyOutput, QueryOutput};
pub use crate::ops_sliding_wide::Pk128;
use furiosa_opt_std::prelude::*;
type Cluster = m![P128 / 64];
type QHeadRows = m![P128 / 4 % 16, Ns, Gs];
type HeadQuery = DmTensor<bf16, Chip, Cluster, QHeadRows, m![P128 % 4, Ds]>;
use crate::{Chip, EPS};

type Rows = m![Ns, P128 / 4 % 16, Gs];
type HeadRows = m![Ns, 1 # 32];
type KeyRows = m![Ns, Ts / 32];
type KeyScores = DmTensor<f32, Chip, Cluster, KeyRows, m![P128 % 64, Gs, Ts % 32]>;
type KeyScalar = DmTensor<f32, Chip, Cluster, KeyRows, m![P128%64,Gs,1#8]>;
type HeadScalar = DmTensor<f32, Chip, Cluster, HeadRows, m![P128%64,Gs,1#8]>;
type Scalar = DmTensor<f32, Chip, Cluster, Rows, m![P128%4,1 # 8]>;

fn scalar_vrf(
    device: &mut Device,
    x: &Scalar,
) -> VrfTensor<f32, Chip, Cluster, Rows, m![P128%4,1 # 8]> {
    device
        .sub
        .begin(x.view())
        .fetch::<m![P128 % 4], m![1 # 8]>()
        .collect::<m![P128 % 4], m![1 # 8]>()
        .to_vrf()
}

fn old_scores(
    device: &mut Device,
    q: &HeadQuery,
    k: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    base: &HbmTensor<i32, Chip, m![P128 / 64]>,
    mask: &HbmTensor<f32, Chip, m![P128, Ts]>,
) -> KeyScores {
    type KeyRows = m![Ns, Ts / 32];
    let q: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Ds]> = q.to_dm(&mut device.tdma);
    let q: DmTensor<f8e4m3, Chip, Cluster, Rows, m![P128 % 4, Ds]> = device.main.begin(q.view())
        .fetch::<m![P128 % 4, Ds / 16], m![Ds % 16]>().fetch_cast::<f32>()
        .collect::<m![P128 % 4, Ds / 8], m![Ds % 8]>()
        .cast::<f8e4m3, m![Ds % 8 # 32]>().commit_trim::<m![Ds % 8]>().commit();
    let q: DmTensor<f8e4m3, Chip, Cluster, KeyRows, m![P128 % 4, P128 / 4 % 16, Gs, Ds]> = device
        .main
        .begin(q.view())
        .fetch::<m![P128 % 4], m![Ds]>()
        .switch::<KeyRows, m![P128 % 4, P128 / 4 % 16, Gs]>(SwitchConfig::CustomBroadcast {
            ring_size: 32,
        })
        .collect::<m![P128 % 4, P128 / 4 % 16, Gs, Ds / 32], m![Ds % 32]>()
        .commit_trim::<m![Ds % 32]>()
        .commit();
    let k: DmTensor<bf16, Chip, Cluster, KeyRows, m![Ts % 32, Ds]> = k
        .gather::<m![Rc64], m![Ts, Ns, Ds]>()
        .by_byte_offsets(base)
        .to_dm(&mut device.tdma);
    let k: DmTensor<f8e4m3, Chip, Cluster, KeyRows, m![Ts % 32, Ds]> = device.main.begin(k.view())
        .fetch::<m![Ts % 32, Ds / 16], m![Ds % 16]>().fetch_cast::<f32>()
        .collect::<m![Ts % 32, Ds / 8], m![Ds % 8]>()
        .cast::<f8e4m3, m![Ds % 8 # 32]>().commit_trim::<m![Ds % 8]>().commit();
    let kt: TrfTensor<f8e4m3, Chip, Cluster, KeyRows, m![Ts % 8], m![Ts / 8 % 4, Ds]> = device
        .sub
        .begin(k.view())
        .fetch::<m![Ts % 8, Ts / 8 % 4, Ds / 32], m![Ds % 32]>()
        .collect::<m![Ts % 8, Ts / 8 % 4, Ds / 32], m![Ds % 32]>()
        .to_trf();
    let scores: DmTensor<bf16, Chip, Cluster, KeyRows, m![P128 % 64, Gs, Ts % 32]> = device
        .main
        .begin(q.view())
        .fetch::<m![P128 % 64, Gs, Ds / 32], m![Ds % 32]>()
        .collect::<m![P128 % 64, Gs, Ds / 32], m![Ds % 32]>()
        .contract_outer::<m![P128 % 64, Gs, Ds / 64, Ts / 8 % 4], m![Ds % 64], _, _, _>(&kt)
        .contract_packet::<m![1]>()
        .contract_time::<m![P128 % 64, Gs, Ts / 8 % 4]>()
        .contract_lane::<m![P128 % 64, Gs, Ts / 8 % 4], m![Ts % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Ts%8#16]>()
        .commit_trim::<m![Ts % 8]>()
        .commit();
    let mask: DmTensor<f32, Chip, Cluster, KeyRows, m![P128 % 64, Ts % 32]> =
        mask.to_dm(&mut device.tdma);
    let mut result: KeyScores = DmTensor::new();
    for part in 0..2 {
        let mask: VrfTensor<f32, Chip, Cluster, KeyRows, m![P128 % 64 = 32, Ts % 32]> = device
            .sub
            .begin(
                mask.view()
                    .tile::<m![P128 % 64], 32, m![P128%64=32#64,Ts%32]>(part * 32),
            )
            .fetch::<m![P128 % 64 = 32, Ts / 8 % 4], m![Ts % 8]>()
            .collect::<m![P128 % 64 = 32, Ts / 8 % 4], m![Ts % 8]>()
            .to_vrf();
        device
            .main
            .begin(
                scores
                    .view()
                    .tile::<m![P128 % 64], 32, m![P128%64=32#64,Gs,Ts%32]>(part * 32),
            )
            .fetch::<m![P128 % 64 = 32, Gs, Ts / 16 % 2], m![Ts % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![P128 % 64 = 32, Gs, Ts / 8 % 4], m![Ts % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &mask)
            .vector_final()
            .commit_trim::<m![Ts % 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![P128 % 64], 32, m![P128%64=32#{!}64,Gs,Ts%32]>(part * 32),
            );
    }
    result
}

fn row_max(device: &mut Device, scores: &KeyScores) -> Scalar {
    let local: KeyScalar = device
        .main
        .begin(scores.view())
        .fetch::<m![P128 % 64, Gs, Ts / 8 % 4], m![Ts % 8]>()
        .collect::<m![P128 % 64, Gs, Ts / 8 % 4], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 64, Gs, Ts / 4 % 8], m![Ts % 4]>()
        .vector_intra_slice_reduce::<Ts, m![P128 % 64, Gs], m![1#4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1#8]>()
        .vector_final()
        .commit_trim::<m![1#8]>()
        .commit();
    let max: HeadScalar = device
        .main
        .begin(local.view())
        .fetch::<m![P128 % 64, Gs], m![1#8]>()
        .collect::<m![P128 % 64, Gs], m![1#8]>()
        .vector_init()
        .vector_inter_slice_reduce::<HeadRows, m![P128 % 64, Gs]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .commit_trim::<m![1#8]>()
        .commit();
    max.to_dm(&mut device.tdma)
}

fn exp_scores(device: &mut Device, scores: &KeyScores, max: &Scalar) -> KeyScores {
    let max: VrfTensor<f32, Chip, Cluster, KeyRows, m![P128%4,P128/4%16,Gs,1#8]> = device
        .sub
        .begin(max.view())
        .fetch::<m![P128 % 4], m![1#8]>()
        .switch::<KeyRows, m![P128 % 4, P128 / 4 % 16, Gs]>(SwitchConfig::CustomBroadcast {
            ring_size: 32,
        })
        .collect::<m![P128 % 4, P128 / 4 % 16, Gs], m![1#8]>()
        .to_vrf();
    device
        .main
        .begin(scores.view())
        .fetch::<m![P128 % 64, Gs, Ts / 8 % 4], m![Ts % 8]>()
        .collect::<m![P128 % 64, Gs, Ts / 8 % 4], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 64, Gs, Ts / 4 % 8], m![Ts % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &max)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![P128 % 64, Gs, Ts / 8 % 4], m![Ts % 8]>()
        .vector_final()
        .commit_trim::<m![Ts % 8]>()
        .commit()
}

fn row_sum(device: &mut Device, exp: &KeyScores) -> Scalar {
    let exp: DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Ts]> = exp.to_dm(&mut device.tdma);
    device
        .main
        .begin(exp.view())
        .fetch::<m![P128 % 4, Ts / 8], m![Ts % 8]>()
        .collect::<m![P128 % 4, Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4, Ts / 4], m![Ts % 4]>()
        .vector_intra_slice_reduce::<Ts, m![P128 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn weighted_values(
    device: &mut Device,
    exp: &KeyScores,
    v: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    base: &HbmTensor<i32, Chip, m![P128 / 64]>,
) -> DmTensor<bf16, Chip, Cluster, HeadRows, m![P128 % 64, Gs, Ds]> {
    let weights: DmTensor<bf16, Chip, Cluster, KeyRows, m![P128 % 64, Gs, Ts % 32]> = device
        .main
        .begin(exp.view())
        .fetch::<m![P128 % 64, Gs, Ts / 8 % 4], m![Ts % 8]>()
        .collect::<m![P128 % 64, Gs, Ts / 8 % 4], m![Ts % 8]>()
        .cast::<bf16, m![Ts%8#16]>()
        .commit_trim::<m![Ts % 8]>()
        .commit();
    let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![Ts % 32, Ds]> = v
        .gather::<m![Rc64], m![Ts, Ns, Ds]>()
        .by_byte_offsets(base)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![Ds, Ts % 32]> = device
        .main
        .begin(v.view())
        .fetch::<m![Ds / 8, Ts % 32], m![Ds % 8 # 16]>()
        .collect::<m![Ds / 8, Ts % 32], m![Ds % 8 # 16]>()
        .transpose::<m![Ds / 8, Ts / 4 % 8, Ds % 8], m![Ts % 4 # 16]>()
        .commit_trim::<m![Ts % 4]>()
        .commit();
    let vt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Ds % 8], m![Ds / 8, Ts % 32]> = device
        .sub
        .begin(v.view())
        .fetch::<m![Ds % 8, Ds / 8, Ts / 16 % 2], m![Ts % 16]>()
        .collect::<m![Ds % 8, Ds / 8, Ts / 16 % 2], m![Ts % 16]>()
        .to_trf();
    let partial: DmTensor<bf16, Chip, Cluster, KeyRows, m![P128 % 64, Gs, Ds]> = device
        .main
        .begin(weights.view())
        .fetch::<m![P128 % 64, Gs, Ts / 16 % 2], m![Ts % 16]>()
        .collect::<m![P128 % 64, Gs, Ts / 16 % 2], m![Ts % 16]>()
        .contract_outer::<m![P128 % 64, Gs, Ds / 8], m![Ts % 32], _, _, _>(&vt)
        .contract_packet::<m![1]>()
        .contract_time::<m![P128 % 64, Gs, Ds / 8]>()
        .contract_lane::<m![P128 % 64, Gs, Ds / 8], m![Ds % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    device
        .main
        .begin(partial.view())
        .fetch::<m![P128 % 64, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 64, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<HeadRows, m![P128 % 64, Gs, Ds / 8]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn chunk_scores(
    device: &mut Device,
    q: &HeadQuery,
    k: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    mask: &HbmTensor<f32, Chip, m![P128, Pk128]>,
) -> DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Pk128]> {
    type KeyRows = m![P128 / 8 % 8, Ns, Pk128 / 32];
    let k = unsafe { k.view().reshape::<Chip, m![Pk128, Ns, Ds]>() };
    let q: DmTensor<bf16, Chip, Cluster, KeyRows, m![P128 % 8, Gs, Ds]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KeyRows, m![Pk128 % 32, Ds]> = k.to_dm(&mut device.tdma);
    let kt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Pk128 % 8], m![Pk128 / 8 % 4, Ds]> = device
        .sub
        .begin(k.view())
        .fetch::<m![Pk128 % 8, Pk128 / 8 % 4, Ds / 16], m![Ds % 16]>()
        .collect::<m![Pk128 % 8, Pk128 / 8 % 4, Ds / 16], m![Ds % 16]>()
        .to_trf();
    let scores: DmTensor<bf16, Chip, Cluster, KeyRows, m![P128 % 8, Gs, Pk128 % 32]> = device
        .main
        .begin(q.view())
        .fetch::<m![P128 % 8, Gs, Ds / 16], m![Ds % 16]>()
        .collect::<m![P128 % 8, Gs, Ds / 16], m![Ds % 16]>()
        .contract_outer::<m![P128 % 8, Gs, Ds / 32, Pk128 / 8 % 4], m![Ds % 32], _, _, _>(&kt)
        .contract_packet::<m![1]>()
        .contract_time::<m![P128 % 8, Gs, Pk128 / 8 % 4]>()
        .contract_lane::<m![P128 % 8, Gs, Pk128 / 8 % 4], m![Pk128 % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Pk128%8#16]>()
        .commit_trim::<m![Pk128 % 8]>()
        .commit();
    let scores: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Pk128]> =
        scores.to_dm(&mut device.tdma);
    let mask: DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Pk128]> =
        mask.to_dm(&mut device.tdma);
    let mask: VrfTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Pk128]> = device
        .sub
        .begin(mask.view())
        .fetch::<m![P128 % 4, Pk128 / 8], m![Pk128 % 8]>()
        .collect::<m![P128 % 4, Pk128 / 8], m![Pk128 % 8]>()
        .to_vrf();
    device
        .main
        .begin(scores.view())
        .fetch::<m![P128 % 4, Pk128 / 16], m![Pk128 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 4, Pk128 / 8], m![Pk128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &mask)
        .vector_final()
        .commit_trim::<m![Pk128 % 8]>()
        .commit()
}

fn chunk_max(
    device: &mut Device,
    scores: &DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Pk128]>,
) -> Scalar {
    device
        .main
        .begin(scores.view())
        .fetch::<m![P128 % 4, Pk128 / 8], m![Pk128 % 8]>()
        .collect::<m![P128 % 4, Pk128 / 8], m![Pk128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4, Pk128 / 4], m![Pk128 % 4]>()
        .vector_intra_slice_reduce::<Pk128, m![P128 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn chunk_exp(
    device: &mut Device,
    scores: &DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Pk128]>,
    max: &Scalar,
) -> DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Pk128]> {
    let max = scalar_vrf(device, max);
    device
        .main
        .begin(scores.view())
        .fetch::<m![P128 % 4, Pk128 / 8], m![Pk128 % 8]>()
        .collect::<m![P128 % 4, Pk128 / 8], m![Pk128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4, Pk128 / 4], m![Pk128 % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &max)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![P128 % 4, Pk128 / 8], m![Pk128 % 8]>()
        .vector_final()
        .commit_trim::<m![Pk128 % 8]>()
        .commit()
}

fn chunk_sum(
    device: &mut Device,
    exp: &DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Pk128]>,
) -> Scalar {
    device
        .main
        .begin(exp.view())
        .fetch::<m![P128 % 4, Pk128 / 8], m![Pk128 % 8]>()
        .collect::<m![P128 % 4, Pk128 / 8], m![Pk128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4, Pk128 / 4], m![Pk128 % 4]>()
        .vector_intra_slice_reduce::<Pk128, m![P128 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn chunk_values(
    device: &mut Device,
    exp: &DmTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Pk128]>,
    v: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Ds]> {
    type KeyRows = m![P128 / 8 % 8, Ns, Pk128 / 32];
    type ChunkHeadRows = m![P128/8%8,Ns,1#4];
    let v = unsafe { v.view().reshape::<Chip, m![Pk128, Ns, Ds]>() };
    let weights: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Pk128]> = device
        .main
        .begin(exp.view())
        .fetch::<m![P128 % 4, Pk128 / 8], m![Pk128 % 8]>()
        .collect::<m![P128 % 4, Pk128 / 8], m![Pk128 % 8]>()
        .cast::<bf16, m![Pk128 % 8 # 16]>()
        .commit_trim::<m![Pk128 % 8]>()
        .commit();
    let weights: DmTensor<bf16, Chip, Cluster, KeyRows, m![P128 % 8, Gs, Pk128 % 32]> =
        weights.to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![Pk128 % 32, Ds]> = v.to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, KeyRows, m![Ds, Pk128 % 32]> = device
        .main
        .begin(v.view())
        .fetch::<m![Ds / 8, Pk128 % 32], m![Ds % 8 # 16]>()
        .collect::<m![Ds / 8, Pk128 % 32], m![Ds % 8 # 16]>()
        .transpose::<m![Ds / 8, Pk128 / 4 % 8, Ds % 8], m![Pk128 % 4 # 16]>()
        .commit_trim::<m![Pk128 % 4]>()
        .commit();
    let vt: TrfTensor<bf16, Chip, Cluster, KeyRows, m![Ds % 8], m![Ds / 8, Pk128 % 32]> = device
        .sub
        .begin(v.view())
        .fetch::<m![Ds % 8, Ds / 8, Pk128 / 16 % 2], m![Pk128 % 16]>()
        .collect::<m![Ds % 8, Ds / 8, Pk128 / 16 % 2], m![Pk128 % 16]>()
        .to_trf();
    let partial: DmTensor<bf16, Chip, Cluster, KeyRows, m![P128 % 8, Gs, Ds]> = device
        .main
        .begin(weights.view())
        .fetch::<m![P128 % 8, Gs, Pk128 / 16 % 2], m![Pk128 % 16]>()
        .collect::<m![P128 % 8, Gs, Pk128 / 16 % 2], m![Pk128 % 16]>()
        .contract_outer::<m![P128 % 8, Gs, Ds / 8], m![Pk128 % 32], _, _, _>(&vt)
        .contract_packet::<m![1]>()
        .contract_time::<m![P128 % 8, Gs, Ds / 8]>()
        .contract_lane::<m![P128 % 8, Gs, Ds / 8], m![Ds % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, ChunkHeadRows, m![P128 % 8, Gs, Ds]> = device
        .main
        .begin(partial.view())
        .fetch::<m![P128 % 8, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 8, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<ChunkHeadRows, m![P128 % 8, Gs, Ds / 8]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    result.to_dm(&mut device.tdma)
}

/// Attend before committing this chunk to the old ring. The masks jointly describe
/// [max(0, query_position - 1023), query_position], with old keys restricted to
/// positions before chunk_start and current keys restricted to valid chunk rows.
fn attend(
    device: &mut Device,
    q: &HeadQuery,
    old_k: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    old_v: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    base: &HbmTensor<i32, Chip, m![P128 / 64]>,
    chunk_k: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    chunk_v: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    old_mask: &HbmTensor<f32, Chip, m![P128, Ts]>,
    chunk_mask: &HbmTensor<f32, Chip, m![P128, Pk128]>,
    out: &mut HbmTensor<bf16, Chip, m![P128, Ns, Gs, Ds]>,
) {
    let old_scores = old_scores(device, q, old_k, base, old_mask);
    let new_scores = chunk_scores(device, q, chunk_k, chunk_mask);
    let old_max = row_max(device, &old_scores);
    let new_max = chunk_max(device, &new_scores);
    let new_max = scalar_vrf(device, &new_max);
    let max: Scalar = device
        .main
        .begin(old_max.view())
        .fetch::<m![P128 % 4], m![1 # 8]>()
        .collect::<m![P128 % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Max, &new_max)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let old_exp = exp_scores(device, &old_scores, &max);
    let new_exp = chunk_exp(device, &new_scores, &max);
    let old_sum = row_sum(device, &old_exp);
    let new_sum = chunk_sum(device, &new_exp);
    let new_sum = scalar_vrf(device, &new_sum);
    let sum: Scalar = device
        .main
        .begin(old_sum.view())
        .fetch::<m![P128 % 4], m![1 # 8]>()
        .collect::<m![P128 % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &new_sum)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let sum = scalar_vrf(device, &sum);
    let old_values = weighted_values(device, &old_exp, old_v, base);
    let old_values: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Ds]> =
        old_values.to_dm(&mut device.tdma);
    let new_values = chunk_values(device, &new_exp, chunk_v);
    let new_values: VrfTensor<f32, Chip, Cluster, Rows, m![P128 % 4, Ds]> = device
        .sub
        .begin(new_values.view())
        .fetch::<m![P128 % 4, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 4, Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let total: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Ds]> = device
        .main
        .begin(old_values.view())
        .fetch::<m![P128 % 4, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 4, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &new_values)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Ds]> = device
        .main
        .begin(total.view())
        .fetch::<m![P128 % 4, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 4, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4, Ds / 4], m![Ds % 4]>()
        .vector_fp_div(&sum)
        .vector_widen_concat::<m![P128 % 4, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// First128 causal queries only: no position precedes this chunk. All rows must
/// be live, with finite normalized Q/K and finite V. The ordinary entry remains
/// the control for every other position and for diagnostic comparison.
fn attend_first(
    device: &mut Device,
    q: &HeadQuery,
    chunk_k: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    chunk_v: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    chunk_mask: &HbmTensor<f32, Chip, m![P128, Pk128]>,
    out: &mut HbmTensor<bf16, Chip, m![P128, Ns, Gs, Ds]>,
) {
    let scores = chunk_scores(device, q, chunk_k, chunk_mask);
    let max = chunk_max(device, &scores);
    let exp = chunk_exp(device, &scores, &max);
    let sum = chunk_sum(device, &exp);
    // The control adds EPS to the all-zero old denominator, then the new sum.
    let sum: Scalar = device
        .main
        .begin(sum.view())
        .fetch::<m![P128 % 4], m![1#8]>()
        .collect::<m![P128 % 4], m![1#8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1#8]>()
        .commit();
    let sum = scalar_vrf(device, &sum);
    let values = chunk_values(device, &exp, chunk_v);
    // Preserve the control's BF16 numerator boundary after adding old zero.
    let total: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Ds]> = device
        .main
        .begin(values.view())
        .fetch::<m![P128 % 4, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 4, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, 0f32)
        .vector_final()
        .cast::<bf16, m![Ds%8#16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, Rows, m![P128 % 4, Ds]> = device
        .main
        .begin(total.view())
        .fetch::<m![P128 % 4, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 % 4, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 % 4, Ds / 4], m![Ds % 4]>()
        .vector_fp_div(&sum)
        .vector_widen_concat::<m![P128 % 4, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds%8#16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

type RopeRows = m![P128/4%16,1#16];
fn prepare_heads(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, QueryOutput>,
    k_raw: &HbmTensor<bf16, Chip, KeyOutput>,
    v_raw: &HbmTensor<bf16, Chip, KeyOutput>,
    q_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![P128]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
) -> HeadQuery {
    type QRows = m![P128 / 4 % 16, Ns, Gs];
    type KRows = m![P128/4%16,Ns,1#2];
    let q = unsafe {
        q_raw
            .view()
            .reshape::<Chip, m![P128 / 64, Ns, Gs, Ds / 64, P128 % 64, Ds % 64]>()
    };
    let k = unsafe {
        k_raw
            .view()
            .reshape::<Chip, m![P128 / 64, Ns, Ds / 32, P128 % 64, Ds % 32]>()
    };
    let q: DmTensor<bf16, Chip, Cluster, QRows, m![P128 % 4, Ds]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KRows, m![P128 % 4, Ds]> = k.to_dm(&mut device.tdma);
    let qr = rms_ds(device, &q);
    let kr = rms_ds(device, &k);
    let v = unsafe {
        v_raw
            .view()
            .reshape::<Chip, m![P128 / 64, Ns, Ds / 32, P128 % 64, Ds % 32]>()
    };
    let v: DmTensor<bf16, Chip, Cluster, KRows, m![P128 % 4, Ds]> = v.to_dm(&mut device.tdma);
    let vr = rms_ds(device, &v);
    let v = value_ds(device, &v, &vr);
    let qw = weight_ds(device, q_weight);
    let kw = weight_ds(device, k_weight);
    let q = norm_ds(device, &q, &qr, &qw);
    let k = norm_ds(device, &k, &kr, &kw);
    let cos: DmTensor<bf16, Chip, Cluster, RopeRows, m![P128 % 4, Ds]> = cos
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let cos: DmTensor<bf16, Chip, Cluster, QRows, m![P128 % 4, Ds]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, RopeRows, m![P128 % 4, Ds]> = sin
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, QRows, m![P128 % 4, Ds]> = sin.to_dm(&mut device.tdma);
    let q = rope_ds(device, &q, &cos, &sin);
    let cos: DmTensor<bf16, Chip, Cluster, KRows, m![P128 % 4, Ds]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KRows, m![P128 % 4, Ds]> = sin.to_dm(&mut device.tdma);
    let k = rope_ds(device, &k, &cos, &sin);
    k.view().to_hbm_view(&mut device.tdma, k_out.view_mut());
    v.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
    q
}

fn commit(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![P128]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
) {
    type Rows = m![P128 % 64, Ns / 2];
    let k: DmTensor<bf16, Chip, Cluster, Rows, m![Ns % 2, Ds]> = k.to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, Rows, m![Ns % 2, Ds]> = v.to_dm(&mut device.tdma);
    k_cache
        .view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(offsets.view())
        .from_dm(&mut device.tdma, k);
    v_cache
        .view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(offsets.view())
        .from_dm(&mut device.tdma, v);
}

#[device(chip = 1)]
pub fn sliding_fused_p128(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, QueryOutput>,
    k_raw: &HbmTensor<bf16, Chip, KeyOutput>,
    v_raw: &HbmTensor<bf16, Chip, KeyOutput>,
    q_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![P128]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    bases: &HbmTensor<i32, Chip, m![P128 / 64]>,
    old_mask: &HbmTensor<f32, Chip, m![P128, Ts]>,
    chunk_mask: &HbmTensor<f32, Chip, m![P128, Pk128]>,
    commit_offsets: &HbmTensor<i32, Chip, m![P128]>,
    k_scratch: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v_scratch: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    out: &mut HbmTensor<bf16, Chip, m![P128, Ns, Gs, Ds]>,
) {
    let q = prepare_heads(
        device,
        q_raw,
        k_raw,
        v_raw,
        q_weight,
        k_weight,
        rope_offsets,
        cos,
        sin,
        k_scratch,
        v_scratch,
    );
    attend(
        device,
        &q,
        &*pool_k,
        &*pool_v,
        bases,
        &*k_scratch,
        &*v_scratch,
        old_mask,
        chunk_mask,
        out,
    );
    commit(
        device,
        &*k_scratch,
        &*v_scratch,
        commit_offsets,
        pool_k,
        pool_v,
    );
}

#[device(chip = 1)]
pub fn sliding_fused_first_p128(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, QueryOutput>,
    k_raw: &HbmTensor<bf16, Chip, KeyOutput>,
    v_raw: &HbmTensor<bf16, Chip, KeyOutput>,
    q_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![P128]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    _bases: &HbmTensor<i32, Chip, m![P128 / 64]>,
    _old_mask: &HbmTensor<f32, Chip, m![P128, Ts]>,
    chunk_mask: &HbmTensor<f32, Chip, m![P128, Pk128]>,
    commit_offsets: &HbmTensor<i32, Chip, m![P128]>,
    k_scratch: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v_scratch: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    out: &mut HbmTensor<bf16, Chip, m![P128, Ns, Gs, Ds]>,
) {
    let q = prepare_heads(
        device,
        q_raw,
        k_raw,
        v_raw,
        q_weight,
        k_weight,
        rope_offsets,
        cos,
        sin,
        k_scratch,
        v_scratch,
    );
    attend_first(device, &q, &*k_scratch, &*v_scratch, chunk_mask, out);
    commit(
        device,
        &*k_scratch,
        &*v_scratch,
        commit_offsets,
        pool_k,
        pool_v,
    );
}

#[device(chip = 1)]
pub fn headed_outputs(
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
    let q = prepare_heads(
        device,
        q_raw,
        k_raw,
        v_raw,
        q_weight,
        k_weight,
        rope_offsets,
        cos,
        sin,
        k_out,
        v_out,
    );
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
}
