//! R64 embedding and exact BF16 LM head. Two clusters split the32request rows,
//! while256slices stream16 vocabulary rows each to fit the input and weights in DM.

use furiosa_opt_std::prelude::*;

use crate::axes::{R64, H, W};
use crate::{Chip, EMBED_SCALE, LOGIT_SOFTCAP};

/// Gather each token independently, scale in F32, then round once to BF16.
/// Offsets are byte offsets into the original `[W, H]` embedding table.
/// All R64 entries must contain valid offsets; callers ignore padded output rows.
#[device(chip = 1)]
pub fn embed_tokens(
    device: &mut Device,
    embedding_table: &HbmTensor<bf16, Chip, m![W, H]>,
    byte_offsets: &HbmTensor<i32, Chip, m![R64]>,
    out: &mut HbmTensor<bf16, Chip, m![R64, H]>,
) {
    type Cluster = m![R64 /32];
    type Rows = m![R64 %32, H /480];
    let rows: DmTensor<bf16, Chip, Cluster, Rows, m![H % 480]> = embedding_table
        .gather::<m![W], m![H]>()
        .by_byte_offsets(byte_offsets)
        .to_dm(&mut device.tdma);
    let scaled: DmTensor<bf16, Chip, Cluster, Rows, m![H % 480]> = device
        .main
        .begin(rows.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), EMBED_SCALE)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    scaled.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// Compute all vocabulary logits for R64 independent normalized hidden rows.
/// The original BF16 head is read once per band and reused by two B8 matrix
/// contractions. Preserve the scalar dot's BF16 rounding before softcapping.
#[device(chip = 1)]
pub fn logits_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![R64, H]>,
    weight: &HbmTensor<bf16, Chip, m![W, H]>,
    out: &mut HbmTensor<bf16, Chip, m![R64, W]>,
) {
    type Cluster = m![R64 /32];
    type Rows = m![W /16 %256];
    let input: DmTensor<bf16, Chip, Cluster, Rows, m![R64 %32, H]> = input.to_dm(&mut device.tdma);
    let mut result: DmTensor<bf16, Chip, Cluster, Rows, m![W /4096, W %16, R64 %32]> = DmTensor::new();
    for band in 0..(W::SIZE /4096) {
        let mut band_result: DmTensor<bf16, Chip, Cluster, Rows, m![W %16, R64 %32]> = DmTensor::new();
        let weight: DmTensor<bf16, Chip, Cluster, Rows, m![W %16, H]> = weight
            .view()
            .tile::<m![W /4096], 1, m![1 #64, W %4096, H]>(band)
            .to_dm(&mut device.tdma);
        for chunk in 0..4 {
            let x: TrfTensor<bf16, Chip, Cluster, Rows, m![R64 %32 = 8], m![H]> = device.sub
                .begin(input.view().tile::<m![R64 %32], 8, m![R64 %32 = 8 #32, H]>(chunk * 8))
                .fetch::<m![R64 %32 = 8, H / 16], m![H % 16]>()
                .collect::<m![R64 %32 = 8, H / 16], m![H % 16]>()
                .to_trf();
            let dot: DmTensor<bf16, Chip, Cluster, Rows, m![W %16, R64 %32 = 8]> = device.main
                .begin(weight.view())
                .fetch::<m![W %16, H / 16], m![H % 16]>()
                .collect::<m![W %16, H / 16], m![H % 16]>()
                .contract_outer::<m![W %16, H / 32], m![H % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![W %16]>()
                .contract_lane::<m![W %16], m![R64 %32 = 8]>(LaneMode::Interleaved)
                .cast::<bf16, m![R64 %32 = 8 # 16]>()
                .commit_trim::<m![R64 %32 = 8]>()
                .commit();
            let scaled: DmTensor<f32, Chip, Cluster, Rows, m![W %16, R64 %32 = 8]> = device.main
                .begin(dot.view())
                .fetch::<m![W %16], m![R64 %32 = 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W %16], m![R64 %32 = 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W %16, R64 %32 = 8 / 4], m![R64 %32 = 8 % 4]>()
                .vector_fp_div(LOGIT_SOFTCAP)
                .vector_widen_concat::<m![W %16], m![R64 %32 = 8]>()
                .vector_final()
                .commit_trim::<m![R64 %32 = 8]>()
                .commit();
            device.main
                .begin(scaled.view())
                .fetch::<m![W %16], m![R64 %32 = 8]>()
                .collect::<m![W %16], m![R64 %32 = 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W %16, R64 %32 = 8 / 4], m![R64 %32 = 8 % 4]>()
                .vector_fp_unary(FpUnaryOp::Tanh)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), LOGIT_SOFTCAP)
                .vector_widen_concat::<m![W %16], m![R64 %32 = 8]>()
                .vector_final()
                .cast::<bf16, m![R64 %32 = 8 # 16]>()
                .commit_trim::<m![R64 %32 = 8]>()
                .commit_view(band_result.view_mut()
                    .tile::<m![R64 %32], 8, m![W %16, R64 %32 = 8 #{!}32]>(chunk * 8));
        }
        band_result.view().to_dm_view(&mut device.tdma, result.view_mut()
            .tile::<m![W /4096], 1, m![1 #{!}64, W %16, R64 %32]>(band));
    }
    let result: DmTensor<bf16, Chip, Cluster, Rows, m![R64 %32, W /4096, W %16]> = device.main
        .begin(result.view())
        .fetch::<m![W /4096, W %16, R64 /16 %2], m![R64 %16]>()
        .collect::<m![W /4096, W %16, R64 /16 %2], m![R64 %16]>()
        .transpose::<m![W /4096, W /4 %4, R64 %32], m![W % 4 # 16]>()
        .commit_trim::<m![W % 4]>()
        .commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// Copy one live row into the caller-owned scalar logit buffer.
/// `byte_offset[0]` is `row * W::SIZE * size_of::<bf16>()`.
#[device(chip = 1)]
pub fn extract_logits_row(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![R64, W]>,
    byte_offset: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![W]>,
) {
    type Cluster = m![W / 8192 % 2];
    type Rows = m![W / 32 % 256];
    let row: DmTensor<bf16, Chip, Cluster, Rows, m![W / 16384, W % 32]> = input
        .gather::<m![R64], m![W]>()
        .by_byte_offsets(byte_offset)
        .to_dm(&mut device.tdma);
    row.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
