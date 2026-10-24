//! Batched token embedding with the scalar path's arithmetic and original weights.

use furiosa_opt_std::prelude::*;

use crate::axes::{B, H, W};
use crate::{Chip, EMBED_SCALE};

/// Gather each token independently, scale in F32, then round once to BF16.
/// Offsets are byte offsets into the original `[W, H]` embedding table.
/// All B entries must contain valid offsets; callers ignore padded output rows.
#[device(chip = 1)]
pub fn embed_tokens(
    device: &mut Device,
    embedding_table: &HbmTensor<bf16, Chip, m![W, H]>,
    byte_offsets: &HbmTensor<i32, Chip, m![B]>,
    out: &mut HbmTensor<bf16, Chip, m![B, H]>,
) {
    type Cluster = m![1 # 2];
    type Rows = m![B, H / 240];
    let rows: DmTensor<bf16, Chip, Cluster, Rows, m![H % 240]> = embedding_table
        .gather::<m![W], m![H]>()
        .by_byte_offsets(byte_offsets)
        .to_dm(&mut device.tdma);
    let scaled: DmTensor<bf16, Chip, Cluster, Rows, m![H % 240]> = device
        .main
        .begin(rows.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), EMBED_SCALE)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    scaled.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
