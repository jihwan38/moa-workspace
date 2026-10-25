//! Exact full-H LM math with a real final W32-to-W128 DM grouping.
use furiosa_opt_std::prelude::*;
use crate::{axes::{B,H,W}, Chip, LOGIT_SOFTCAP};

#[device(chip = 1)]
pub fn logits_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<bf16, Chip, m![W, H]>,
    out: &mut HbmTensor<bf16, Chip, m![B, W]>,
) {
    type Cluster = m![W / 8192 % 2];
    type Rows = m![W / 32 % 256];
    let input: DmTensor<bf16, Chip, Cluster, Rows, m![B, H]> = input.to_dm(&mut device.tdma);
    let mut result: DmTensor<bf16, Chip, Cluster, Rows, m![W / 16384, W % 32, B]> = DmTensor::new();
    for band in 0..(W::SIZE / 16384) {
        let mut band_result: DmTensor<bf16, Chip, Cluster, Rows, m![W % 32, B]> = DmTensor::new();
        let weight: DmTensor<bf16, Chip, Cluster, Rows, m![W % 32, H]> = weight
            .view()
            .tile::<m![W / 16384], 1, m![1 # 16, W % 16384, H]>(band)
            .to_dm(&mut device.tdma);
        for chunk in 0..2 {
            let x: TrfTensor<bf16, Chip, Cluster, Rows, m![B = 8], m![H]> = device.sub
                .begin(input.view().tile::<m![B], 8, m![B = 8 # 16, H]>(chunk * 8))
                .fetch::<m![B = 8, H / 16], m![H % 16]>()
                .collect::<m![B = 8, H / 16], m![H % 16]>()
                .to_trf();
            let dot: DmTensor<bf16, Chip, Cluster, Rows, m![W % 32, B = 8]> = device.main
                .begin(weight.view())
                .fetch::<m![W % 32, H / 16], m![H % 16]>()
                .collect::<m![W % 32, H / 16], m![H % 16]>()
                .contract_outer::<m![W % 32, H / 32], m![H % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![W % 32]>()
                .contract_lane::<m![W % 32], m![B = 8]>(LaneMode::Interleaved)
                .cast::<bf16, m![B = 8 # 16]>()
                .commit_trim::<m![B = 8]>()
                .commit();
            let scaled: DmTensor<f32, Chip, Cluster, Rows, m![W % 32, B = 8]> = device.main
                .begin(dot.view())
                .fetch::<m![W % 32], m![B = 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % 32], m![B = 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W % 32, B = 8 / 4], m![B = 8 % 4]>()
                .vector_fp_div(LOGIT_SOFTCAP)
                .vector_widen_concat::<m![W % 32], m![B = 8]>()
                .vector_final()
                .commit_trim::<m![B = 8]>()
                .commit();
            device.main
                .begin(scaled.view())
                .fetch::<m![W % 32], m![B = 8]>()
                .collect::<m![W % 32], m![B = 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W % 32, B = 8 / 4], m![B = 8 % 4]>()
                .vector_fp_unary(FpUnaryOp::Tanh)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), LOGIT_SOFTCAP)
                .vector_widen_concat::<m![W % 32], m![B = 8]>()
                .vector_final()
                .cast::<bf16, m![B = 8 # 16]>()
                .commit_trim::<m![B = 8]>()
                .commit_view(band_result.view_mut()
                    .tile::<m![B], 8, m![W % 32, B = 8 #{!} 16]>(chunk * 8));
        }
        band_result.view().to_dm_view(&mut device.tdma, result.view_mut()
            .tile::<m![W / 16384], 1, m![1 #{!} 16, W % 32, B]>(band));
    }
    let result: DmTensor<bf16, Chip, Cluster, Rows, m![B, W / 16384, W % 32]> = device.main
        .begin(result.view())
        .fetch::<m![W / 16384, W % 32], m![B]>()
        .collect::<m![W / 16384, W % 32], m![B]>()
        .transpose::<m![W / 16384, W / 4 % 8, B], m![W % 4 # 16]>()
        .commit_trim::<m![W % 4]>()
        .commit();
    // Real named-axis movement groups four W32 source slices into one W128
    // live leader. All dot and softcap operations above are unchanged.
    type StoreRows = m![W / 128 % 64, 1 # 4];
    let result: DmTensor<bf16, Chip, Cluster, StoreRows, m![B, W / 16384, W % 128]> =
        result.to_dm(&mut device.tdma);
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

