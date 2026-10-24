//! Exact R64 logits stored as contiguous per-slice bands; row extraction restores vocabulary order.
use furiosa_opt_std::prelude::*;
use crate::{Chip,LOGIT_SOFTCAP,axes::{R64,H,W}};
pub type BlockedLogits = m![W/8192,W/32%256,R64,W%32];
#[device(chip = 1)]
pub fn logits_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![R64, H]>,
    weight: &HbmTensor<bf16, Chip, m![W, H]>,
    out: &mut HbmTensor<bf16, Chip, BlockedLogits>,
) {
    type Cluster = m![R64 /32];
    type Rows = m![W /32 %256];
    let input: DmTensor<bf16, Chip, Cluster, Rows, m![R64 %32, H]> = input.to_dm(&mut device.tdma);
    for band in 0..(W::SIZE /8192) {
        let mut band_result: DmTensor<bf16, Chip, Cluster, Rows, m![W %32, R64 %32]> = DmTensor::new();
        let weight: DmTensor<bf16, Chip, Cluster, Rows, m![W %32, H]> = weight
            .view()
            .tile::<m![W /8192], 1, m![1 #32, W %8192, H]>(band)
            .to_dm(&mut device.tdma);
        for chunk in 0..4 {
            let x: TrfTensor<bf16, Chip, Cluster, Rows, m![R64 %32 = 8], m![H]> = device.sub
                .begin(input.view().tile::<m![R64 %32], 8, m![R64 %32 = 8 #32, H]>(chunk * 8))
                .fetch::<m![R64 %32 = 8, H / 16], m![H % 16]>()
                .collect::<m![R64 %32 = 8, H / 16], m![H % 16]>()
                .to_trf();
            let dot: DmTensor<bf16, Chip, Cluster, Rows, m![W %32, R64 %32 = 8]> = device.main
                .begin(weight.view())
                .fetch::<m![W %32, H / 16], m![H % 16]>()
                .collect::<m![W %32, H / 16], m![H % 16]>()
                .contract_outer::<m![W %32, H / 32], m![H % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![W %32]>()
                .contract_lane::<m![W %32], m![R64 %32 = 8]>(LaneMode::Interleaved)
                .cast::<bf16, m![R64 %32 = 8 # 16]>()
                .commit_trim::<m![R64 %32 = 8]>()
                .commit();
            let scaled: DmTensor<f32, Chip, Cluster, Rows, m![W %32, R64 %32 = 8]> = device.main
                .begin(dot.view())
                .fetch::<m![W %32], m![R64 %32 = 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W %32], m![R64 %32 = 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W %32, R64 %32 = 8 / 4], m![R64 %32 = 8 % 4]>()
                .vector_fp_div(LOGIT_SOFTCAP)
                .vector_widen_concat::<m![W %32], m![R64 %32 = 8]>()
                .vector_final()
                .commit_trim::<m![R64 %32 = 8]>()
                .commit();
            device.main
                .begin(scaled.view())
                .fetch::<m![W %32], m![R64 %32 = 8]>()
                .collect::<m![W %32], m![R64 %32 = 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W %32, R64 %32 = 8 / 4], m![R64 %32 = 8 % 4]>()
                .vector_fp_unary(FpUnaryOp::Tanh)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), LOGIT_SOFTCAP)
                .vector_widen_concat::<m![W %32], m![R64 %32 = 8]>()
                .vector_final()
                .cast::<bf16, m![R64 %32 = 8 # 16]>()
                .commit_trim::<m![R64 %32 = 8]>()
                .commit_view(band_result.view_mut()
                    .tile::<m![R64 %32], 8, m![W %32, R64 %32 = 8 #{!}32]>(chunk * 8));
        }
        let tile:DmTensor<bf16,Chip,Cluster,Rows,m![R64%32,W%32]>=device.main
            .begin(band_result.view())
            .fetch::<m![W%32,R64/16%2],m![R64%16]>()
            .collect::<m![W%32,R64/16%2],m![R64%16]>()
            .transpose::<m![W/4%8,R64%32],m![W%4#16]>()
            .commit_trim::<m![W%4]>().commit();
        tile.view().to_hbm_view(&mut device.tdma,out.view_mut()
            .tile::<m![W/8192],1,m![1#{!}32,W/32%256,R64,W%32]>(band));
    }
}

/// Extract logical request row0..63 into canonical vocabulary order.
#[device(chip=1)]
pub fn extract_logits_row(
    device:&mut Device,
    input:&HbmTensor<bf16,Chip,BlockedLogits>,
    row_index:&HbmTensor<i32,Chip,m![1]>,
    out:&mut HbmTensor<bf16,Chip,m![W]>,
){
    type Cluster=m![W/8192%2];
    type Rows=m![W/32%256];
    let index:DmTensor<i32,Chip,m![1#2],m![1#256],m![1#8]>=row_index.to_dm(&mut device.tdma);
    let index:SpmTensor<i32,Chip,m![1#2],m![1#4],m![1]>=index.to_spm(&mut device.tdma);
    let row:DmTensor<bf16,Chip,Cluster,Rows,m![W/16384,W%32]>=input
        .gather::<m![R64],m![W/8192,W/32%256,W%32]>()
        .by_positions::<m![1]>(&index).to_dm(&mut device.tdma);
    row.view().to_hbm_view(&mut device.tdma,out.view_mut());
}
