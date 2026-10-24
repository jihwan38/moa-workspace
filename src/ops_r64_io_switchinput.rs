//! Hold original vocabulary bands and load all query inputs once in H-sharded DM.
//! Preserves full-H dot arithmetic and scalar BF16 softcap boundaries.
use furiosa_opt_std::prelude::*;
use crate::{Chip,LOGIT_SOFTCAP,axes::{R64,H,W}};
#[device(chip = 1)]
pub fn logits_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![R64, H]>,
    weight: &HbmTensor<bf16, Chip, m![W, H]>,
    out: &mut HbmTensor<bf16, Chip, m![R64, W]>,
) {
    type Cluster = m![W / 8192 % 2];
    type Rows = m![W / 32 % 256];
    let input:DmTensor<bf16,Chip,Cluster,m![1#32,H/480],m![R64,H%480]>=input.to_dm(&mut device.tdma);
    let mut result: DmTensor<bf16, Chip, Cluster, Rows, m![W / 16384, W % 32, R64]> = DmTensor::new();
    for band in 0..(W::SIZE / 16384) {
        let mut band_result: DmTensor<bf16, Chip, Cluster, Rows, m![W % 32, R64]> = DmTensor::new();
        let weight: DmTensor<bf16, Chip, Cluster, Rows, m![W % 32, H]> = weight
            .view()
            .tile::<m![W / 16384], 1, m![1 # 16, W % 16384, H]>(band)
            .to_dm(&mut device.tdma);
        for chunk in 0..8 {
            let gathered:DmTensor<bf16,Chip,Cluster,m![8#256],m![R64=8,H]>=device.sub
                .begin(input.view().tile::<m![R64],8,m![R64=8#64,H%480]>(8*chunk))
                .fetch::<m![R64=8],m![H%480]>()
                .switch::<m![8#256],m![R64=8,H/480]>(SwitchConfig::Broadcast1{slice1:8,slice0:1})
                .collect::<m![R64=8,H/16],m![H%16]>().commit_trim::<m![H%16]>().commit();
            let first:DmTensorView<'_,bf16,Chip,Cluster,m![1#256],m![R64=8,H]>=unsafe{gathered.view().reshape()};
            let x:TrfTensor<bf16,Chip,Cluster,Rows,m![R64=8],m![H]>=device.sub.begin(first)
                .fetch::<m![R64=8],m![H]>()
                .switch::<Rows,m![R64=8]>(SwitchConfig::CustomBroadcast{ring_size:256})
                .collect::<m![R64=8,H/16],m![H%16]>().to_trf();
            let dot: DmTensor<bf16, Chip, Cluster, Rows, m![W % 32, R64 = 8]> = device.main
                .begin(weight.view())
                .fetch::<m![W % 32, H / 16], m![H % 16]>()
                .collect::<m![W % 32, H / 16], m![H % 16]>()
                .contract_outer::<m![W % 32, H / 32], m![H % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![W % 32]>()
                .contract_lane::<m![W % 32], m![R64 = 8]>(LaneMode::Interleaved)
                .cast::<bf16, m![R64 = 8 # 16]>()
                .commit_trim::<m![R64 = 8]>()
                .commit();
            let scaled: DmTensor<f32, Chip, Cluster, Rows, m![W % 32, R64 = 8]> = device.main
                .begin(dot.view())
                .fetch::<m![W % 32], m![R64 = 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % 32], m![R64 = 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W % 32, R64 = 8 / 4], m![R64 = 8 % 4]>()
                .vector_fp_div(LOGIT_SOFTCAP)
                .vector_widen_concat::<m![W % 32], m![R64 = 8]>()
                .vector_final()
                .commit_trim::<m![R64 = 8]>()
                .commit();
            device.main
                .begin(scaled.view())
                .fetch::<m![W % 32], m![R64 = 8]>()
                .collect::<m![W % 32], m![R64 = 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W % 32, R64 = 8 / 4], m![R64 = 8 % 4]>()
                .vector_fp_unary(FpUnaryOp::Tanh)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), LOGIT_SOFTCAP)
                .vector_widen_concat::<m![W % 32], m![R64 = 8]>()
                .vector_final()
                .cast::<bf16, m![R64 = 8 # 16]>()
                .commit_trim::<m![R64 = 8]>()
                .commit_view(band_result.view_mut()
                    .tile::<m![R64], 8, m![W % 32, R64 =8 #{!}64]>(chunk * 8));
        }
        band_result.view().to_dm_view(&mut device.tdma, result.view_mut()
            .tile::<m![W / 16384], 1, m![1 #{!} 16, W % 32, R64]>(band));
    }
    let result: DmTensor<bf16, Chip, Cluster, Rows, m![R64, W / 16384, W % 32]> = device.main
        .begin(result.view())
        .fetch::<m![R64 /16,W /16384,W %32],m![R64 %16]>()
        .collect::<m![R64 /16,W /16384,W %32],m![R64 %16]>()
        .transpose::<m![R64 /16,W /16384,W /4 %8,R64 %16], m![W % 4 # 16]>()
        .commit_trim::<m![W % 4]>()
        .commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

