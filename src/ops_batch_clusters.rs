//! Distribute the two B8 halves across clusters instead of replicating both.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{B,H,Qs}};
type Cluster=m![B / 8];
type QRows=m![Qs/16];
#[device(chip = 1)]
pub fn project_query_clusters(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    scale: &HbmTensor<bf16, Chip, m![Qs]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Qs]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, QRows, m![B % 8, H]> = input.to_dm(&mut device.tdma);
    let weight: DmTensor<f8e4m3, Chip, Cluster, QRows, m![Qs % 16, H]> =
        weight.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, QRows, m![Qs % 16, H]> = device
        .main
        .begin(weight.view())
        .fetch::<m![Qs % 16, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 16, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    let scale: DmTensor<bf16, Chip, Cluster, QRows, m![Qs % 16]> = scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, QRows, m![Qs % 16]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 2], m![Qs % 8]>()
        .to_vrf();
    let mut result: DmTensor<bf16, Chip, Cluster, QRows, m![Qs % 16, B % 8]> = DmTensor::new();
    
        let x: TrfTensor<bf16, Chip, Cluster, QRows, m![B % 8], m![H]> = device
            .sub
            .begin(input.view())
            .fetch::<m![B % 8, H / 16], m![H % 16]>()
            .collect::<m![B % 8, H / 16], m![H % 16]>()
            .to_trf();
        let dot: DmTensor<bf16, Chip, Cluster, QRows, m![Qs % 16, B % 8]> = device
            .main
            .begin(weight.view())
            .fetch::<m![Qs % 16, H / 16], m![H % 16]>()
            .collect::<m![Qs % 16, H / 16], m![H % 16]>()
            .contract_outer::<m![Qs % 16, H / 32], m![H % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![Qs % 16]>()
            .contract_lane::<m![Qs % 16], m![B % 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B % 8 # 16]>()
            .commit_trim::<m![B % 8]>()
            .commit();
        device
            .main
            .begin(dot.view())
            .fetch::<m![Qs % 16], m![B % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Qs % 16], m![B % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Qs % 16, B % 8 / 4], m![B % 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![Qs % 16], m![B % 8]>()
            .vector_final()
            .cast::<bf16, m![B % 8 # 16]>()
            .commit_trim::<m![B % 8]>()
            .commit_view(result.view_mut());
    let result: DmTensor<bf16,Chip,Cluster,QRows,m![B % 8,Qs % 16]> = device.main.begin(result.view())
        .fetch::<m![Qs % 16],m![B % 8 # 16]>().collect::<m![Qs % 16],m![B % 8 # 16]>()
        .transpose::<m![Qs / 4 % 4,B % 8],m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
