//! Wide prompt linear kernels. P128 is split into64 rows per cluster; every
//! weight shard is reused across eight B8 activation tiles in each cluster.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,H,L,Qs,Ps,Qf,Pf,Ns,Gs,Ds,Gf,Df}};
type Cluster=m![P128 / 64];
type UpRows=m![L / 60];
type DownColumns=m![H / 120,L / 1920];
type DownRows=m![H / 120,1 # 8];

#[device(chip = 1)]
pub fn project_query_batch(
    device:&mut Device,
    input:&HbmTensor<bf16,Chip,m![P128,H]>,
    weight:&HbmTensor<f8e4m3,Chip,m![Qs,H]>,
    scale:&HbmTensor<bf16,Chip,m![Qs]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,Qs]>,
) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,m![Qs / 16 # 256],m![Qs % 16,H]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,m![Qs / 16 # 256],m![Qs % 16,H]>=device.main.begin(weight.view())
        .fetch::<m![Qs % 16,H / 32],m![H % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 16,H / 16],m![H % 16]>().commit_trim::<m![H % 16]>().commit();
    let scale:DmTensor<bf16,Chip,Cluster,m![Qs / 16 # 256],m![Qs % 16]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,m![Qs / 16 # 256],m![Qs % 16]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 2],m![Qs % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,m![Qs / 16 # 256],m![Qs % 16,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let input:DmTensor<bf16,Chip,Cluster,m![Qs / 16 # 256],m![P128 % 64 = 8,H]>=input.view()
            .tile::<m![P128 % 64],8,m![P128 / 64,P128 % 64 = 8 # 64,H]>(8*chunk).to_dm(&mut device.tdma);
        let x:TrfTensor<bf16,Chip,Cluster,m![Qs / 16 # 256],m![P128 % 64 = 8],m![H]>=device.sub.begin(input.view())
            .fetch::<m![P128 % 64 = 8,H / 16],m![H % 16]>()
            .collect::<m![P128 % 64 = 8,H / 16],m![H % 16]>().to_trf();
        let dot:DmTensor<bf16,Chip,Cluster,m![Qs / 16 # 256],m![Qs % 16,P128 % 64 = 8]>=device.main.begin(weight.view())
            .fetch::<m![Qs % 16,H / 16],m![H % 16]>().collect::<m![Qs % 16,H / 16],m![H % 16]>()
            .contract_outer::<m![Qs % 16,H / 32],m![H % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![Qs % 16]>()
            .contract_lane::<m![Qs % 16],m![P128 % 64 = 8]>(LaneMode::Interleaved)
            .cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>().commit();
        device.main.begin(dot.view())
            .fetch::<m![Qs % 16],m![P128 % 64 = 8]>().fetch_cast::<f32>()
            .collect::<m![Qs % 16],m![P128 % 64 = 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Qs % 16,P128 % 64 = 8 / 4],m![P128 % 64 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
            .vector_widen_concat::<m![Qs % 16],m![P128 % 64 = 8]>()
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(result.view_mut().tile::<m![P128 % 64],8,m![Qs % 16,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    let result:DmTensor<bf16,Chip,Cluster,m![Qs / 16 # 256],m![P128 % 64,Qs % 16]>=device.main.begin(result.view())
        .fetch::<m![Qs % 16,P128 / 16 % 4],m![P128 % 16]>()
        .collect::<m![Qs % 16,P128 / 16 % 4],m![P128 % 16]>()
        .transpose::<m![Qs / 4 % 4,P128 % 64],m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

#[device(chip = 1)]
pub fn project_sliding_kv_batch(
    device:&mut Device,
    input:&HbmTensor<bf16,Chip,m![P128,H]>,
    weight:&HbmTensor<f8e4m3,Chip,m![Ps,H]>,
    scale:&HbmTensor<bf16,Chip,m![Ps]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,Ps]>,
) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,m![Ps / 8 # 256],m![Ps % 8,H]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,m![Ps / 8 # 256],m![Ps % 8,H]>=device.main.begin(weight.view())
        .fetch::<m![Ps % 8,H / 32],m![H % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 8,H / 16],m![H % 16]>().commit_trim::<m![H % 16]>().commit();
    let scale:DmTensor<bf16,Chip,Cluster,m![Ps / 8 # 256],m![Ps % 8]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,m![Ps / 8 # 256],m![Ps % 8]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![Ps % 8]>().fetch_cast::<f32>()
        .collect::<m![Ps / 8 % 1],m![Ps % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,m![Ps / 8 # 256],m![Ps % 8,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let input:DmTensor<bf16,Chip,Cluster,m![Ps / 8 # 256],m![P128 % 64 = 8,H]>=input.view()
            .tile::<m![P128 % 64],8,m![P128 / 64,P128 % 64 = 8 # 64,H]>(8*chunk).to_dm(&mut device.tdma);
        let x:TrfTensor<bf16,Chip,Cluster,m![Ps / 8 # 256],m![P128 % 64 = 8],m![H]>=device.sub.begin(input.view())
            .fetch::<m![P128 % 64 = 8,H / 16],m![H % 16]>()
            .collect::<m![P128 % 64 = 8,H / 16],m![H % 16]>().to_trf();
        let dot:DmTensor<bf16,Chip,Cluster,m![Ps / 8 # 256],m![Ps % 8,P128 % 64 = 8]>=device.main.begin(weight.view())
            .fetch::<m![Ps % 8,H / 16],m![H % 16]>().collect::<m![Ps % 8,H / 16],m![H % 16]>()
            .contract_outer::<m![Ps % 8,H / 32],m![H % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![Ps % 8]>()
            .contract_lane::<m![Ps % 8],m![P128 % 64 = 8]>(LaneMode::Interleaved)
            .cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>().commit();
        device.main.begin(dot.view())
            .fetch::<m![Ps % 8],m![P128 % 64 = 8]>().fetch_cast::<f32>()
            .collect::<m![Ps % 8],m![P128 % 64 = 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Ps % 8,P128 % 64 = 8 / 4],m![P128 % 64 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
            .vector_widen_concat::<m![Ps % 8],m![P128 % 64 = 8]>()
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(result.view_mut().tile::<m![P128 % 64],8,m![Ps % 8,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    let result:DmTensor<bf16,Chip,Cluster,m![Ps / 8 # 256],m![P128 % 64,Ps % 8]>=device.main.begin(result.view())
        .fetch::<m![Ps % 8,P128 / 16 % 4],m![P128 % 16]>()
        .collect::<m![Ps % 8,P128 / 16 % 4],m![P128 % 16]>()
        .transpose::<m![Ps / 4 % 2,P128 % 64],m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

#[device(chip = 1)]
pub fn project_full_query_batch(
    device:&mut Device,
    input:&HbmTensor<bf16,Chip,m![P128,H]>,
    weight:&HbmTensor<f8e4m3,Chip,m![Qf,H]>,
    scale:&HbmTensor<bf16,Chip,m![Qf]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,Qf]>,
) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,m![Qf / 32 # 256],m![Qf % 32,H]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,m![Qf / 32 # 256],m![Qf % 32,H]>=device.main.begin(weight.view())
        .fetch::<m![Qf % 32,H / 32],m![H % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![Qf % 32,H / 16],m![H % 16]>().commit_trim::<m![H % 16]>().commit();
    let scale:DmTensor<bf16,Chip,Cluster,m![Qf / 32 # 256],m![Qf % 32]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,m![Qf / 32 # 256],m![Qf % 32]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![Qf % 32]>().fetch_cast::<f32>()
        .collect::<m![Qf / 8 % 4],m![Qf % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,m![Qf / 32 # 256],m![Qf % 32,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let input:DmTensor<bf16,Chip,Cluster,m![Qf / 32 # 256],m![P128 % 64 = 8,H]>=input.view()
            .tile::<m![P128 % 64],8,m![P128 / 64,P128 % 64 = 8 # 64,H]>(8*chunk).to_dm(&mut device.tdma);
        let x:TrfTensor<bf16,Chip,Cluster,m![Qf / 32 # 256],m![P128 % 64 = 8],m![H]>=device.sub.begin(input.view())
            .fetch::<m![P128 % 64 = 8,H / 16],m![H % 16]>()
            .collect::<m![P128 % 64 = 8,H / 16],m![H % 16]>().to_trf();
        let dot:DmTensor<bf16,Chip,Cluster,m![Qf / 32 # 256],m![Qf % 32,P128 % 64 = 8]>=device.main.begin(weight.view())
            .fetch::<m![Qf % 32,H / 16],m![H % 16]>().collect::<m![Qf % 32,H / 16],m![H % 16]>()
            .contract_outer::<m![Qf % 32,H / 32],m![H % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![Qf % 32]>()
            .contract_lane::<m![Qf % 32],m![P128 % 64 = 8]>(LaneMode::Interleaved)
            .cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>().commit();
        device.main.begin(dot.view())
            .fetch::<m![Qf % 32],m![P128 % 64 = 8]>().fetch_cast::<f32>()
            .collect::<m![Qf % 32],m![P128 % 64 = 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Qf % 32,P128 % 64 = 8 / 4],m![P128 % 64 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
            .vector_widen_concat::<m![Qf % 32],m![P128 % 64 = 8]>()
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(result.view_mut().tile::<m![P128 % 64],8,m![Qf % 32,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    let result:DmTensor<bf16,Chip,Cluster,m![Qf / 32 # 256],m![P128 % 64,Qf % 32]>=device.main.begin(result.view())
        .fetch::<m![Qf % 32,P128 / 16 % 4],m![P128 % 16]>()
        .collect::<m![Qf % 32,P128 / 16 % 4],m![P128 % 16]>()
        .transpose::<m![Qf / 4 % 8,P128 % 64],m![Qf % 4 # 16]>()
        .commit_trim::<m![Qf % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

#[device(chip = 1)]
pub fn project_full_key_batch(
    device:&mut Device,
    input:&HbmTensor<bf16,Chip,m![P128,H]>,
    weight:&HbmTensor<f8e4m3,Chip,m![Pf,H]>,
    scale:&HbmTensor<bf16,Chip,m![Pf]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,Pf]>,
) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,m![Pf / 8 # 256],m![Pf % 8,H]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,m![Pf / 8 # 256],m![Pf % 8,H]>=device.main.begin(weight.view())
        .fetch::<m![Pf % 8,H / 32],m![H % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![Pf % 8,H / 16],m![H % 16]>().commit_trim::<m![H % 16]>().commit();
    let scale:DmTensor<bf16,Chip,Cluster,m![Pf / 8 # 256],m![Pf % 8]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,m![Pf / 8 # 256],m![Pf % 8]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![Pf % 8]>().fetch_cast::<f32>()
        .collect::<m![Pf / 8 % 1],m![Pf % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,m![Pf / 8 # 256],m![Pf % 8,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let input:DmTensor<bf16,Chip,Cluster,m![Pf / 8 # 256],m![P128 % 64 = 8,H]>=input.view()
            .tile::<m![P128 % 64],8,m![P128 / 64,P128 % 64 = 8 # 64,H]>(8*chunk).to_dm(&mut device.tdma);
        let x:TrfTensor<bf16,Chip,Cluster,m![Pf / 8 # 256],m![P128 % 64 = 8],m![H]>=device.sub.begin(input.view())
            .fetch::<m![P128 % 64 = 8,H / 16],m![H % 16]>()
            .collect::<m![P128 % 64 = 8,H / 16],m![H % 16]>().to_trf();
        let dot:DmTensor<bf16,Chip,Cluster,m![Pf / 8 # 256],m![Pf % 8,P128 % 64 = 8]>=device.main.begin(weight.view())
            .fetch::<m![Pf % 8,H / 16],m![H % 16]>().collect::<m![Pf % 8,H / 16],m![H % 16]>()
            .contract_outer::<m![Pf % 8,H / 32],m![H % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![Pf % 8]>()
            .contract_lane::<m![Pf % 8],m![P128 % 64 = 8]>(LaneMode::Interleaved)
            .cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>().commit();
        device.main.begin(dot.view())
            .fetch::<m![Pf % 8],m![P128 % 64 = 8]>().fetch_cast::<f32>()
            .collect::<m![Pf % 8],m![P128 % 64 = 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Pf % 8,P128 % 64 = 8 / 4],m![P128 % 64 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
            .vector_widen_concat::<m![Pf % 8],m![P128 % 64 = 8]>()
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(result.view_mut().tile::<m![P128 % 64],8,m![Pf % 8,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    let result:DmTensor<bf16,Chip,Cluster,m![Pf / 8 # 256],m![P128 % 64,Pf % 8]>=device.main.begin(result.view())
        .fetch::<m![Pf % 8,P128 / 16 % 4],m![P128 % 16]>()
        .collect::<m![Pf % 8,P128 / 16 % 4],m![P128 % 16]>()
        .transpose::<m![Pf / 4 % 2,P128 % 64],m![Pf % 4 # 16]>()
        .commit_trim::<m![Pf % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
