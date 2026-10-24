//! Exact BF16-weight FFN experiment. Host predecodes each original FP4×E4 block
//! product to BF16; dot/global/GELU boundaries remain those of the serial model.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{B,H,L}};
use crate::ops_batch::{Cluster,UpRows,DownRows,DownColumns,geglu_batch};

fn upgate(
    device:&mut Device,
    input:&HbmTensor<bf16,Chip,m![B,H]>,
    weight:&HbmTensor<bf16,Chip,m![L,H]>,
) -> DmTensor<bf16,Chip,Cluster,UpRows,m![L % 60,B]> {
    let input:DmTensor<bf16,Chip,Cluster,UpRows,m![B,H]>=input.to_dm(&mut device.tdma);
    let mut output:DmTensor<bf16,Chip,Cluster,UpRows,m![L % 60,B]>=DmTensor::new();
    // A complete60-row BF16 weight shard plus B16 inputs exceeds512KiB DM.
    // Each15-row shard is loaded once and shared by both B8 contractions.
    for pass in 0..4 {
        let weights:DmTensor<bf16,Chip,Cluster,UpRows,m![L % 60 = 15,H]>=weight.view()
            .tile::<m![L % 60],15,m![L / 60,L % 60 = 15 # 60,H]>(15*pass).to_dm(&mut device.tdma);
        let mut partial:DmTensor<bf16,Chip,Cluster,UpRows,m![L % 60 = 15,B]>=DmTensor::new();
        for chunk in 0..2 {
            let x:TrfTensor<bf16,Chip,Cluster,UpRows,m![B = 8],m![H]>=device.sub
                .begin(input.view().tile::<m![B],8,m![B = 8 # 16,H]>(8*chunk))
                .fetch::<m![B = 8,H / 16],m![H % 16]>()
                .collect::<m![B = 8,H / 16],m![H % 16]>().to_trf();
            device.main.begin(weights.view())
                .fetch::<m![L % 60 = 15,H / 16],m![H % 16]>()
                .collect::<m![L % 60 = 15,H / 16],m![H % 16]>()
                .contract_outer::<m![L % 60 = 15,H / 32],m![H % 32],_,_,_>(&x)
                .contract_packet::<m![1]>().contract_time::<m![L % 60 = 15]>()
                .contract_lane::<m![L % 60 = 15],m![B = 8]>(LaneMode::Interleaved)
                .cast::<bf16,m![B = 8 # 16]>().commit_trim::<m![B = 8]>()
                .commit_view(partial.view_mut().tile::<m![B],8,m![L % 60 = 15,B = 8 #{!} 16]>(8*chunk));
        }
        partial.view().to_dm_view(&mut device.tdma,output.view_mut().tile::<m![L % 60],15,m![L % 60 = 15 #{!} 60,B]>(15*pass));
    }
    output
}

fn down(
    device:&mut Device,
    input:&DmTensor<bf16,Chip,Cluster,UpRows,m![L % 60,B]>,
    weight:&HbmTensor<bf16,Chip,m![H,L]>,
    global_scale:&HbmTensor<f32,Chip,m![1]>,
    output:&mut HbmTensor<bf16,Chip,m![B,H]>,
) {
    let input:DmTensor<bf16,Chip,Cluster,UpRows,m![B,L % 60]>=device.main.begin(input.view())
        .fetch::<m![L % 60],m![B]>().collect::<m![L % 60],m![B]>()
        .transpose::<m![L / 4 % 15,B],m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit();
    let input:DmTensor<bf16,Chip,Cluster,DownColumns,m![B,L % 1920]>=input.to_dm(&mut device.tdma);
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,B]>=DmTensor::new();
    for pass in 0..4 {
        let weights:DmTensor<bf16,Chip,Cluster,DownColumns,m![H % 120 = 30,L % 1920]>=weight.view()
            .tile::<m![H % 120],30,m![H / 120,H % 120 = 30 # 120,L]>(30*pass).to_dm(&mut device.tdma);
        let mut partial:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120 = 30,B]>=DmTensor::new();
        for chunk in 0..2 {
            let x:TrfTensor<bf16,Chip,Cluster,DownColumns,m![B = 8],m![L % 1920]>=device.sub
                .begin(input.view().tile::<m![B],8,m![B = 8 # 16,L % 1920]>(8*chunk))
                .fetch::<m![B = 8,L / 16 % 120],m![L % 16]>()
                .collect::<m![B = 8,L / 16 % 120],m![L % 16]>().to_trf();
            device.main.begin(weights.view())
                .fetch::<m![H % 120 = 30,L / 16 % 120],m![L % 16]>()
                .collect::<m![H % 120 = 30,L / 16 % 120],m![L % 16]>()
                .contract_outer::<m![H % 120 = 30,L / 32 % 60],m![L % 32],_,_,_>(&x)
                .contract_packet::<m![1]>().contract_time::<m![H % 120 = 30]>()
                .contract_lane::<m![H % 120 = 30],m![B = 8]>(LaneMode::Interleaved)
                .vector_init().vector_inter_slice_reduce::<DownRows,m![H % 120 = 30]>(InterSliceReduceOpF32::Add)
                .vector_final().cast::<bf16,m![B = 8 # 16]>().commit_trim::<m![B = 8]>()
                .commit_view(partial.view_mut().tile::<m![B],8,m![H % 120 = 30,B = 8 #{!} 16]>(8*chunk));
        }
        partial.view().to_dm_view(&mut device.tdma,result.view_mut().tile::<m![H % 120],30,m![H % 120 = 30 #{!} 120,B]>(30*pass));
    }
    let scale:DmTensor<f32,Chip,Cluster,DownRows,m![1 # 8]>=global_scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,DownRows,m![1 # 8]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>().to_vrf();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,B]>=device.main.begin(result.view())
        .fetch::<m![H % 120,B / 8],m![B % 8]>().fetch_cast::<f32>()
        .collect::<m![H % 120,B / 8],m![B % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120,B / 4],m![B % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
        .vector_widen_concat::<m![H % 120,B / 8],m![B % 8]>()
        .vector_final().cast::<bf16,m![B % 8 # 16]>().commit_trim::<m![B % 8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![B,H % 120]>=device.main.begin(result.view())
        .fetch::<m![H % 120],m![B]>().collect::<m![H % 120],m![B]>()
        .transpose::<m![H / 4 % 30,B],m![H % 4 # 16]>().commit_trim::<m![H % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

/// Exact FFN core accepting weights already expanded to the baseline BF16 format.
#[device(chip = 1)]
pub fn batch_ffn(
    device:&mut Device,
    input:&HbmTensor<bf16,Chip,m![B,H]>,
    up_weight:&HbmTensor<bf16,Chip,m![L,H]>,
    gate_weight:&HbmTensor<bf16,Chip,m![L,H]>,
    down_weight:&HbmTensor<bf16,Chip,m![H,L]>,
    up_global_scale:&HbmTensor<f32,Chip,m![1]>,
    gate_global_scale:&HbmTensor<f32,Chip,m![1]>,
    down_global_scale:&HbmTensor<f32,Chip,m![1]>,
    output:&mut HbmTensor<bf16,Chip,m![B,H]>,
) {
    let up=upgate(device,input,up_weight);
    let gate=upgate(device,input,gate_weight);
    let hidden=geglu_batch(device,&up,&gate,up_global_scale,gate_global_scale);
    down(device,&hidden,down_weight,down_global_scale,output);
}
