//! Exact split-cluster B16 FFN with original FP4×E4 weights predecoded to BF16.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{B,H,L}};
type Cluster=m![B/8];
type UpRows=m![L/60];
type DownColumns=m![H/120,L/1920];
type DownRows=m![H/120,1 # 8];
fn upgate(device:&mut Device,input:&DmTensor<bf16,Chip,Cluster,UpRows,m![B % 8,H]>,
    weight:&HbmTensor<bf16,Chip,m![L,H]>)->DmTensor<bf16,Chip,Cluster,UpRows,m![L % 60,B % 8]> {
    let x:TrfTensor<bf16,Chip,Cluster,UpRows,m![B % 8],m![H]>=device.sub.begin(input.view())
        .fetch::<m![B % 8,H / 16],m![H % 16]>().collect::<m![B % 8,H / 16],m![H % 16]>().to_trf();
    let mut result:DmTensor<bf16,Chip,Cluster,UpRows,m![L % 60,B % 8]>=DmTensor::new();
    for part in 0..2 {
        let weights:DmTensor<bf16,Chip,Cluster,UpRows,m![L % 60 = 30,H]>=weight.view()
            .tile::<m![L % 60],30,m![L / 60,L % 60 = 30 # 60,H]>(30*part).to_dm(&mut device.tdma);
        device.main.begin(weights.view()).fetch::<m![L % 60 = 30,H / 16],m![H % 16]>()
            .collect::<m![L % 60 = 30,H / 16],m![H % 16]>()
            .contract_outer::<m![L % 60 = 30,H / 32],m![H % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![L % 60 = 30]>()
            .contract_lane::<m![L % 60 = 30],m![B % 8]>(LaneMode::Interleaved)
            .cast::<bf16,m![B % 8 # 16]>().commit_trim::<m![B % 8]>()
            .commit_view(result.view_mut().tile::<m![L % 60],30,m![L % 60 = 30 #{!} 60,B % 8]>(30*part));
    }
    result
}
fn down(device:&mut Device,input:&DmTensor<bf16,Chip,Cluster,UpRows,m![L % 60,B % 8]>,
    weight:&HbmTensor<bf16,Chip,m![H,L]>,global:&HbmTensor<f32,Chip,m![1]>,
    output:&mut HbmTensor<bf16,Chip,m![B,H]>) {
    let input:DmTensor<bf16,Chip,Cluster,UpRows,m![B % 8,L % 60]>=device.main.begin(input.view())
        .fetch::<m![L % 60],m![B % 8]>().collect::<m![L % 60],m![B % 8 # 16]>()
        .transpose::<m![L / 4 % 15,B % 8],m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit();
    let input:DmTensor<bf16,Chip,Cluster,DownColumns,m![B % 8,L % 1920]>=input.to_dm(&mut device.tdma);
    let x:TrfTensor<bf16,Chip,Cluster,DownColumns,m![B % 8],m![L % 1920]>=device.sub.begin(input.view())
        .fetch::<m![B % 8,L / 16 % 120],m![L % 16]>().collect::<m![B % 8,L / 16 % 120],m![L % 16]>().to_trf();
    let weight:DmTensor<bf16,Chip,Cluster,DownColumns,m![H % 120,L % 1920]>=weight.to_dm(&mut device.tdma);
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,B % 8]>=device.main.begin(weight.view())
        .fetch::<m![H % 120,L / 16 % 120],m![L % 16]>().collect::<m![H % 120,L / 16 % 120],m![L % 16]>()
        .contract_outer::<m![H % 120,L / 32 % 60],m![L % 32],_,_,_>(&x)
        .contract_packet::<m![1]>().contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120],m![B % 8]>(LaneMode::Interleaved)
        .vector_init().vector_inter_slice_reduce::<DownRows,m![H % 120]>(InterSliceReduceOpF32::Add)
        .vector_final().cast::<bf16,m![B % 8 # 16]>().commit_trim::<m![B % 8]>().commit();
    let global:DmTensor<f32,Chip,Cluster,DownRows,m![1 # 8]>=global.to_dm(&mut device.tdma);
    let global:VrfTensor<f32,Chip,Cluster,DownRows,m![1 # 8]>=device.sub.begin(global.view())
        .fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>().to_vrf();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,B % 8]>=device.main.begin(result.view())
        .fetch::<m![H % 120],m![B % 8]>().fetch_cast::<f32>().collect::<m![H % 120],m![B % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120,B % 8 / 4],m![B % 8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&global)
        .vector_widen_concat::<m![H % 120],m![B % 8]>().vector_final()
        .cast::<bf16,m![B % 8 # 16]>().commit_trim::<m![B % 8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![B % 8,H % 120]>=device.main.begin(result.view())
        .fetch::<m![H % 120],m![B % 8 # 16]>().collect::<m![H % 120],m![B % 8 # 16]>()
        .transpose::<m![H / 4 % 30,B % 8],m![H % 4 # 16]>().commit_trim::<m![H % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
#[device(chip = 1)]
pub fn batch_ffn(device:&mut Device,input:&HbmTensor<bf16,Chip,m![B,H]>,
    up:&HbmTensor<bf16,Chip,m![L,H]>,gate:&HbmTensor<bf16,Chip,m![L,H]>,down_weight:&HbmTensor<bf16,Chip,m![H,L]>,
    ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
    output:&mut HbmTensor<bf16,Chip,m![B,H]>) {
    let input:DmTensor<bf16,Chip,Cluster,UpRows,m![B % 8,H]>=input.to_dm(&mut device.tdma);
    let up=upgate(device,&input,up);
    let gate=upgate(device,&input,gate);
    let hidden=crate::ops_batch_clusters::geglu_batch(device,&up,&gate,ug,gg);
    down(device,&hidden,down_weight,dg,output);
}
