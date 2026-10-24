//! Packed down L960 split; same weights, different16-way F32 reduction tree.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,H,L}};
type Cluster=m![P128/64];
type UpParts=m![L/240,H/960];
type UpRows=m![L/60];
type DownParts=m![H/240,L/960];
type DownReduced=m![H/240,1 # 16];
fn down960_dot(device:&mut Device,input:&DmTensor<bf16,Chip,Cluster,DownParts,m![P128 % 64,L % 960]>,
    packed:&HbmTensor<f4e2m1,Chip,m![H,L]>,scale:&HbmTensor<f8e4m3,Chip,m![H,L / 16]>)
    ->DmTensor<bf16,Chip,Cluster,DownReduced,m![H % 240,P128 % 64]> {
    let mut result:DmTensor<bf16,Chip,Cluster,DownReduced,m![H % 240,P128 % 64]>=DmTensor::new();
    for pass in 0..4 {
        // Expand60 full-K scale rows at a time so expansion plus input fits DM.
        let scale:DmTensor<f8e4m3,Chip,Cluster,DownReduced,m![H % 240 = 60,L / 16]>=scale.view()
            .tile::<m![H % 240],60,m![H / 240,H % 240 = 60 # 240,L / 16]>(60*pass).to_dm(&mut device.tdma);
        let scale:DmTensor<bf16,Chip,Cluster,DownReduced,m![H % 240 = 60,L / 16]>=device.main.begin(scale.view())
            .fetch::<m![H % 240 = 60,L / 256],m![L / 16 % 16]>().fetch_table_lookup::<bf16>()
            .collect::<m![H % 240 = 60,L / 256],m![L / 16 % 16]>().commit_trim::<m![L / 16 % 16]>().commit();
        let scale:DmTensor<bf16,Chip,Cluster,DownParts,m![H % 240 = 60,L / 16 % 60]>=scale.to_dm(&mut device.tdma);
        let packed:DmTensor<f4e2m1,Chip,Cluster,DownParts,m![H % 240 = 60,L % 960]>=packed.view()
            .tile::<m![H % 240],60,m![H / 240,H % 240 = 60 # 240,L]>(60*pass).to_dm(&mut device.tdma);
        let decoded:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![H % 240 = 60,L % 960]>=device.main.begin(packed.view())
            .fetch::<m![H % 240 = 60],m![L % 960]>().fetch_table_lookup::<f8e4m3>()
            .collect::<m![H % 240 = 60,L / 32 % 30],m![L % 32]>().commit_trim::<m![L % 32]>().commit();
        let mut weights:DmTensor<bf16,Chip,Cluster,DownParts,m![H % 240 = 60,L % 960]>=DmTensor::new();
        for part in 0..15 {
            let scales:VrfTensor<f32,Chip,Cluster,DownParts,m![H % 240 = 4,L / 16 % 60 # 64]>=device.sub
                .begin(scale.view().tile::<m![H % 240 = 60],4,m![H % 240 = 60 = 4 # 60,L / 16 % 60]>(4*part))
                .fetch::<m![H % 240 = 4],m![L / 16 % 60]>().fetch_cast::<f32>()
                .collect::<m![H % 240 = 4,(L / 16 % 60 # 64) / 8],m![(L / 16 % 60 # 64) % 8]>().to_vrf();
            device.main.begin(decoded.view().tile::<m![H % 240 = 60],4,m![H % 240 = 60 = 4 # 60,L % 960]>(4*part))
                .fetch::<m![H % 240 = 4,L / 32 % 30],m![L % 32]>().fetch_cast::<f32>()
                .collect::<m![H % 240 = 4,L / 8 % 120],m![L % 8]>()
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H % 240 = 4,L / 4 % 240],m![L % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scales)
                .vector_widen_concat::<m![H % 240 = 4,L / 8 % 120],m![L % 8]>()
                .vector_final().cast::<bf16,m![L % 8 # 16]>().commit_trim::<m![L % 8]>()
                .commit_view(weights.view_mut().tile::<m![H % 240 = 60],4,m![H % 240 = 60 = 4 #{!} 60,L % 960]>(4*part));
        }
        let mut partial:DmTensor<bf16,Chip,Cluster,DownReduced,m![H % 240 = 60,P128 % 64]>=DmTensor::new();
        for chunk in 0..8 {
            let x:TrfTensor<bf16,Chip,Cluster,DownParts,m![P128 % 64 = 8],m![L % 960]>=device.sub
                .begin(input.view().tile::<m![P128 % 64],8,m![P128 % 64 = 8 # 64,L % 960]>(8*chunk))
                .fetch::<m![P128 % 64 = 8,L / 16 % 60],m![L % 16]>()
                .collect::<m![P128 % 64 = 8,L / 16 % 60],m![L % 16]>().to_trf();
            device.main.begin(weights.view())
                .fetch::<m![H % 240 = 60,L / 16 % 60],m![L % 16]>()
                .collect::<m![H % 240 = 60,L / 16 % 60],m![L % 16]>()
                .contract_outer::<m![H % 240 = 60,L / 32 % 30],m![L % 32],_,_,_>(&x)
                .contract_packet::<m![1]>().contract_time::<m![H % 240 = 60]>()
                .contract_lane::<m![H % 240 = 60],m![P128 % 64 = 8]>(LaneMode::Interleaved)
                .vector_init().vector_inter_slice_reduce::<DownReduced,m![H % 240 = 60]>(InterSliceReduceOpF32::Add)
                .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
                .commit_view(partial.view_mut().tile::<m![P128 % 64],8,m![H % 240 = 60,P128 % 64 = 8 #{!} 64]>(8*chunk));
        }
        partial.view().to_dm_view(&mut device.tdma,result.view_mut().tile::<m![H % 240],60,m![H % 240 = 60 #{!} 240,P128 % 64]>(60*pass));
    }
    result
}


fn down960(device:&mut Device,input:&DmTensor<bf16,Chip,Cluster,UpRows,m![L % 60,P128 % 64]>,
    weight:&HbmTensor<f4e2m1,Chip,m![H,L]>,scale:&HbmTensor<f8e4m3,Chip,m![H,L / 16]>,
    global:&HbmTensor<f32,Chip,m![1]>,out:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    let input: DmTensor<bf16, Chip, Cluster, UpRows, m![P128 % 64, L % 60]> = device
        .main
        .begin(input.view())
        .fetch::<m![P128 / 16 % 4, L % 60], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, L % 60], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, L / 4 % 15, P128 % 16], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();
    let input:DmTensor<bf16,Chip,Cluster,DownParts,m![P128 % 64,L % 960]>=input.to_dm(&mut device.tdma);
    let result=down960_dot(device,&input,weight,scale);
    let global: DmTensor<f32, Chip, Cluster, DownReduced, m![1 # 8]> = global.to_dm(&mut device.tdma);
    let global: VrfTensor<f32, Chip, Cluster, DownReduced, m![1 # 8]> = device
        .sub
        .begin(global.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, DownReduced, m![H % 240, P128 % 64]> = device
        .main
        .begin(result.view())
        .fetch::<m![H % 240, P128 / 8 % 8], m![P128 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 240, P128 / 8 % 8], m![P128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 240, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &global)
        .vector_widen_concat::<m![H % 240, P128 / 8 % 8], m![P128 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128 % 8 # 16]>()
        .commit_trim::<m![P128 % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, DownReduced, m![P128 % 64, H % 240]> = device
        .main
        .begin(result.view())
        .fetch::<m![P128 / 16 % 4, H % 240], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, H % 240], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, H / 4 % 60, P128 % 16], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn batch_ffn(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,H]>,
    up:&HbmTensor<f4e2m1,Chip,m![L,H]>,gate:&HbmTensor<f4e2m1,Chip,m![L,H]>,down:&HbmTensor<f4e2m1,Chip,m![H,L]>,
    us:&HbmTensor<f8e4m3,Chip,m![L,H / 16]>,gs:&HbmTensor<f8e4m3,Chip,m![L,H / 16]>,ds:&HbmTensor<f8e4m3,Chip,m![H,L / 16]>,
    ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    let input:DmTensor<bf16,Chip,Cluster,UpParts,m![P128 % 64,H % 960]>=input.to_dm(&mut device.tdma);
    let up=crate::ops_batch_kpart::upgate_k4(device,&input,up,us);
    let gate=crate::ops_batch_kpart::upgate_k4(device,&input,gate,gs);
    let hidden=crate::ops_batch_kpart::geglu_k4(device,&up,&gate,ug,gg);
    let hidden:DmTensor<bf16,Chip,Cluster,UpRows,m![L % 60,P128 % 64]>=hidden.to_dm(&mut device.tdma);
    down960(device,&hidden,down,ds,dg,output);
}
