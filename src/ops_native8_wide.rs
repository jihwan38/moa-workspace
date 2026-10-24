//! Prepacked native FP8 linear throughput bound; caller supplies scaled FP8 input/weights.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,H,L}};
type Cluster=m![P128 / 64];
type Parts=m![L / 240,H / 960];
type Rows=m![L / 240,1 # 4];

#[device(chip = 1)]
pub fn up_native8(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![P128,H]>,
    weight:&HbmTensor<f8e4m3,Chip,m![L,H]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,L]>) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,Parts,m![L % 240,H % 960]>=weight.to_dm(&mut device.tdma);
    let input:DmTensor<f8e4m3,Chip,Cluster,Parts,m![P128 % 64,H % 960]>=input.to_dm(&mut device.tdma);
    let mut result:DmTensor<bf16,Chip,Cluster,Rows,m![L % 240,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let x:TrfTensor<f8e4m3,Chip,Cluster,Parts,m![P128 % 64 = 8],m![H % 960]>=device.sub
            .begin(input.view().tile::<m![P128 % 64],8,m![P128 % 64 = 8 # 64,H % 960]>(8*chunk))
            .fetch::<m![P128 % 64 = 8,H / 32 % 30],m![H % 32]>()
            .collect::<m![P128 % 64 = 8,H / 32 % 30],m![H % 32]>().to_trf();
        let dot:DmTensor<bf16,Chip,Cluster,Rows,m![L % 240,P128 % 64 = 8]>=device.main.begin(weight.view())
            .fetch::<m![L % 240,H / 32 % 30],m![H % 32]>()
            .collect::<m![L % 240,H / 32 % 30],m![H % 32]>()
            .contract_outer::<m![L % 240,H / 64 % 15],m![H % 64],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![L % 240]>()
            .contract_lane::<m![L % 240],m![P128 % 64 = 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<Rows,m![L % 240]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>().commit();
        dot.view().to_dm_view(&mut device.tdma,result.view_mut().tile::<m![P128 % 64],8,m![L % 240,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    let result:DmTensor<bf16,Chip,Cluster,Rows,m![P128 % 64,L % 240]>=device.main.begin(result.view())
        .fetch::<m![P128 / 16 % 4,L % 240],m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4,L % 240],m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4,L / 4 % 60,P128 % 16],m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

type DownParts=m![H / 240,L / 960];
type DownRows=m![H / 240,1 # 16];
#[device(chip = 1)]
pub fn down_native8(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![P128,L]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,L]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![H % 240,L % 960]>=weight.to_dm(&mut device.tdma);
    let input:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![P128 % 64,L % 960]>=input.to_dm(&mut device.tdma);
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 240,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let x:TrfTensor<f8e4m3,Chip,Cluster,DownParts,m![P128 % 64 = 8],m![L % 960]>=device.sub
            .begin(input.view().tile::<m![P128 % 64],8,m![P128 % 64 = 8 # 64,L % 960]>(8*chunk))
            .fetch::<m![P128 % 64 = 8,L / 32 % 30],m![L % 32]>()
            .collect::<m![P128 % 64 = 8,L / 32 % 30],m![L % 32]>().to_trf();
        let dot:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 240,P128 % 64 = 8]>=device.main.begin(weight.view())
            .fetch::<m![H % 240,L / 32 % 30],m![L % 32]>()
            .collect::<m![H % 240,L / 32 % 30],m![L % 32]>()
            .contract_outer::<m![H % 240,L / 64 % 15],m![L % 64],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H % 240]>()
            .contract_lane::<m![H % 240],m![P128 % 64 = 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<DownRows,m![H % 240]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>().commit();
        dot.view().to_dm_view(&mut device.tdma,result.view_mut().tile::<m![P128 % 64],8,m![H % 240,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![P128 % 64,H % 240]>=device.main.begin(result.view())
        .fetch::<m![P128 / 16 % 4,H % 240],m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4,H % 240],m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4,H / 4 % 60,P128 % 16],m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
