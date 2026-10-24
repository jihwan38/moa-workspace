//! Contracting-dimension partition experiment. It preserves the BF16 dot
//! boundary, but the preceding F32 reduction tree differs from the full-H dot.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,H,Qs,Ps,Qf,Pf,L}};
type Cluster=m![P128 / 64];
type Parts=m![Qs / 64,H / 960];
type Rows=m![Qs / 64,1 # 4];

#[device(chip = 1)]
pub fn project_query_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,H]>,
    weight:&HbmTensor<f8e4m3,Chip,m![Qs,H]>,scale:&HbmTensor<bf16,Chip,m![Qs]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,Qs]>) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,Parts,m![Qs % 64,H % 960]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,Parts,m![Qs % 64,H % 960]>=device.main.begin(weight.view())
        .fetch::<m![Qs % 64,H / 32 % 30],m![H % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 64,H / 16 % 60],m![H % 16]>().commit_trim::<m![H % 16]>().commit();
    let input:DmTensor<bf16,Chip,Cluster,Parts,m![P128 % 64,H % 960]>=input.to_dm(&mut device.tdma);
    let scale:DmTensor<bf16,Chip,Cluster,Rows,m![Qs % 64]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,Rows,m![Qs % 64]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![Qs % 64]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 8],m![Qs % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,Rows,m![Qs % 64,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let x:TrfTensor<bf16,Chip,Cluster,Parts,m![P128 % 64 = 8],m![H % 960]>=device.sub
            .begin(input.view().tile::<m![P128 % 64],8,m![P128 % 64 = 8 # 64,H % 960]>(8*chunk))
            .fetch::<m![P128 % 64 = 8,H / 16 % 60],m![H % 16]>()
            .collect::<m![P128 % 64 = 8,H / 16 % 60],m![H % 16]>().to_trf();
        let dot:DmTensor<bf16,Chip,Cluster,Rows,m![Qs % 64,P128 % 64 = 8]>=device.main.begin(weight.view())
            .fetch::<m![Qs % 64,H / 16 % 60],m![H % 16]>()
            .collect::<m![Qs % 64,H / 16 % 60],m![H % 16]>()
            .contract_outer::<m![Qs % 64,H / 32 % 30],m![H % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![Qs % 64]>()
            .contract_lane::<m![Qs % 64],m![P128 % 64 = 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<Rows,m![Qs % 64]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>().commit();
        device.main.begin(dot.view()).fetch::<m![Qs % 64],m![P128 % 64 = 8]>().fetch_cast::<f32>()
            .collect::<m![Qs % 64],m![P128 % 64 = 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Qs % 64,P128 % 64 = 8 / 4],m![P128 % 64 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
            .vector_widen_concat::<m![Qs % 64],m![P128 % 64 = 8]>()
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(result.view_mut().tile::<m![P128 % 64],8,m![Qs % 64,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    let result:DmTensor<bf16,Chip,Cluster,Rows,m![P128 % 64,Qs % 64]>=device.main.begin(result.view())
        .fetch::<m![P128 / 16 % 4,Qs % 64],m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4,Qs % 64],m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4,Qs / 4 % 16,P128 % 16],m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

#[device(chip = 1)]
pub fn project_sliding_kv_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,H]>,
    weight:&HbmTensor<f8e4m3,Chip,m![Ps,H]>,scale:&HbmTensor<bf16,Chip,m![Ps]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,Ps]>) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,m![Ps / 32,H / 960],m![Ps % 32,H % 960]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,m![Ps / 32,H / 960],m![Ps % 32,H % 960]>=device.main.begin(weight.view())
        .fetch::<m![Ps % 32,H / 32 % 30],m![H % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 32,H / 16 % 60],m![H % 16]>().commit_trim::<m![H % 16]>().commit();
    let input:DmTensor<bf16,Chip,Cluster,m![Ps / 32,H / 960],m![P128 % 64,H % 960]>=input.to_dm(&mut device.tdma);
    let scale:DmTensor<bf16,Chip,Cluster,m![Ps / 32,1 # 4],m![Ps % 32]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,m![Ps / 32,1 # 4],m![Ps % 32]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![Ps % 32]>().fetch_cast::<f32>()
        .collect::<m![Ps / 8 % 4],m![Ps % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,m![Ps / 32,1 # 4],m![Ps % 32,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let x:TrfTensor<bf16,Chip,Cluster,m![Ps / 32,H / 960],m![P128 % 64 = 8],m![H % 960]>=device.sub
            .begin(input.view().tile::<m![P128 % 64],8,m![P128 % 64 = 8 # 64,H % 960]>(8*chunk))
            .fetch::<m![P128 % 64 = 8,H / 16 % 60],m![H % 16]>()
            .collect::<m![P128 % 64 = 8,H / 16 % 60],m![H % 16]>().to_trf();
        let dot:DmTensor<bf16,Chip,Cluster,m![Ps / 32,1 # 4],m![Ps % 32,P128 % 64 = 8]>=device.main.begin(weight.view())
            .fetch::<m![Ps % 32,H / 16 % 60],m![H % 16]>()
            .collect::<m![Ps % 32,H / 16 % 60],m![H % 16]>()
            .contract_outer::<m![Ps % 32,H / 32 % 30],m![H % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![Ps % 32]>()
            .contract_lane::<m![Ps % 32],m![P128 % 64 = 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<m![Ps / 32,1 # 4],m![Ps % 32]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>().commit();
        device.main.begin(dot.view()).fetch::<m![Ps % 32],m![P128 % 64 = 8]>().fetch_cast::<f32>()
            .collect::<m![Ps % 32],m![P128 % 64 = 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Ps % 32,P128 % 64 = 8 / 4],m![P128 % 64 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
            .vector_widen_concat::<m![Ps % 32],m![P128 % 64 = 8]>()
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(result.view_mut().tile::<m![P128 % 64],8,m![Ps % 32,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    let result:DmTensor<bf16,Chip,Cluster,m![Ps / 32,1 # 4],m![P128 % 64,Ps % 32]>=device.main.begin(result.view())
        .fetch::<m![P128 / 16 % 4,Ps % 32],m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4,Ps % 32],m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4,Ps / 4 % 8,P128 % 16],m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

#[device(chip = 1)]
pub fn project_full_query_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,H]>,
    weight:&HbmTensor<f8e4m3,Chip,m![Qf,H]>,scale:&HbmTensor<bf16,Chip,m![Qf]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,Qf]>) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,m![Qf / 128,H / 960],m![Qf % 128,H % 960]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,m![Qf / 128,H / 960],m![Qf % 128,H % 960]>=device.main.begin(weight.view())
        .fetch::<m![Qf % 128,H / 32 % 30],m![H % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![Qf % 128,H / 16 % 60],m![H % 16]>().commit_trim::<m![H % 16]>().commit();
    let input:DmTensor<bf16,Chip,Cluster,m![Qf / 128,H / 960],m![P128 % 64,H % 960]>=input.to_dm(&mut device.tdma);
    let scale:DmTensor<bf16,Chip,Cluster,m![Qf / 128,1 # 4],m![Qf % 128]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,m![Qf / 128,1 # 4],m![Qf % 128]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![Qf % 128]>().fetch_cast::<f32>()
        .collect::<m![Qf / 8 % 16],m![Qf % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,m![Qf / 128,1 # 4],m![Qf % 128,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let x:TrfTensor<bf16,Chip,Cluster,m![Qf / 128,H / 960],m![P128 % 64 = 8],m![H % 960]>=device.sub
            .begin(input.view().tile::<m![P128 % 64],8,m![P128 % 64 = 8 # 64,H % 960]>(8*chunk))
            .fetch::<m![P128 % 64 = 8,H / 16 % 60],m![H % 16]>()
            .collect::<m![P128 % 64 = 8,H / 16 % 60],m![H % 16]>().to_trf();
        let dot:DmTensor<bf16,Chip,Cluster,m![Qf / 128,1 # 4],m![Qf % 128,P128 % 64 = 8]>=device.main.begin(weight.view())
            .fetch::<m![Qf % 128,H / 16 % 60],m![H % 16]>()
            .collect::<m![Qf % 128,H / 16 % 60],m![H % 16]>()
            .contract_outer::<m![Qf % 128,H / 32 % 30],m![H % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![Qf % 128]>()
            .contract_lane::<m![Qf % 128],m![P128 % 64 = 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<m![Qf / 128,1 # 4],m![Qf % 128]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>().commit();
        device.main.begin(dot.view()).fetch::<m![Qf % 128],m![P128 % 64 = 8]>().fetch_cast::<f32>()
            .collect::<m![Qf % 128],m![P128 % 64 = 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Qf % 128,P128 % 64 = 8 / 4],m![P128 % 64 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
            .vector_widen_concat::<m![Qf % 128],m![P128 % 64 = 8]>()
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(result.view_mut().tile::<m![P128 % 64],8,m![Qf % 128,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    let result:DmTensor<bf16,Chip,Cluster,m![Qf / 128,1 # 4],m![P128 % 64,Qf % 128]>=device.main.begin(result.view())
        .fetch::<m![P128 / 16 % 4,Qf % 128],m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4,Qf % 128],m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4,Qf / 4 % 32,P128 % 16],m![Qf % 4 # 16]>()
        .commit_trim::<m![Qf % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

#[device(chip = 1)]
pub fn project_full_key_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,H]>,
    weight:&HbmTensor<f8e4m3,Chip,m![Pf,H]>,scale:&HbmTensor<bf16,Chip,m![Pf]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,Pf]>) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,m![Pf / 8,H / 960],m![Pf % 8,H % 960]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,m![Pf / 8,H / 960],m![Pf % 8,H % 960]>=device.main.begin(weight.view())
        .fetch::<m![Pf % 8,H / 32 % 30],m![H % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![Pf % 8,H / 16 % 60],m![H % 16]>().commit_trim::<m![H % 16]>().commit();
    let input:DmTensor<bf16,Chip,Cluster,m![Pf / 8,H / 960],m![P128 % 64,H % 960]>=input.to_dm(&mut device.tdma);
    let scale:DmTensor<bf16,Chip,Cluster,m![Pf / 8,1 # 4],m![Pf % 8]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,m![Pf / 8,1 # 4],m![Pf % 8]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![Pf % 8]>().fetch_cast::<f32>()
        .collect::<m![Pf / 8 % 1],m![Pf % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,m![Pf / 8,1 # 4],m![Pf % 8,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let x:TrfTensor<bf16,Chip,Cluster,m![Pf / 8,H / 960],m![P128 % 64 = 8],m![H % 960]>=device.sub
            .begin(input.view().tile::<m![P128 % 64],8,m![P128 % 64 = 8 # 64,H % 960]>(8*chunk))
            .fetch::<m![P128 % 64 = 8,H / 16 % 60],m![H % 16]>()
            .collect::<m![P128 % 64 = 8,H / 16 % 60],m![H % 16]>().to_trf();
        let dot:DmTensor<bf16,Chip,Cluster,m![Pf / 8,1 # 4],m![Pf % 8,P128 % 64 = 8]>=device.main.begin(weight.view())
            .fetch::<m![Pf % 8,H / 16 % 60],m![H % 16]>()
            .collect::<m![Pf % 8,H / 16 % 60],m![H % 16]>()
            .contract_outer::<m![Pf % 8,H / 32 % 30],m![H % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![Pf % 8]>()
            .contract_lane::<m![Pf % 8],m![P128 % 64 = 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<m![Pf / 8,1 # 4],m![Pf % 8]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>().commit();
        device.main.begin(dot.view()).fetch::<m![Pf % 8],m![P128 % 64 = 8]>().fetch_cast::<f32>()
            .collect::<m![Pf % 8],m![P128 % 64 = 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Pf % 8,P128 % 64 = 8 / 4],m![P128 % 64 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
            .vector_widen_concat::<m![Pf % 8],m![P128 % 64 = 8]>()
            .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
            .commit_view(result.view_mut().tile::<m![P128 % 64],8,m![Pf % 8,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    let result:DmTensor<bf16,Chip,Cluster,m![Pf / 8,1 # 4],m![P128 % 64,Pf % 8]>=device.main.begin(result.view())
        .fetch::<m![P128 / 16 % 4,Pf % 8],m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4,Pf % 8],m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4,Pf / 4 % 2,P128 % 16],m![Pf % 4 # 16]>()
        .commit_trim::<m![Pf % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

pub use crate::ops_batch_wide::{project_sliding_output_batch,project_full_output_batch};

type UpParts=m![L / 240,H / 960];
type UpReduced=m![L / 240,1 # 4];

pub(crate) fn upgate_k4(device:&mut Device,input:&DmTensor<bf16,Chip,Cluster,UpParts,m![P128 % 64,H % 960]>,
    packed:&HbmTensor<f4e2m1,Chip,m![L,H]>,scale:&HbmTensor<f8e4m3,Chip,m![L,H / 16]>)
    ->DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64]> {
    // An H960 shard has60 E4 scale bytes, below DMA's eight-byte tail alignment.
    // Expand exact E4 scales to BF16 before the split:60 BF16 values are120 bytes.
    let scale:DmTensor<f8e4m3,Chip,Cluster,UpReduced,m![L % 240,H / 16]>=scale.to_dm(&mut device.tdma);
    let scale:DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,H / 16]>=device.main.begin(scale.view())
        .fetch::<m![L % 240,H / 256],m![H / 16 % 16]>().fetch_table_lookup::<bf16>()
        .collect::<m![L % 240,H / 256],m![H / 16 % 16]>().commit_trim::<m![H / 16 % 16]>().commit();
    let scale:DmTensor<bf16,Chip,Cluster,UpParts,m![L % 240,H / 16 % 60]>=scale.to_dm(&mut device.tdma);
    let mut result:DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64]>=DmTensor::new();
    for pass in 0..5 {
        let packed:DmTensor<f4e2m1,Chip,Cluster,UpParts,m![L % 240 = 48,H % 960]>=packed.view()
            .tile::<m![L % 240],48,m![L / 240,L % 240 = 48 # 240,H]>(48*pass).to_dm(&mut device.tdma);
        let decoded:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![L % 240 = 48,H % 960]>=device.main.begin(packed.view())
            .fetch::<m![L % 240 = 48],m![H % 960]>().fetch_table_lookup::<f8e4m3>()
            .collect::<m![L % 240 = 48,H / 32 % 30],m![H % 32]>().commit_trim::<m![H % 32]>().commit();
        let mut weights:DmTensor<bf16,Chip,Cluster,UpParts,m![L % 240 = 48,H % 960]>=DmTensor::new();
        for part in 0..12 {
            let scales:VrfTensor<f32,Chip,Cluster,UpParts,m![L % 240 = 4,H / 16 % 60 # 64]>=device.sub
                .begin(scale.view().tile::<m![L % 240],4,m![L % 240 = 4 # 240,H / 16 % 60]>(48*pass+4*part))
                .fetch::<m![L % 240 = 4],m![H / 16 % 60]>().fetch_cast::<f32>()
                .collect::<m![L % 240 = 4,(H / 16 % 60 # 64) / 8],m![(H / 16 % 60 # 64) % 8]>().to_vrf();
            device.main.begin(decoded.view().tile::<m![L % 240 = 48],4,m![L % 240 = 48 = 4 # 48,H % 960]>(4*part))
                .fetch::<m![L % 240 = 4,H / 32 % 30],m![H % 32]>().fetch_cast::<f32>()
                .collect::<m![L % 240 = 4,H / 8 % 120],m![H % 8]>()
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![L % 240 = 4,H / 4 % 240],m![H % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scales)
                .vector_widen_concat::<m![L % 240 = 4,H / 8 % 120],m![H % 8]>()
                .vector_final().cast::<bf16,m![H % 8 # 16]>().commit_trim::<m![H % 8]>()
                .commit_view(weights.view_mut().tile::<m![L % 240 = 48],4,m![L % 240 = 48 = 4 #{!} 48,H % 960]>(4*part));
        }
        let mut partial:DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240 = 48,P128 % 64]>=DmTensor::new();
        for chunk in 0..8 {
            let x:TrfTensor<bf16,Chip,Cluster,UpParts,m![P128 % 64 = 8],m![H % 960]>=device.sub
                .begin(input.view().tile::<m![P128 % 64],8,m![P128 % 64 = 8 # 64,H % 960]>(8*chunk))
                .fetch::<m![P128 % 64 = 8,H / 16 % 60],m![H % 16]>()
                .collect::<m![P128 % 64 = 8,H / 16 % 60],m![H % 16]>().to_trf();
            device.main.begin(weights.view())
                .fetch::<m![L % 240 = 48,H / 16 % 60],m![H % 16]>()
                .collect::<m![L % 240 = 48,H / 16 % 60],m![H % 16]>()
                .contract_outer::<m![L % 240 = 48,H / 32 % 30],m![H % 32],_,_,_>(&x)
                .contract_packet::<m![1]>().contract_time::<m![L % 240 = 48]>()
                .contract_lane::<m![L % 240 = 48],m![P128 % 64 = 8]>(LaneMode::Interleaved)
                .vector_init().vector_inter_slice_reduce::<UpReduced,m![L % 240 = 48]>(InterSliceReduceOpF32::Add)
                .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
                .commit_view(partial.view_mut().tile::<m![P128 % 64],8,m![L % 240 = 48,P128 % 64 = 8 #{!} 64]>(8*chunk));
        }
        partial.view().to_dm_view(&mut device.tdma,result.view_mut().tile::<m![L % 240],48,m![L % 240 = 48 #{!} 240,P128 % 64]>(48*pass));
    }
    result
}

fn scale_upgate_tile(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, UpReduced, m![L % 240, P128 % 64 = 8]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpReduced, m![L % 240, P128 % 64 = 8]> {
    let scale: DmTensor<f32, Chip, Cluster, UpReduced, m![1 # 8]> =
        global_scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, UpReduced, m![1 # 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    device
        .main
        .begin(input.view())
        .fetch::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 240, P128 % 64 = 8 / 4], m![P128 % 64 = 8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128 % 64 = 8 % 8 # 16]>()
        .commit_trim::<m![P128 % 64 = 8 % 8]>()
        .commit()
}
fn geglu_tile(
    device: &mut Device,
    up: &DmTensor<bf16, Chip, Cluster, UpReduced, m![L % 240, P128 % 64 = 8]>,
    gate: &DmTensor<bf16, Chip, Cluster, UpReduced, m![L % 240, P128 % 64 = 8]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpReduced, m![L % 240, P128 % 64 = 8]> {
    let up = scale_upgate_tile(device, up, up_global_scale);
    let gate = scale_upgate_tile(device, gate, gate_global_scale);
    let gelu: DmTensor<f32, Chip, Cluster, UpReduced, m![L % 240, P128 % 64 = 8]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 240, P128 % 64 = 8 / 4], m![P128 % 64 = 8 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.70710678118f32)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .vector_final()
        .commit_trim::<m![P128 % 64 = 8 % 8]>()
        .commit();
    let gelu: VrfTensor<f32, Chip, Cluster, UpReduced, m![L % 240, P128 % 64 = 8]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .collect::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .to_vrf();
    device
        .main
        .begin(up.view())
        .fetch::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 240, P128 % 64 = 8 / 4], m![P128 % 64 = 8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L % 240, P128 % 64 = 8 / 8], m![P128 % 64 = 8 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128 % 64 = 8 % 8 # 16]>()
        .commit_trim::<m![P128 % 64 = 8 % 8]>()
        .commit()
}

fn geglu_k4(device:&mut Device,up:&DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64]>,
    gate:&DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64]>,
    us:&HbmTensor<f32,Chip,m![1]>,gs:&HbmTensor<f32,Chip,m![1]>)
    ->DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64]> {
    let mut result:DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let mut u:DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64 = 8]>=DmTensor::new();
        up.view().tile::<m![P128 % 64],8,m![L % 240,P128 % 64 = 8 # 64]>(8*chunk).to_dm_view(&mut device.tdma,u.view_mut());
        let mut g:DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64 = 8]>=DmTensor::new();
        gate.view().tile::<m![P128 % 64],8,m![L % 240,P128 % 64 = 8 # 64]>(8*chunk).to_dm_view(&mut device.tdma,g.view_mut());
        let y=geglu_tile(device,&u,&g,us,gs);
        y.view().to_dm_view(&mut device.tdma,result.view_mut().tile::<m![P128 % 64],8,m![L % 240,P128 % 64 = 8 #{!} 64]>(8*chunk));
    }
    result
}

#[device(chip = 1)]
pub fn batch_ffn(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,H]>,
    up:&HbmTensor<f4e2m1,Chip,m![L,H]>,gate:&HbmTensor<f4e2m1,Chip,m![L,H]>,down:&HbmTensor<f4e2m1,Chip,m![H,L]>,
    us:&HbmTensor<f8e4m3,Chip,m![L,H / 16]>,gs:&HbmTensor<f8e4m3,Chip,m![L,H / 16]>,ds:&HbmTensor<f8e4m3,Chip,m![H,L / 16]>,
    ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    // Both up and gate reuse this single H960-partitioned activation cache.
    let input:DmTensor<bf16,Chip,Cluster,UpParts,m![P128 % 64,H % 960]>=input.to_dm(&mut device.tdma);
    let up=upgate_k4(device,&input,up,us);
    let gate=upgate_k4(device,&input,gate,gs);
    let hidden=geglu_k4(device,&up,&gate,ug,gg);
    let hidden:DmTensor<bf16,Chip,Cluster,m![L / 60],m![L % 60,P128 % 64]>=hidden.to_dm(&mut device.tdma);
    crate::ops_batch_wide::down_cached(device,&hidden,down,ds,dg,output);
}
