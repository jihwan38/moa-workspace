//! Native E4 activation × original FP4 blockdot; weights and blockscales are unchanged.
//! Input is caller-quantized with one F32 power-of-two scale per token.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,H,L}};
type Cluster=m![P128/64];
type Parts=m![L/240,H/960];
type Rows=m![L/240,1 # 4];

#[device(chip = 1)]
pub fn project_up(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![P128,H]>,
    xscale:&HbmTensor<f32,Chip,m![P128]>,weight:&HbmTensor<f4e2m1,Chip,m![L,H]>,
    scale:&HbmTensor<f8e4m3,Chip,m![L,H / 16]>,output:&mut HbmTensor<bf16,Chip,m![P128,L]>) {
    let scale:DmTensor<f8e4m3,Chip,Cluster,Rows,m![L % 240,H / 16]>=scale.to_dm(&mut device.tdma);
    let scale:DmTensor<bf16,Chip,Cluster,Rows,m![L % 240,H / 16]>=device.main.begin(scale.view())
        .fetch::<m![L % 240,H / 256],m![H / 16 % 16]>().fetch_table_lookup::<bf16>()
        .collect::<m![L % 240,H / 256],m![H / 16 % 16]>().commit_trim::<m![H / 16 % 16]>().commit();
    let scale:DmTensor<bf16,Chip,Cluster,Parts,m![L % 240,H / 16 % 60]>=scale.to_dm(&mut device.tdma);
    let input:DmTensor<f8e4m3,Chip,Cluster,Parts,m![P128 % 64,H % 960]>=input.to_dm(&mut device.tdma);
    let weight:DmTensor<f4e2m1,Chip,Cluster,Parts,m![L % 240,H % 960]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<f8e4m3,Chip,Cluster,Parts,m![L % 240,H % 960]>=device.main.begin(weight.view())
        .fetch::<m![L % 240],m![H % 960]>().fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 240,H / 32 % 30],m![H % 32]>().commit_trim::<m![H % 32]>().commit();
    let mut result:DmTensor<f32,Chip,Cluster,Rows,m![L % 240,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let x:TrfTensor<f8e4m3,Chip,Cluster,Parts,m![P128 % 64 = 8],m![H % 960]>=device.sub
            .begin(input.view().tile::<m![P128 % 64],8,m![P128 % 64 = 8 # 64,H % 960]>(8*chunk))
            .fetch::<m![P128 % 64 = 8,H / 32 % 30],m![H % 32]>()
            .collect::<m![P128 % 64 = 8,H / 32 % 30],m![H % 32]>().to_trf();
        for row in 0..8 {
            let scales:VrfTensor<f32,Chip,Cluster,Parts,m![L % 240 = 30,H / 16 % 60 # 64]>=device.sub
                .begin(scale.view().tile::<m![L % 240],30,m![L % 240 = 30 # 240,H / 16 % 60]>(30*row))
                .fetch::<m![L % 240 = 30],m![H / 16 % 60]>().fetch_cast::<f32>()
                .collect::<m![L % 240 = 30,(H / 16 % 60 # 64) / 8],m![(H / 16 % 60 # 64) % 8]>().to_vrf();
            device.main.begin(weight.view().tile::<m![L % 240],30,m![L % 240 = 30 # 240,H % 960]>(30*row))
                .fetch::<m![L % 240 = 30,H / 32 % 30],m![H % 32]>()
                .collect::<m![L % 240 = 30,H / 32 % 30],m![H % 32]>()
                .contract_outer::<m![L % 240 = 30,H / 64 % 15],m![H % 64],_,_,_>(&x)
                .contract_packet::<m![H / 16 % 4]>()
                .contract_time::<m![L % 240 = 30,H / 64 % 15]>()
                .contract_lane::<m![L % 240 = 30,H / 64 % 15,P128 % 64 = 8],m![H / 16 % 4 # 8]>(LaneMode::Sequential)
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![H / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scales)
                .vector_intra_slice_reduce::<H,m![L % 240 = 30,P128 % 64 = 8],m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<Rows,m![L % 240 = 30,P128 % 64 = 8]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .transpose::<m![L % 240 = 30,P128 % 64 = 8 / 2],m![P128 % 64 = 8 % 2 # 8]>()
                .commit_trim::<m![P128 % 64 = 8 % 2]>()
                .commit_view(result.view_mut()
                    .tile::<m![L % 240],30,m![L % 240 = 30 #{!} 240,P128 % 64]>(30*row)
                    .tile::<m![P128 % 64],8,m![L % 240 = 30 #{!} 240,P128 % 64 = 8 #{!} 64]>(8*chunk));
        }
    }
    let xscale:DmTensor<f32,Chip,Cluster,Rows,m![P128 % 64]>=xscale.to_dm(&mut device.tdma);
    let xscale:VrfTensor<f32,Chip,Cluster,Rows,m![P128 % 64]>=device.sub.begin(xscale.view())
        .fetch::<m![P128 / 8 % 8],m![P128 % 8]>().collect::<m![P128 / 8 % 8],m![P128 % 8]>().to_vrf();
    let result:DmTensor<bf16,Chip,Cluster,Rows,m![L % 240,P128 % 64]>=device.main.begin(result.view())
        .fetch::<m![L % 240,P128 / 8 % 8],m![P128 % 8]>().collect::<m![L % 240,P128 / 8 % 8],m![P128 % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 240,P128 / 4 % 16],m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xscale)
        .vector_widen_concat::<m![L % 240,P128 / 8 % 8],m![P128 % 8]>()
        .vector_final().cast::<bf16,m![P128 % 8 # 16]>().commit_trim::<m![P128 % 8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,Rows,m![P128 % 64,L % 240]>=device.main.begin(result.view())
        .fetch::<m![P128 / 16 % 4,L % 240],m![P128 % 16]>().collect::<m![P128 / 16 % 4,L % 240],m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4,L / 4 % 60,P128 % 16],m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}


#[device(chip = 1)]
pub fn project_up_rows6(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![P128,H]>,
    xscale:&HbmTensor<f32,Chip,m![P128]>,weight:&HbmTensor<f4e2m1,Chip,m![L,H]>,
    scale:&HbmTensor<f8e4m3,Chip,m![L,H / 16]>,output:&mut HbmTensor<bf16,Chip,m![P128,L]>) {
    let scale:DmTensor<f8e4m3,Chip,Cluster,Rows,m![L % 240,H / 16]>=scale.to_dm(&mut device.tdma);
    let scale:DmTensor<bf16,Chip,Cluster,Rows,m![L % 240,H / 16]>=device.main.begin(scale.view())
        .fetch::<m![L % 240,H / 256],m![H / 16 % 16]>().fetch_table_lookup::<bf16>()
        .collect::<m![L % 240,H / 256],m![H / 16 % 16]>().commit_trim::<m![H / 16 % 16]>().commit();
    let scale:DmTensor<bf16,Chip,Cluster,Parts,m![L % 240,H / 16 % 60]>=scale.to_dm(&mut device.tdma);
    let input:DmTensor<f8e4m3,Chip,Cluster,Parts,m![P128 % 64,H % 960]>=input.to_dm(&mut device.tdma);
    let weight:DmTensor<f4e2m1,Chip,Cluster,Parts,m![L % 240,H % 960]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<f8e4m3,Chip,Cluster,Parts,m![L % 240,H % 960]>=device.main.begin(weight.view())
        .fetch::<m![L % 240],m![H % 960]>().fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 240,H / 32 % 30],m![H % 32]>().commit_trim::<m![H % 32]>().commit();
    let mut result:DmTensor<f32,Chip,Cluster,Rows,m![L % 240,P128 % 64]>=DmTensor::new();
    for chunk in 0..8 {
        let x:TrfTensor<f8e4m3,Chip,Cluster,Parts,m![P128 % 64 = 8],m![H % 960]>=device.sub
            .begin(input.view().tile::<m![P128 % 64],8,m![P128 % 64 = 8 # 64,H % 960]>(8*chunk))
            .fetch::<m![P128 % 64 = 8,H / 32 % 30],m![H % 32]>()
            .collect::<m![P128 % 64 = 8,H / 32 % 30],m![H % 32]>().to_trf();
        for row in 0..40 {
            let scales:VrfTensor<f32,Chip,Cluster,Parts,m![L % 240 = 6,H / 16 % 60 # 64]>=device.sub
                .begin(scale.view().tile::<m![L % 240],6,m![L % 240 = 6 # 240,H / 16 % 60]>(6*row))
                .fetch::<m![L % 240 = 6],m![H / 16 % 60]>().fetch_cast::<f32>()
                .collect::<m![L % 240 = 6,(H / 16 % 60 # 64) / 8],m![(H / 16 % 60 # 64) % 8]>().to_vrf();
            device.main.begin(weight.view().tile::<m![L % 240],6,m![L % 240 = 6 # 240,H % 960]>(6*row))
                .fetch::<m![L % 240 = 6,H / 32 % 30],m![H % 32]>()
                .collect::<m![L % 240 = 6,H / 32 % 30],m![H % 32]>()
                .contract_outer::<m![L % 240 = 6,H / 64 % 15],m![H % 64],_,_,_>(&x)
                .contract_packet::<m![H / 16 % 4]>()
                .contract_time::<m![L % 240 = 6,H / 64 % 15]>()
                .contract_lane::<m![L % 240 = 6,H / 64 % 15,P128 % 64 = 8],m![H / 16 % 4 # 8]>(LaneMode::Sequential)
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![H / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scales)
                .vector_intra_slice_reduce::<H,m![L % 240 = 6,P128 % 64 = 8],m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<Rows,m![L % 240 = 6,P128 % 64 = 8]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .transpose::<m![L % 240 = 6,P128 % 64 = 8 / 2],m![P128 % 64 = 8 % 2 # 8]>()
                .commit_trim::<m![P128 % 64 = 8 % 2]>()
                .commit_view(result.view_mut()
                    .tile::<m![L % 240],6,m![L % 240 = 6 #{!} 240,P128 % 64]>(6*row)
                    .tile::<m![P128 % 64],8,m![L % 240 = 6 #{!} 240,P128 % 64 = 8 #{!} 64]>(8*chunk));
        }
    }
    let xscale:DmTensor<f32,Chip,Cluster,Rows,m![P128 % 64]>=xscale.to_dm(&mut device.tdma);
    let xscale:VrfTensor<f32,Chip,Cluster,Rows,m![P128 % 64]>=device.sub.begin(xscale.view())
        .fetch::<m![P128 / 8 % 8],m![P128 % 8]>().collect::<m![P128 / 8 % 8],m![P128 % 8]>().to_vrf();
    let result:DmTensor<bf16,Chip,Cluster,Rows,m![L % 240,P128 % 64]>=device.main.begin(result.view())
        .fetch::<m![L % 240,P128 / 8 % 8],m![P128 % 8]>().collect::<m![L % 240,P128 / 8 % 8],m![P128 % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 240,P128 / 4 % 16],m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xscale)
        .vector_widen_concat::<m![L % 240,P128 / 8 % 8],m![P128 % 8]>()
        .vector_final().cast::<bf16,m![P128 % 8 # 16]>().commit_trim::<m![P128 % 8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,Rows,m![P128 % 64,L % 240]>=device.main.begin(result.view())
        .fetch::<m![P128 / 16 % 4,L % 240],m![P128 % 16]>().collect::<m![P128 / 16 % 4,L % 240],m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4,L / 4 % 60,P128 % 16],m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

/// Same weighted-BF16 arithmetic as H960 FFN; probe reconstructs the FP8 inputs exactly.
#[device(chip = 1)]
pub fn project_up_reference(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,H]>,
    weight:&HbmTensor<f4e2m1,Chip,m![L,H]>,scale:&HbmTensor<f8e4m3,Chip,m![L,H / 16]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,L]>) {
    let input:DmTensor<bf16,Chip,Cluster,Parts,m![P128 % 64,H % 960]>=input.to_dm(&mut device.tdma);
    let result=crate::ops_batch_kpart::upgate_k4(device,&input,weight,scale);
    let result:DmTensor<bf16,Chip,Cluster,Rows,m![P128 % 64,L % 240]>=device.main.begin(result.view())
        .fetch::<m![P128 / 16 % 4,L % 240],m![P128 % 16]>().collect::<m![P128 / 16 % 4,L % 240],m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4,L / 4 % 60,P128 % 16],m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
