//! P512 staged FFN with output-channel cluster partition: no duplicated weights.
//! Same FP8/BF16 arithmetic and staged HBM buffers as token-cluster baseline.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P512,H,L}};
pub use crate::ops_native_fp8_blocked::{UpWeight,DownWeight};
pub use crate::ops_native_fp8_p512::{quantize_hidden,geglu_quantize};
type UpCluster=m![L/7680];
type DownCluster=m![H/1920];
type UpParts=m![L/120%64,H/960];
type UpRows=m![L/120%64,1#4];
type DownParts=m![H/120%16,L/960];
type DownRows=m![H/120%16,1#16];
#[device(chip=1)]
pub fn project_up_scaled(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![P512,H]>,
 xscale:&HbmTensor<f32,Chip,m![P512]>,weight:&HbmTensor<f8e4m3,Chip,UpWeight>,
 weight_dequant_scale:&HbmTensor<f32,Chip,m![L]>,output:&mut HbmTensor<bf16,Chip,m![P512,L]>) {
    let weight:DmTensor<f8e4m3,Chip,UpCluster,UpParts,m![L % 120,H % 960]>=weight.to_dm(&mut device.tdma);
    let ws:DmTensor<f32,Chip,UpCluster,UpRows,m![L % 120]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,UpCluster,UpRows,m![L % 120]>=device.sub.begin(ws.view())
        .fetch::<m![L / 8 % 15],m![L % 8]>().collect::<m![L / 8 % 15],m![L % 8]>().to_vrf();
    for chunk in 0..8 {
        let input:DmTensor<f8e4m3,Chip,UpCluster,UpParts,m![P512=64,H%960]>=input.view()
            .tile::<m![P512],64,m![P512=64#512,H]>(64*chunk).to_dm(&mut device.tdma);
        let xscale:DmTensor<f32,Chip,UpCluster,UpRows,m![P512=64]>=xscale.view()
            .tile::<m![P512],64,m![P512=64#512]>(64*chunk).to_dm(&mut device.tdma);
    let xs:VrfTensor<f32,Chip,UpCluster,UpRows,m![P512=64]>=device.sub.begin(xscale.view())
        .fetch::<m![P512=64/8],m![P512=64%8]>().collect::<m![P512=64/8],m![P512=64%8]>().to_vrf();
    let x:TrfTensor<f8e4m3,Chip,UpCluster,UpParts,m![P512=64%8],m![P512=64/8,H%960]>=device.sub.begin(input.view())
        .fetch::<m![P512=64%8,P512=64/8,H/32%30],m![H%32]>()
        .collect::<m![P512=64%8,P512=64/8,H/32%30],m![H%32]>().to_trf();
    let result:DmTensor<bf16,Chip,UpCluster,UpRows,m![L%120,P512=64]>=device.main.begin(weight.view())
        .fetch::<m![L%120,H/32%30],m![H%32]>().collect::<m![L%120,H/32%30],m![H%32]>()
        .contract_outer::<m![L%120,H/64%15,P512=64/8],m![H%64],_,_,_>(&x)
        .contract_packet::<m![1]>().contract_time::<m![L%120,P512=64/8]>()
        .contract_lane::<m![L%120,P512=64/8],m![P512=64%8]>(LaneMode::Interleaved)
        .vector_init().vector_inter_slice_reduce::<UpRows,m![L%120,P512=64/8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L%120,P512=64/4],m![P512=64%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![L%120,P512=64/8],m![P512=64%8]>()
        .vector_final().cast::<bf16,m![P512=64%8#16]>().commit_trim::<m![P512=64%8]>().commit();
        let result:DmTensor<bf16,Chip,UpCluster,UpRows,m![P512=64,L%120]>=device.main.begin(result.view())
            .fetch::<m![P512=64/16,L%120],m![P512=64%16]>().collect::<m![P512=64/16,L%120],m![P512=64%16]>()
            .transpose::<m![P512=64/16,L/4%30,P512=64%16],m![L%4#16]>().commit_trim::<m![L%4]>().commit();
        result.view().to_hbm_view(&mut device.tdma,output.view_mut()
            .tile::<m![P512],64,m![P512=64#{!}512,L]>(64*chunk));
    }
}
#[device(chip=1)]
pub fn project_down_scaled(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![P512,L]>,
 xscale:&HbmTensor<f32,Chip,m![P512]>,weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
 weight_dequant_scale:&HbmTensor<f32,Chip,m![H]>,global_scale:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,m![P512,H]>) {
    let weight:DmTensor<f8e4m3,Chip,DownCluster,DownParts,m![H % 120,L % 960]>=weight.to_dm(&mut device.tdma);
    let ws:DmTensor<f32,Chip,DownCluster,DownRows,m![H % 120]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,DownCluster,DownRows,m![H % 120]>=device.sub.begin(ws.view())
        .fetch::<m![H / 8 % 15],m![H % 8]>().collect::<m![H / 8 % 15],m![H % 8]>().to_vrf();
    let global:DmTensor<f32,Chip,DownCluster,DownRows,m![1#8]>=global_scale.to_dm(&mut device.tdma);
    let global:VrfTensor<f32,Chip,DownCluster,DownRows,m![1#8]>=device.sub.begin(global.view())
        .fetch::<m![1],m![1#8]>().collect::<m![1],m![1#8]>().to_vrf();
    for chunk in 0..8 {
        let input:DmTensor<f8e4m3,Chip,DownCluster,DownParts,m![P512=64,L%960]>=input.view()
            .tile::<m![P512],64,m![P512=64#512,L]>(64*chunk).to_dm(&mut device.tdma);
        let xscale:DmTensor<f32,Chip,DownCluster,DownRows,m![P512=64]>=xscale.view()
            .tile::<m![P512],64,m![P512=64#512]>(64*chunk).to_dm(&mut device.tdma);
    let xs:VrfTensor<f32,Chip,DownCluster,DownRows,m![P512=64]>=device.sub.begin(xscale.view())
        .fetch::<m![P512=64/8],m![P512=64%8]>().collect::<m![P512=64/8],m![P512=64%8]>().to_vrf();
    let x:TrfTensor<f8e4m3,Chip,DownCluster,DownParts,m![P512=64%8],m![P512=64/8,L%960]>=device.sub.begin(input.view())
        .fetch::<m![P512=64%8,P512=64/8,L/32%30],m![L%32]>()
        .collect::<m![P512=64%8,P512=64/8,L/32%30],m![L%32]>().to_trf();
    let result:DmTensor<bf16,Chip,DownCluster,DownRows,m![H%120,P512=64]>=device.main.begin(weight.view())
        .fetch::<m![H%120,L/32%30],m![L%32]>().collect::<m![H%120,L/32%30],m![L%32]>()
        .contract_outer::<m![H%120,L/64%15,P512=64/8],m![L%64],_,_,_>(&x)
        .contract_packet::<m![1]>().contract_time::<m![H%120,P512=64/8]>()
        .contract_lane::<m![H%120,P512=64/8],m![P512=64%8]>(LaneMode::Interleaved)
        .vector_init().vector_inter_slice_reduce::<DownRows,m![H%120,P512=64/8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H%120,P512=64/4],m![P512=64%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![H%120,P512=64/8],m![P512=64%8]>()
        .vector_final().cast::<bf16,m![P512=64%8#16]>().commit_trim::<m![P512=64%8]>().commit();
        let result:DmTensor<bf16,Chip,DownCluster,DownRows,m![H%120,P512=64]>=device.main.begin(result.view())
            .fetch::<m![H%120,P512=64/8],m![P512=64%8]>().fetch_cast::<f32>()
            .collect::<m![H%120,P512=64/8],m![P512=64%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H%120,P512=64/4],m![P512=64%4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&global)
            .vector_widen_concat::<m![H%120,P512=64/8],m![P512=64%8]>()
            .vector_final().cast::<bf16,m![P512=64%8#16]>().commit_trim::<m![P512=64%8]>().commit();
        let result:DmTensor<bf16,Chip,DownCluster,DownRows,m![P512=64,H%120]>=device.main.begin(result.view())
            .fetch::<m![P512=64/16,H%120],m![P512=64%16]>().collect::<m![P512=64/16,H%120],m![P512=64%16]>()
            .transpose::<m![P512=64/16,H/4%30,P512=64%16],m![H%4#16]>().commit_trim::<m![H%4]>().commit();
        result.view().to_hbm_view(&mut device.tdma,output.view_mut()
            .tile::<m![P512],64,m![P512=64#{!}512,H]>(64*chunk));
    }
}
