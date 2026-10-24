//! Physically blocked FP8 weights; arithmetic is identical to ops_native_fp8.
//! Dot F32*x_dequant_scale*weight_dequant_scale rounds to BF16 before original globals.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P256,H,L}};
pub use crate::ops_native_fp8_blocked::{UpWeight,DownWeight};
type Cluster=m![P256/128];
type QuantRows=m![P256 % 128 # 256];
type UpParts=m![L/240,H/960];
type UpRows=m![L/240,1 # 4];
type DownParts=m![H/240,L/960];
type DownRows=m![H/240,1 # 16];

fn quantize_hidden(device:&mut Device,x:&DmTensor<bf16,Chip,Cluster,QuantRows,m![H]>)
    ->(DmTensor<f8e4m3,Chip,Cluster,QuantRows,m![H]>,DmTensor<f32,Chip,Cluster,QuantRows,m![1 # 8]>) {
            let amax:DmTensor<f32,Chip,Cluster,QuantRows,m![1 # 8]>=device.main.begin(x.view())
                .fetch::<m![H /16],m![H %16]>().fetch_cast::<f32>()
                .collect::<m![H /8],m![H %8]>()
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32)
                .vector_reinterpret::<f32>()
                .vector_narrow_split::<m![H /4],m![H %4]>()
                .vector_intra_slice_reduce::<H,m![1],m![1 #4]>(IntraSliceReduceOpF32::Max)
                .vector_widen_pad::<m![1 #8]>()
                .vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
                .vector_final().commit_trim::<m![1 #8]>().commit();
            let scale:DmTensor<f32,Chip,Cluster,QuantRows,m![1 #8]>=device.main.begin(amax.view())
                .fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>()
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7f80_0000i32)
                .vector_reinterpret::<f32>().vector_narrow_trim::<m![1 #4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),0.0078125f32)
                .vector_widen_pad::<m![1 #8]>().vector_final()
                .commit_trim::<m![1 #8]>().commit();
            let sv:VrfTensor<f32,Chip,Cluster,QuantRows,m![1 #8]>=device.sub.begin(scale.view())
                .fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
            let q:DmTensor<f8e4m3,Chip,Cluster,QuantRows,m![H]>=device.main.begin(x.view())
                .fetch::<m![H /16],m![H %16]>().fetch_cast::<f32>()
                .collect::<m![H /8],m![H %8]>()
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H /4],m![H %4]>()
                .vector_fp_binary(FpBinaryOp::DivF,&sv)
                .vector_widen_concat::<m![H /8],m![H %8]>().vector_final()
                .cast::<f8e4m3,m![H %8 #32]>().commit_trim::<m![H %8]>().commit();
    (q,scale)
}

fn quantize_ffn(device:&mut Device,x:&DmTensor<bf16,Chip,Cluster,QuantRows,m![L]>)
    ->(DmTensor<f8e4m3,Chip,Cluster,QuantRows,m![L]>,DmTensor<f32,Chip,Cluster,QuantRows,m![1 # 8]>) {
            let amax:DmTensor<f32,Chip,Cluster,QuantRows,m![1 # 8]>=device.main.begin(x.view())
                .fetch::<m![L /16],m![L %16]>().fetch_cast::<f32>()
                .collect::<m![L /8],m![L %8]>()
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32)
                .vector_reinterpret::<f32>()
                .vector_narrow_split::<m![L /4],m![L %4]>()
                .vector_intra_slice_reduce::<L,m![1],m![1 #4]>(IntraSliceReduceOpF32::Max)
                .vector_widen_pad::<m![1 #8]>()
                .vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
                .vector_final().commit_trim::<m![1 #8]>().commit();
            let scale:DmTensor<f32,Chip,Cluster,QuantRows,m![1 #8]>=device.main.begin(amax.view())
                .fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>()
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7f80_0000i32)
                .vector_reinterpret::<f32>().vector_narrow_trim::<m![1 #4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),0.0078125f32)
                .vector_widen_pad::<m![1 #8]>().vector_final()
                .commit_trim::<m![1 #8]>().commit();
            let sv:VrfTensor<f32,Chip,Cluster,QuantRows,m![1 #8]>=device.sub.begin(scale.view())
                .fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
            let q:DmTensor<f8e4m3,Chip,Cluster,QuantRows,m![L]>=device.main.begin(x.view())
                .fetch::<m![L /16],m![L %16]>().fetch_cast::<f32>()
                .collect::<m![L /8],m![L %8]>()
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![L /4],m![L %4]>()
                .vector_fp_binary(FpBinaryOp::DivF,&sv)
                .vector_widen_concat::<m![L /8],m![L %8]>().vector_final()
                .cast::<f8e4m3,m![L %8 #32]>().commit_trim::<m![L %8]>().commit();
    (q,scale)
}

fn pack_up_scales(device:&mut Device,xscale:&DmTensor<f32,Chip,Cluster,UpRows,m![P256 % 128,1 # 8]>)
    ->DmTensor<f32,Chip,Cluster,UpRows,m![P256 % 128]> {
    // Keep the 32-byte scalar slot for aligned inter-slice DMA, then pack by TU.
    let xscale:DmTensor<f32,Chip,Cluster,UpRows,m![P256 % 128]>=device.sub.begin(xscale.view())
        .fetch::<m![P256 % 128],m![1 # 8]>().collect::<m![P256 % 128],m![1 # 8]>()
        .transpose::<m![P256 % 128 / 2],m![P256 % 128 % 2 # 8]>().commit_trim::<m![P256 % 128 % 2]>().commit();
    xscale
}
fn matrix_up(device:&mut Device,input:&DmTensor<f8e4m3,Chip,Cluster,UpParts,m![P256 % 128,H % 960]>,
    xscale:&DmTensor<f32,Chip,Cluster,UpRows,m![P256 % 128]>,weight:&HbmTensor<f8e4m3,Chip,UpWeight>,
    weight_dequant_scale:&HbmTensor<f32,Chip,m![L]>) ->DmTensor<bf16,Chip,Cluster,UpRows,m![L % 240,P256 % 128]> {
    let weight:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![L % 240,H % 960]>=weight.to_dm(&mut device.tdma);
    let ws:DmTensor<f32,Chip,Cluster,UpRows,m![L % 240]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,Cluster,UpRows,m![L % 240]>=device.sub.begin(ws.view())
        .fetch::<m![L / 8 % 30],m![L % 8]>().collect::<m![L / 8 % 30],m![L % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,UpRows,m![L % 240,P256 % 128]>=DmTensor::new();
    for chunk in 0..16 {
        let xs:VrfTensor<f32,Chip,Cluster,UpRows,m![P256 % 128 = 8]>=device.sub
            .begin(xscale.view().tile::<m![P256 % 128],8,m![P256 % 128 = 8 # 128]>(8*chunk))
            .fetch::<m![1],m![P256 % 128 = 8]>().collect::<m![1],m![P256 % 128 = 8]>().to_vrf();
        let x:TrfTensor<f8e4m3,Chip,Cluster,UpParts,m![P256 % 128 = 8],m![H % 960]>=device.sub
            .begin(input.view().tile::<m![P256 % 128],8,m![P256 % 128 = 8 # 128,H % 960]>(8*chunk))
            .fetch::<m![P256 % 128 = 8,H / 32 % 30],m![H % 32]>()
            .collect::<m![P256 % 128 = 8,H / 32 % 30],m![H % 32]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![L % 240,H / 32 % 30],m![H % 32]>()
            .collect::<m![L % 240,H / 32 % 30],m![H % 32]>()
            .contract_outer::<m![L % 240,H / 64 % 15],m![H % 64],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![L % 240]>()
            .contract_lane::<m![L % 240],m![P256 % 128 = 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<UpRows,m![L % 240]>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 240,P256 % 128 = 8 / 4],m![P256 % 128 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
            .vector_widen_concat::<m![L % 240],m![P256 % 128 = 8]>()
            .vector_final().cast::<bf16,m![P256 % 128 = 8 # 16]>().commit_trim::<m![P256 % 128 = 8]>()
            .commit_view(result.view_mut().tile::<m![P256 % 128],8,m![L % 240,P256 % 128 = 8 #{!} 128]>(8*chunk));
    }
    result
}

fn pack_down_scales(device:&mut Device,xscale:&DmTensor<f32,Chip,Cluster,DownRows,m![P256 % 128,1 # 8]>)
    ->DmTensor<f32,Chip,Cluster,DownRows,m![P256 % 128]> {
    // Keep the 32-byte scalar slot for aligned inter-slice DMA, then pack by TU.
    let xscale:DmTensor<f32,Chip,Cluster,DownRows,m![P256 % 128]>=device.sub.begin(xscale.view())
        .fetch::<m![P256 % 128],m![1 # 8]>().collect::<m![P256 % 128],m![1 # 8]>()
        .transpose::<m![P256 % 128 / 2],m![P256 % 128 % 2 # 8]>().commit_trim::<m![P256 % 128 % 2]>().commit();
    xscale
}
fn matrix_down(device:&mut Device,input:&DmTensor<f8e4m3,Chip,Cluster,DownParts,m![P256 % 128,L % 960]>,
    xscale:&DmTensor<f32,Chip,Cluster,DownRows,m![P256 % 128]>,weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
    weight_dequant_scale:&HbmTensor<f32,Chip,m![H]>) ->DmTensor<bf16,Chip,Cluster,DownRows,m![H % 240,P256 % 128]> {
    let weight:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![H % 240,L % 960]>=weight.to_dm(&mut device.tdma);
    let ws:DmTensor<f32,Chip,Cluster,DownRows,m![H % 240]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 240]>=device.sub.begin(ws.view())
        .fetch::<m![H / 8 % 30],m![H % 8]>().collect::<m![H / 8 % 30],m![H % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 240,P256 % 128]>=DmTensor::new();
    for chunk in 0..16 {
        let xs:VrfTensor<f32,Chip,Cluster,DownRows,m![P256 % 128 = 8]>=device.sub
            .begin(xscale.view().tile::<m![P256 % 128],8,m![P256 % 128 = 8 # 128]>(8*chunk))
            .fetch::<m![1],m![P256 % 128 = 8]>().collect::<m![1],m![P256 % 128 = 8]>().to_vrf();
        let x:TrfTensor<f8e4m3,Chip,Cluster,DownParts,m![P256 % 128 = 8],m![L % 960]>=device.sub
            .begin(input.view().tile::<m![P256 % 128],8,m![P256 % 128 = 8 # 128,L % 960]>(8*chunk))
            .fetch::<m![P256 % 128 = 8,L / 32 % 30],m![L % 32]>()
            .collect::<m![P256 % 128 = 8,L / 32 % 30],m![L % 32]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H % 240,L / 32 % 30],m![L % 32]>()
            .collect::<m![H % 240,L / 32 % 30],m![L % 32]>()
            .contract_outer::<m![H % 240,L / 64 % 15],m![L % 64],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H % 240]>()
            .contract_lane::<m![H % 240],m![P256 % 128 = 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<DownRows,m![H % 240]>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 240,P256 % 128 = 8 / 4],m![P256 % 128 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
            .vector_widen_concat::<m![H % 240],m![P256 % 128 = 8]>()
            .vector_final().cast::<bf16,m![P256 % 128 = 8 # 16]>().commit_trim::<m![P256 % 128 = 8]>()
            .commit_view(result.view_mut().tile::<m![P256 % 128],8,m![H % 240,P256 % 128 = 8 #{!} 128]>(8*chunk));
    }
    result
}

#[device(chip = 1)]
pub fn project_up_scaled(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![P256,H]>,xscale:&HbmTensor<f32,Chip,m![P256]>,
    weight:&HbmTensor<f8e4m3,Chip,UpWeight>,weight_scale:&HbmTensor<f32,Chip,m![L]>,output:&mut HbmTensor<bf16,Chip,m![P256,L]>) {
    let input:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![P256 % 128,H % 960]>=input.to_dm(&mut device.tdma);
    let xscale:DmTensor<f32,Chip,Cluster,UpRows,m![P256 % 128]>=xscale.to_dm(&mut device.tdma);
    let result=matrix_up(device,&input,&xscale,weight,weight_scale);
    let result:DmTensor<bf16,Chip,Cluster,UpRows,m![P256 % 128,L % 240]>=device.main.begin(result.view())
        .fetch::<m![P256 / 16 % 8,L % 240],m![P256 % 16]>().collect::<m![P256 / 16 % 8,L % 240],m![P256 % 16]>()
        .transpose::<m![P256 / 16 % 8,L / 4 % 60,P256 % 16],m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
#[device(chip = 1)]
pub fn project_down_scaled(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![P256,L]>,xscale:&HbmTensor<f32,Chip,m![P256]>,
    weight:&HbmTensor<f8e4m3,Chip,DownWeight>,weight_scale:&HbmTensor<f32,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![P256,H]>) {
    let input:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![P256 % 128,L % 960]>=input.to_dm(&mut device.tdma);
    let xscale:DmTensor<f32,Chip,Cluster,DownRows,m![P256 % 128]>=xscale.to_dm(&mut device.tdma);
    let result=matrix_down(device,&input,&xscale,weight,weight_scale);
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![P256 % 128,H % 240]>=device.main.begin(result.view())
        .fetch::<m![P256 / 16 % 8,H % 240],m![P256 % 16]>().collect::<m![P256 / 16 % 8,H % 240],m![P256 % 16]>()
        .transpose::<m![P256 / 16 % 8,H / 4 % 60,P256 % 16],m![H % 4 # 16]>().commit_trim::<m![H % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
#[device(chip = 1)]
pub fn batch_ffn(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P256,H]>,
    up:&HbmTensor<f8e4m3,Chip,UpWeight>,gate:&HbmTensor<f8e4m3,Chip,UpWeight>,down:&HbmTensor<f8e4m3,Chip,DownWeight>,
    up_dequant:&HbmTensor<f32,Chip,m![L]>,gate_dequant:&HbmTensor<f32,Chip,m![L]>,down_dequant:&HbmTensor<f32,Chip,m![H]>,
    up_global:&HbmTensor<f32,Chip,m![1]>,gate_global:&HbmTensor<f32,Chip,m![1]>,down_global:&HbmTensor<f32,Chip,m![1]>,
    output:&mut HbmTensor<bf16,Chip,m![P256,H]>) {
    let input:DmTensor<bf16,Chip,Cluster,QuantRows,m![H]>=input.to_dm(&mut device.tdma);
    let (input,xscale)=quantize_hidden(device,&input);
    let input:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![P256 % 128,H % 960]>=input.to_dm(&mut device.tdma);
    let xscale:DmTensor<f32,Chip,Cluster,UpRows,m![P256 % 128,1 # 8]>=xscale.to_dm(&mut device.tdma);
    let xscale=pack_up_scales(device,&xscale);
    let up=matrix_up(device,&input,&xscale,up,up_dequant);
    let gate=matrix_up(device,&input,&xscale,gate,gate_dequant);
    let hidden=crate::ops_geglu_direct_p256::geglu(device,&up,&gate,up_global,gate_global);
    let hidden:DmTensor<bf16,Chip,Cluster,UpRows,m![P256 % 128,L % 240]>=device.main.begin(hidden.view())
        .fetch::<m![P256 / 16 % 8,L % 240],m![P256 % 16]>().collect::<m![P256 / 16 % 8,L % 240],m![P256 % 16]>()
        .transpose::<m![P256 / 16 % 8,L / 4 % 60,P256 % 16],m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit();
    let hidden:DmTensor<bf16,Chip,Cluster,QuantRows,m![L]>=hidden.to_dm(&mut device.tdma);
    let (hidden,hscale)=quantize_ffn(device,&hidden);
    let hidden:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![P256 % 128,L % 960]>=hidden.to_dm(&mut device.tdma);
    let hscale:DmTensor<f32,Chip,Cluster,DownRows,m![P256 % 128,1 # 8]>=hscale.to_dm(&mut device.tdma);
    let hscale=pack_down_scales(device,&hscale);
    let result=matrix_down(device,&hidden,&hscale,down,down_dequant);
    let global:DmTensor<f32,Chip,Cluster,DownRows,m![1 # 8]>=down_global.to_dm(&mut device.tdma);
    let global:VrfTensor<f32,Chip,Cluster,DownRows,m![1 # 8]>=device.sub.begin(global.view())
        .fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>().to_vrf();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 240,P256 % 128]>=device.main.begin(result.view())
        .fetch::<m![H % 240,P256 / 8 % 16],m![P256 % 8]>().fetch_cast::<f32>()
        .collect::<m![H % 240,P256 / 8 % 16],m![P256 % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 240,P256 / 4 % 32],m![P256 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&global)
        .vector_widen_concat::<m![H % 240,P256 / 8 % 16],m![P256 % 8]>()
        .vector_final().cast::<bf16,m![P256 % 8 # 16]>().commit_trim::<m![P256 % 8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![P256 % 128,H % 240]>=device.main.begin(result.view())
        .fetch::<m![P256 / 16 % 8,H % 240],m![P256 % 16]>().collect::<m![P256 / 16 % 8,H % 240],m![P256 % 16]>()
        .transpose::<m![P256 / 16 % 8,H / 4 % 60,P256 % 16],m![H % 4 # 16]>().commit_trim::<m![H % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

