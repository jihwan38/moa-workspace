//! Full B16 FFN with startup-prepacked FP8 weights and per-token dynamic FP8 activations.
//! Dot F32*x_dequant_scale*weight_dequant_scale rounds to BF16 before original globals.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{B,H,L}};
type Cluster=m![B/8];
type QuantRows=m![B % 8 # 256];
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

fn pack_up_scales(device:&mut Device,xscale:&DmTensor<f32,Chip,Cluster,UpRows,m![B % 8,1 # 8]>)
    ->DmTensor<f32,Chip,Cluster,UpRows,m![B % 8]> {
    // Keep the 32-byte scalar slot for aligned inter-slice DMA, then pack by TU.
    let xscale:DmTensor<f32,Chip,Cluster,UpRows,m![B % 8]>=device.sub.begin(xscale.view())
        .fetch::<m![B % 8],m![1 # 8]>().collect::<m![B % 8],m![1 # 8]>()
        .transpose::<m![B % 8 / 2],m![B % 8 % 2 # 8]>().commit_trim::<m![B % 8 % 2]>().commit();
    xscale
}
fn matrix_up(device:&mut Device,input:&DmTensor<f8e4m3,Chip,Cluster,UpParts,m![B % 8,H % 960]>,
    xscale:&DmTensor<f32,Chip,Cluster,UpRows,m![B % 8]>,weight:&HbmTensor<f8e4m3,Chip,m![L,H]>,
    weight_dequant_scale:&HbmTensor<f32,Chip,m![L]>) ->DmTensor<bf16,Chip,Cluster,UpRows,m![L % 240,B % 8]> {
    let weight:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![L % 240,H % 960]>=weight.to_dm(&mut device.tdma);
    let ws:DmTensor<f32,Chip,Cluster,UpRows,m![L % 240]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,Cluster,UpRows,m![L % 240]>=device.sub.begin(ws.view())
        .fetch::<m![L / 8 % 30],m![L % 8]>().collect::<m![L / 8 % 30],m![L % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,UpRows,m![L % 240,B % 8]>=DmTensor::new();
    {
        let xs:VrfTensor<f32,Chip,Cluster,UpRows,m![B % 8 = 8]>=device.sub
            .begin(xscale.view().tile::<m![B % 8],8,m![B % 8 = 8 # 8]>(0))
            .fetch::<m![1],m![B % 8 = 8]>().collect::<m![1],m![B % 8 = 8]>().to_vrf();
        let x:TrfTensor<f8e4m3,Chip,Cluster,UpParts,m![B % 8 = 8],m![H % 960]>=device.sub
            .begin(input.view().tile::<m![B % 8],8,m![B % 8 = 8 # 8,H % 960]>(0))
            .fetch::<m![B % 8 = 8,H / 32 % 30],m![H % 32]>()
            .collect::<m![B % 8 = 8,H / 32 % 30],m![H % 32]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![L % 240,H / 32 % 30],m![H % 32]>()
            .collect::<m![L % 240,H / 32 % 30],m![H % 32]>()
            .contract_outer::<m![L % 240,H / 64 % 15],m![H % 64],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![L % 240]>()
            .contract_lane::<m![L % 240],m![B % 8 = 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<UpRows,m![L % 240]>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 240,B % 8 = 8 / 4],m![B % 8 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
            .vector_widen_concat::<m![L % 240],m![B % 8 = 8]>()
            .vector_final().cast::<bf16,m![B % 8 = 8 # 16]>().commit_trim::<m![B % 8 = 8]>()
            .commit_view(result.view_mut().tile::<m![B % 8],8,m![L % 240,B % 8 = 8 #{!} 8]>(0));
    }
    result
}

fn pack_down_scales(device:&mut Device,xscale:&DmTensor<f32,Chip,Cluster,DownRows,m![B % 8,1 # 8]>)
    ->DmTensor<f32,Chip,Cluster,DownRows,m![B % 8]> {
    // Keep the 32-byte scalar slot for aligned inter-slice DMA, then pack by TU.
    let xscale:DmTensor<f32,Chip,Cluster,DownRows,m![B % 8]>=device.sub.begin(xscale.view())
        .fetch::<m![B % 8],m![1 # 8]>().collect::<m![B % 8],m![1 # 8]>()
        .transpose::<m![B % 8 / 2],m![B % 8 % 2 # 8]>().commit_trim::<m![B % 8 % 2]>().commit();
    xscale
}
fn matrix_down(device:&mut Device,input:&DmTensor<f8e4m3,Chip,Cluster,DownParts,m![B % 8,L % 960]>,
    xscale:&DmTensor<f32,Chip,Cluster,DownRows,m![B % 8]>,weight:&HbmTensor<f8e4m3,Chip,m![H,L]>,
    weight_dequant_scale:&HbmTensor<f32,Chip,m![H]>) ->DmTensor<bf16,Chip,Cluster,DownRows,m![H % 240,B % 8]> {
    let weight:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![H % 240,L % 960]>=weight.to_dm(&mut device.tdma);
    let ws:DmTensor<f32,Chip,Cluster,DownRows,m![H % 240]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 240]>=device.sub.begin(ws.view())
        .fetch::<m![H / 8 % 30],m![H % 8]>().collect::<m![H / 8 % 30],m![H % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 240,B % 8]>=DmTensor::new();
    {
        let xs:VrfTensor<f32,Chip,Cluster,DownRows,m![B % 8 = 8]>=device.sub
            .begin(xscale.view().tile::<m![B % 8],8,m![B % 8 = 8 # 8]>(0))
            .fetch::<m![1],m![B % 8 = 8]>().collect::<m![1],m![B % 8 = 8]>().to_vrf();
        let x:TrfTensor<f8e4m3,Chip,Cluster,DownParts,m![B % 8 = 8],m![L % 960]>=device.sub
            .begin(input.view().tile::<m![B % 8],8,m![B % 8 = 8 # 8,L % 960]>(0))
            .fetch::<m![B % 8 = 8,L / 32 % 30],m![L % 32]>()
            .collect::<m![B % 8 = 8,L / 32 % 30],m![L % 32]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H % 240,L / 32 % 30],m![L % 32]>()
            .collect::<m![H % 240,L / 32 % 30],m![L % 32]>()
            .contract_outer::<m![H % 240,L / 64 % 15],m![L % 64],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H % 240]>()
            .contract_lane::<m![H % 240],m![B % 8 = 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<DownRows,m![H % 240]>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 240,B % 8 = 8 / 4],m![B % 8 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
            .vector_widen_concat::<m![H % 240],m![B % 8 = 8]>()
            .vector_final().cast::<bf16,m![B % 8 = 8 # 16]>().commit_trim::<m![B % 8 = 8]>()
            .commit_view(result.view_mut().tile::<m![B % 8],8,m![H % 240,B % 8 = 8 #{!} 8]>(0));
    }
    result
}

#[device(chip = 1)]
pub fn project_up_scaled(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![B,H]>,xscale:&HbmTensor<f32,Chip,m![B]>,
    weight:&HbmTensor<f8e4m3,Chip,m![L,H]>,weight_scale:&HbmTensor<f32,Chip,m![L]>,output:&mut HbmTensor<bf16,Chip,m![B,L]>) {
    let input:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![B % 8,H % 960]>=input.to_dm(&mut device.tdma);
    let xscale:DmTensor<f32,Chip,Cluster,UpRows,m![B % 8]>=xscale.to_dm(&mut device.tdma);
    let result=matrix_up(device,&input,&xscale,weight,weight_scale);
    let result:DmTensor<bf16,Chip,Cluster,UpRows,m![B % 8,L % 240]>=device.main.begin(result.view())
        .fetch::<m![L % 240],m![B % 8 # 16]>().collect::<m![L % 240],m![B % 8 # 16]>()
        .transpose::<m![L / 4 % 60,B % 8],m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
#[device(chip = 1)]
pub fn project_down_scaled(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![B,L]>,xscale:&HbmTensor<f32,Chip,m![B]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,L]>,weight_scale:&HbmTensor<f32,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![B,H]>) {
    let input:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![B % 8,L % 960]>=input.to_dm(&mut device.tdma);
    let xscale:DmTensor<f32,Chip,Cluster,DownRows,m![B % 8]>=xscale.to_dm(&mut device.tdma);
    let result=matrix_down(device,&input,&xscale,weight,weight_scale);
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![B % 8,H % 240]>=device.main.begin(result.view())
        .fetch::<m![H % 240],m![B % 8 # 16]>().collect::<m![H % 240],m![B % 8 # 16]>()
        .transpose::<m![H / 4 % 60,B % 8],m![H % 4 # 16]>().commit_trim::<m![H % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
#[device(chip = 1)]
pub fn batch_ffn(device:&mut Device,input:&HbmTensor<bf16,Chip,m![B,H]>,
    up:&HbmTensor<f8e4m3,Chip,m![L,H]>,gate:&HbmTensor<f8e4m3,Chip,m![L,H]>,down:&HbmTensor<f8e4m3,Chip,m![H,L]>,
    up_dequant:&HbmTensor<f32,Chip,m![L]>,gate_dequant:&HbmTensor<f32,Chip,m![L]>,down_dequant:&HbmTensor<f32,Chip,m![H]>,
    up_global:&HbmTensor<f32,Chip,m![1]>,gate_global:&HbmTensor<f32,Chip,m![1]>,down_global:&HbmTensor<f32,Chip,m![1]>,
    output:&mut HbmTensor<bf16,Chip,m![B,H]>) {
    let input:DmTensor<bf16,Chip,Cluster,QuantRows,m![H]>=input.to_dm(&mut device.tdma);
    let (input,xscale)=quantize_hidden(device,&input);
    let input:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![B % 8,H % 960]>=input.to_dm(&mut device.tdma);
    let xscale:DmTensor<f32,Chip,Cluster,UpRows,m![B % 8,1 # 8]>=xscale.to_dm(&mut device.tdma);
    let xscale=pack_up_scales(device,&xscale);
    let up=matrix_up(device,&input,&xscale,up,up_dequant);
    let gate=matrix_up(device,&input,&xscale,gate,gate_dequant);
    let hidden=crate::ops_batch_kpart16::geglu_k4(device,&up,&gate,up_global,gate_global);
    let hidden:DmTensor<bf16,Chip,Cluster,UpRows,m![B % 8,L % 240]>=device.main.begin(hidden.view())
        .fetch::<m![L % 240],m![B % 8 # 16]>().collect::<m![L % 240],m![B % 8 # 16]>()
        .transpose::<m![L / 4 % 60,B % 8],m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit();
    let hidden:DmTensor<bf16,Chip,Cluster,QuantRows,m![L]>=hidden.to_dm(&mut device.tdma);
    let (hidden,hscale)=quantize_ffn(device,&hidden);
    let hidden:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![B % 8,L % 960]>=hidden.to_dm(&mut device.tdma);
    let hscale:DmTensor<f32,Chip,Cluster,DownRows,m![B % 8,1 # 8]>=hscale.to_dm(&mut device.tdma);
    let hscale=pack_down_scales(device,&hscale);
    let result=matrix_down(device,&hidden,&hscale,down,down_dequant);
    let global:DmTensor<f32,Chip,Cluster,DownRows,m![1 # 8]>=down_global.to_dm(&mut device.tdma);
    let global:VrfTensor<f32,Chip,Cluster,DownRows,m![1 # 8]>=device.sub.begin(global.view())
        .fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>().to_vrf();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 240,B % 8]>=device.main.begin(result.view())
        .fetch::<m![H % 240,B / 8 % 1],m![B % 8]>().fetch_cast::<f32>()
        .collect::<m![H % 240,B / 8 % 1],m![B % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 240,B / 4 % 2],m![B % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&global)
        .vector_widen_concat::<m![H % 240,B / 8 % 1],m![B % 8]>()
        .vector_final().cast::<bf16,m![B % 8 # 16]>().commit_trim::<m![B % 8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![B % 8,H % 240]>=device.main.begin(result.view())
        .fetch::<m![H % 240],m![B % 8 # 16]>().collect::<m![H % 240],m![B % 8 # 16]>()
        .transpose::<m![H / 4 % 60,B % 8],m![H % 4 # 16]>().commit_trim::<m![H % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
