//! P512 FFN with physically blocked FP8 inputs and BF16 up/gate outputs; arithmetic unchanged.
//! Uses exactly the established quantizer, dot BF16 boundary and global scales.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P512,H,L}};
use crate::ops_batch_layouts::{P512H960,P512L960,P512L240};
pub use crate::ops_native_fp8_blocked::{UpWeight,DownWeight};
type Cluster=m![P512/256];
type QuantRows=m![P512%256];
type UpParts=m![L/240,H/960];
type UpRows=m![L/240,1 #4];
type DownParts=m![H/240,L/960];
type DownRows=m![H/240,1 #16];
fn quantize_hidden_dm(device:&mut Device,x:&DmTensor<bf16,Chip,Cluster,QuantRows,m![H]>)
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

fn quantize_ffn_dm(device:&mut Device,x:&DmTensor<bf16,Chip,Cluster,QuantRows,m![L]>)
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


#[device(chip=1)]
pub fn quantize_hidden(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P512,H]>,
    output:&mut HbmTensor<f8e4m3,Chip,P512H960>,scales:&mut HbmTensor<f32,Chip,m![P512]>) {
    let input:DmTensor<bf16,Chip,Cluster,QuantRows,m![H]>=input.to_dm(&mut device.tdma);
    let (q,scale)=quantize_hidden_dm(device,&input);
    q.view().to_hbm_view(&mut device.tdma,output.view_mut());
    scale.view().to_hbm_view(&mut device.tdma,scales.view_mut());
}

#[device(chip=1)]
pub fn project_up_scaled(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512H960>,
 xscale:&HbmTensor<f32,Chip,m![P512]>,weight:&HbmTensor<f8e4m3,Chip,UpWeight>,
 weight_dequant_scale:&HbmTensor<f32,Chip,m![L]>,output:&mut HbmTensor<bf16,Chip,P512L240>) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![L % 240,H % 960]>=weight.to_dm(&mut device.tdma);
    let ws:DmTensor<f32,Chip,Cluster,UpRows,m![L % 240]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,Cluster,UpRows,m![L % 240]>=device.sub.begin(ws.view())
        .fetch::<m![L / 8 % 30],m![L % 8]>().collect::<m![L / 8 % 30],m![L % 8]>().to_vrf();
    for chunk in 0..4 {
        let input:DmTensor<f8e4m3,Chip,Cluster,UpParts,m![P512%256=64,H%960]>=input.view()
            .tile::<m![P512%256],64,m![P512/256,H/960,P512%256=64#256,H%960]>(64*chunk).to_dm(&mut device.tdma);
        let xscale:DmTensor<f32,Chip,Cluster,UpRows,m![P512%256=64]>=xscale.view()
            .tile::<m![P512%256],64,m![P512/256,P512%256=64#256]>(64*chunk).to_dm(&mut device.tdma);
    let xs:VrfTensor<f32,Chip,Cluster,UpRows,m![P512%256=64]>=device.sub.begin(xscale.view())
        .fetch::<m![P512%256=64/8],m![P512%256=64%8]>().collect::<m![P512%256=64/8],m![P512%256=64%8]>().to_vrf();
    let x:TrfTensor<f8e4m3,Chip,Cluster,UpParts,m![P512%256=64%8],m![P512%256=64/8,H%960]>=device.sub.begin(input.view())
        .fetch::<m![P512%256=64%8,P512%256=64/8,H/32%30],m![H%32]>()
        .collect::<m![P512%256=64%8,P512%256=64/8,H/32%30],m![H%32]>().to_trf();
    let result:DmTensor<bf16,Chip,Cluster,UpRows,m![L%240,P512%256=64]>=device.main.begin(weight.view())
        .fetch::<m![L%240,H/32%30],m![H%32]>().collect::<m![L%240,H/32%30],m![H%32]>()
        .contract_outer::<m![L%240,H/64%15,P512%256=64/8],m![H%64],_,_,_>(&x)
        .contract_packet::<m![1]>().contract_time::<m![L%240,P512%256=64/8]>()
        .contract_lane::<m![L%240,P512%256=64/8],m![P512%256=64%8]>(LaneMode::Interleaved)
        .vector_init().vector_inter_slice_reduce::<UpRows,m![L%240,P512%256=64/8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L%240,P512%256=64/4],m![P512%256=64%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![L%240,P512%256=64/8],m![P512%256=64%8]>()
        .vector_final().cast::<bf16,m![P512%256=64%8#16]>().commit_trim::<m![P512%256=64%8]>().commit();
        let result:DmTensor<bf16,Chip,Cluster,UpRows,m![P512%256=64,L%240]>=device.main.begin(result.view())
            .fetch::<m![P512%256=64/16,L%240],m![P512%256=64%16]>().collect::<m![P512%256=64/16,L%240],m![P512%256=64%16]>()
            .transpose::<m![P512%256=64/16,L/4%60,P512%256=64%16],m![L%4#16]>().commit_trim::<m![L%4]>().commit();
        result.view().to_hbm_view(&mut device.tdma,output.view_mut()
            .tile::<m![P512%256],64,m![P512/256,L/240,P512%256=64#{!}256,L%240]>(64*chunk));
    }
}

#[device(chip=1)]
pub fn project_down_scaled(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512L960>,
 xscale:&HbmTensor<f32,Chip,m![P512]>,weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
 weight_dequant_scale:&HbmTensor<f32,Chip,m![H]>,global_scale:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,m![P512,H]>) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![H % 240,L % 960]>=weight.to_dm(&mut device.tdma);
    let ws:DmTensor<f32,Chip,Cluster,DownRows,m![H % 240]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 240]>=device.sub.begin(ws.view())
        .fetch::<m![H / 8 % 30],m![H % 8]>().collect::<m![H / 8 % 30],m![H % 8]>().to_vrf();
    let global:DmTensor<f32,Chip,Cluster,DownRows,m![1#8]>=global_scale.to_dm(&mut device.tdma);
    let global:VrfTensor<f32,Chip,Cluster,DownRows,m![1#8]>=device.sub.begin(global.view())
        .fetch::<m![1],m![1#8]>().collect::<m![1],m![1#8]>().to_vrf();
    for chunk in 0..4 {
        let input:DmTensor<f8e4m3,Chip,Cluster,DownParts,m![P512%256=64,L%960]>=input.view()
            .tile::<m![P512%256],64,m![P512/256,L/960,P512%256=64#256,L%960]>(64*chunk).to_dm(&mut device.tdma);
        let xscale:DmTensor<f32,Chip,Cluster,DownRows,m![P512%256=64]>=xscale.view()
            .tile::<m![P512%256],64,m![P512/256,P512%256=64#256]>(64*chunk).to_dm(&mut device.tdma);
    let xs:VrfTensor<f32,Chip,Cluster,DownRows,m![P512%256=64]>=device.sub.begin(xscale.view())
        .fetch::<m![P512%256=64/8],m![P512%256=64%8]>().collect::<m![P512%256=64/8],m![P512%256=64%8]>().to_vrf();
    let x:TrfTensor<f8e4m3,Chip,Cluster,DownParts,m![P512%256=64%8],m![P512%256=64/8,L%960]>=device.sub.begin(input.view())
        .fetch::<m![P512%256=64%8,P512%256=64/8,L/32%30],m![L%32]>()
        .collect::<m![P512%256=64%8,P512%256=64/8,L/32%30],m![L%32]>().to_trf();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H%240,P512%256=64]>=device.main.begin(weight.view())
        .fetch::<m![H%240,L/32%30],m![L%32]>().collect::<m![H%240,L/32%30],m![L%32]>()
        .contract_outer::<m![H%240,L/64%15,P512%256=64/8],m![L%64],_,_,_>(&x)
        .contract_packet::<m![1]>().contract_time::<m![H%240,P512%256=64/8]>()
        .contract_lane::<m![H%240,P512%256=64/8],m![P512%256=64%8]>(LaneMode::Interleaved)
        .vector_init().vector_inter_slice_reduce::<DownRows,m![H%240,P512%256=64/8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H%240,P512%256=64/4],m![P512%256=64%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![H%240,P512%256=64/8],m![P512%256=64%8]>()
        .vector_final().cast::<bf16,m![P512%256=64%8#16]>().commit_trim::<m![P512%256=64%8]>().commit();
        let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H%240,P512%256=64]>=device.main.begin(result.view())
            .fetch::<m![H%240,P512%256=64/8],m![P512%256=64%8]>().fetch_cast::<f32>()
            .collect::<m![H%240,P512%256=64/8],m![P512%256=64%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H%240,P512%256=64/4],m![P512%256=64%4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&global)
            .vector_widen_concat::<m![H%240,P512%256=64/8],m![P512%256=64%8]>()
            .vector_final().cast::<bf16,m![P512%256=64%8#16]>().commit_trim::<m![P512%256=64%8]>().commit();
        let result:DmTensor<bf16,Chip,Cluster,DownRows,m![P512%256=64,H%240]>=device.main.begin(result.view())
            .fetch::<m![P512%256=64/16,H%240],m![P512%256=64%16]>().collect::<m![P512%256=64/16,H%240],m![P512%256=64%16]>()
            .transpose::<m![P512%256=64/16,H/4%60,P512%256=64%16],m![H%4#16]>().commit_trim::<m![H%4]>().commit();
        result.view().to_hbm_view(&mut device.tdma,output.view_mut()
            .tile::<m![P512%256],64,m![P512/256,P512%256=64#{!}256,H]>(64*chunk));
    }
}

fn scale_tile(
    device: &mut Device,
    input: DmTensorView<'_,bf16,Chip,Cluster,QuantRows,m![L=1920#15360]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, QuantRows, m![L=1920]> {
    let scale: DmTensor<f32, Chip, Cluster, QuantRows, m![1 # 8]> =
        global_scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, QuantRows, m![1 # 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    device
        .main
        .begin(input)
        .fetch::<m![1], m![L=1920]>()
        .fetch_cast::<f32>()
        .collect::<m![L=1920/8], m![L=1920%8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L=1920/4], m![L=1920%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![L=1920/8], m![L=1920%8]>()
        .vector_final()
        .cast::<bf16, m![L=1920%8#16]>()
        .commit_trim::<m![L=1920%8]>()
        .commit()
}
fn geglu_tile_into(
    device: &mut Device,
    up: DmTensorView<'_,bf16,Chip,Cluster,QuantRows,m![L=1920#15360]>,
    gate: DmTensorView<'_,bf16,Chip,Cluster,QuantRows,m![L=1920#15360]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    output:DmTensorViewMut<'_,bf16,Chip,Cluster,QuantRows,m![L=1920#{!}15360]>,
) {
    let up = scale_tile(device, up, up_global_scale);
    let gate = scale_tile(device, gate, gate_global_scale);
    let gelu: DmTensor<f32, Chip, Cluster, QuantRows, m![L=1920]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![L=1920/8], m![L=1920%8]>()
        .fetch_cast::<f32>()
        .collect::<m![L=1920/8], m![L=1920%8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L=1920/4], m![L=1920%4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.70710678118f32)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L=1920/8], m![L=1920%8]>()
        .vector_final()
        .commit_trim::<m![L=1920%8]>()
        .commit();
    let gelu: VrfTensor<f32, Chip, Cluster, QuantRows, m![L=1920]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![L=1920/8], m![L=1920%8]>()
        .collect::<m![L=1920/8], m![L=1920%8]>()
        .to_vrf();
    device
        .main
        .begin(up.view())
        .fetch::<m![L=1920/8], m![L=1920%8]>()
        .fetch_cast::<f32>()
        .collect::<m![L=1920/8], m![L=1920%8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L=1920/4], m![L=1920%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L=1920/8], m![L=1920%8]>()
        .vector_final()
        .cast::<bf16, m![L=1920%8#16]>()
        .commit_trim::<m![L=1920%8]>()
        .commit_view(output);
}

#[device(chip=1)]
pub fn geglu_quantize(device:&mut Device,up:&HbmTensor<bf16,Chip,P512L240>,
    gate:&HbmTensor<bf16,Chip,P512L240>,up_global:&HbmTensor<f32,Chip,m![1]>,gate_global:&HbmTensor<f32,Chip,m![1]>,
    output:&mut HbmTensor<f8e4m3,Chip,P512L960>,scales:&mut HbmTensor<f32,Chip,m![P512]>) {
    let up:DmTensor<bf16,Chip,Cluster,QuantRows,m![L]>=up.to_dm(&mut device.tdma);
    let gate:DmTensor<bf16,Chip,Cluster,QuantRows,m![L]>=gate.to_dm(&mut device.tdma);
    let mut hidden:DmTensor<bf16,Chip,Cluster,QuantRows,m![L]>=DmTensor::new();
    for tile in 0..8 {
        geglu_tile_into(device,
            up.view().tile::<m![L],1920,m![L=1920#15360]>(1920*tile),
            gate.view().tile::<m![L],1920,m![L=1920#15360]>(1920*tile),up_global,gate_global,
            hidden.view_mut().tile::<m![L],1920,m![L=1920#{!}15360]>(1920*tile));
    }
    let (q,scale)=quantize_ffn_dm(device,&hidden);
    q.view().to_hbm_view(&mut device.tdma,output.view_mut());
    scale.view().to_hbm_view(&mut device.tdma,scales.view_mut());
}

#[device(chip=1)]
pub fn quantize_ffn(device:&mut Device,input:&HbmTensor<bf16,Chip,P512L240>,
    output:&mut HbmTensor<f8e4m3,Chip,P512L960>,scales:&mut HbmTensor<f32,Chip,m![P512]>) {
    let input:DmTensor<bf16,Chip,Cluster,QuantRows,m![L]>=input.to_dm(&mut device.tdma);
    let (q,scale)=quantize_ffn_dm(device,&input);
    q.view().to_hbm_view(&mut device.tdma,output.view_mut());
    scale.view().to_hbm_view(&mut device.tdma,scales.view_mut());
}
