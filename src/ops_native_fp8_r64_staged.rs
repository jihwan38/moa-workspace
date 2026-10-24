//! R64 FFN with HBM activation exchange and exact row quantization.
//! Four kernels, shared output-cluster blocked weights, established BF16 boundaries.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{R64,H,L}};
pub use crate::ops_native_fp8_blocked::{UpWeight,DownWeight};
type Cluster=m![R64/32];
type QuantRows=m![R64%32#256];
type UpCluster=m![L/7680];
type DownCluster=m![H/1920];
type UpParts=m![L/120%64,H/960];
type UpRows=m![L/120%64,1#4];
type DownParts=m![H/120%16,L/960];
type DownRows=m![H/120%16,1#16];

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
pub fn quantize_hidden(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,H]>,
    output:&mut HbmTensor<f8e4m3,Chip,m![R64,H]>,scales:&mut HbmTensor<f32,Chip,m![R64]>) {
    let input:DmTensor<bf16,Chip,Cluster,QuantRows,m![H]>=input.to_dm(&mut device.tdma);
    let (q,scale)=quantize_hidden_dm(device,&input);
    q.view().to_hbm_view(&mut device.tdma,output.view_mut());
    scale.view().to_hbm_view(&mut device.tdma,scales.view_mut());
}

#[device(chip=1)]
pub fn quantize_ffn(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,L]>,
    output:&mut HbmTensor<f8e4m3,Chip,m![R64,L]>,scales:&mut HbmTensor<f32,Chip,m![R64]>) {
    let input:DmTensor<bf16,Chip,Cluster,QuantRows,m![L]>=input.to_dm(&mut device.tdma);
    let (q,scale)=quantize_ffn_dm(device,&input);
    q.view().to_hbm_view(&mut device.tdma,output.view_mut());
    scale.view().to_hbm_view(&mut device.tdma,scales.view_mut());
}
fn matrix_tile(device:&mut Device,
    input:&TrfTensor<f8e4m3,Chip,UpCluster,UpParts,m![R64%8],m![R64/8,H%960]>,
    weight:&DmTensor<f8e4m3,Chip,UpCluster,UpParts,m![L%120,H%960]>,
    xs:&DmTensor<f32,Chip,UpCluster,UpRows,m![R64]>,
    ws:&DmTensor<f32,Chip,UpCluster,UpRows,m![L%120]>)
    ->DmTensor<bf16,Chip,UpCluster,UpRows,m![L%120,R64]> {
    let xs:VrfTensor<f32,Chip,UpCluster,UpRows,m![R64]>=device.sub.begin(xs.view())
        .fetch::<m![R64/8],m![R64%8]>().collect::<m![R64/8],m![R64%8]>().to_vrf();
    let ws:VrfTensor<f32,Chip,UpCluster,UpRows,m![L%120]>=device.sub.begin(ws.view())
        .fetch::<m![L/8%15],m![L%8]>().collect::<m![L/8%15],m![L%8]>().to_vrf();
    device.main.begin(weight.view())
        .fetch::<m![L%120,H/32%30],m![H%32]>().collect::<m![L%120,H/32%30],m![H%32]>()
        .contract_outer::<m![L%120,H/64%15,R64/8],m![H%64],_,_,_>(input)
        .contract_packet::<m![1]>().contract_time::<m![L%120,R64/8]>()
        .contract_lane::<m![L%120,R64/8],m![R64%8]>(LaneMode::Interleaved)
        .vector_init().vector_inter_slice_reduce::<UpRows,m![L%120,R64/8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L%120,R64/4],m![R64%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![L%120,R64/8],m![R64%8]>()
        .vector_final().cast::<bf16,m![R64%8#16]>().commit_trim::<m![R64%8]>().commit()
}

fn scale_tile(
    device: &mut Device,
    input: DmTensorView<'_,bf16,Chip,UpCluster,UpRows,m![L % 120,R64=8 # 64]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, UpCluster, UpRows, m![L % 120, R64=8]> {
    let scale: DmTensor<f32, Chip, UpCluster, UpRows, m![1 # 8]> =
        global_scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, UpCluster, UpRows, m![1 # 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    device
        .main
        .begin(input)
        .fetch::<m![L % 120], m![R64=8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 120, R64=8 / 8], m![R64=8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 120, R64=8 / 4], m![R64=8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![L % 120, R64=8 / 8], m![R64=8 % 8]>()
        .vector_final()
        .cast::<bf16, m![R64=8 % 8 # 16]>()
        .commit_trim::<m![R64=8 % 8]>()
        .commit()
}
fn geglu_tile_into(
    device: &mut Device,
    up: DmTensorView<'_,bf16,Chip,UpCluster,UpRows,m![L % 120,R64=8 # 64]>,
    gate: DmTensorView<'_,bf16,Chip,UpCluster,UpRows,m![L % 120,R64=8 # 64]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    output:DmTensorViewMut<'_,bf16,Chip,UpCluster,UpRows,m![L % 120,R64=8 #{!} 64]>,
) {
    let up = scale_tile(device, up, up_global_scale);
    let gate = scale_tile(device, gate, gate_global_scale);
    let gelu: DmTensor<f32, Chip, UpCluster, UpRows, m![L % 120, R64=8]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![L % 120, R64=8 / 8], m![R64=8 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 120, R64=8 / 8], m![R64=8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 120, R64=8 / 4], m![R64=8 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.70710678118f32)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L % 120, R64=8 / 8], m![R64=8 % 8]>()
        .vector_final()
        .commit_trim::<m![R64=8 % 8]>()
        .commit();
    let gelu: VrfTensor<f32, Chip, UpCluster, UpRows, m![L % 120, R64=8]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![L % 120, R64=8 / 8], m![R64=8 % 8]>()
        .collect::<m![L % 120, R64=8 / 8], m![R64=8 % 8]>()
        .to_vrf();
    device
        .main
        .begin(up.view())
        .fetch::<m![L % 120, R64=8 / 8], m![R64=8 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 120, R64=8 / 8], m![R64=8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 120, R64=8 / 4], m![R64=8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L % 120, R64=8 / 8], m![R64=8 % 8]>()
        .vector_final()
        .cast::<bf16, m![R64=8 % 8 # 16]>()
        .commit_trim::<m![R64=8 % 8]>()
        .commit_view(output);
}

fn geglu(device:&mut Device,up:&DmTensor<bf16,Chip,UpCluster,UpRows,m![L % 120,R64]>,
    gate:&DmTensor<bf16,Chip,UpCluster,UpRows,m![L % 120,R64]>,
    us:&HbmTensor<f32,Chip,m![1]>,gs:&HbmTensor<f32,Chip,m![1]>)
    ->DmTensor<bf16,Chip,UpCluster,UpRows,m![L % 120,R64]> {
    let mut result:DmTensor<bf16,Chip,UpCluster,UpRows,m![L % 120,R64]>=DmTensor::new();
    for chunk in 0..8 {
        geglu_tile_into(device,
            up.view().tile::<m![R64],8,m![L % 120,R64=8 # 64]>(8*chunk),
            gate.view().tile::<m![R64],8,m![L % 120,R64=8 # 64]>(8*chunk),us,gs,
            result.view_mut().tile::<m![R64],8,m![L % 120,R64=8 #{!} 64]>(8*chunk));
    }
    result
}


#[device(chip=1)]
pub fn upgate_geglu_prequantized(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![R64,H]>, input_scale:&HbmTensor<f32,Chip,m![R64]>,
    up:&HbmTensor<f8e4m3,Chip,UpWeight>,gate:&HbmTensor<f8e4m3,Chip,UpWeight>,
    up_dequant:&HbmTensor<f32,Chip,m![L]>,gate_dequant:&HbmTensor<f32,Chip,m![L]>,
    up_global:&HbmTensor<f32,Chip,m![1]>,gate_global:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,m![R64,L]>) {
    let input:DmTensor<f8e4m3,Chip,UpCluster,UpParts,m![R64,H%960]>=input.to_dm(&mut device.tdma);
    let xscale:DmTensor<f32,Chip,UpCluster,UpRows,m![R64]>=input_scale.to_dm(&mut device.tdma);
    let up:DmTensor<f8e4m3,Chip,UpCluster,UpParts,m![L%120,H%960]>=up.to_dm(&mut device.tdma);
    let gate:DmTensor<f8e4m3,Chip,UpCluster,UpParts,m![L%120,H%960]>=gate.to_dm(&mut device.tdma);
    let us:DmTensor<f32,Chip,UpCluster,UpRows,m![L%120]>=up_dequant.to_dm(&mut device.tdma);
    let gs:DmTensor<f32,Chip,UpCluster,UpRows,m![L%120]>=gate_dequant.to_dm(&mut device.tdma);
    let x:TrfTensor<f8e4m3,Chip,UpCluster,UpParts,m![R64%8],m![R64/8,H%960]>=device.sub.begin(input.view())
        .fetch::<m![R64%8,R64/8,H/32%30],m![H%32]>()
        .collect::<m![R64%8,R64/8,H/32%30],m![H%32]>().to_trf();
    let a=matrix_tile(device,&x,&up,&xscale,&us);
    let b=matrix_tile(device,&x,&gate,&xscale,&gs);
    let hidden=geglu(device,&a,&b,up_global,gate_global);
    let hidden:DmTensor<bf16,Chip,UpCluster,UpRows,m![R64,L%120]>=device.main.begin(hidden.view())
        .fetch::<m![R64/16,L%120],m![R64%16]>().collect::<m![R64/16,L%120],m![R64%16]>()
        .transpose::<m![R64/16,L/4%30,R64%16],m![L%4#16]>().commit_trim::<m![L%4]>().commit();
    hidden.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
#[device(chip=1)]
pub fn down_prequantized(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![R64,L]>, input_scale:&HbmTensor<f32,Chip,m![R64]>,
    weight:&HbmTensor<f8e4m3,Chip,DownWeight>,weight_dequant_scale:&HbmTensor<f32,Chip,m![H]>,
    global_scale:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,m![R64,H]>) {
    let input:DmTensor<f8e4m3,Chip,DownCluster,DownParts,m![R64,L%960]>=input.to_dm(&mut device.tdma);
    let xscale:DmTensor<f32,Chip,DownCluster,DownRows,m![R64]>=input_scale.to_dm(&mut device.tdma);
    let weight:DmTensor<f8e4m3,Chip,DownCluster,DownParts,m![H % 120,L % 960]>=weight.to_dm(&mut device.tdma);
    let ws:DmTensor<f32,Chip,DownCluster,DownRows,m![H % 120]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,DownCluster,DownRows,m![H % 120]>=device.sub.begin(ws.view())
        .fetch::<m![H / 8 % 15],m![H % 8]>().collect::<m![H / 8 % 15],m![H % 8]>().to_vrf();
    let global:DmTensor<f32,Chip,DownCluster,DownRows,m![1#8]>=global_scale.to_dm(&mut device.tdma);
    let global:VrfTensor<f32,Chip,DownCluster,DownRows,m![1#8]>=device.sub.begin(global.view())
        .fetch::<m![1],m![1#8]>().collect::<m![1],m![1#8]>().to_vrf();
    let xs:VrfTensor<f32,Chip,DownCluster,DownRows,m![R64]>=device.sub.begin(xscale.view())
        .fetch::<m![R64/8],m![R64%8]>().collect::<m![R64/8],m![R64%8]>().to_vrf();
    let x:TrfTensor<f8e4m3,Chip,DownCluster,DownParts,m![R64%8],m![R64/8,L%960]>=device.sub.begin(input.view())
        .fetch::<m![R64%8,R64/8,L/32%30],m![L%32]>()
        .collect::<m![R64%8,R64/8,L/32%30],m![L%32]>().to_trf();
    let result:DmTensor<bf16,Chip,DownCluster,DownRows,m![H%120,R64]>=device.main.begin(weight.view())
        .fetch::<m![H%120,L/32%30],m![L%32]>().collect::<m![H%120,L/32%30],m![L%32]>()
        .contract_outer::<m![H%120,L/64%15,R64/8],m![L%64],_,_,_>(&x)
        .contract_packet::<m![1]>().contract_time::<m![H%120,R64/8]>()
        .contract_lane::<m![H%120,R64/8],m![R64%8]>(LaneMode::Interleaved)
        .vector_init().vector_inter_slice_reduce::<DownRows,m![H%120,R64/8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H%120,R64/4],m![R64%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![H%120,R64/8],m![R64%8]>()
        .vector_final().cast::<bf16,m![R64%8#16]>().commit_trim::<m![R64%8]>().commit();
        let result:DmTensor<bf16,Chip,DownCluster,DownRows,m![H%120,R64]>=device.main.begin(result.view())
            .fetch::<m![H%120,R64/8],m![R64%8]>().fetch_cast::<f32>()
            .collect::<m![H%120,R64/8],m![R64%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H%120,R64/4],m![R64%4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&global)
            .vector_widen_concat::<m![H%120,R64/8],m![R64%8]>()
            .vector_final().cast::<bf16,m![R64%8#16]>().commit_trim::<m![R64%8]>().commit();
        let result:DmTensor<bf16,Chip,DownCluster,DownRows,m![R64,H%120]>=device.main.begin(result.view())
            .fetch::<m![R64/16,H%120],m![R64%16]>().collect::<m![R64/16,H%120],m![R64%16]>()
            .transpose::<m![R64/16,H/4%30,R64%16],m![H%4#16]>().commit_trim::<m![H%4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
