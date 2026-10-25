//! Resource-only F4/group960 weights for the current first32-query FFN.
//! Original activation quantizers, R64 prefix ABI, GeGLU and output stores remain.
//! Approximate static weights: local group scale precedes the H4/L16 F32 sum.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{R32,R64,H,L}};
pub use crate::ops_native_fp8_blocked::{UpWeight,DownWeight};
pub type UpBlockScale=m![L/240,H/960,L%240];
pub type DownBlockScale=m![H/240,L/960,H%240];
pub type Input=m![H/960,R32,H%960];
pub type Hidden=m![L/120,R32,L%120];
pub type QuantHidden=m![L/960,R32,L%960];
type Cluster=m![R32/16];
type QuantRows=m![R32%16#256];
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



fn stage_quantize_hidden(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,H]>,
    output:&mut HbmTensor<f8e4m3,Chip,Input>,scales:&mut HbmTensor<f32,Chip,m![R32]>) {
    let input:DmTensor<bf16,Chip,m![R64/16%2],m![R64%16#256],m![H]>=input.view()
        .tile::<m![R64],32,m![R64=32 # 64,H]>(0).to_dm(&mut device.tdma);
    let input:DmTensor<bf16,Chip,Cluster,QuantRows,m![H]>=unsafe { input.reshape() };
    let (q,scale)=quantize_hidden_dm(device,&input);
    q.view().to_hbm_view(&mut device.tdma,output.view_mut());
    scale.view().to_hbm_view(&mut device.tdma,scales.view_mut());
}

fn matrix_tile(device:&mut Device,
    input:&TrfTensor<f8e4m3,Chip,UpCluster,UpParts,m![R32%8],m![R32/8,H%960]>,
    weight:&DmTensor<f4e2m1,Chip,UpCluster,UpParts,m![L%120,H%960]>,
    group:&DmTensor<f32,Chip,UpCluster,UpParts,m![L%120]>,
    xs:&DmTensor<f32,Chip,UpCluster,UpRows,m![R32]>,
    ws:&DmTensor<f32,Chip,UpCluster,UpRows,m![L%120]>)
    ->DmTensor<bf16,Chip,UpCluster,UpRows,m![L%120,R32]> {
    let xs:VrfTensor<f32,Chip,UpCluster,UpRows,m![R32]>=device.sub.begin(xs.view())
        .fetch::<m![R32/8],m![R32%8]>().collect::<m![R32/8],m![R32%8]>().to_vrf();
    let ws:VrfTensor<f32,Chip,UpCluster,UpRows,m![L%120]>=device.sub.begin(ws.view())
        .fetch::<m![L/8%15],m![L%8]>().collect::<m![L/8%15],m![L%8]>().to_vrf();
    let group:VrfTensor<f32,Chip,UpCluster,UpParts,m![L%120]>=device.sub.begin(group.view())
        .fetch::<m![L/8%15],m![L%8]>().collect::<m![L/8%15],m![L%8]>().to_vrf();
    let reduced:DmTensor<f32,Chip,UpCluster,UpRows,m![L%120,R32]>=device.main.begin(weight.view())
        .fetch::<m![L%120],m![H%960]>().fetch_table_lookup::<f8e4m3>()
        .collect::<m![L%120,H/32%30],m![H%32]>()
        .contract_outer::<m![L%120,H/64%15,R32/8],m![H%64],_,_,_>(input)
        .contract_packet::<m![1]>().contract_time::<m![L%120,R32/8]>()
        .contract_lane::<m![L%120,R32/8],m![R32%8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L%120,R32/4],m![R32%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&group)
        .vector_widen_concat::<m![L%120,R32/8],m![R32%8]>()
        .vector_inter_slice_reduce::<UpRows,m![L%120,R32/8]>(InterSliceReduceOpF32::Add)
        .vector_final().commit_trim::<m![R32%8]>().commit();
    device.main.begin(reduced.view())
        .fetch::<m![L%120,R32/8],m![R32%8]>().collect::<m![L%120,R32/8],m![R32%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L%120,R32/4],m![R32%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![L%120,R32/8],m![R32%8]>()
        .vector_final().cast::<bf16,m![R32%8#16]>().commit_trim::<m![R32%8]>().commit()
}

fn scale_tile(
    device: &mut Device,
    input: DmTensorView<'_,bf16,Chip,UpCluster,UpRows,m![L % 120,R32=8 # 32]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, UpCluster, UpRows, m![L % 120, R32=8]> {
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
        .fetch::<m![L % 120], m![R32=8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 120, R32=8 / 8], m![R32=8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 120, R32=8 / 4], m![R32=8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![L % 120, R32=8 / 8], m![R32=8 % 8]>()
        .vector_final()
        .cast::<bf16, m![R32=8 % 8 # 16]>()
        .commit_trim::<m![R32=8 % 8]>()
        .commit()
}
fn geglu_tile_into(
    device: &mut Device,
    up: DmTensorView<'_,bf16,Chip,UpCluster,UpRows,m![L % 120,R32=8 # 32]>,
    gate: DmTensorView<'_,bf16,Chip,UpCluster,UpRows,m![L % 120,R32=8 # 32]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    output:DmTensorViewMut<'_,bf16,Chip,UpCluster,UpRows,m![L % 120,R32=8 #{!} 32]>,
) {
    let up = scale_tile(device, up, up_global_scale);
    let gate = scale_tile(device, gate, gate_global_scale);
    let gelu: DmTensor<f32, Chip, UpCluster, UpRows, m![L % 120, R32=8]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![L % 120, R32=8 / 8], m![R32=8 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 120, R32=8 / 8], m![R32=8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 120, R32=8 / 4], m![R32=8 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.70710678118f32)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L % 120, R32=8 / 8], m![R32=8 % 8]>()
        .vector_final()
        .commit_trim::<m![R32=8 % 8]>()
        .commit();
    let gelu: VrfTensor<f32, Chip, UpCluster, UpRows, m![L % 120, R32=8]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![L % 120, R32=8 / 8], m![R32=8 % 8]>()
        .collect::<m![L % 120, R32=8 / 8], m![R32=8 % 8]>()
        .to_vrf();
    device
        .main
        .begin(up.view())
        .fetch::<m![L % 120, R32=8 / 8], m![R32=8 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 120, R32=8 / 8], m![R32=8 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 120, R32=8 / 4], m![R32=8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L % 120, R32=8 / 8], m![R32=8 % 8]>()
        .vector_final()
        .cast::<bf16, m![R32=8 % 8 # 16]>()
        .commit_trim::<m![R32=8 % 8]>()
        .commit_view(output);
}

fn geglu(device:&mut Device,up:&DmTensor<bf16,Chip,UpCluster,UpRows,m![L % 120,R32]>,
    gate:&DmTensor<bf16,Chip,UpCluster,UpRows,m![L % 120,R32]>,
    us:&HbmTensor<f32,Chip,m![1]>,gs:&HbmTensor<f32,Chip,m![1]>)
    ->DmTensor<bf16,Chip,UpCluster,UpRows,m![L % 120,R32]> {
    let mut result:DmTensor<bf16,Chip,UpCluster,UpRows,m![L % 120,R32]>=DmTensor::new();
    for chunk in 0..4 {
        geglu_tile_into(device,
            up.view().tile::<m![R32],8,m![L % 120,R32=8 # 32]>(8*chunk),
            gate.view().tile::<m![R32],8,m![L % 120,R32=8 # 32]>(8*chunk),us,gs,
            result.view_mut().tile::<m![R32],8,m![L % 120,R32=8 #{!} 32]>(8*chunk));
    }
    result
}


fn stage_upgate(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,Input>, input_scale:&HbmTensor<f32,Chip,m![R32]>,
    up:&HbmTensor<f4e2m1,Chip,UpWeight>,gate:&HbmTensor<f4e2m1,Chip,UpWeight>,
    up_block:&HbmTensor<f32,Chip,UpBlockScale>,gate_block:&HbmTensor<f32,Chip,UpBlockScale>,
    up_dequant:&HbmTensor<f32,Chip,m![L]>,gate_dequant:&HbmTensor<f32,Chip,m![L]>,
    up_global:&HbmTensor<f32,Chip,m![1]>,gate_global:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,Hidden>) {
    let input:DmTensor<f8e4m3,Chip,UpCluster,UpParts,m![R32,H%960]>=input.to_dm(&mut device.tdma);
    let xscale:DmTensor<f32,Chip,UpCluster,UpRows,m![R32]>=input_scale.to_dm(&mut device.tdma);
    let up:DmTensor<f4e2m1,Chip,UpCluster,UpParts,m![L%120,H%960]>=up.to_dm(&mut device.tdma);
    let gate:DmTensor<f4e2m1,Chip,UpCluster,UpParts,m![L%120,H%960]>=gate.to_dm(&mut device.tdma);
    let up_block:DmTensor<f32,Chip,UpCluster,UpParts,m![L%120]>=up_block.to_dm(&mut device.tdma);
    let gate_block:DmTensor<f32,Chip,UpCluster,UpParts,m![L%120]>=gate_block.to_dm(&mut device.tdma);
    let us:DmTensor<f32,Chip,UpCluster,UpRows,m![L%120]>=up_dequant.to_dm(&mut device.tdma);
    let gs:DmTensor<f32,Chip,UpCluster,UpRows,m![L%120]>=gate_dequant.to_dm(&mut device.tdma);
    let x:TrfTensor<f8e4m3,Chip,UpCluster,UpParts,m![R32%8],m![R32/8,H%960]>=device.sub.begin(input.view())
        .fetch::<m![R32%8,R32/8,H/32%30],m![H%32]>()
        .collect::<m![R32%8,R32/8,H/32%30],m![H%32]>().to_trf();
    let a=matrix_tile(device,&x,&up,&up_block,&xscale,&us);
    let b=matrix_tile(device,&x,&gate,&gate_block,&xscale,&gs);
    let hidden=geglu(device,&a,&b,up_global,gate_global);
    let hidden:DmTensor<bf16,Chip,UpCluster,UpRows,m![R32,L%120]>=device.main.begin(hidden.view())
        .fetch::<m![R32/16,L%120],m![R32%16]>().collect::<m![R32/16,L%120],m![R32%16]>()
        .transpose::<m![R32/16,L/4%30,R32%16],m![L%4#16]>().commit_trim::<m![L%4]>().commit();
    hidden.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
fn stage_down(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,QuantHidden>, input_scale:&HbmTensor<f32,Chip,m![R32]>,
    weight:&HbmTensor<f4e2m1,Chip,DownWeight>,block:&HbmTensor<f32,Chip,DownBlockScale>,weight_dequant_scale:&HbmTensor<f32,Chip,m![H]>,
    global_scale:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,m![R64,H]>) {
    let input:DmTensor<f8e4m3,Chip,DownCluster,DownParts,m![R32,L%960]>=input.to_dm(&mut device.tdma);
    let xscale:DmTensor<f32,Chip,DownCluster,DownRows,m![R32]>=input_scale.to_dm(&mut device.tdma);
    let weight:DmTensor<f4e2m1,Chip,DownCluster,DownParts,m![H % 120,L % 960]>=weight.to_dm(&mut device.tdma);
    let block:DmTensor<f32,Chip,DownCluster,DownParts,m![H%120]>=block.to_dm(&mut device.tdma);
    let block:VrfTensor<f32,Chip,DownCluster,DownParts,m![H%120]>=device.sub.begin(block.view())
        .fetch::<m![H/8%15],m![H%8]>().collect::<m![H/8%15],m![H%8]>().to_vrf();
    let ws:DmTensor<f32,Chip,DownCluster,DownRows,m![H % 120]>=weight_dequant_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,DownCluster,DownRows,m![H % 120]>=device.sub.begin(ws.view())
        .fetch::<m![H / 8 % 15],m![H % 8]>().collect::<m![H / 8 % 15],m![H % 8]>().to_vrf();
    let global:DmTensor<f32,Chip,DownCluster,DownRows,m![1#8]>=global_scale.to_dm(&mut device.tdma);
    let global:VrfTensor<f32,Chip,DownCluster,DownRows,m![1#8]>=device.sub.begin(global.view())
        .fetch::<m![1],m![1#8]>().collect::<m![1],m![1#8]>().to_vrf();
    let xs:VrfTensor<f32,Chip,DownCluster,DownRows,m![R32]>=device.sub.begin(xscale.view())
        .fetch::<m![R32/8],m![R32%8]>().collect::<m![R32/8],m![R32%8]>().to_vrf();
    let x:TrfTensor<f8e4m3,Chip,DownCluster,DownParts,m![R32%8],m![R32/8,L%960]>=device.sub.begin(input.view())
        .fetch::<m![R32%8,R32/8,L/32%30],m![L%32]>()
        .collect::<m![R32%8,R32/8,L/32%30],m![L%32]>().to_trf();
    let reduced:DmTensor<f32,Chip,DownCluster,DownRows,m![H%120,R32]>=device.main.begin(weight.view())
        .fetch::<m![H%120],m![L%960]>().fetch_table_lookup::<f8e4m3>()
        .collect::<m![H%120,L/32%30],m![L%32]>()
        .contract_outer::<m![H%120,L/64%15,R32/8],m![L%64],_,_,_>(&x)
        .contract_packet::<m![1]>().contract_time::<m![H%120,R32/8]>()
        .contract_lane::<m![H%120,R32/8],m![R32%8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H%120,R32/4],m![R32%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&block)
        .vector_widen_concat::<m![H%120,R32/8],m![R32%8]>()
        .vector_inter_slice_reduce::<DownRows,m![H%120,R32/8]>(InterSliceReduceOpF32::Add)
        .vector_final().commit_trim::<m![R32%8]>().commit();
    let result:DmTensor<bf16,Chip,DownCluster,DownRows,m![H%120,R32]>=device.main.begin(reduced.view())
        .fetch::<m![H%120,R32/8],m![R32%8]>().collect::<m![H%120,R32/8],m![R32%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H%120,R32/4],m![R32%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![H%120,R32/8],m![R32%8]>()
        .vector_final().cast::<bf16,m![R32%8#16]>().commit_trim::<m![R32%8]>().commit();
        let result:DmTensor<bf16,Chip,DownCluster,DownRows,m![H%120,R32]>=device.main.begin(result.view())
            .fetch::<m![H%120,R32/8],m![R32%8]>().fetch_cast::<f32>()
            .collect::<m![H%120,R32/8],m![R32%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H%120,R32/4],m![R32%4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&global)
            .vector_widen_concat::<m![H%120,R32/8],m![R32%8]>()
            .vector_final().cast::<bf16,m![R32%8#16]>().commit_trim::<m![R32%8]>().commit();
        let result:DmTensor<bf16,Chip,DownCluster,DownRows,m![R32,H%120]>=device.main.begin(result.view())
            .fetch::<m![R32/16,H%120],m![R32%16]>().collect::<m![R32/16,H%120],m![R32%16]>()
            .transpose::<m![R32/16,H/4%30,R32%16],m![H%4#16]>().commit_trim::<m![H%4]>().commit();
    let result:DmTensor<bf16,Chip,DownCluster,DownRows,m![R64=32,H%120]>=unsafe { result.reshape() };
    result.view().to_hbm_view(&mut device.tdma,output.view_mut()
        .tile::<m![R64],32,m![R64=32 #{!} 64,H]>(0));
}

type Shards=m![R32/2%8 #16,L/960];
type MaxParts=Shards;
type MaxRows=m![R32/2%8 #16,1#16];
fn stage_quantize_ffn(device:&mut Device,input:&HbmTensor<bf16,Chip,Hidden>,
 output:&mut HbmTensor<f8e4m3,Chip,QuantHidden>,scales:&mut HbmTensor<f32,Chip,m![R32]>) {
 let hidden:DmTensor<bf16,Chip,Cluster,Shards,m![R32%2,L%960]>=input.to_dm(&mut device.tdma);
        let amax:DmTensor<f32,Chip,Cluster,Shards,m![R32%2,1#8]>=device.main.begin(hidden.view())
            .fetch::<m![R32%2,L/16%60],m![L%16]>().fetch_cast::<f32>()
            .collect::<m![R32%2,L/8%120],m![L%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32).vector_reinterpret::<f32>()
            .vector_narrow_split::<m![R32%2,L/4%240],m![L%4]>()
            .vector_intra_slice_reduce::<L,m![R32%2],m![1#4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_pad::<m![1#8]>().vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
            .vector_final().commit_trim::<m![1#8]>().commit();
        let amax:DmTensor<f32,Chip,Cluster,MaxParts,m![R32%2,1#8]>=amax.to_dm(&mut device.tdma);
        let amax:DmTensor<f32,Chip,Cluster,MaxRows,m![R32%2,1#8]>=device.main.begin(amax.view())
            .fetch::<m![R32%2],m![1#8]>().collect::<m![R32%2],m![1#8]>()
            .vector_init().vector_inter_slice_reduce::<MaxRows,m![R32%2]>(InterSliceReduceOpF32::Max)
            .vector_final().commit_trim::<m![1#8]>().commit();
        let scale:DmTensor<f32,Chip,Cluster,MaxRows,m![R32%2,1#8]>=device.main.begin(amax.view())
            .fetch::<m![R32%2],m![1#8]>().collect::<m![R32%2],m![1#8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7f80_0000i32).vector_reinterpret::<f32>()
            .vector_narrow_trim::<m![1#4]>().vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),0.0078125f32)
            .vector_widen_pad::<m![1#8]>().vector_final().commit_trim::<m![1#8]>().commit();
        let broadcast:DmTensor<f32,Chip,Cluster,Shards,m![R32%2,1#8]>=scale.to_dm(&mut device.tdma);
        let packed:DmTensor<f32,Chip,Cluster,Shards,m![R32%2]>=device.main.begin(broadcast.view())
            .fetch::<m![R32%2],m![1#8]>().collect::<m![R32%2],m![1#8]>()
            .transpose::<m![1],m![R32%2#8]>().commit_trim::<m![R32%2]>().commit();
        let sv:VrfTensor<f32,Chip,Cluster,Shards,m![R32%2#8]>=device.sub.begin(packed.view())
            .fetch::<m![1],m![R32%2#8]>().collect::<m![1],m![R32%2#8]>().to_vrf();
        let q:DmTensor<f8e4m3,Chip,Cluster,Shards,m![R32%2,L%960]>=device.main.begin(hidden.view())
            .fetch::<m![R32%2,L/16%60],m![L%16]>().fetch_cast::<f32>()
            .collect::<m![R32%2,L/8%120],m![L%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R32%2,L/4%240],m![L%4]>()
            .vector_fp_binary(FpBinaryOp::DivF,&sv)
            .vector_widen_concat::<m![R32%2,L/8%120],m![L%8]>().vector_final()
            .cast::<f8e4m3,m![L%8#32]>().commit_trim::<m![L%8]>().commit();
        q.view().to_hbm_view(&mut device.tdma, output.view_mut());
        let scale:DmTensor<f32,Chip,Cluster,MaxRows,m![R32%2]>=device.main.begin(scale.view())
            .fetch::<m![R32%2],m![1#8]>().collect::<m![R32%2],m![1#8]>()
            .transpose::<m![1],m![R32%2#8]>().commit_trim::<m![R32%2]>().commit();
        scale.view().to_hbm_view(&mut device.tdma,scales.view_mut());
}

#[device(chip=1)]
pub fn batch_ffn(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,H]>,
 up:&HbmTensor<f4e2m1,Chip,UpWeight>,gate:&HbmTensor<f4e2m1,Chip,UpWeight>,down:&HbmTensor<f4e2m1,Chip,DownWeight>,
 up_block:&HbmTensor<f32,Chip,UpBlockScale>,gate_block:&HbmTensor<f32,Chip,UpBlockScale>,down_block:&HbmTensor<f32,Chip,DownBlockScale>,
 us:&HbmTensor<f32,Chip,m![L]>,gs:&HbmTensor<f32,Chip,m![L]>,ds:&HbmTensor<f32,Chip,m![H]>,
 ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
 qinput:&mut HbmTensor<f8e4m3,Chip,Input>,xscale:&mut HbmTensor<f32,Chip,m![R32]>,
 hidden:&mut HbmTensor<bf16,Chip,Hidden>,qhidden:&mut HbmTensor<f8e4m3,Chip,QuantHidden>,hscale:&mut HbmTensor<f32,Chip,m![R32]>,
 output:&mut HbmTensor<bf16,Chip,m![R64,H]>) {
 stage_quantize_hidden(device,input,qinput,xscale);
 stage_upgate(device,qinput,xscale,up,gate,up_block,gate_block,us,gs,ug,gg,hidden);
 stage_quantize_ffn(device,hidden,qhidden,hscale);
 stage_down(device,qhidden,hscale,down,down_block,ds,dg,output);
}
