//! Explicit per-row-group copies repair padded-seed replication.
//! Two-kernel R64 FFN with output-cluster weights and seeded exact row quantization.
//! The BF16 GeGLU hidden HBM stage makes full-row scale computation cluster-local.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{R64,H,L}};
pub use crate::ops_native_fp8_blocked::{UpWeight,DownWeight};
type UpCluster=m![L/7680];
type DownCluster=m![H/1920];
type UpParts=m![L/120%64,H/960];
type UpRows=m![L/120%64,1#4];
type DownParts=m![H/120%16,L/960];
type DownRows=m![H/120%16,1#16];
type UpSeedParts=m![1#64,H/960];
type DownSeedParts=m![1#16,L/960];
type SeedRows=m![1#256];

fn pack_up_seed_scales(device:&mut Device,xscale:&DmTensor<f32,Chip,UpCluster,SeedRows,m![R64,1 # 8]>)
    ->DmTensor<f32,Chip,UpCluster,SeedRows,m![R64]> {
    // Keep the 32-byte scalar slot for aligned inter-slice DMA, then pack by TU.
    let xscale:DmTensor<f32,Chip,UpCluster,SeedRows,m![R64]>=device.sub.begin(xscale.view())
        .fetch::<m![R64],m![1 # 8]>().collect::<m![R64],m![1 # 8]>()
        .transpose::<m![R64 / 2],m![R64 % 2 # 8]>().commit_trim::<m![R64 % 2]>().commit();
    xscale
}
fn quantize_up_seed(device:&mut Device,x:&DmTensor<bf16,Chip,UpCluster,UpSeedParts,m![R64,H % 960]>)
 ->(DmTensor<f8e4m3,Chip,UpCluster,UpSeedParts,m![R64,H % 960]>,DmTensor<f32,Chip,UpCluster,SeedRows,m![R64]>) {
 let amax:DmTensor<f32,Chip,UpCluster,SeedRows,m![R64,1 # 8]>=device.main.begin(x.view())
  .fetch::<m![R64,H /16 %60],m![H %16]>().fetch_cast::<f32>()
  .collect::<m![R64,H /8 %120],m![H %8]>()
  .vector_init().vector_intra_slice_tag(TagMode::Zero)
  .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32).vector_reinterpret::<f32>()
  .vector_narrow_split::<m![R64,H /4 %240],m![H %4]>()
  .vector_intra_slice_reduce::<H,m![R64],m![1 #4]>(IntraSliceReduceOpF32::Max)
  .vector_widen_pad::<m![1 #8]>()
  .vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
  .vector_inter_slice_reduce::<SeedRows,m![R64]>(InterSliceReduceOpF32::Max)
  .vector_final().commit_trim::<m![1 #8]>().commit();
 let scale:DmTensor<f32,Chip,UpCluster,SeedRows,m![R64,1 #8]>=device.main.begin(amax.view())
  .fetch::<m![R64],m![1 #8]>().collect::<m![R64],m![1 #8]>()
  .vector_init().vector_intra_slice_tag(TagMode::Zero)
  .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7f80_0000i32).vector_reinterpret::<f32>()
  .vector_narrow_trim::<m![1 #4]>().vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),0.0078125f32)
  .vector_widen_pad::<m![1 #8]>().vector_final().commit_trim::<m![1 #8]>().commit();
 let scale=pack_up_seed_scales(device,&scale);
 let broadcast:DmTensor<f32,Chip,UpCluster,UpSeedParts,m![R64]>=scale.to_dm(&mut device.tdma);
 let sv:VrfTensor<f32,Chip,UpCluster,UpSeedParts,m![R64]>=device.sub.begin(broadcast.view())
  .fetch::<m![R64 /8],m![R64 %8]>().collect::<m![R64 /8],m![R64 %8]>().to_vrf();
 let q:DmTensor<f8e4m3,Chip,UpCluster,UpSeedParts,m![R64,H %960]>=device.main.begin(x.view())
  .fetch::<m![R64,H /16 %60],m![H %16]>().fetch_cast::<f32>()
  .collect::<m![R64,H /8 %120],m![H %8]>()
  .vector_init().vector_intra_slice_tag(TagMode::Zero)
  .vector_narrow_split::<m![R64,H /4 %240],m![H %4]>()
  .vector_fp_binary(FpBinaryOp::DivF,&sv)
  .vector_widen_concat::<m![R64,H /8 %120],m![H %8]>().vector_final()
  .cast::<f8e4m3,m![H %8 #32]>().commit_trim::<m![H %8]>().commit();
 (q,scale)
}

fn pack_down_seed_scales(device:&mut Device,xscale:&DmTensor<f32,Chip,DownCluster,SeedRows,m![R64,1 # 8]>)
    ->DmTensor<f32,Chip,DownCluster,SeedRows,m![R64]> {
    // Keep the 32-byte scalar slot for aligned inter-slice DMA, then pack by TU.
    let xscale:DmTensor<f32,Chip,DownCluster,SeedRows,m![R64]>=device.sub.begin(xscale.view())
        .fetch::<m![R64],m![1 # 8]>().collect::<m![R64],m![1 # 8]>()
        .transpose::<m![R64 / 2],m![R64 % 2 # 8]>().commit_trim::<m![R64 % 2]>().commit();
    xscale
}
fn quantize_down_seed(device:&mut Device,x:&DmTensor<bf16,Chip,DownCluster,DownSeedParts,m![R64,L % 960]>)
 ->(DmTensor<f8e4m3,Chip,DownCluster,DownSeedParts,m![R64,L % 960]>,DmTensor<f32,Chip,DownCluster,SeedRows,m![R64]>) {
 let amax:DmTensor<f32,Chip,DownCluster,SeedRows,m![R64,1 # 8]>=device.main.begin(x.view())
  .fetch::<m![R64,L /16 %60],m![L %16]>().fetch_cast::<f32>()
  .collect::<m![R64,L /8 %120],m![L %8]>()
  .vector_init().vector_intra_slice_tag(TagMode::Zero)
  .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32).vector_reinterpret::<f32>()
  .vector_narrow_split::<m![R64,L /4 %240],m![L %4]>()
  .vector_intra_slice_reduce::<L,m![R64],m![1 #4]>(IntraSliceReduceOpF32::Max)
  .vector_widen_pad::<m![1 #8]>()
  .vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
  .vector_inter_slice_reduce::<SeedRows,m![R64]>(InterSliceReduceOpF32::Max)
  .vector_final().commit_trim::<m![1 #8]>().commit();
 let scale:DmTensor<f32,Chip,DownCluster,SeedRows,m![R64,1 #8]>=device.main.begin(amax.view())
  .fetch::<m![R64],m![1 #8]>().collect::<m![R64],m![1 #8]>()
  .vector_init().vector_intra_slice_tag(TagMode::Zero)
  .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7f80_0000i32).vector_reinterpret::<f32>()
  .vector_narrow_trim::<m![1 #4]>().vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),0.0078125f32)
  .vector_widen_pad::<m![1 #8]>().vector_final().commit_trim::<m![1 #8]>().commit();
 let scale=pack_down_seed_scales(device,&scale);
 let broadcast:DmTensor<f32,Chip,DownCluster,DownSeedParts,m![R64]>=scale.to_dm(&mut device.tdma);
 let sv:VrfTensor<f32,Chip,DownCluster,DownSeedParts,m![R64]>=device.sub.begin(broadcast.view())
  .fetch::<m![R64 /8],m![R64 %8]>().collect::<m![R64 /8],m![R64 %8]>().to_vrf();
 let q:DmTensor<f8e4m3,Chip,DownCluster,DownSeedParts,m![R64,L %960]>=device.main.begin(x.view())
  .fetch::<m![R64,L /16 %60],m![L %16]>().fetch_cast::<f32>()
  .collect::<m![R64,L /8 %120],m![L %8]>()
  .vector_init().vector_intra_slice_tag(TagMode::Zero)
  .vector_narrow_split::<m![R64,L /4 %240],m![L %4]>()
  .vector_fp_binary(FpBinaryOp::DivF,&sv)
  .vector_widen_concat::<m![R64,L /8 %120],m![L %8]>().vector_final()
  .cast::<f8e4m3,m![L %8 #32]>().commit_trim::<m![L %8]>().commit();
 (q,scale)
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
pub fn upgate_geglu(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,H]>,
    up:&HbmTensor<f8e4m3,Chip,UpWeight>,gate:&HbmTensor<f8e4m3,Chip,UpWeight>,
    up_dequant:&HbmTensor<f32,Chip,m![L]>,gate_dequant:&HbmTensor<f32,Chip,m![L]>,
    up_global:&HbmTensor<f32,Chip,m![1]>,gate_global:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,m![R64,L]>) {
    let input:DmTensor<bf16,Chip,UpCluster,UpSeedParts,m![R64,H%960]>=input.to_dm(&mut device.tdma);
    let (input,xscale)=quantize_up_seed(device,&input);
    let (input,xscale)=replicate_up(device,&input,&xscale);
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
pub fn down(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,L]>,
    weight:&HbmTensor<f8e4m3,Chip,DownWeight>,weight_dequant_scale:&HbmTensor<f32,Chip,m![H]>,
    global_scale:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,m![R64,H]>) {
    let input:DmTensor<bf16,Chip,DownCluster,DownSeedParts,m![R64,L%960]>=input.to_dm(&mut device.tdma);
    let (input,xscale)=quantize_down_seed(device,&input);
    let (input,xscale)=replicate_down(device,&input,&xscale);
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

fn replicate_up(device:&mut Device,q:&DmTensor<f8e4m3,Chip,UpCluster,UpSeedParts,m![R64,H%960]>,
    scale:&DmTensor<f32,Chip,UpCluster,SeedRows,m![R64]>)
    ->(DmTensor<f8e4m3,Chip,UpCluster,UpParts,m![R64,H%960]>,DmTensor<f32,Chip,UpCluster,UpRows,m![R64]>) {
    let mut out:DmTensor<f8e4m3,Chip,UpCluster,UpParts,m![R64,H%960]>=DmTensor::new();
    let mut scales:DmTensor<f32,Chip,UpCluster,UpRows,m![R64]>=DmTensor::new();
    for group in 0..64 {
        q.view().to_dm_view(&mut device.tdma,out.view_mut()
            .slice_tile::<m![L/120%64],1,m![L/120%64=1#{!}64,H/960]>(group));
        scale.view().to_dm_view(&mut device.tdma,scales.view_mut()
            .slice_tile::<m![L/120%64],1,m![L/120%64=1#{!}64,1# 4]>(group));
    }
    (out,scales)
}

fn replicate_down(device:&mut Device,q:&DmTensor<f8e4m3,Chip,DownCluster,DownSeedParts,m![R64,L%960]>,
    scale:&DmTensor<f32,Chip,DownCluster,SeedRows,m![R64]>)
    ->(DmTensor<f8e4m3,Chip,DownCluster,DownParts,m![R64,L%960]>,DmTensor<f32,Chip,DownCluster,DownRows,m![R64]>) {
    let mut out:DmTensor<f8e4m3,Chip,DownCluster,DownParts,m![R64,L%960]>=DmTensor::new();
    let mut scales:DmTensor<f32,Chip,DownCluster,DownRows,m![R64]>=DmTensor::new();
    for group in 0..16 {
        q.view().to_dm_view(&mut device.tdma,out.view_mut()
            .slice_tile::<m![H/120%16],1,m![H/120%16=1#{!}16,L/960]>(group));
        scale.view().to_dm_view(&mut device.tdma,scales.view_mut()
            .slice_tile::<m![H/120%16],1,m![H/120%16=1#{!}16,1# 16]>(group));
    }
    (out,scales)
}

#[device(chip=1)]
pub fn debug_up_quant(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,H]>,
    seed_q:&mut HbmTensor<f8e4m3,Chip,m![L/7680,R64,H]>,seed_s:&mut HbmTensor<f32,Chip,m![L/7680,R64]>,
    copies_q:&mut HbmTensor<f8e4m3,Chip,m![L/120,R64,H]>,copies_s:&mut HbmTensor<f32,Chip,m![L/120,R64]>) {
    let input:DmTensor<bf16,Chip,UpCluster,UpSeedParts,m![R64,H%960]>=input.to_dm(&mut device.tdma);
    let (q,scale)=quantize_up_seed(device,&input);
    q.view().to_hbm_view(&mut device.tdma,seed_q.view_mut());
    scale.view().to_hbm_view(&mut device.tdma,seed_s.view_mut());
    let (q,scale)=replicate_up(device,&q,&scale);
    q.view().to_hbm_view(&mut device.tdma,copies_q.view_mut());
    scale.view().to_hbm_view(&mut device.tdma,copies_s.view_mut());
}

#[device(chip=1)]
pub fn debug_down_quant(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,L]>,
    seed_q:&mut HbmTensor<f8e4m3,Chip,m![H/1920,R64,L]>,seed_s:&mut HbmTensor<f32,Chip,m![H/1920,R64]>,
    copies_q:&mut HbmTensor<f8e4m3,Chip,m![H/120,R64,L]>,copies_s:&mut HbmTensor<f32,Chip,m![H/120,R64]>) {
    let input:DmTensor<bf16,Chip,DownCluster,DownSeedParts,m![R64,L%960]>=input.to_dm(&mut device.tdma);
    let (q,scale)=quantize_down_seed(device,&input);
    q.view().to_hbm_view(&mut device.tdma,seed_q.view_mut());
    scale.view().to_hbm_view(&mut device.tdma,seed_s.view_mut());
    let (q,scale)=replicate_down(device,&q,&scale);
    q.view().to_hbm_view(&mut device.tdma,copies_q.view_mut());
    scale.view().to_hbm_view(&mut device.tdma,copies_s.view_mut());
}
