//! Approximate full-H B16 LM with an E4 shadow of the original BF16 embedding.
//! Both power-of-two dequantizations precede the original BF16 dot boundary.
use furiosa_opt_std::prelude::*;
use crate::{axes::{B,H,W},Chip,LOGIT_SOFTCAP};
type Cluster=m![W/8192%2];
type Rows=m![W/32%256];

/// Returned allocations avoid uploading a zero-filled 1GB shadow at startup.
#[device(chip=1)]
pub fn quantize_embedding(device:&mut Device,weight:&HbmTensor<bf16,Chip,m![W,H]>)
 ->(HbmTensor<f8e4m3,Chip,m![W,H]>,HbmTensor<f32,Chip,m![W]>) {
 let mut quantized=HbmTensor::new();let mut scales=HbmTensor::new();
 for band in 0..16 {
  let x:DmTensor<bf16,Chip,Cluster,Rows,m![W%32,H]>=weight.view()
   .tile::<m![W/16384],1,m![1#16,W%16384,H]>(band).to_dm(&mut device.tdma);
  let amax:DmTensor<f32,Chip,Cluster,Rows,m![W%32]>=device.main.begin(x.view())
   .fetch::<m![W%32,H/16],m![H%16]>().fetch_cast::<f32>()
   .collect::<m![W%32,H/8],m![H%8]>()
   .vector_init().vector_intra_slice_tag(TagMode::Zero)
   .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32).vector_reinterpret::<f32>()
   .vector_narrow_split::<m![W%32,H/4],m![H%4]>()
   .vector_intra_slice_reduce::<H,m![W%32],m![1#4]>(IntraSliceReduceOpF32::Max)
   .vector_widen_pad::<m![1#8]>().vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
   .vector_final().transpose::<m![W/2%16],m![W%2#8]>()
   .commit_trim::<m![W%2]>().commit();
  let scale:DmTensor<f32,Chip,Cluster,Rows,m![W%32]>=device.main.begin(amax.view())
   .fetch::<m![W/2%16],m![W%2#8]>().collect::<m![W/2%16],m![W%2#8]>()
   .vector_init().vector_intra_slice_tag(TagMode::Zero)
   .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7f80_0000i32).vector_reinterpret::<f32>()
   .vector_narrow_trim::<m![W%2]>().vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),0.0078125f32)
   .vector_widen_pad::<m![W%2#8]>().vector_final().commit_trim::<m![W%2]>().commit();
  let sv:VrfTensor<f32,Chip,Cluster,Rows,m![W%32]>=device.sub.begin(scale.view())
   .fetch::<m![1],m![W%32]>().collect::<m![W/8%4],m![W%8]>().to_vrf();
  let q:DmTensor<f8e4m3,Chip,Cluster,Rows,m![W%32,H]>=device.main.begin(x.view())
   .fetch::<m![W%32,H/16],m![H%16]>().fetch_cast::<f32>()
   .collect::<m![W%32,H/8],m![H%8]>()
   .vector_init().vector_intra_slice_tag(TagMode::Zero)
   .vector_narrow_split::<m![W%32,H/4],m![H%4]>()
   .vector_fp_binary(FpBinaryOp::DivF,&sv)
   .vector_widen_concat::<m![W%32,H/8],m![H%8]>().vector_final()
   .cast::<f8e4m3,m![H%8#32]>().commit_trim::<m![H%8]>().commit();
  q.view().to_hbm_view(&mut device.tdma,quantized.view_mut()
   .tile::<m![W/16384],1,m![1#{!}16,W%16384,H]>(band));
  scale.view().to_hbm_view(&mut device.tdma,scales.view_mut()
   .tile::<m![W/16384],1,m![1#{!}16,W%16384]>(band));
 }
 (quantized,scales)
}

#[device(chip=1)]
pub fn logits_quantized(device:&mut Device,
 input:&HbmTensor<f8e4m3,Chip,m![B,H]>,
 input_scale:&HbmTensor<f32,Chip,m![B,1#8]>,
 weight:&HbmTensor<f8e4m3,Chip,m![W,H]>,
 weight_scale:&HbmTensor<f32,Chip,m![W]>,
 out:&mut HbmTensor<bf16,Chip,m![B,W]>) {
 let input:DmTensor<f8e4m3,Chip,Cluster,Rows,m![B,H]>=input.to_dm(&mut device.tdma);
 let xs:DmTensor<f32,Chip,Cluster,Rows,m![B,1#8]>=input_scale.to_dm(&mut device.tdma);
 let xs:DmTensor<f32,Chip,Cluster,Rows,m![B]>=device.sub.begin(xs.view())
  .fetch::<m![B],m![1#8]>().collect::<m![B],m![1#8]>()
  .transpose::<m![B/2],m![B%2#8]>().commit_trim::<m![B%2]>().commit();
 let xs:VrfTensor<f32,Chip,Cluster,Rows,m![B]>=device.sub.begin(xs.view())
  .fetch::<m![B/8],m![B%8]>().collect::<m![B/8],m![B%8]>().to_vrf();
 let x:TrfTensor<f8e4m3,Chip,Cluster,Rows,m![B%8],m![B/8,H]>=device.sub.begin(input.view())
  .fetch::<m![B%8,B/8,H/32],m![H%32]>()
  .collect::<m![B%8,B/8,H/32],m![H%32]>().to_trf();
 let mut result:DmTensor<bf16,Chip,Cluster,Rows,m![W/16384,W%32,B]>=DmTensor::new();
 for band in 0..16 {
  let weight:DmTensor<f8e4m3,Chip,Cluster,Rows,m![W%32,H]>=weight.view()
   .tile::<m![W/16384],1,m![1#16,W%16384,H]>(band).to_dm(&mut device.tdma);
  let ws:DmTensor<f32,Chip,Cluster,Rows,m![W%32]>=weight_scale.view()
   .tile::<m![W/16384],1,m![1#16,W%16384]>(band).to_dm(&mut device.tdma);
  let ws:VrfTensor<f32,Chip,Cluster,Rows,m![W%32]>=device.sub.begin(ws.view())
   .fetch::<m![W/8%4],m![W%8]>().collect::<m![W/8%4],m![W%8]>().to_vrf();
  let dot:DmTensor<bf16,Chip,Cluster,Rows,m![W%32,B]>=device.main.begin(weight.view())
   .fetch::<m![W%32,H/32],m![H%32]>().collect::<m![W%32,H/32],m![H%32]>()
   .contract_outer::<m![W%32,H/64,B/8],m![H%64],_,_,_>(&x)
   .contract_packet::<m![1]>().contract_time::<m![W%32,B/8]>()
   .contract_lane::<m![W%32,B/8],m![B%8]>(LaneMode::Interleaved)
   .vector_init().vector_intra_slice_tag(TagMode::Zero)
   .vector_narrow_split::<m![W%32,B/4],m![B%4]>()
   .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
   .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
   .vector_widen_concat::<m![W%32,B/8],m![B%8]>().vector_final()
   .cast::<bf16,m![B%8#16]>().commit_trim::<m![B%8]>().commit();
  let scaled:DmTensor<f32,Chip,Cluster,Rows,m![W%32,B]>=device.main.begin(dot.view())
   .fetch::<m![W%32,B/8],m![B%8]>().fetch_cast::<f32>()
   .collect::<m![W%32,B/8],m![B%8]>()
   .vector_init().vector_intra_slice_tag(TagMode::Zero)
   .vector_narrow_split::<m![W%32,B/4],m![B%4]>().vector_fp_div(LOGIT_SOFTCAP)
   .vector_widen_concat::<m![W%32,B/8],m![B%8]>().vector_final().commit_trim::<m![B%8]>().commit();
  device.main.begin(scaled.view()).fetch::<m![W%32,B/8],m![B%8]>()
   .collect::<m![W%32,B/8],m![B%8]>().vector_init().vector_intra_slice_tag(TagMode::Zero)
   .vector_narrow_split::<m![W%32,B/4],m![B%4]>().vector_fp_unary(FpUnaryOp::Tanh)
   .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),LOGIT_SOFTCAP)
   .vector_widen_concat::<m![W%32,B/8],m![B%8]>().vector_final()
   .cast::<bf16,m![B%8#16]>().commit_trim::<m![B%8]>()
   .commit_view(result.view_mut().tile::<m![W/16384],1,m![1#{!}16,W%32,B]>(band));
 }
 let result:DmTensor<bf16,Chip,Cluster,Rows,m![B,W/16384,W%32]>=device.main.begin(result.view())
  .fetch::<m![W/16384,W%32],m![B]>().collect::<m![W/16384,W%32],m![B]>()
  .transpose::<m![W/16384,W/4%8,B],m![W%4#16]>().commit_trim::<m![W%4]>().commit();
 type StoreRows=m![W/128%64,1#4];
 let result:DmTensor<bf16,Chip,Cluster,StoreRows,m![B,W/16384,W%128]>=result.to_dm(&mut device.tdma);
 result.view().to_hbm_view(&mut device.tdma,out.view_mut());
}
