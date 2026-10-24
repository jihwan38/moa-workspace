//! Vocabulary-sharded BF16 LM head: each embedding weight is read by one cluster.
//! H960 F32 partials reduce before the original BF16 dot/softcap boundaries.
use furiosa_opt_std::prelude::*;
use crate::{Chip, LOGIT_SOFTCAP, axes::{R64,H,W}, ops_r64_io_blocked::BlockedLogits};
type Cluster=m![W/4096%2];
type Parts=m![W/64%64,H/960];
type Rows=m![W/64%64,1#4];
#[device(chip=1)]
pub fn logits_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,H]>,weight:&HbmTensor<bf16,Chip,m![W,H]>,out:&mut HbmTensor<bf16,Chip,BlockedLogits>) {
 let input:DmTensor<bf16,Chip,Cluster,Parts,m![R64,H%960]>=input.to_dm(&mut device.tdma);
 for band in 0..W::SIZE/8192 {
  let weight:DmTensor<bf16,Chip,Cluster,Parts,m![W%64,H%960]>=weight.view()
   .tile::<m![W/8192],1,m![1#32,W%8192,H]>(band).to_dm(&mut device.tdma);
  let mut result:DmTensor<bf16,Chip,Cluster,Rows,m![W%64,R64]>=DmTensor::new();
  for chunk in 0..2 {
   let x:TrfTensor<bf16,Chip,Cluster,Parts,m![R64%32%8],m![R64%32/8,H%960]>=device.sub
    .begin(input.view().tile::<m![R64/32],1,m![1#2,R64%32,H%960]>(chunk))
    .fetch::<m![R64%32%8,R64%32/8,H/16%60],m![H%16]>()
    .collect::<m![R64%32%8,R64%32/8,H/16%60],m![H%16]>().to_trf();
   let dot:DmTensor<bf16,Chip,Cluster,Rows,m![W%64,R64%32]>=device.main.begin(weight.view())
    .fetch::<m![W%64,H/16%60],m![H%16]>().collect::<m![W%64,H/16%60],m![H%16]>()
    .contract_outer::<m![W%64,H/32%30,R64%32/8],m![H%32],_,_,_>(&x)
    .contract_packet::<m![1]>().contract_time::<m![W%64,R64%32/8]>()
    .contract_lane::<m![W%64,R64%32/8],m![R64%32%8]>(LaneMode::Interleaved)
    .vector_init().vector_inter_slice_reduce::<Rows,m![W%64,R64%32/8]>(InterSliceReduceOpF32::Add)
    .vector_final().cast::<bf16,m![R64%32%8#16]>().commit_trim::<m![R64%32%8]>().commit();
   let scaled:DmTensor<f32,Chip,Cluster,Rows,m![W%64,R64%32]>=device.main.begin(dot.view())
    .fetch::<m![W%64,R64%32/8],m![R64%32%8]>().fetch_cast::<f32>()
    .collect::<m![W%64,R64%32/8],m![R64%32%8]>()
    .vector_init().vector_intra_slice_tag(TagMode::Zero)
    .vector_narrow_split::<m![W%64,R64%32/4],m![R64%32%4]>()
    .vector_fp_div(LOGIT_SOFTCAP)
    .vector_widen_concat::<m![W%64,R64%32/8],m![R64%32%8]>()
    .vector_final().commit_trim::<m![R64%32%8]>().commit();
   device.main.begin(scaled.view())
    .fetch::<m![W%64,R64%32/8],m![R64%32%8]>().collect::<m![W%64,R64%32/8],m![R64%32%8]>()
    .vector_init().vector_intra_slice_tag(TagMode::Zero)
    .vector_narrow_split::<m![W%64,R64%32/4],m![R64%32%4]>()
    .vector_fp_unary(FpUnaryOp::Tanh)
    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),LOGIT_SOFTCAP)
    .vector_widen_concat::<m![W%64,R64%32/8],m![R64%32%8]>()
    .vector_final().cast::<bf16,m![R64%32%8#16]>().commit_trim::<m![R64%32%8]>()
    .commit_view(result.view_mut().tile::<m![R64/32],1,m![W%64,1#{!}2,R64%32]>(chunk));
  }
  // Transpose each W32 band separately so its whole R64xW32 payload is contiguous.
  for half in 0..2 {
   let tile:DmTensor<bf16,Chip,Cluster,Rows,m![R64,W%32]>=device.main
    .begin(result.view().tile::<m![W/32%2],1,m![1#2,W%32,R64]>(half))
    .fetch::<m![W%32,R64/16],m![R64%16]>()
    .collect::<m![W%32,R64/16],m![R64%16]>()
    .transpose::<m![R64/16,W/4%8,R64%16],m![W%4#16]>()
    .commit_trim::<m![W%4]>().commit();
   tile.view().to_hbm_view(&mut device.tdma,out.view_mut()
    .tile::<m![W/8192],1,m![1#{!}32,W/32%256,R64,W%32]>(band)
    .tile::<m![W/32%2],1,m![1#{!}32,W/64%128,1#{!}2,R64,W%32]>(half));
  }
 }
}
