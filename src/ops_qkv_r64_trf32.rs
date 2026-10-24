//! BF16 H960 QKV with32 token rows in TRF and exact temporal packet reuse.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{R64,H,Qs,Ps,Qf,Pf}};
type Cluster=m![R64/32];

#[device(chip=1)]
pub fn project_query_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,H]>,weight:&HbmTensor<f8e4m3,Chip,m![Qs,H]>,scale:&HbmTensor<bf16,Chip,m![Qs]>,output:&mut HbmTensor<bf16,Chip,m![R64,Qs]>) {
 let packed:DmTensor<f8e4m3,Chip,Cluster,m![Qs/64,H/960],m![Qs%64,H%960]>=weight.to_dm(&mut device.tdma);
 let weight:DmTensor<bf16,Chip,Cluster,m![Qs/64,H/960],m![Qs%64,H%960]>=device.main.begin(packed.view())
  .fetch::<m![Qs%64,H/32%30],m![H%32]>().fetch_table_lookup::<bf16>()
  .collect::<m![Qs%64,H/16%60],m![H%16]>().commit_trim::<m![H%16]>().commit();
 let input:DmTensor<bf16,Chip,Cluster,m![Qs/64,H/960],m![R64%32,H%960]>=input.to_dm(&mut device.tdma);
 let scale:DmTensor<bf16,Chip,Cluster,m![Qs/64,1 #4],m![Qs%64]>=scale.to_dm(&mut device.tdma);
 let scale:VrfTensor<f32,Chip,Cluster,m![Qs/64,1 #4],m![Qs%64]>=device.sub.begin(scale.view())
  .fetch::<m![1],m![Qs%64]>().fetch_cast::<f32>().collect::<m![Qs/8%8],m![Qs%8]>().to_vrf();
 let x:TrfTensor<bf16,Chip,Cluster,m![Qs/64,H/960],m![R64%32%8],m![R64%32/8,H%960]>=device.sub.begin(input.view())
  .fetch::<m![R64%32%8,R64%32/8,H/16%60],m![H%16]>().collect::<m![R64%32%8,R64%32/8,H/16%60],m![H%16]>().to_trf();
 let dot:DmTensor<bf16,Chip,Cluster,m![Qs/64,1 #4],m![Qs%64,R64%32]>=device.main.begin(weight.view())
  .fetch::<m![Qs%64,H/16%60],m![H%16]>().collect::<m![Qs%64,H/16%60],m![H%16]>()
  .contract_outer::<m![Qs%64,H/32%30,R64%32/8],m![H%32],_,_,_>(&x)
  .contract_packet::<m![1]>().contract_time::<m![Qs%64,R64%32/8]>()
  .contract_lane::<m![Qs%64,R64%32/8],m![R64%32%8]>(LaneMode::Interleaved)
  .vector_init().vector_inter_slice_reduce::<m![Qs/64,1 #4],m![Qs%64,R64%32/8]>(InterSliceReduceOpF32::Add)
  .vector_final().cast::<bf16,m![R64%32%8#16]>().commit_trim::<m![R64%32%8]>().commit();
 let result:DmTensor<bf16,Chip,Cluster,m![Qs/64,1 #4],m![Qs%64,R64%32]>=device.main.begin(dot.view()).fetch::<m![Qs%64,R64%32/8],m![R64%32%8]>().fetch_cast::<f32>()
  .collect::<m![Qs%64,R64%32/8],m![R64%32%8]>().vector_init().vector_intra_slice_tag(TagMode::Zero)
  .vector_narrow_split::<m![Qs%64,R64%32/4],m![R64%32%4]>()
  .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
  .vector_widen_concat::<m![Qs%64,R64%32/8],m![R64%32%8]>()
  .vector_final().cast::<bf16,m![R64%32%8#16]>().commit_trim::<m![R64%32%8]>()
  .commit();
 let result:DmTensor<bf16,Chip,Cluster,m![Qs/64,1 #4],m![R64%32,Qs%64]>=device.main.begin(result.view())
  .fetch::<m![R64/16%2,Qs%64],m![R64%16]>().collect::<m![R64/16%2,Qs%64],m![R64%16]>()
  .transpose::<m![R64/16%2,Qs/4%16,R64%16],m![Qs%4#16]>().commit_trim::<m![Qs%4]>().commit();
 result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

#[device(chip=1)]
pub fn project_sliding_kv_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,H]>,weight:&HbmTensor<f8e4m3,Chip,m![Ps,H]>,scale:&HbmTensor<bf16,Chip,m![Ps]>,output:&mut HbmTensor<bf16,Chip,m![R64,Ps]>) {
 let packed:DmTensor<f8e4m3,Chip,Cluster,m![Ps/32,H/960],m![Ps%32,H%960]>=weight.to_dm(&mut device.tdma);
 let weight:DmTensor<bf16,Chip,Cluster,m![Ps/32,H/960],m![Ps%32,H%960]>=device.main.begin(packed.view())
  .fetch::<m![Ps%32,H/32%30],m![H%32]>().fetch_table_lookup::<bf16>()
  .collect::<m![Ps%32,H/16%60],m![H%16]>().commit_trim::<m![H%16]>().commit();
 let input:DmTensor<bf16,Chip,Cluster,m![Ps/32,H/960],m![R64%32,H%960]>=input.to_dm(&mut device.tdma);
 let scale:DmTensor<bf16,Chip,Cluster,m![Ps/32,1 #4],m![Ps%32]>=scale.to_dm(&mut device.tdma);
 let scale:VrfTensor<f32,Chip,Cluster,m![Ps/32,1 #4],m![Ps%32]>=device.sub.begin(scale.view())
  .fetch::<m![1],m![Ps%32]>().fetch_cast::<f32>().collect::<m![Ps/8%4],m![Ps%8]>().to_vrf();
 let x:TrfTensor<bf16,Chip,Cluster,m![Ps/32,H/960],m![R64%32%8],m![R64%32/8,H%960]>=device.sub.begin(input.view())
  .fetch::<m![R64%32%8,R64%32/8,H/16%60],m![H%16]>().collect::<m![R64%32%8,R64%32/8,H/16%60],m![H%16]>().to_trf();
 let dot:DmTensor<bf16,Chip,Cluster,m![Ps/32,1 #4],m![Ps%32,R64%32]>=device.main.begin(weight.view())
  .fetch::<m![Ps%32,H/16%60],m![H%16]>().collect::<m![Ps%32,H/16%60],m![H%16]>()
  .contract_outer::<m![Ps%32,H/32%30,R64%32/8],m![H%32],_,_,_>(&x)
  .contract_packet::<m![1]>().contract_time::<m![Ps%32,R64%32/8]>()
  .contract_lane::<m![Ps%32,R64%32/8],m![R64%32%8]>(LaneMode::Interleaved)
  .vector_init().vector_inter_slice_reduce::<m![Ps/32,1 #4],m![Ps%32,R64%32/8]>(InterSliceReduceOpF32::Add)
  .vector_final().cast::<bf16,m![R64%32%8#16]>().commit_trim::<m![R64%32%8]>().commit();
 let result:DmTensor<bf16,Chip,Cluster,m![Ps/32,1 #4],m![Ps%32,R64%32]>=device.main.begin(dot.view()).fetch::<m![Ps%32,R64%32/8],m![R64%32%8]>().fetch_cast::<f32>()
  .collect::<m![Ps%32,R64%32/8],m![R64%32%8]>().vector_init().vector_intra_slice_tag(TagMode::Zero)
  .vector_narrow_split::<m![Ps%32,R64%32/4],m![R64%32%4]>()
  .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
  .vector_widen_concat::<m![Ps%32,R64%32/8],m![R64%32%8]>()
  .vector_final().cast::<bf16,m![R64%32%8#16]>().commit_trim::<m![R64%32%8]>()
  .commit();
 let result:DmTensor<bf16,Chip,Cluster,m![Ps/32,1 #4],m![R64%32,Ps%32]>=device.main.begin(result.view())
  .fetch::<m![R64/16%2,Ps%32],m![R64%16]>().collect::<m![R64/16%2,Ps%32],m![R64%16]>()
  .transpose::<m![R64/16%2,Ps/4%8,R64%16],m![Ps%4#16]>().commit_trim::<m![Ps%4]>().commit();
 result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

#[device(chip=1)]
pub fn project_full_query_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,H]>,weight:&HbmTensor<f8e4m3,Chip,m![Qf,H]>,scale:&HbmTensor<bf16,Chip,m![Qf]>,output:&mut HbmTensor<bf16,Chip,m![R64,Qf]>) {
 let packed:DmTensor<f8e4m3,Chip,Cluster,m![Qf/128,H/960],m![Qf%128,H%960]>=weight.to_dm(&mut device.tdma);
 let weight:DmTensor<bf16,Chip,Cluster,m![Qf/128,H/960],m![Qf%128,H%960]>=device.main.begin(packed.view())
  .fetch::<m![Qf%128,H/32%30],m![H%32]>().fetch_table_lookup::<bf16>()
  .collect::<m![Qf%128,H/16%60],m![H%16]>().commit_trim::<m![H%16]>().commit();
 let input:DmTensor<bf16,Chip,Cluster,m![Qf/128,H/960],m![R64%32,H%960]>=input.to_dm(&mut device.tdma);
 let scale:DmTensor<bf16,Chip,Cluster,m![Qf/128,1 #4],m![Qf%128]>=scale.to_dm(&mut device.tdma);
 let scale:VrfTensor<f32,Chip,Cluster,m![Qf/128,1 #4],m![Qf%128]>=device.sub.begin(scale.view())
  .fetch::<m![1],m![Qf%128]>().fetch_cast::<f32>().collect::<m![Qf/8%16],m![Qf%8]>().to_vrf();
 let x:TrfTensor<bf16,Chip,Cluster,m![Qf/128,H/960],m![R64%32%8],m![R64%32/8,H%960]>=device.sub.begin(input.view())
  .fetch::<m![R64%32%8,R64%32/8,H/16%60],m![H%16]>().collect::<m![R64%32%8,R64%32/8,H/16%60],m![H%16]>().to_trf();
 let dot:DmTensor<bf16,Chip,Cluster,m![Qf/128,1 #4],m![Qf%128,R64%32]>=device.main.begin(weight.view())
  .fetch::<m![Qf%128,H/16%60],m![H%16]>().collect::<m![Qf%128,H/16%60],m![H%16]>()
  .contract_outer::<m![Qf%128,H/32%30,R64%32/8],m![H%32],_,_,_>(&x)
  .contract_packet::<m![1]>().contract_time::<m![Qf%128,R64%32/8]>()
  .contract_lane::<m![Qf%128,R64%32/8],m![R64%32%8]>(LaneMode::Interleaved)
  .vector_init().vector_inter_slice_reduce::<m![Qf/128,1 #4],m![Qf%128,R64%32/8]>(InterSliceReduceOpF32::Add)
  .vector_final().cast::<bf16,m![R64%32%8#16]>().commit_trim::<m![R64%32%8]>().commit();
 let result:DmTensor<bf16,Chip,Cluster,m![Qf/128,1 #4],m![Qf%128,R64%32]>=device.main.begin(dot.view()).fetch::<m![Qf%128,R64%32/8],m![R64%32%8]>().fetch_cast::<f32>()
  .collect::<m![Qf%128,R64%32/8],m![R64%32%8]>().vector_init().vector_intra_slice_tag(TagMode::Zero)
  .vector_narrow_split::<m![Qf%128,R64%32/4],m![R64%32%4]>()
  .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
  .vector_widen_concat::<m![Qf%128,R64%32/8],m![R64%32%8]>()
  .vector_final().cast::<bf16,m![R64%32%8#16]>().commit_trim::<m![R64%32%8]>()
  .commit();
 let result:DmTensor<bf16,Chip,Cluster,m![Qf/128,1 #4],m![R64%32,Qf%128]>=device.main.begin(result.view())
  .fetch::<m![R64/16%2,Qf%128],m![R64%16]>().collect::<m![R64/16%2,Qf%128],m![R64%16]>()
  .transpose::<m![R64/16%2,Qf/4%32,R64%16],m![Qf%4#16]>().commit_trim::<m![Qf%4]>().commit();
 result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

#[device(chip=1)]
pub fn project_full_key_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,H]>,weight:&HbmTensor<f8e4m3,Chip,m![Pf,H]>,scale:&HbmTensor<bf16,Chip,m![Pf]>,output:&mut HbmTensor<bf16,Chip,m![R64,Pf]>) {
 let packed:DmTensor<f8e4m3,Chip,Cluster,m![Pf/8,H/960],m![Pf%8,H%960]>=weight.to_dm(&mut device.tdma);
 let weight:DmTensor<bf16,Chip,Cluster,m![Pf/8,H/960],m![Pf%8,H%960]>=device.main.begin(packed.view())
  .fetch::<m![Pf%8,H/32%30],m![H%32]>().fetch_table_lookup::<bf16>()
  .collect::<m![Pf%8,H/16%60],m![H%16]>().commit_trim::<m![H%16]>().commit();
 let input:DmTensor<bf16,Chip,Cluster,m![Pf/8,H/960],m![R64%32,H%960]>=input.to_dm(&mut device.tdma);
 let scale:DmTensor<bf16,Chip,Cluster,m![Pf/8,1 #4],m![Pf%8]>=scale.to_dm(&mut device.tdma);
 let scale:VrfTensor<f32,Chip,Cluster,m![Pf/8,1 #4],m![Pf%8]>=device.sub.begin(scale.view())
  .fetch::<m![1],m![Pf%8]>().fetch_cast::<f32>().collect::<m![Pf/8%1],m![Pf%8]>().to_vrf();
 let x:TrfTensor<bf16,Chip,Cluster,m![Pf/8,H/960],m![R64%32%8],m![R64%32/8,H%960]>=device.sub.begin(input.view())
  .fetch::<m![R64%32%8,R64%32/8,H/16%60],m![H%16]>().collect::<m![R64%32%8,R64%32/8,H/16%60],m![H%16]>().to_trf();
 let dot:DmTensor<bf16,Chip,Cluster,m![Pf/8,1 #4],m![Pf%8,R64%32]>=device.main.begin(weight.view())
  .fetch::<m![Pf%8,H/16%60],m![H%16]>().collect::<m![Pf%8,H/16%60],m![H%16]>()
  .contract_outer::<m![Pf%8,H/32%30,R64%32/8],m![H%32],_,_,_>(&x)
  .contract_packet::<m![1]>().contract_time::<m![Pf%8,R64%32/8]>()
  .contract_lane::<m![Pf%8,R64%32/8],m![R64%32%8]>(LaneMode::Interleaved)
  .vector_init().vector_inter_slice_reduce::<m![Pf/8,1 #4],m![Pf%8,R64%32/8]>(InterSliceReduceOpF32::Add)
  .vector_final().cast::<bf16,m![R64%32%8#16]>().commit_trim::<m![R64%32%8]>().commit();
 let result:DmTensor<bf16,Chip,Cluster,m![Pf/8,1 #4],m![Pf%8,R64%32]>=device.main.begin(dot.view()).fetch::<m![Pf%8,R64%32/8],m![R64%32%8]>().fetch_cast::<f32>()
  .collect::<m![Pf%8,R64%32/8],m![R64%32%8]>().vector_init().vector_intra_slice_tag(TagMode::Zero)
  .vector_narrow_split::<m![Pf%8,R64%32/4],m![R64%32%4]>()
  .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
  .vector_widen_concat::<m![Pf%8,R64%32/8],m![R64%32%8]>()
  .vector_final().cast::<bf16,m![R64%32%8#16]>().commit_trim::<m![R64%32%8]>()
  .commit();
 let result:DmTensor<bf16,Chip,Cluster,m![Pf/8,1 #4],m![R64%32,Pf%8]>=device.main.begin(result.view())
  .fetch::<m![R64/16%2,Pf%8],m![R64%16]>().collect::<m![R64/16%2,Pf%8],m![R64%16]>()
  .transpose::<m![R64/16%2,Pf/4%2,R64%16],m![Pf%4#16]>().commit_trim::<m![Pf%4]>().commit();
 result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
