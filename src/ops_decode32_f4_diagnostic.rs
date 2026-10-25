//! Probe-only local partial exports; production R32 F4 kernel remains unchanged.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{R32,H,L}};
use crate::ops_decode32_f4_partialscale::{Input,QuantHidden,UpWeight,DownWeight,UpBlockScale,DownBlockScale};
type UpCluster=m![L/7680];
type UpParts=m![L/120%64,H/960];
type DownCluster=m![H/1920];
type DownParts=m![H/120%16,L/960];
/// Probe-only local scaled F32 partials, before the four-part reduction.
#[device(chip=1)]
pub fn debug_up_partials(device:&mut Device,
 input:&HbmTensor<f8e4m3,Chip,Input>,weight:&HbmTensor<f4e2m1,Chip,UpWeight>,
 scales:&HbmTensor<f32,Chip,UpBlockScale>,output:&mut HbmTensor<f32,Chip,m![R32,L,H/960]>) {
 let input:DmTensor<f8e4m3,Chip,UpCluster,UpParts,m![R32,H%960]>=input.to_dm(&mut device.tdma);
 let weight:DmTensor<f4e2m1,Chip,UpCluster,UpParts,m![L%120,H%960]>=weight.to_dm(&mut device.tdma);
 let scales:DmTensor<f32,Chip,UpCluster,UpParts,m![L%120]>=scales.to_dm(&mut device.tdma);
 let scales:VrfTensor<f32,Chip,UpCluster,UpParts,m![L%120]>=device.sub.begin(scales.view())
  .fetch::<m![L/8%15],m![L%8]>().collect::<m![L/8%15],m![L%8]>().to_vrf();
 let x:TrfTensor<f8e4m3,Chip,UpCluster,UpParts,m![R32%8],m![R32/8,H%960]>=device.sub.begin(input.view())
  .fetch::<m![R32%8,R32/8,H/32%30],m![H%32]>()
  .collect::<m![R32%8,R32/8,H/32%30],m![H%32]>().to_trf();
 let partial:DmTensor<f32,Chip,UpCluster,UpParts,m![L%120,R32]>=device.main.begin(weight.view())
  .fetch::<m![L%120],m![H%960]>().fetch_table_lookup::<f8e4m3>()
  .collect::<m![L%120,H/32%30],m![H%32]>()
  .contract_outer::<m![L%120,H/64%15,R32/8],m![H%64],_,_,_>(&x)
  .contract_packet::<m![1]>().contract_time::<m![L%120,R32/8]>()
  .contract_lane::<m![L%120,R32/8],m![R32%8]>(LaneMode::Interleaved)
  .vector_init().vector_intra_slice_tag(TagMode::Zero)
  .vector_narrow_split::<m![L%120,R32/4],m![R32%4]>()
  .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scales)
  .vector_widen_concat::<m![L%120,R32/8],m![R32%8]>().vector_final()
  .commit_trim::<m![R32%8]>().commit();
 partial.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
/// Probe-only local scaled F32 partials, before the sixteen-part reduction.
#[device(chip=1)]
pub fn debug_down_partials(device:&mut Device,
 input:&HbmTensor<f8e4m3,Chip,QuantHidden>,weight:&HbmTensor<f4e2m1,Chip,DownWeight>,
 scales:&HbmTensor<f32,Chip,DownBlockScale>,output:&mut HbmTensor<f32,Chip,m![R32,H,L/960]>) {
 let input:DmTensor<f8e4m3,Chip,DownCluster,DownParts,m![R32,L%960]>=input.to_dm(&mut device.tdma);
 let weight:DmTensor<f4e2m1,Chip,DownCluster,DownParts,m![H%120,L%960]>=weight.to_dm(&mut device.tdma);
 let scales:DmTensor<f32,Chip,DownCluster,DownParts,m![H%120]>=scales.to_dm(&mut device.tdma);
 let scales:VrfTensor<f32,Chip,DownCluster,DownParts,m![H%120]>=device.sub.begin(scales.view())
  .fetch::<m![H/8%15],m![H%8]>().collect::<m![H/8%15],m![H%8]>().to_vrf();
 let x:TrfTensor<f8e4m3,Chip,DownCluster,DownParts,m![R32%8],m![R32/8,L%960]>=device.sub.begin(input.view())
  .fetch::<m![R32%8,R32/8,L/32%30],m![L%32]>().collect::<m![R32%8,R32/8,L/32%30],m![L%32]>().to_trf();
 let partial:DmTensor<f32,Chip,DownCluster,DownParts,m![H%120,R32]>=device.main.begin(weight.view())
  .fetch::<m![H%120],m![L%960]>().fetch_table_lookup::<f8e4m3>()
  .collect::<m![H%120,L/32%30],m![L%32]>()
  .contract_outer::<m![H%120,L/64%15,R32/8],m![L%64],_,_,_>(&x)
  .contract_packet::<m![1]>().contract_time::<m![H%120,R32/8]>()
  .contract_lane::<m![H%120,R32/8],m![R32%8]>(LaneMode::Interleaved)
  .vector_init().vector_intra_slice_tag(TagMode::Zero)
  .vector_narrow_split::<m![H%120,R32/4],m![R32%4]>()
  .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scales)
  .vector_widen_concat::<m![H%120,R32/8],m![R32%8]>().vector_final()
  .commit_trim::<m![R32%8]>().commit();
 partial.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
