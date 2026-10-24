//! Same P512 down arithmetic with a physically blocked final H240 output.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P512,P128,H,L}};
use crate::ops_batch_layouts::P512L960;
use crate::ops_native_fp8_blocked::DownWeight;
pub type Output=m![P512/256,H/240,P512%256,H%240];
type Cluster=m![P512/256];
type DownParts=m![H/240,L/960];
type DownRows=m![H/240,1#16];
#[device(chip=1)]
pub fn project_down_scaled(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,P512L960>,
 xscale:&HbmTensor<f32,Chip,m![P512]>,weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
 weight_dequant_scale:&HbmTensor<f32,Chip,m![H]>,global_scale:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,Output>) {
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
            .tile::<m![P512%256],64,m![P512/256,H/240,P512%256=64#{!}256,H%240]>(64*chunk));
    }
}

fn unpack_one(device:&mut Device,input:&HbmTensor<bf16,Chip,Output>,cluster:usize,half:usize,
 output:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
 let x:DmTensor<bf16,Chip,m![1#2],m![P512%256=128,1#2],m![H]>=input.view()
  .tile::<m![P512/256],1,m![P512/256=1#2,H/240,P512%256,H%240]>(cluster)
  .tile::<m![P512%256],128,m![P512/256=1#2,H/240,P512%256=128#256,H%240]>(128*half)
  .to_dm(&mut device.tdma);
 let x:DmTensor<bf16,Chip,m![1#2],m![P128,1#2],m![H]>=unsafe{x.reshape()};
 x.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
#[device(chip=1)]
pub fn unpack_hidden_group(device:&mut Device,input:&HbmTensor<bf16,Chip,Output>,
 a:&mut HbmTensor<bf16,Chip,m![P128,H]>,b:&mut HbmTensor<bf16,Chip,m![P128,H]>,
 c:&mut HbmTensor<bf16,Chip,m![P128,H]>,d:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
 unpack_one(device,input,0,0,a);unpack_one(device,input,0,1,b);
 unpack_one(device,input,1,0,c);unpack_one(device,input,1,1,d);
}
