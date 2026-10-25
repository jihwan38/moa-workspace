//! Exact H64 O with 240 live slices instead of128: keep each K shard in a private Switch ring; preserve the BF16 partial gather/tree.
//! Four P128 output projections retain one E4 weight read and BF16 conversion.
//! Identical K1024 BF16 partial tree and per-channel scaling; no activation quantization.
//! Exact output projection with K1024 partials parallel across slices.
//! Gather four contiguous partial planes, then preserve the original BF16 addition tree.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,H,Qs,Qf,Ns,Gs,Ds,Gf,Df}};
type Cluster=m![P128/64];
type Parts=m![Qs/1024,H/64#64];
type DownRows=m![H/64#64,1 # 4];
fn add_partials_wide(device:&mut Device,
    left:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 64,P128%64]>,
    right:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 64,P128%64]>)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 64,P128%64]> {
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 64,P128%64]>=DmTensor::new();
    for chunk in 0..4 {
        let left:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 64,P128%64 = 16]>=device.sub
            .begin(left.view().tile::<m![P128%64],16,m![H % 64,P128%64 = 16 # 64]>(16*chunk))
            .fetch::<m![H % 64],m![P128%64 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 64,P128%64 = 16 / 8],m![P128%64 = 16 % 8]>().to_vrf();
        device.main.begin(right.view().tile::<m![P128%64],16,m![H % 64,P128%64 = 16 # 64]>(16*chunk))
            .fetch::<m![H % 64],m![P128%64 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 64,P128%64 = 16 / 8],m![P128%64 = 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add,&left).vector_final()
            .cast::<bf16,m![P128%64 = 16 % 8 # 16]>().commit_trim::<m![P128%64 = 16 % 8]>()
            .commit_view(result.view_mut().tile::<m![P128%64],16,m![H % 64,P128%64 = 16 #{!} 64]>(16*chunk));
    }
    result
}

fn add_gathered_pair(device:&mut Device,
    input:&DmTensor<bf16,Chip,Cluster,DownRows,m![Qs/1024,H%64,P128%64]>,offset:usize)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 64,P128%64]> {
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 64,P128%64]>=DmTensor::new();
    for chunk in 0..4 {
        let left:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 64,P128%64 = 16]>=device.sub
            .begin(input.view().tile::<m![Qs/1024],1,m![1 # 4,H%64,P128%64]>(offset).tile::<m![P128%64],16,m![H % 64 # 256,P128%64 = 16 # 64]>(16*chunk))
            .fetch::<m![H % 64],m![P128%64 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 64,P128%64 = 16 / 8],m![P128%64 = 16 % 8]>().to_vrf();
        device.main.begin(input.view().tile::<m![Qs/1024],1,m![1 # 4,H%64,P128%64]>(offset+1).tile::<m![P128%64],16,m![H % 64 # 256,P128%64 = 16 # 64]>(16*chunk))
            .fetch::<m![H % 64],m![P128%64 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 64,P128%64 = 16 / 8],m![P128%64 = 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add,&left).vector_final()
            .cast::<bf16,m![P128%64 = 16 % 8 # 16]>().commit_trim::<m![P128%64 = 16 % 8]>()
            .commit_view(result.view_mut().tile::<m![P128%64],16,m![H % 64,P128%64 = 16 #{!} 64]>(16*chunk));
    }
    result
}

fn scale_output_wide(device:&mut Device,input:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 64,P128%64]>,
    scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    let scale:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 64]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 64]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![H % 64]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 8],m![H % 8]>().to_vrf();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 64,P128%64]>=device.main.begin(input.view())
        .fetch::<m![H % 64,P128/8%8],m![P128 % 8]>().fetch_cast::<f32>()
        .collect::<m![H % 64,P128/8%8],m![P128 % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 64,P128/4%16],m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
        .vector_widen_concat::<m![H % 64,P128/8%8],m![P128 % 8]>()
        .vector_final().cast::<bf16,m![P128 % 8 # 16]>().commit_trim::<m![P128 % 8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![P128%64,H % 64]>=device.main.begin(result.view())
        .fetch::<m![P128/16%4,H % 64],m![P128 % 16]>()
        .collect::<m![P128/16%4,H % 64],m![P128 % 16]>()
        .transpose::<m![P128/16%4,H / 4 % 16,P128 % 16],m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

fn project_sliding_one(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
 weight:&DmTensor<bf16,Chip,Cluster,Parts,m![H%64,Qs%1024]>,
 scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    let flat:HbmTensorView<'_,bf16,Chip,m![P128,Qs]>=unsafe{input.view().reshape()};

    let mut partial:DmTensor<bf16,Chip,Cluster,Parts,m![H%64,P128%64]>=DmTensor::new();
    for chunk in 0..2 {
        let input:DmTensor<bf16,Chip,Cluster,m![Qs/1024,P128%8,P128/8%4,1#2],m![Qs%1024]>=flat
            .tile::<m![P128/32%2],1,m![P128/64,1#2,P128%32,Qs]>(chunk).to_dm(&mut device.tdma);
        let x:TrfTensor<bf16,Chip,Cluster,Parts,m![P128%8],m![P128/8%4,Qs%1024]>=device.sub
            .begin(input.view())
            .fetch::<m![1],m![Qs%1024]>()
            .switch::<Parts,m![P128%8,P128/8%4]>(SwitchConfig::CustomBroadcast { ring_size: 64 })
            .collect::<m![P128%8,P128/8%4,Qs/16%64],m![Qs%16]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H%64,Qs/16%64],m![Qs%16]>().collect::<m![H%64,Qs/16%64],m![Qs%16]>()
            .contract_outer::<m![H%64,Qs/32%32,P128/8%4],m![Qs%32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H%64,P128/8%4]>()
            .contract_lane::<m![H%64,P128/8%4],m![P128%8]>(LaneMode::Interleaved)
            .cast::<bf16,m![P128%8#16]>().commit_trim::<m![P128%8]>()
            .commit_view(partial.view_mut().tile::<m![P128/32%2],1,m![H%64,1#{!}2,P128%32]>(chunk));
    }
    let gathered:DmTensor<bf16,Chip,Cluster,DownRows,m![Qs/1024,H%64,P128%64]>=partial.to_dm(&mut device.tdma);
    let p01=add_gathered_pair(device,&gathered,0);
    let p23=add_gathered_pair(device,&gathered,2);
    let result=add_partials_wide(device,&p01,&p23);
    scale_output_wide(device,&result,scale,output);
}

#[device(chip=1)]
pub fn project_sliding_output_four(device:&mut Device,a:&HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
 b:&HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
 c:&HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
 d:&HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
 weight:&HbmTensor<f8e4m3,Chip,m![H,Qs]>,scale:&HbmTensor<bf16,Chip,m![H]>,ao:&mut HbmTensor<bf16,Chip,m![P128,H]>,
 bo:&mut HbmTensor<bf16,Chip,m![P128,H]>,
 co:&mut HbmTensor<bf16,Chip,m![P128,H]>,
 out_d:&mut HbmTensor<bf16,Chip,m![P128,H]>) {

    let packed:DmTensor<f8e4m3,Chip,Cluster,Parts,m![H%64,Qs%1024]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,Parts,m![H%64,Qs%1024]>=device.main.begin(packed.view())
        .fetch::<m![H%64,Qs/32%32],m![Qs%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H%64,Qs/16%64],m![Qs%16]>().commit_trim::<m![Qs%16]>().commit();
    project_sliding_one(device,a,&weight,scale,ao);
    project_sliding_one(device,b,&weight,scale,bo);
    project_sliding_one(device,c,&weight,scale,co);
    project_sliding_one(device,d,&weight,scale,out_d);
}

