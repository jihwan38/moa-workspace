//! Exact retained output with staged pairwise inter-slice BF16 reductions.
//! Each pair adds two BF16 partials in F32 then rounds to BF16, preserving the original tree.
//! Four P128 output projections retain one E4 weight read and BF16 conversion.
//! Identical K1024 BF16 partial tree and per-channel scaling; no activation quantization.
//! Exact output projection with K1024 partials parallel across slices.
//! Gather four contiguous partial planes, then preserve the original BF16 addition tree.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,H,Qs,Qf,Ns,Gs,Ds,Gf,Df}};
type Cluster=m![P128/64];
type Parts=m![H/120,Qs/1024,1 # 2];
type FullParts=m![H/120,Qf/1024];
type DownRows=m![H/120,1 # 8];
fn add_partials_wide(device:&mut Device,
    left:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128%64]>,
    right:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128%64]>)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128%64]> {
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128%64]>=DmTensor::new();
    for chunk in 0..4 {
        let left:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 120,P128%64 = 16]>=device.sub
            .begin(left.view().tile::<m![P128%64],16,m![H % 120,P128%64 = 16 # 64]>(16*chunk))
            .fetch::<m![H % 120],m![P128%64 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 120,P128%64 = 16 / 8],m![P128%64 = 16 % 8]>().to_vrf();
        device.main.begin(right.view().tile::<m![P128%64],16,m![H % 120,P128%64 = 16 # 64]>(16*chunk))
            .fetch::<m![H % 120],m![P128%64 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 120,P128%64 = 16 / 8],m![P128%64 = 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add,&left).vector_final()
            .cast::<bf16,m![P128%64 = 16 % 8 # 16]>().commit_trim::<m![P128%64 = 16 % 8]>()
            .commit_view(result.view_mut().tile::<m![P128%64],16,m![H % 120,P128%64 = 16 #{!} 64]>(16*chunk));
    }
    result
}

fn add_gathered_pair(device:&mut Device,
    input:&DmTensor<bf16,Chip,Cluster,DownRows,m![Qs/1024,H%120,P128%64]>,offset:usize)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128%64]> {
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128%64]>=DmTensor::new();
    for chunk in 0..4 {
        let left:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 120,P128%64 = 16]>=device.sub
            .begin(input.view().tile::<m![Qs/1024],1,m![1 # 4,H%120,P128%64]>(offset).tile::<m![P128%64],16,m![H % 120 # 480,P128%64 = 16 # 64]>(16*chunk))
            .fetch::<m![H % 120],m![P128%64 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 120,P128%64 = 16 / 8],m![P128%64 = 16 % 8]>().to_vrf();
        device.main.begin(input.view().tile::<m![Qs/1024],1,m![1 # 4,H%120,P128%64]>(offset+1).tile::<m![P128%64],16,m![H % 120 # 480,P128%64 = 16 # 64]>(16*chunk))
            .fetch::<m![H % 120],m![P128%64 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 120,P128%64 = 16 / 8],m![P128%64 = 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add,&left).vector_final()
            .cast::<bf16,m![P128%64 = 16 % 8 # 16]>().commit_trim::<m![P128%64 = 16 % 8]>()
            .commit_view(result.view_mut().tile::<m![P128%64],16,m![H % 120,P128%64 = 16 #{!} 64]>(16*chunk));
    }
    result
}

fn scale_output_wide(device:&mut Device,input:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128%64]>,
    scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    let scale:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 120]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![H % 120]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 15],m![H % 8]>().to_vrf();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128%64]>=device.main.begin(input.view())
        .fetch::<m![H % 120,P128/8%8],m![P128 % 8]>().fetch_cast::<f32>()
        .collect::<m![H % 120,P128/8%8],m![P128 % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120,P128/4%16],m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
        .vector_widen_concat::<m![H % 120,P128/8%8],m![P128 % 8]>()
        .vector_final().cast::<bf16,m![P128 % 8 # 16]>().commit_trim::<m![P128 % 8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![P128%64,H % 120]>=device.main.begin(result.view())
        .fetch::<m![P128/16%4,H % 120],m![P128 % 16]>()
        .collect::<m![P128/16%4,H % 120],m![P128 % 16]>()
        .transpose::<m![P128/16%4,H / 4 % 30,P128 % 16],m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

fn add_gathered_pair_full(device:&mut Device,
    input:&DmTensor<bf16,Chip,Cluster,DownRows,m![Qf/1024,H%120,P128%64]>,offset:usize)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128%64]> {
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,P128%64]>=DmTensor::new();
    for chunk in 0..4 {
        let left:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 120,P128%64 = 16]>=device.sub
            .begin(input.view().tile::<m![Qf/1024],1,m![1 # 8,H%120,P128%64]>(offset).tile::<m![P128%64],16,m![H % 120 # 960,P128%64 = 16 # 64]>(16*chunk))
            .fetch::<m![H % 120],m![P128%64 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 120,P128%64 = 16 / 8],m![P128%64 = 16 % 8]>().to_vrf();
        device.main.begin(input.view().tile::<m![Qf/1024],1,m![1 # 8,H%120,P128%64]>(offset+1).tile::<m![P128%64],16,m![H % 120 # 960,P128%64 = 16 # 64]>(16*chunk))
            .fetch::<m![H % 120],m![P128%64 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 120,P128%64 = 16 / 8],m![P128%64 = 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add,&left).vector_final()
            .cast::<bf16,m![P128%64 = 16 % 8 # 16]>().commit_trim::<m![P128%64 = 16 % 8]>()
            .commit_view(result.view_mut().tile::<m![P128%64],16,m![H % 120,P128%64 = 16 #{!} 64]>(16*chunk));
    }
    result
}

fn reduce_pair<Source: M, Target: M>(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, Source, m![H%120,P128%64]>,
) -> DmTensor<bf16, Chip, Cluster, Target, m![H%120,P128%64]> {
    let mut output: DmTensor<bf16, Chip, Cluster, Target, m![H%120,P128%64]> = DmTensor::new();
    for chunk in 0..4 {
        device.main.begin(input.view().tile::<m![P128%64],16,m![H%120,P128%64=16#64]>(16*chunk))
            .fetch::<m![H%120,P128%64=16/8],m![P128%64=16%8]>()
            .fetch_cast::<f32>()
            .collect::<m![H%120,P128%64=16/8],m![P128%64=16%8]>()
            .vector_init()
            .vector_inter_slice_reduce::<Target,m![H%120,P128%64=16/8]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16,m![P128%64=16%8#16]>()
            .commit_trim::<m![P128%64=16%8]>()
            .commit_view(output.view_mut().tile::<m![P128%64],16,m![H%120,P128%64=16#{!}64]>(16*chunk));
    }
    output
}

fn project_sliding_one(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
 weight:&DmTensor<bf16,Chip,Cluster,Parts,m![H%120,Qs%1024]>,
 scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    let flat:HbmTensorView<'_,bf16,Chip,m![P128,Qs]>=unsafe{input.view().reshape()};
    let input:DmTensor<bf16,Chip,Cluster,Parts,m![P128%64,Qs%1024]>=flat.to_dm(&mut device.tdma);
    let mut partial:DmTensor<bf16,Chip,Cluster,Parts,m![H%120,P128%64]>=DmTensor::new();
    for chunk in 0..2 {
        let x:TrfTensor<bf16,Chip,Cluster,Parts,m![P128%8],m![P128/8%4,Qs%1024]>=device.sub
            .begin(input.view().tile::<m![P128/32%2],1,m![1#2,P128%32,Qs%1024]>(chunk))
            .fetch::<m![P128%8,P128/8%4,Qs/16%64],m![Qs%16]>().collect::<m![P128%8,P128/8%4,Qs/16%64],m![Qs%16]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H%120,Qs/16%64],m![Qs%16]>().collect::<m![H%120,Qs/16%64],m![Qs%16]>()
            .contract_outer::<m![H%120,Qs/32%32,P128/8%4],m![Qs%32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H%120,P128/8%4]>()
            .contract_lane::<m![H%120,P128/8%4],m![P128%8]>(LaneMode::Interleaved)
            .cast::<bf16,m![P128%8#16]>().commit_trim::<m![P128%8]>()
            .commit_view(partial.view_mut().tile::<m![P128/32%2],1,m![H%120,1#{!}2,P128%32]>(chunk));
    }
    let pairs = reduce_pair::<Parts,m![H/120,Qs/2048,1#4]>(device,&partial);
    let result = reduce_pair::<m![H/120,Qs/2048,1#4],DownRows>(device,&pairs);
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

    let packed:DmTensor<f8e4m3,Chip,Cluster,Parts,m![H%120,Qs%1024]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,Parts,m![H%120,Qs%1024]>=device.main.begin(packed.view())
        .fetch::<m![H%120,Qs/32%32],m![Qs%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H%120,Qs/16%64],m![Qs%16]>().commit_trim::<m![Qs%16]>().commit();
    project_sliding_one(device,a,&weight,scale,ao);
    project_sliding_one(device,b,&weight,scale,bo);
    project_sliding_one(device,c,&weight,scale,co);
    project_sliding_one(device,d,&weight,scale,out_d);
}

fn project_full_one(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,Gf,Df]>,
 weight:&DmTensor<bf16,Chip,Cluster,FullParts,m![H%120,Qf%1024]>,
 scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    let flat:HbmTensorView<'_,bf16,Chip,m![P128,Qf]>=unsafe{input.view().reshape()};
    let input:DmTensor<bf16,Chip,Cluster,FullParts,m![P128%64,Qf%1024]>=flat.to_dm(&mut device.tdma);
    let mut partial:DmTensor<bf16,Chip,Cluster,FullParts,m![H%120,P128%64]>=DmTensor::new();
    for chunk in 0..2 {
        let x:TrfTensor<bf16,Chip,Cluster,FullParts,m![P128%8],m![P128/8%4,Qf%1024]>=device.sub
            .begin(input.view().tile::<m![P128/32%2],1,m![1#2,P128%32,Qf%1024]>(chunk))
            .fetch::<m![P128%8,P128/8%4,Qf/16%64],m![Qf%16]>().collect::<m![P128%8,P128/8%4,Qf/16%64],m![Qf%16]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H%120,Qf/16%64],m![Qf%16]>().collect::<m![H%120,Qf/16%64],m![Qf%16]>()
            .contract_outer::<m![H%120,Qf/32%32,P128/8%4],m![Qf%32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H%120,P128/8%4]>()
            .contract_lane::<m![H%120,P128/8%4],m![P128%8]>(LaneMode::Interleaved)
            .cast::<bf16,m![P128%8#16]>().commit_trim::<m![P128%8]>()
            .commit_view(partial.view_mut().tile::<m![P128/32%2],1,m![H%120,1#{!}2,P128%32]>(chunk));
    }
    let pairs = reduce_pair::<FullParts,m![H/120,Qf/2048,1#2]>(device,&partial);
    let quads = reduce_pair::<m![H/120,Qf/2048,1#2],m![H/120,Qf/4096,1#4]>(device,&pairs);
    let result = reduce_pair::<m![H/120,Qf/4096,1#4],DownRows>(device,&quads);
    scale_output_wide(device,&result,scale,output);
}

#[device(chip=1)]
pub fn project_full_output_four(device:&mut Device,a:&HbmTensor<bf16,Chip,m![P128,Gf,Df]>,
 b:&HbmTensor<bf16,Chip,m![P128,Gf,Df]>,
 c:&HbmTensor<bf16,Chip,m![P128,Gf,Df]>,
 d:&HbmTensor<bf16,Chip,m![P128,Gf,Df]>,
 weight:&HbmTensor<f8e4m3,Chip,m![H,Qf]>,scale:&HbmTensor<bf16,Chip,m![H]>,ao:&mut HbmTensor<bf16,Chip,m![P128,H]>,
 bo:&mut HbmTensor<bf16,Chip,m![P128,H]>,
 co:&mut HbmTensor<bf16,Chip,m![P128,H]>,
 out_d:&mut HbmTensor<bf16,Chip,m![P128,H]>) {

    let packed:DmTensor<f8e4m3,Chip,Cluster,FullParts,m![H%120,Qf%1024]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,FullParts,m![H%120,Qf%1024]>=device.main.begin(packed.view())
        .fetch::<m![H%120,Qf/32%32],m![Qf%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H%120,Qf/16%64],m![Qf%16]>().commit_trim::<m![Qf%16]>().commit();
    project_full_one(device,a,&weight,scale,ao);
    project_full_one(device,b,&weight,scale,bo);
    project_full_one(device,c,&weight,scale,co);
    project_full_one(device,d,&weight,scale,out_d);
}
