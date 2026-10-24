//! Exact R64 output projection with direct sum-operand views.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{R64,H,Qs,Qf,Ns,Gs,Ds,Gf,Df}};
type Cluster=m![R64/32];
type DownRows=m![H/120,1 # 8];
fn add_partials_wide(device:&mut Device,
    left:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R64 % 32]>,
    right:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R64 % 32]>)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R64 % 32]> {
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R64 % 32]>=DmTensor::new();
    for chunk in 0..4 {
        let left:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 120,R64 % 32 = 8]>=device.sub
            .begin(left.view().tile::<m![R64 % 32],8,m![H % 120,R64 % 32 = 8 # 32]>(8*chunk))
            .fetch::<m![H % 120,R64 % 32 = 8 / 8],m![R64 % 32 = 8 % 8]>().fetch_cast::<f32>()
            .collect::<m![H % 120,R64 % 32 = 8 / 8],m![R64 % 32 = 8 % 8]>().to_vrf();
        device.main.begin(right.view().tile::<m![R64 % 32],8,m![H % 120,R64 % 32 = 8 # 32]>(8*chunk))
            .fetch::<m![H % 120,R64 % 32 = 8 / 8],m![R64 % 32 = 8 % 8]>().fetch_cast::<f32>()
            .collect::<m![H % 120,R64 % 32 = 8 / 8],m![R64 % 32 = 8 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add,&left).vector_final()
            .cast::<bf16,m![R64 % 32 = 8 % 8 # 16]>().commit_trim::<m![R64 % 32 = 8 % 8]>()
            .commit_view(result.view_mut().tile::<m![R64 % 32],8,m![H % 120,R64 % 32 = 8 #{!} 32]>(8*chunk));
    }
    result
}

fn scale_output_wide(device:&mut Device,input:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R64 % 32]>,
    scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![R64,H]>) {
    let scale:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 120]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![H % 120]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 15],m![H % 8]>().to_vrf();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R64 % 32]>=device.main.begin(input.view())
        .fetch::<m![H % 120,R64 / 8 % 4],m![R64 % 8]>().fetch_cast::<f32>()
        .collect::<m![H % 120,R64 / 8 % 4],m![R64 % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120,R64 / 4 % 8],m![R64 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
        .vector_widen_concat::<m![H % 120,R64 / 8 % 4],m![R64 % 8]>()
        .vector_final().cast::<bf16,m![R64 % 8 # 16]>().commit_trim::<m![R64 % 8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![R64 % 32,H % 120]>=device.main.begin(result.view())
        .fetch::<m![R64 / 16 % 2,H % 120],m![R64 % 16]>()
        .collect::<m![R64 / 16 % 2,H % 120],m![R64 % 16]>()
        .transpose::<m![R64 / 16 % 2,H / 4 % 30,R64 % 16],m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

fn sliding_partial(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,Ns,Gs,Ds]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qs]>,offset:usize)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R64 % 32]> {
    let packed:DmTensor<f8e4m3,Chip,Cluster,DownRows,m![H % 120,Qs = 1024]>=weight.view()
        .tile::<m![Qs],1024,m![H,Qs = 1024 # 4096]>(offset).to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,Qs = 1024]>=device.main.begin(packed.view())
        .fetch::<m![H % 120,Qs = 1024 / 32],m![Qs = 1024 % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H % 120,Qs = 1024 / 16],m![Qs = 1024 % 16]>()
        .commit_trim::<m![Qs = 1024 % 16]>().commit();
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R64 % 32]>=DmTensor::new();
    for chunk in 0..4 {
        let flat:HbmTensorView<'_,bf16,Chip,m![R64,Qs]>=unsafe{input.view().reshape()};
        let x:DmTensor<bf16,Chip,Cluster,DownRows,m![R64 % 32 = 8,Qs = 1024]>=flat
            .tile::<m![Qs],1024,m![R64,Qs = 1024 # 4096]>(offset)
            .tile::<m![R64 % 32],8,m![R64 / 32,R64 % 32 = 8 # 32,Qs = 1024 # 4096]>(8*chunk).to_dm(&mut device.tdma);
        let x:TrfTensor<bf16,Chip,Cluster,DownRows,m![R64 % 32 = 8],m![Qs = 1024]>=device.sub.begin(x.view())
            .fetch::<m![R64 % 32 = 8,Qs = 1024 / 16],m![Qs = 1024 % 16]>()
            .collect::<m![R64 % 32 = 8,Qs = 1024 / 16],m![Qs = 1024 % 16]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H % 120,Qs = 1024 / 16],m![Qs = 1024 % 16]>()
            .collect::<m![H % 120,Qs = 1024 / 16],m![Qs = 1024 % 16]>()
            .contract_outer::<m![H % 120,Qs = 1024 / 32],m![Qs = 1024 % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H % 120]>()
            .contract_lane::<m![H % 120],m![R64 % 32 = 8]>(LaneMode::Interleaved)
            .cast::<bf16,m![R64 % 32 = 8 # 16]>().commit_trim::<m![R64 % 32 = 8]>()
            .commit_view(result.view_mut().tile::<m![R64 % 32],8,m![H % 120,R64 % 32 = 8 #{!} 32]>(8*chunk));
    }
    result
}

#[device(chip = 1)]
pub fn project_sliding_output_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,Ns,Gs,Ds]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qs]>,scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![R64,H]>){
    let p0=sliding_partial(device,input,weight,0);
    let p1=sliding_partial(device,input,weight,1024);
    let p2=sliding_partial(device,input,weight,2048);
    let p3=sliding_partial(device,input,weight,3072);
    let p0p1=add_partials_wide(device,&p0,&p1);
    let p2p3=add_partials_wide(device,&p2,&p3);
    let p0p1p2p3=add_partials_wide(device,&p0p1,&p2p3);
    scale_output_wide(device,&p0p1p2p3,scale,output);
}

fn full_partial(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,Gf,Df]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qf]>,offset:usize)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R64 % 32]> {
    let packed:DmTensor<f8e4m3,Chip,Cluster,DownRows,m![H % 120,Qf = 1024]>=weight.view()
        .tile::<m![Qf],1024,m![H,Qf = 1024 # 8192]>(offset).to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,Qf = 1024]>=device.main.begin(packed.view())
        .fetch::<m![H % 120,Qf = 1024 / 32],m![Qf = 1024 % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H % 120,Qf = 1024 / 16],m![Qf = 1024 % 16]>()
        .commit_trim::<m![Qf = 1024 % 16]>().commit();
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R64 % 32]>=DmTensor::new();
    for chunk in 0..4 {
        let flat:HbmTensorView<'_,bf16,Chip,m![R64,Qf]>=unsafe{input.view().reshape()};
        let x:DmTensor<bf16,Chip,Cluster,DownRows,m![R64 % 32 = 8,Qf = 1024]>=flat
            .tile::<m![Qf],1024,m![R64,Qf = 1024 # 8192]>(offset)
            .tile::<m![R64 % 32],8,m![R64 / 32,R64 % 32 = 8 # 32,Qf = 1024 # 8192]>(8*chunk).to_dm(&mut device.tdma);
        let x:TrfTensor<bf16,Chip,Cluster,DownRows,m![R64 % 32 = 8],m![Qf = 1024]>=device.sub.begin(x.view())
            .fetch::<m![R64 % 32 = 8,Qf = 1024 / 16],m![Qf = 1024 % 16]>()
            .collect::<m![R64 % 32 = 8,Qf = 1024 / 16],m![Qf = 1024 % 16]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H % 120,Qf = 1024 / 16],m![Qf = 1024 % 16]>()
            .collect::<m![H % 120,Qf = 1024 / 16],m![Qf = 1024 % 16]>()
            .contract_outer::<m![H % 120,Qf = 1024 / 32],m![Qf = 1024 % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H % 120]>()
            .contract_lane::<m![H % 120],m![R64 % 32 = 8]>(LaneMode::Interleaved)
            .cast::<bf16,m![R64 % 32 = 8 # 16]>().commit_trim::<m![R64 % 32 = 8]>()
            .commit_view(result.view_mut().tile::<m![R64 % 32],8,m![H % 120,R64 % 32 = 8 #{!} 32]>(8*chunk));
    }
    result
}

#[device(chip = 1)]
pub fn project_full_output_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,Gf,Df]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qf]>,scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![R64,H]>){
    let p0=full_partial(device,input,weight,0);
    let p1=full_partial(device,input,weight,1024);
    let p2=full_partial(device,input,weight,2048);
    let p3=full_partial(device,input,weight,3072);
    let p4=full_partial(device,input,weight,4096);
    let p5=full_partial(device,input,weight,5120);
    let p6=full_partial(device,input,weight,6144);
    let p7=full_partial(device,input,weight,7168);
    let p0p1=add_partials_wide(device,&p0,&p1);
    let p2p3=add_partials_wide(device,&p2,&p3);
    let p4p5=add_partials_wide(device,&p4,&p5);
    let p6p7=add_partials_wide(device,&p6,&p7);
    let p0p1p2p3=add_partials_wide(device,&p0p1,&p2p3);
    let p4p5p6p7=add_partials_wide(device,&p4p5,&p6p7);
    let p0p1p2p3p4p5p6p7=add_partials_wide(device,&p0p1p2p3,&p4p5p6p7);
    scale_output_wide(device,&p0p1p2p3p4p5p6p7,scale,output);
}

