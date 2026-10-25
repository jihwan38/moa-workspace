//! Exact first32 O within the original64-row HBM ABI; two16-row query clusters.
//! Rows32..63 remain untouched. Weight reads and all K1024 BF16 boundaries are unchanged.
//! Exact output projection with K1024 partials parallel across slices.
//! Gather four contiguous partial planes, then preserve the original BF16 addition tree.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{R32,R64,H,Qs,Qf,Ns,Gs,Ds,Gf,Df}};
type Cluster=m![R32/16];
type Parts=m![H/120,Qs/1024,1 # 2];
type FullParts=m![H/120,Qf/1024];
type DownRows=m![H/120,1 # 8];
fn add_partials_wide(device:&mut Device,
    left:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R32 % 16]>,
    right:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R32 % 16]>)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R32 % 16]> {
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R32 % 16]>=DmTensor::new();
    {
        let left:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 120,R32 % 16 = 16]>=device.sub
            .begin(left.view().tile::<m![R32 % 16],16,m![H % 120,R32 % 16 = 16 # 16]>(0))
            .fetch::<m![H % 120],m![R32 % 16 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 120,R32 % 16 = 16 / 8],m![R32 % 16 = 16 % 8]>().to_vrf();
        device.main.begin(right.view().tile::<m![R32 % 16],16,m![H % 120,R32 % 16 = 16 # 16]>(0))
            .fetch::<m![H % 120],m![R32 % 16 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 120,R32 % 16 = 16 / 8],m![R32 % 16 = 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add,&left).vector_final()
            .cast::<bf16,m![R32 % 16 = 16 % 8 # 16]>().commit_trim::<m![R32 % 16 = 16 % 8]>()
            .commit_view(result.view_mut().tile::<m![R32 % 16],16,m![H % 120,R32 % 16 = 16 #{!} 16]>(0));
    }
    result
}

fn add_gathered_pair(device:&mut Device,
    input:&DmTensor<bf16,Chip,Cluster,DownRows,m![Qs/1024,H%120,R32%16]>,offset:usize)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R32 % 16]> {
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R32 % 16]>=DmTensor::new();
    {
        let left:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 120,R32 % 16 = 16]>=device.sub
            .begin(input.view().tile::<m![Qs/1024],1,m![1 # 4,H%120,R32%16]>(offset).tile::<m![R32 % 16],16,m![H % 120 # 480,R32 % 16 = 16 # 16]>(0))
            .fetch::<m![H % 120],m![R32 % 16 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 120,R32 % 16 = 16 / 8],m![R32 % 16 = 16 % 8]>().to_vrf();
        device.main.begin(input.view().tile::<m![Qs/1024],1,m![1 # 4,H%120,R32%16]>(offset+1).tile::<m![R32 % 16],16,m![H % 120 # 480,R32 % 16 = 16 # 16]>(0))
            .fetch::<m![H % 120],m![R32 % 16 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 120,R32 % 16 = 16 / 8],m![R32 % 16 = 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add,&left).vector_final()
            .cast::<bf16,m![R32 % 16 = 16 % 8 # 16]>().commit_trim::<m![R32 % 16 = 16 % 8]>()
            .commit_view(result.view_mut().tile::<m![R32 % 16],16,m![H % 120,R32 % 16 = 16 #{!} 16]>(0));
    }
    result
}

fn scale_output_wide(device:&mut Device,input:&DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R32 % 16]>,
    scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![R64,H]>) {
    let scale:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 120]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![H % 120]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 15],m![H % 8]>().to_vrf();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R32 % 16]>=device.main.begin(input.view())
        .fetch::<m![H % 120,R32 / 8 % 2],m![R32 % 8]>().fetch_cast::<f32>()
        .collect::<m![H % 120,R32 / 8 % 2],m![R32 % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120,R32 / 4 % 4],m![R32 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
        .vector_widen_concat::<m![H % 120,R32 / 8 % 2],m![R32 % 8]>()
        .vector_final().cast::<bf16,m![R32 % 8 # 16]>().commit_trim::<m![R32 % 8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![R32 % 16,H % 120]>=device.main.begin(result.view())
        .fetch::<m![R32 / 16 % 1,H % 120],m![R32 % 16]>()
        .collect::<m![R32 / 16 % 1,H % 120],m![R32 % 16]>()
        .transpose::<m![R32 / 16 % 1,H / 4 % 30,R32 % 16],m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>().commit();
    let result:DmTensor<bf16,Chip,m![R64/16%2],DownRows,m![R64%16,H%120]>=unsafe{result.reshape()};
    result.view().to_hbm_view(&mut device.tdma,output.view_mut()
        .tile::<m![R64],32,m![R64=32 #{!}64,H]>(0));
}

#[device(chip=1)]
pub fn project_sliding_output_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,Ns,Gs,Ds]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qs]>,scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![R64,H]>) {
    let packed:DmTensor<f8e4m3,Chip,Cluster,Parts,m![H%120,Qs%1024]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,Parts,m![H%120,Qs%1024]>=device.main.begin(packed.view())
        .fetch::<m![H%120,Qs/32%32],m![Qs%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H%120,Qs/16%64],m![Qs%16]>().commit_trim::<m![Qs%16]>().commit();
    let flat:HbmTensorView<'_,bf16,Chip,m![R64,Qs]>=unsafe{input.view().reshape()};
    let input:DmTensor<bf16,Chip,m![R64/16%2],Parts,m![R64%16,Qs%1024]>=flat
        .tile::<m![R64],32,m![R64=32 #64,Qs]>(0).to_dm(&mut device.tdma);
    let input:DmTensor<bf16,Chip,Cluster,Parts,m![R32%16,Qs%1024]>=unsafe{input.reshape()};
    let mut partial:DmTensor<bf16,Chip,Cluster,Parts,m![H%120,R32%16]>=DmTensor::new();
    {
        let x:TrfTensor<bf16,Chip,Cluster,Parts,m![R32%8],m![R32/8%2,Qs%1024]>=device.sub
            .begin(input.view())
            .fetch::<m![R32%8,R32/8%2,Qs/16%64],m![Qs%16]>().collect::<m![R32%8,R32/8%2,Qs/16%64],m![Qs%16]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H%120,Qs/16%64],m![Qs%16]>().collect::<m![H%120,Qs/16%64],m![Qs%16]>()
            .contract_outer::<m![H%120,Qs/32%32,R32/8%2],m![Qs%32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H%120,R32/8%2]>()
            .contract_lane::<m![H%120,R32/8%2],m![R32%8]>(LaneMode::Interleaved)
            .cast::<bf16,m![R32%8#16]>().commit_trim::<m![R32%8]>()
            .commit_view(partial.view_mut());
    }
    let gathered:DmTensor<bf16,Chip,Cluster,DownRows,m![Qs/1024,H%120,R32%16]>=partial.to_dm(&mut device.tdma);
    let p01=add_gathered_pair(device,&gathered,0);
    let p23=add_gathered_pair(device,&gathered,2);
    let result=add_partials_wide(device,&p01,&p23);
    scale_output_wide(device,&result,scale,output);
}

#[device(chip=1)]
pub fn project_sliding_partial_dots(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,Ns,Gs,Ds]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qs]>,output:&mut HbmTensor<bf16,Chip,m![R64,Qs/1024,H]>) {
    let packed:DmTensor<f8e4m3,Chip,Cluster,Parts,m![H%120,Qs%1024]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,Parts,m![H%120,Qs%1024]>=device.main.begin(packed.view())
        .fetch::<m![H%120,Qs/32%32],m![Qs%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H%120,Qs/16%64],m![Qs%16]>().commit_trim::<m![Qs%16]>().commit();
    let flat:HbmTensorView<'_,bf16,Chip,m![R64,Qs]>=unsafe{input.view().reshape()};
    let input:DmTensor<bf16,Chip,m![R64/16%2],Parts,m![R64%16,Qs%1024]>=flat
        .tile::<m![R64],32,m![R64=32 #64,Qs]>(0).to_dm(&mut device.tdma);
    let input:DmTensor<bf16,Chip,Cluster,Parts,m![R32%16,Qs%1024]>=unsafe{input.reshape()};
    let mut partial:DmTensor<bf16,Chip,Cluster,Parts,m![H%120,R32%16]>=DmTensor::new();
    {
        let x:TrfTensor<bf16,Chip,Cluster,Parts,m![R32%8],m![R32/8%2,Qs%1024]>=device.sub
            .begin(input.view())
            .fetch::<m![R32%8,R32/8%2,Qs/16%64],m![Qs%16]>().collect::<m![R32%8,R32/8%2,Qs/16%64],m![Qs%16]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H%120,Qs/16%64],m![Qs%16]>().collect::<m![H%120,Qs/16%64],m![Qs%16]>()
            .contract_outer::<m![H%120,Qs/32%32,R32/8%2],m![Qs%32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H%120,R32/8%2]>()
            .contract_lane::<m![H%120,R32/8%2],m![R32%8]>(LaneMode::Interleaved)
            .cast::<bf16,m![R32%8#16]>().commit_trim::<m![R32%8]>()
            .commit_view(partial.view_mut());
    }
    let partial:DmTensor<bf16,Chip,m![R64/16%2],Parts,m![H%120,R64%16]>=unsafe{partial.reshape()};
    partial.view().to_hbm_view(&mut device.tdma,output.view_mut()
        .tile::<m![R64],32,m![R64=32 #{!}64,Qs/1024,H]>(0));
}

fn add_gathered_pair_full(device:&mut Device,
    input:&DmTensor<bf16,Chip,Cluster,DownRows,m![Qf/1024,H%120,R32%16]>,offset:usize)
    -> DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R32 % 16]> {
    let mut result:DmTensor<bf16,Chip,Cluster,DownRows,m![H % 120,R32 % 16]>=DmTensor::new();
    {
        let left:VrfTensor<f32,Chip,Cluster,DownRows,m![H % 120,R32 % 16 = 16]>=device.sub
            .begin(input.view().tile::<m![Qf/1024],1,m![1 # 8,H%120,R32%16]>(offset).tile::<m![R32 % 16],16,m![H % 120 # 960,R32 % 16 = 16 # 16]>(0))
            .fetch::<m![H % 120],m![R32 % 16 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 120,R32 % 16 = 16 / 8],m![R32 % 16 = 16 % 8]>().to_vrf();
        device.main.begin(input.view().tile::<m![Qf/1024],1,m![1 # 8,H%120,R32%16]>(offset+1).tile::<m![R32 % 16],16,m![H % 120 # 960,R32 % 16 = 16 # 16]>(0))
            .fetch::<m![H % 120],m![R32 % 16 = 16]>().fetch_cast::<f32>()
            .collect::<m![H % 120,R32 % 16 = 16 / 8],m![R32 % 16 = 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add,&left).vector_final()
            .cast::<bf16,m![R32 % 16 = 16 % 8 # 16]>().commit_trim::<m![R32 % 16 = 16 % 8]>()
            .commit_view(result.view_mut().tile::<m![R32 % 16],16,m![H % 120,R32 % 16 = 16 #{!} 16]>(0));
    }
    result
}


#[device(chip=1)]
pub fn project_full_output_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,Gf,Df]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qf]>,scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![R64,H]>) {
    let packed:DmTensor<f8e4m3,Chip,Cluster,FullParts,m![H%120,Qf%1024]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,FullParts,m![H%120,Qf%1024]>=device.main.begin(packed.view())
        .fetch::<m![H%120,Qf/32%32],m![Qf%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H%120,Qf/16%64],m![Qf%16]>().commit_trim::<m![Qf%16]>().commit();
    let flat:HbmTensorView<'_,bf16,Chip,m![R64,Qf]>=unsafe{input.view().reshape()};
    let input:DmTensor<bf16,Chip,m![R64/16%2],FullParts,m![R64%16,Qf%1024]>=flat
        .tile::<m![R64],32,m![R64=32 #64,Qf]>(0).to_dm(&mut device.tdma);
    let input:DmTensor<bf16,Chip,Cluster,FullParts,m![R32%16,Qf%1024]>=unsafe{input.reshape()};
    let mut partial:DmTensor<bf16,Chip,Cluster,FullParts,m![H%120,R32%16]>=DmTensor::new();
    {
        let x:TrfTensor<bf16,Chip,Cluster,FullParts,m![R32%8],m![R32/8%2,Qf%1024]>=device.sub
            .begin(input.view())
            .fetch::<m![R32%8,R32/8%2,Qf/16%64],m![Qf%16]>().collect::<m![R32%8,R32/8%2,Qf/16%64],m![Qf%16]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H%120,Qf/16%64],m![Qf%16]>().collect::<m![H%120,Qf/16%64],m![Qf%16]>()
            .contract_outer::<m![H%120,Qf/32%32,R32/8%2],m![Qf%32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H%120,R32/8%2]>()
            .contract_lane::<m![H%120,R32/8%2],m![R32%8]>(LaneMode::Interleaved)
            .cast::<bf16,m![R32%8#16]>().commit_trim::<m![R32%8]>()
            .commit_view(partial.view_mut());
    }
    let gathered:DmTensor<bf16,Chip,Cluster,DownRows,m![Qf/1024,H%120,R32%16]>=partial.to_dm(&mut device.tdma);
    let p01=add_gathered_pair_full(device,&gathered,0);
    let p23=add_gathered_pair_full(device,&gathered,2);
    let p45=add_gathered_pair_full(device,&gathered,4);
    let p67=add_gathered_pair_full(device,&gathered,6);
    let p0123=add_partials_wide(device,&p01,&p23);
    let p4567=add_partials_wide(device,&p45,&p67);
    let result=add_partials_wide(device,&p0123,&p4567);
    scale_output_wide(device,&result,scale,output);
}


#[device(chip=1)]
pub fn project_full_partial_dots(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,Gf,Df]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qf]>,output:&mut HbmTensor<bf16,Chip,m![R64,Qf/1024,H]>) {
    let packed:DmTensor<f8e4m3,Chip,Cluster,FullParts,m![H%120,Qf%1024]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,FullParts,m![H%120,Qf%1024]>=device.main.begin(packed.view())
        .fetch::<m![H%120,Qf/32%32],m![Qf%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H%120,Qf/16%64],m![Qf%16]>().commit_trim::<m![Qf%16]>().commit();
    let flat:HbmTensorView<'_,bf16,Chip,m![R64,Qf]>=unsafe{input.view().reshape()};
    let input:DmTensor<bf16,Chip,m![R64/16%2],FullParts,m![R64%16,Qf%1024]>=flat
        .tile::<m![R64],32,m![R64=32 #64,Qf]>(0).to_dm(&mut device.tdma);
    let input:DmTensor<bf16,Chip,Cluster,FullParts,m![R32%16,Qf%1024]>=unsafe{input.reshape()};
    let mut partial:DmTensor<bf16,Chip,Cluster,FullParts,m![H%120,R32%16]>=DmTensor::new();
    {
        let x:TrfTensor<bf16,Chip,Cluster,FullParts,m![R32%8],m![R32/8%2,Qf%1024]>=device.sub
            .begin(input.view())
            .fetch::<m![R32%8,R32/8%2,Qf/16%64],m![Qf%16]>().collect::<m![R32%8,R32/8%2,Qf/16%64],m![Qf%16]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H%120,Qf/16%64],m![Qf%16]>().collect::<m![H%120,Qf/16%64],m![Qf%16]>()
            .contract_outer::<m![H%120,Qf/32%32,R32/8%2],m![Qf%32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H%120,R32/8%2]>()
            .contract_lane::<m![H%120,R32/8%2],m![R32%8]>(LaneMode::Interleaved)
            .cast::<bf16,m![R32%8#16]>().commit_trim::<m![R32%8]>()
            .commit_view(partial.view_mut());
    }
    let partial:DmTensor<bf16,Chip,m![R64/16%2],FullParts,m![H%120,R64%16]>=unsafe{partial.reshape()};
    partial.view().to_hbm_view(&mut device.tdma,output.view_mut()
        .tile::<m![R64],32,m![R64=32 #{!}64,Qf/1024,H]>(0));
}
