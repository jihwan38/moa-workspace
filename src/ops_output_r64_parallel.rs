//! Exact output projection with K1024 partials parallel across slices.
//! Every partial and every pairwise sum still rounds to BF16.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{R64,H,Qs,Ns,Gs,Ds}};
type Cluster=m![R64/32];
type Parts=m![H/120,1 # 2,Qs/1024];
type Pairs=m![H/120,1 # 4,Qs/2048];
type DownRows=m![H/120,1 # 8];
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

#[device(chip=1)]
pub fn project_sliding_output_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,Ns,Gs,Ds]>,
    weight:&HbmTensor<f8e4m3,Chip,m![H,Qs]>,scale:&HbmTensor<bf16,Chip,m![H]>,output:&mut HbmTensor<bf16,Chip,m![R64,H]>) {
    let packed:DmTensor<f8e4m3,Chip,Cluster,Parts,m![H%120,Qs%1024]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,Parts,m![H%120,Qs%1024]>=device.main.begin(packed.view())
        .fetch::<m![H%120,Qs/32%32],m![Qs%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H%120,Qs/16%64],m![Qs%16]>().commit_trim::<m![Qs%16]>().commit();
    let flat:HbmTensorView<'_,bf16,Chip,m![R64,Qs]>=unsafe{input.view().reshape()};
    let input:DmTensor<bf16,Chip,Cluster,Parts,m![R64%32,Qs%1024]>=flat.to_dm(&mut device.tdma);
    let mut partial:DmTensor<bf16,Chip,Cluster,Parts,m![H%120,R64%32]>=DmTensor::new();
    for chunk in 0..4 {
        let x:TrfTensor<bf16,Chip,Cluster,Parts,m![R64%32=8],m![Qs%1024]>=device.sub
            .begin(input.view().tile::<m![R64%32],8,m![R64%32=8#32,Qs%1024]>(8*chunk))
            .fetch::<m![R64%32=8,Qs/16%64],m![Qs%16]>().collect::<m![R64%32=8,Qs/16%64],m![Qs%16]>().to_trf();
        device.main.begin(weight.view())
            .fetch::<m![H%120,Qs/16%64],m![Qs%16]>().collect::<m![H%120,Qs/16%64],m![Qs%16]>()
            .contract_outer::<m![H%120,Qs/32%32],m![Qs%32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H%120]>()
            .contract_lane::<m![H%120],m![R64%32=8]>(LaneMode::Interleaved)
            .cast::<bf16,m![R64%32=8#16]>().commit_trim::<m![R64%32=8]>()
            .commit_view(partial.view_mut().tile::<m![R64%32],8,m![H%120,R64%32=8#{!}32]>(8*chunk));
    }
    let pairs:DmTensor<bf16,Chip,Cluster,Pairs,m![H%120,R64%32]>=device.main.begin(partial.view())
        .fetch::<m![H%120,R64/8%4],m![R64%8]>().fetch_cast::<f32>()
        .collect::<m![H%120,R64/8%4],m![R64%8]>()
        .vector_init().vector_inter_slice_reduce::<Pairs,m![H%120,R64/8%4]>(InterSliceReduceOpF32::Add)
        .vector_final().cast::<bf16,m![R64%8#16]>().commit_trim::<m![R64%8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,DownRows,m![H%120,R64%32]>=device.main.begin(pairs.view())
        .fetch::<m![H%120,R64/8%4],m![R64%8]>().fetch_cast::<f32>()
        .collect::<m![H%120,R64/8%4],m![R64%8]>()
        .vector_init().vector_inter_slice_reduce::<DownRows,m![H%120,R64/8%4]>(InterSliceReduceOpF32::Add)
        .vector_final().cast::<bf16,m![R64%8#16]>().commit_trim::<m![R64%8]>().commit();
    scale_output_wide(device,&result,scale,output);
}
