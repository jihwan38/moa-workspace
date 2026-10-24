//! Contracting-dimension partition experiment. It preserves the BF16 dot
//! boundary, but the preceding F32 reduction tree differs from the full-H dot.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{R64,H,Qs}};
pub type QueryWeight=m![Qs/64,H/960,Qs%64,H%960];
type Cluster=m![R64 / 32];
type Parts=m![Qs / 64,H / 960];
type Rows=m![Qs / 64,1 # 4];

#[device(chip = 1)]
pub fn project_query_batch(device:&mut Device,input:&HbmTensor<bf16,Chip,m![R64,H]>,
    weight:&HbmTensor<f8e4m3,Chip,QueryWeight>,scale:&HbmTensor<bf16,Chip,m![Qs]>,
    output:&mut HbmTensor<bf16,Chip,m![R64,Qs]>) {
    let weight:DmTensor<f8e4m3,Chip,Cluster,Parts,m![Qs % 64,H % 960]>=weight.to_dm(&mut device.tdma);
    let weight:DmTensor<bf16,Chip,Cluster,Parts,m![Qs % 64,H % 960]>=device.main.begin(weight.view())
        .fetch::<m![Qs % 64,H / 32 % 30],m![H % 32]>().fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 64,H / 16 % 60],m![H % 16]>().commit_trim::<m![H % 16]>().commit();
    let input:DmTensor<bf16,Chip,Cluster,Parts,m![R64 % 32,H % 960]>=input.to_dm(&mut device.tdma);
    let scale:DmTensor<bf16,Chip,Cluster,Rows,m![Qs % 64]>=scale.to_dm(&mut device.tdma);
    let scale:VrfTensor<f32,Chip,Cluster,Rows,m![Qs % 64]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![Qs % 64]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 8],m![Qs % 8]>().to_vrf();
    let mut result:DmTensor<bf16,Chip,Cluster,Rows,m![Qs % 64,R64 % 32]>=DmTensor::new();
    for chunk in 0..4 {
        let x:TrfTensor<bf16,Chip,Cluster,Parts,m![R64 % 32 = 8],m![H % 960]>=device.sub
            .begin(input.view().tile::<m![R64 % 32],8,m![R64 % 32 = 8 # 32,H % 960]>(8*chunk))
            .fetch::<m![R64 % 32 = 8,H / 16 % 60],m![H % 16]>()
            .collect::<m![R64 % 32 = 8,H / 16 % 60],m![H % 16]>().to_trf();
        let dot:DmTensor<bf16,Chip,Cluster,Rows,m![Qs % 64,R64 % 32 = 8]>=device.main.begin(weight.view())
            .fetch::<m![Qs % 64,H / 16 % 60],m![H % 16]>()
            .collect::<m![Qs % 64,H / 16 % 60],m![H % 16]>()
            .contract_outer::<m![Qs % 64,H / 32 % 30],m![H % 32],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![Qs % 64]>()
            .contract_lane::<m![Qs % 64],m![R64 % 32 = 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<Rows,m![Qs % 64]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16,m![R64 % 32 = 8 # 16]>().commit_trim::<m![R64 % 32 = 8]>().commit();
        device.main.begin(dot.view()).fetch::<m![Qs % 64],m![R64 % 32 = 8]>().fetch_cast::<f32>()
            .collect::<m![Qs % 64],m![R64 % 32 = 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Qs % 64,R64 % 32 = 8 / 4],m![R64 % 32 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
            .vector_widen_concat::<m![Qs % 64],m![R64 % 32 = 8]>()
            .vector_final().cast::<bf16,m![R64 % 32 = 8 # 16]>().commit_trim::<m![R64 % 32 = 8]>()
            .commit_view(result.view_mut().tile::<m![R64 % 32],8,m![Qs % 64,R64 % 32 = 8 #{!} 32]>(8*chunk));
    }
    let result:DmTensor<bf16,Chip,Cluster,Rows,m![R64 % 32,Qs % 64]>=device.main.begin(result.view())
        .fetch::<m![R64 / 16 % 2,Qs % 64],m![R64 % 16]>()
        .collect::<m![R64 / 16 % 2,Qs % 64],m![R64 % 16]>()
        .transpose::<m![R64 / 16 % 2,Qs / 4 % 16,R64 % 16],m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}


/// Startup permutation only; source E4 values are unchanged.
#[device(chip=1)]
pub fn reblock_query(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![Qs,H]>,output:&mut HbmTensor<f8e4m3,Chip,QueryWeight>) {
    let weights:DmTensor<f8e4m3,Chip,m![Qs/2048],m![Qs/64%32,H/960,1 # 2],m![Qs%64,H%960]>=input.to_dm(&mut device.tdma);
    weights.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
