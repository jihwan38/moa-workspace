//! Predecoded-BF16 control for H960 P128 FFN: original FP4×E4 products,
//! same H960 reduction tree and all existing BF16/global/GeGLU boundaries.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,H,L}};
type Cluster=m![P128/64];
type UpParts=m![L/240,H/960];
type UpReduced=m![L/240,1 # 4];
type UpRows=m![L/60];
type DownColumns=m![H/120,L/1920];
type DownRows=m![H/120,1 # 8];
fn upgate_predecoded(device:&mut Device,input:&DmTensor<bf16,Chip,Cluster,UpParts,m![P128 % 64,H % 960]>,
    weight:&HbmTensor<bf16,Chip,m![L,H]>)
    ->DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64]> {
    let mut result:DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240,P128 % 64]>=DmTensor::new();
    for pass in 0..2 {
        let weights:DmTensor<bf16,Chip,Cluster,UpParts,m![L % 240 = 120,H % 960]>=weight.view()
            .tile::<m![L % 240],120,m![L / 240,L % 240 = 120 # 240,H]>(120*pass).to_dm(&mut device.tdma);
        let mut partial:DmTensor<bf16,Chip,Cluster,UpReduced,m![L % 240 = 120,P128 % 64]>=DmTensor::new();
        for chunk in 0..8 {
            let x:TrfTensor<bf16,Chip,Cluster,UpParts,m![P128 % 64 = 8],m![H % 960]>=device.sub
                .begin(input.view().tile::<m![P128 % 64],8,m![P128 % 64 = 8 # 64,H % 960]>(8*chunk))
                .fetch::<m![P128 % 64 = 8,H / 16 % 60],m![H % 16]>()
                .collect::<m![P128 % 64 = 8,H / 16 % 60],m![H % 16]>().to_trf();
            device.main.begin(weights.view())
                .fetch::<m![L % 240 = 120,H / 16 % 60],m![H % 16]>()
                .collect::<m![L % 240 = 120,H / 16 % 60],m![H % 16]>()
                .contract_outer::<m![L % 240 = 120,H / 32 % 30],m![H % 32],_,_,_>(&x)
                .contract_packet::<m![1]>().contract_time::<m![L % 240 = 120]>()
                .contract_lane::<m![L % 240 = 120],m![P128 % 64 = 8]>(LaneMode::Interleaved)
                .vector_init().vector_inter_slice_reduce::<UpReduced,m![L % 240 = 120]>(InterSliceReduceOpF32::Add)
                .vector_final().cast::<bf16,m![P128 % 64 = 8 # 16]>().commit_trim::<m![P128 % 64 = 8]>()
                .commit_view(partial.view_mut().tile::<m![P128 % 64],8,m![L % 240 = 120,P128 % 64 = 8 #{!} 64]>(8*chunk));
        }
        partial.view().to_dm_view(&mut device.tdma,result.view_mut().tile::<m![L % 240],120,m![L % 240 = 120 #{!} 240,P128 % 64]>(120*pass));
    }
    result
}


fn down_predecoded(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, UpRows, m![L % 60, P128 % 64]>,
    weight: &HbmTensor<bf16, Chip, m![H, L]>,
    global: &HbmTensor<f32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![P128, H]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, UpRows, m![P128 % 64, L % 60]> = device
        .main
        .begin(input.view())
        .fetch::<m![P128 / 16 % 4, L % 60], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, L % 60], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, L / 4 % 15, P128 % 16], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();
    let input:DmTensor<bf16,Chip,Cluster,DownColumns,m![P128 % 64,L % 1920]>=input.to_dm(&mut device.tdma);
    let mut result: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, P128 % 64]> =
        DmTensor::new();
    for pass in 0..3 {
        let weights:DmTensor<bf16,Chip,Cluster,DownColumns,m![H % 120 = 40,L % 1920]>=weight.view()
            .tile::<m![H % 120],40,m![H / 120,H % 120 = 40 # 120,L]>(40*pass).to_dm(&mut device.tdma);
        let mut partial: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120 = 40, P128 % 64]> =
            DmTensor::new();
        for chunk in 0..8 {
            let x:TrfTensor<bf16,Chip,Cluster,DownColumns,m![P128 % 64 = 8],m![L % 1920]>=device.sub
                .begin(input.view().tile::<m![P128 % 64],8,m![P128 % 64 = 8 # 64,L % 1920]>(8*chunk))
                .fetch::<m![P128 % 64 = 8,L / 16 % 120],m![L % 16]>()
                .collect::<m![P128 % 64 = 8,L / 16 % 120],m![L % 16]>().to_trf();
            device
                .main
                .begin(weights.view())
                .fetch::<m![H % 120 = 40, L / 16 % 120], m![L % 16]>()
                .collect::<m![H % 120 = 40, L / 16 % 120], m![L % 16]>()
                .contract_outer::<m![H % 120 = 40, L / 32 % 60], m![L % 32], _, _, _>(&x)
                .contract_packet::<m![1]>()
                .contract_time::<m![H % 120 = 40]>()
                .contract_lane::<m![H % 120 = 40], m![P128 % 64 = 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<DownRows, m![H % 120 = 40]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![P128 % 64 = 8 # 16]>()
                .commit_trim::<m![P128 % 64 = 8]>()
                .commit_view(
                    partial
                        .view_mut()
                        .tile::<m![P128 % 64], 8, m![H % 120 = 40,P128 % 64 = 8 #{!} 64]>(
                            8 * chunk,
                        ),
                );
        }
        partial.view().to_dm_view(
            &mut device.tdma,
            result
                .view_mut()
                .tile::<m![H % 120], 40, m![H % 120 = 40 #{!} 120,P128 % 64]>(40 * pass),
        );
    }
    let global: DmTensor<f32, Chip, Cluster, DownRows, m![1 # 8]> = global.to_dm(&mut device.tdma);
    let global: VrfTensor<f32, Chip, Cluster, DownRows, m![1 # 8]> = device
        .sub
        .begin(global.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, P128 % 64]> = device
        .main
        .begin(result.view())
        .fetch::<m![H % 120, P128 / 8 % 8], m![P128 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120, P128 / 8 % 8], m![P128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &global)
        .vector_widen_concat::<m![H % 120, P128 / 8 % 8], m![P128 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128 % 8 # 16]>()
        .commit_trim::<m![P128 % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, DownRows, m![P128 % 64, H % 120]> = device
        .main
        .begin(result.view())
        .fetch::<m![P128 / 16 % 4, H % 120], m![P128 % 16]>()
        .collect::<m![P128 / 16 % 4, H % 120], m![P128 % 16]>()
        .transpose::<m![P128 / 16 % 4, H / 4 % 30, P128 % 16], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn batch_ffn(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,H]>,
    up:&HbmTensor<bf16,Chip,m![L,H]>,gate:&HbmTensor<bf16,Chip,m![L,H]>,down:&HbmTensor<bf16,Chip,m![H,L]>,
    ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
    output:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    let input:DmTensor<bf16,Chip,Cluster,UpParts,m![P128 % 64,H % 960]>=input.to_dm(&mut device.tdma);
    let up=upgate_predecoded(device,&input,up);
    let gate=upgate_predecoded(device,&input,gate);
    let hidden=crate::ops_batch_kpart::geglu_k4(device,&up,&gate,ug,gg);
    let hidden:DmTensor<bf16,Chip,Cluster,UpRows,m![L % 60,P128 % 64]>=hidden.to_dm(&mut device.tdma);
    down_predecoded(device,&hidden,down,dg,output);
}
