//! Startup FP8 weight prepack from the provided FP4 and E4 block scales.
//! Reconstruct as q * weight_dequant_scale; original global scale stays post-dot.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{H,L}};

type UpCluster=m![L / 7680];
type UpSlice=m![L / 30 % 256 # 256];
#[device(chip = 1)]
pub fn prepack_up(device:&mut Device,packed:&HbmTensor<f4e2m1,Chip,m![L,H]>,
    block_scales:&HbmTensor<f8e4m3,Chip,m![L,H / 16]>,
    quantized:&mut HbmTensor<f8e4m3,Chip,m![L,H]>,
    weight_dequant_scale:&mut HbmTensor<f32,Chip,m![L]>) {
    let packed:DmTensor<f4e2m1,Chip,UpCluster,UpSlice,m![L % 30,H]>=packed.to_dm(&mut device.tdma);
    let decoded:DmTensor<f8e4m3,Chip,UpCluster,UpSlice,m![L % 30,H]>=device.main.begin(packed.view())
        .fetch::<m![L % 30],m![H]>().fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 30,H / 32],m![H % 32]>().commit_trim::<m![H % 32]>().commit();
    let scales:DmTensor<f8e4m3,Chip,UpCluster,UpSlice,m![L % 30,H / 16]>=block_scales.to_dm(&mut device.tdma);
    let mut weights:DmTensor<bf16,Chip,UpCluster,UpSlice,m![L % 30,H]>=DmTensor::new();
    for row in 0..6 {
        let scales:VrfTensor<f32,Chip,UpCluster,UpSlice,m![L % 30 = 5,H / 16]>=device.sub
            .begin(scales.view().tile::<m![L % 30],5,m![L % 30 = 5 # 30,H / 16]>(5*row))
            .fetch::<m![L % 30 = 5],m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![L % 30 = 5,H / 128],m![H / 16 % 8]>().to_vrf();
        device.main.begin(decoded.view().tile::<m![L % 30],5,m![L % 30 = 5 # 30,H]>(5*row))
            .fetch::<m![L % 30 = 5,H / 32],m![H % 32]>().fetch_cast::<f32>()
            .collect::<m![L % 30 = 5,H / 8],m![H % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 30 = 5,H / 4],m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scales)
            .vector_widen_concat::<m![L % 30 = 5,H / 8],m![H % 8]>()
            .vector_final().cast::<bf16,m![H % 8 # 16]>().commit_trim::<m![H % 8]>()
            .commit_view(weights.view_mut().tile::<m![L % 30],5,m![L % 30 = 5 #{!} 30,H]>(5*row));
    }
    let amax:DmTensor<f32,Chip,UpCluster,UpSlice,m![L % 30]>=device.main.begin(weights.view())
        .fetch::<m![L % 30,H / 16],m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![L % 30,H / 8],m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32).vector_reinterpret::<f32>()
        .vector_narrow_split::<m![L % 30,H / 4],m![H % 4]>()
        .vector_intra_slice_reduce::<H,m![L % 30],m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>().vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
        .vector_final().transpose::<m![L % 30 / 2],m![L % 30 % 2 # 8]>()
        .commit_trim::<m![L % 30 % 2]>().commit();
    let scale:DmTensor<f32,Chip,UpCluster,UpSlice,m![L % 30]>=device.main.begin(amax.view())
        .fetch::<m![L % 30 / 2],m![L % 30 % 2 # 8]>()
        .collect::<m![L % 30 / 2],m![L % 30 % 2 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7f80_0000i32).vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![L % 30 % 2]>().vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),0.0078125f32)
        .vector_widen_pad::<m![L % 30 % 2 # 8]>().vector_final().commit_trim::<m![L % 30 % 2]>().commit();
    let sv:VrfTensor<f32,Chip,UpCluster,UpSlice,m![L % 30 # 32]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![L % 30]>()
        .collect::<m![(L % 30 # 32) / 8],m![(L % 30 # 32) % 8]>().to_vrf();
    let q:DmTensor<f8e4m3,Chip,UpCluster,UpSlice,m![L % 30,H]>=device.main.begin(weights.view())
        .fetch::<m![L % 30,H / 16],m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![L % 30,H / 8],m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 30,H / 4],m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF,&sv)
        .vector_widen_concat::<m![L % 30,H / 8],m![H % 8]>().vector_final()
        .cast::<f8e4m3,m![H % 8 # 32]>().commit_trim::<m![H % 8]>().commit();
    q.view().to_hbm_view(&mut device.tdma,quantized.view_mut());
    scale.view().to_hbm_view(&mut device.tdma,weight_dequant_scale.view_mut());
}

type DownCluster=m![H / 1920];
type DownSlice=m![H / 30 % 64,1 # 4];
#[device(chip = 1)]
pub fn prepack_down(device:&mut Device,packed:&HbmTensor<f4e2m1,Chip,m![H,L]>,
    block_scales:&HbmTensor<f8e4m3,Chip,m![H,L / 16]>,
    quantized:&mut HbmTensor<f8e4m3,Chip,m![H,L]>,
    weight_dequant_scale:&mut HbmTensor<f32,Chip,m![H]>) {
    for pass in 0..15 {
    let packed:DmTensor<f4e2m1,Chip,DownCluster,DownSlice,m![H % 30 = 2,L]>=packed.view().tile::<m![H % 30],2,m![H / 30,H % 30 = 2 # 30,L]>(2*pass).to_dm(&mut device.tdma);
    let decoded:DmTensor<f8e4m3,Chip,DownCluster,DownSlice,m![H % 30 = 2,L]>=device.main.begin(packed.view())
        .fetch::<m![H % 30 = 2],m![L]>().fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 30 = 2,L / 32],m![L % 32]>().commit_trim::<m![L % 32]>().commit();
    let scales:DmTensor<f8e4m3,Chip,DownCluster,DownSlice,m![H % 30 = 2,L / 16]>=block_scales.view().tile::<m![H % 30],2,m![H / 30,H % 30 = 2 # 30,L / 16]>(2*pass).to_dm(&mut device.tdma);
    let mut weights:DmTensor<bf16,Chip,DownCluster,DownSlice,m![H % 30 = 2,L]>=DmTensor::new();
    {
        let scales:VrfTensor<f32,Chip,DownCluster,DownSlice,m![H % 30 = 2 = 2,L / 16]>=device.sub
            .begin(scales.view().tile::<m![H % 30 = 2],2,m![H % 30 = 2 = 2 # 2,L / 16]>(0))
            .fetch::<m![H % 30 = 2 = 2],m![L / 16]>().fetch_cast::<f32>()
            .collect::<m![H % 30 = 2 = 2,L / 128],m![L / 16 % 8]>().to_vrf();
        device.main.begin(decoded.view().tile::<m![H % 30 = 2],2,m![H % 30 = 2 = 2 # 2,L]>(0))
            .fetch::<m![H % 30 = 2 = 2,L / 32],m![L % 32]>().fetch_cast::<f32>()
            .collect::<m![H % 30 = 2 = 2,L / 8],m![L % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 30 = 2 = 2,L / 4],m![L % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scales)
            .vector_widen_concat::<m![H % 30 = 2 = 2,L / 8],m![L % 8]>()
            .vector_final().cast::<bf16,m![L % 8 # 16]>().commit_trim::<m![L % 8]>()
            .commit_view(weights.view_mut().tile::<m![H % 30 = 2],2,m![H % 30 = 2 = 2 #{!} 2,L]>(0));
    }
    let amax:DmTensor<f32,Chip,DownCluster,DownSlice,m![H % 30 = 2]>=device.main.begin(weights.view())
        .fetch::<m![H % 30 = 2,L / 16],m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![H % 30 = 2,L / 8],m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32).vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H % 30 = 2,L / 4],m![L % 4]>()
        .vector_intra_slice_reduce::<L,m![H % 30 = 2],m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>().vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
        .vector_final().transpose::<m![H % 30 = 2 / 2],m![H % 30 = 2 % 2 # 8]>()
        .commit_trim::<m![H % 30 = 2 % 2]>().commit();
    let scale:DmTensor<f32,Chip,DownCluster,DownSlice,m![H % 30 = 2]>=device.main.begin(amax.view())
        .fetch::<m![H % 30 = 2 / 2],m![H % 30 = 2 % 2 # 8]>()
        .collect::<m![H % 30 = 2 / 2],m![H % 30 = 2 % 2 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7f80_0000i32).vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![H % 30 = 2 % 2]>().vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),0.0078125f32)
        .vector_widen_pad::<m![H % 30 = 2 % 2 # 8]>().vector_final().commit_trim::<m![H % 30 = 2 % 2]>().commit();
    let sv:VrfTensor<f32,Chip,DownCluster,DownSlice,m![H % 30 = 2 # 8]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![H % 30 = 2]>()
        .collect::<m![(H % 30 = 2 # 8) / 8],m![(H % 30 = 2 # 8) % 8]>().to_vrf();
    let q:DmTensor<f8e4m3,Chip,DownCluster,DownSlice,m![H % 30 = 2,L]>=device.main.begin(weights.view())
        .fetch::<m![H % 30 = 2,L / 16],m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![H % 30 = 2,L / 8],m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 30 = 2,L / 4],m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF,&sv)
        .vector_widen_concat::<m![H % 30 = 2,L / 8],m![L % 8]>().vector_final()
        .cast::<f8e4m3,m![L % 8 # 32]>().commit_trim::<m![L % 8]>().commit();
    q.view().to_hbm_view(&mut device.tdma,quantized.view_mut().tile::<m![H % 30],2,m![H / 30,H % 30 = 2 #{!} 30,L]>(2*pass));
    scale.view().to_hbm_view(&mut device.tdma,weight_dequant_scale.view_mut().tile::<m![H % 30],2,m![H / 30,H % 30 = 2 #{!} 30]>(2*pass));
    }
}
