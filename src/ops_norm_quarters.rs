//! Four independent normalization quarters share one DM domain and output transfer.
//! H480 arithmetic and all BF16 boundaries remain unchanged.
use crate::{Chip, EPS, axes::{P128,P512,H,Dummy8}};
use crate::ops_batch_layouts::P512H960;
use crate::ops_output_p128_h64_blocked::Output;
use furiosa_opt_std::prelude::*;
axes![Q4=4];
type Cluster=m![P128/64];
type Shards=m![P128/2%32,H/480];
type Bridge=m![P128/2%32,H/960,1#2];
const H_F32:f32=H::SIZE as f32;
pub(crate) fn normalize(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Shards, m![Q4,P128%2,H % 480]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Shards, m![Q4,P128%2,H % 480]> {
    type ReducingSlices = Shards;

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![Q4,P128%2,1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Q4,P128%2,H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Q4,P128%2,H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Q4,P128%2,H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![Q4,P128%2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![P128/2%32,Dummy8], m![Q4,P128%2,1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![Q4,P128%2], m![1 # 8]>()
        .collect::<m![Q4,P128%2], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![P128/2%32,Dummy8], m![Q4,P128%2]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![P128/2%32,Dummy8], m![Q4,P128%2,1 # 8]> = device
        .main
        .begin(reduced_mean_square.view())
        .fetch::<m![Q4,P128%2], m![1 # 8]>()
        .collect::<m![Q4,P128%2], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![Q4,P128%2,1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> =
        rms_weight.to_dm(&mut device.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![Q4,P128%2,1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![Q4,P128%2], m![1 # 8]>()
        .collect::<m![Q4,P128%2], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<f32, Chip, Cluster, ReducingSlices, m![Q4,P128%2,H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![Q4,P128%2,H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Q4,P128%2,H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Q4,P128%2,H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Q4,P128%2,H / 8 % 60], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();

    device
        .main
        .begin(normalized.view())
        .fetch::<m![Q4,P128%2,H / 8 % 60], m![H % 8]>()
        .collect::<m![Q4,P128%2,H / 8 % 60], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
pub(crate) fn residual_add(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Shards, m![Q4,P128%2,H % 480]>,
    previous: &DmTensor<bf16, Chip, Cluster, Shards, m![Q4,P128%2,H%480]>,
) -> DmTensor<bf16, Chip, Cluster, Shards, m![Q4,P128%2,H % 480]> {
    let previous: VrfTensor<f32, Chip, Cluster, Shards, m![Q4,P128%2,H % 480]> = device
        .sub
        .begin(previous.view())
        .fetch::<m![Q4,P128%2,H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Q4,P128%2,H / 8 % 60], m![H % 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![Q4,P128%2,H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Q4,P128%2,H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &previous)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
type ScalarRows = m![P128/2%32,1#8];
fn quantize_shards(device:&mut Device,
    x:&DmTensor<bf16,Chip,Cluster,Shards,m![Q4,P128%2,H%480]>)
    -> (DmTensor<f8e4m3,Chip,Cluster,Shards,m![Q4,P128%2,H%480]>,
        DmTensor<f32,Chip,Cluster,ScalarRows,m![Q4,P128%2,1#8]>) {
    let amax:DmTensor<f32,Chip,Cluster,ScalarRows,m![Q4,P128%2,1#8]>=device.main.begin(x.view())
        .fetch::<m![Q4,P128%2,H/16%30],m![H%16]>().fetch_cast::<f32>()
        .collect::<m![Q4,P128%2,H/8%60],m![H%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32).vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Q4,P128%2,H/4%120],m![H%4]>()
        .vector_intra_slice_reduce::<H,m![Q4,P128%2],m![1#4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1#8]>()
        .vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
        .vector_inter_slice_reduce::<ScalarRows,m![Q4,P128%2]>(InterSliceReduceOpF32::Max)
        .vector_final().commit_trim::<m![1#8]>().commit();
    let scale:DmTensor<f32,Chip,Cluster,ScalarRows,m![Q4,P128%2,1#8]>=device.main.begin(amax.view())
        .fetch::<m![Q4,P128%2],m![1#8]>().collect::<m![Q4,P128%2],m![1#8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7f80_0000i32).vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1#4]>().vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),0.0078125f32)
        .vector_widen_pad::<m![1#8]>().vector_final().commit_trim::<m![1#8]>().commit();
    let broadcast:DmTensor<f32,Chip,Cluster,Shards,m![Q4,P128%2,1#8]>=scale.to_dm(&mut device.tdma);
    let sv:VrfTensor<f32,Chip,Cluster,Shards,m![Q4,P128%2,1#8]>=device.sub.begin(broadcast.view())
        .fetch::<m![Q4,P128%2],m![1#8]>().collect::<m![Q4,P128%2],m![1#8]>().to_vrf();
    let q:DmTensor<f8e4m3,Chip,Cluster,Shards,m![Q4,P128%2,H%480]>=device.main.begin(x.view())
        .fetch::<m![Q4,P128%2,H/16%30],m![H%16]>().fetch_cast::<f32>()
        .collect::<m![Q4,P128%2,H/8%60],m![H%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Q4,P128%2,H/4%120],m![H%4]>()
        .vector_fp_binary(FpBinaryOp::DivF,&sv)
        .vector_widen_concat::<m![Q4,P128%2,H/8%60],m![H%8]>()
        .vector_final().cast::<f8e4m3,m![H%8#32]>().commit_trim::<m![H%8]>().commit();
    (q,scale)
}

fn load_projected(device:&mut Device,input:&HbmTensor<bf16,Chip,Output>,group:usize,
    target:&mut DmTensor<bf16,Chip,Cluster,Bridge,m![Q4,P128%2,H%960]>) {
    input.view().to_dm_view(&mut device.tdma,
        target.view_mut().tile::<m![Q4],1,m![Q4=1#{!}4,P128%2,H%960]>(group));
}
fn load_residual(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,H]>,group:usize,
    target:&mut DmTensor<bf16,Chip,Cluster,Shards,m![Q4,P128%2,H%480]>) {
    input.view().to_dm_view(&mut device.tdma,
        target.view_mut().tile::<m![Q4],1,m![Q4=1#{!}4,P128%2,H%480]>(group));
}
fn store_residual(device:&mut Device,input:&DmTensor<bf16,Chip,Cluster,Shards,m![Q4,P128%2,H%480]>,group:usize,
    output:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    input.view().tile::<m![Q4],1,m![Q4=1#4,P128%2,H%480]>(group)
        .to_hbm_view(&mut device.tdma,output.view_mut());
}
#[device(chip=1)]
pub fn post_attention_ffn_input_four_blocked(device:&mut Device,
    a:&HbmTensor<bf16,Chip,Output>,b:&HbmTensor<bf16,Chip,Output>,
    c:&HbmTensor<bf16,Chip,Output>,d:&HbmTensor<bf16,Chip,Output>,
    post:&HbmTensor<bf16,Chip,m![H]>,pre:&HbmTensor<bf16,Chip,m![H]>,
    ra:&mut HbmTensor<bf16,Chip,m![P128,H]>,rb:&mut HbmTensor<bf16,Chip,m![P128,H]>,
    rc:&mut HbmTensor<bf16,Chip,m![P128,H]>,rd:&mut HbmTensor<bf16,Chip,m![P128,H]>,
    quantized:&mut HbmTensor<f8e4m3,Chip,P512H960>,scales:&mut HbmTensor<f32,Chip,m![P512]>) {
    let mut projected:DmTensor<bf16,Chip,Cluster,Bridge,m![Q4,P128%2,H%960]>=DmTensor::new();
    load_projected(device,a,0,&mut projected);load_projected(device,b,1,&mut projected);
    load_projected(device,c,2,&mut projected);load_projected(device,d,3,&mut projected);
    let projected:DmTensor<bf16,Chip,Cluster,Shards,m![Q4,P128%2,H%480]>=projected.to_dm(&mut device.tdma);
    let projected=normalize(device,&projected,post);
    let mut previous:DmTensor<bf16,Chip,Cluster,Shards,m![Q4,P128%2,H%480]>=DmTensor::new();
    load_residual(device,ra,0,&mut previous);load_residual(device,rb,1,&mut previous);
    load_residual(device,rc,2,&mut previous);load_residual(device,rd,3,&mut previous);
    let residual=residual_add(device,&projected,&previous);
    store_residual(device,&residual,0,ra);store_residual(device,&residual,1,rb);
    store_residual(device,&residual,2,rc);store_residual(device,&residual,3,rd);
    let pre=normalize(device,&residual,pre);
    let (q,scale)=quantize_shards(device,&pre);
    // Only relabel the independent quarter and row factors; no physical permutation.
    let q:DmTensor<f8e4m3,Chip,m![P512/64%2],m![P512/2%32,H/480],m![P512/128,P512%2,H%480]>=unsafe{q.reshape()};
    q.view().to_hbm_view(&mut device.tdma,quantized.view_mut());
    let scale:DmTensor<f32,Chip,m![P512/64%2],m![P512/2%32,1#8],m![P512/128,P512%2,1#8]>=unsafe{scale.reshape()};
    scale.view().to_hbm_view(&mut device.tdma,scales.view_mut());
}
