//! Independent audio frames share projection weights; the scalar640-wide RMS
//! reduction and unsplit BF16 dot are retained. No frame/cache reuse.
use furiosa_opt_std::prelude::*;
use crate::{Chip, EPS, axes::{Aa,B,P128,H}};
const AA_F32:f32=Aa::SIZE as f32;
type HiddenRows=m![H/120,1#8];
fn mean_square<Cl: M, Sl: M>(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cl, Sl, m![Aa]>,
) -> DmTensor<f32, Chip, Cl, Sl, m![1 # 8]> {
    device
        .main
        .begin(input.view())
        .fetch::<m![Aa / 16], m![Aa % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Aa / 8], m![Aa % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Aa / 4], m![Aa % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Aa, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(AA_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn rms<Cl: M, Sl: M>(
    device: &mut Device,
    mean_square: &DmTensor<f32, Chip, Cl, Sl, m![1 # 8]>,
) -> DmTensor<f32, Chip, Cl, Sl, m![1 # 8]> {
    device
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn rmsnorm_without_weight<Cl: M, Sl: M>(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cl, Sl, m![Aa]>,
) -> DmTensor<bf16, Chip, Cl, Sl, m![Aa]> {
    let mean_square = mean_square(device, input);
    let rms = rms(device, &mean_square);
    let rms_vrf: VrfTensor<f32, Chip, Cl, Sl, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    device
        .main
        .begin(input.view())
        .fetch::<m![Aa / 16], m![Aa % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Aa / 8], m![Aa % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Aa / 4], m![Aa % 4]>()
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Aa / 8], m![Aa % 8]>()
        .vector_final()
        .cast::<bf16, m![Aa % 8 # 16]>()
        .commit_trim::<m![Aa % 8]>()
        .commit()
}

fn project_b16(device:&mut Device,input:&HbmTensor<bf16,Chip,m![B,Aa]>,
 weight:&HbmTensor<bf16,Chip,m![H,Aa]>,out:&mut HbmTensor<bf16,Chip,m![B,H]>) {
 type Cluster=m![B/8];
 type NormRows=m![B%8,1#32];
 let input:DmTensor<bf16,Chip,Cluster,NormRows,m![Aa]>=input.to_dm(&mut device.tdma);
 let normalized=rmsnorm_without_weight(device,&input);
 let input:DmTensor<bf16,Chip,Cluster,HiddenRows,m![B%8,Aa]>=normalized.to_dm(&mut device.tdma);
 let weight:DmTensor<bf16,Chip,Cluster,HiddenRows,m![H%120,Aa]>=weight.to_dm(&mut device.tdma);
 let x:TrfTensor<bf16,Chip,Cluster,HiddenRows,m![B%8],m![Aa]>=device.sub.begin(input.view())
  .fetch::<m![B%8,Aa/16],m![Aa%16]>().collect::<m![B%8,Aa/16],m![Aa%16]>().to_trf();
 let dot:DmTensor<bf16,Chip,Cluster,HiddenRows,m![H%120,B%8]>=device.main.begin(weight.view())
  .fetch::<m![H%120,Aa/16],m![Aa%16]>().collect::<m![H%120,Aa/16],m![Aa%16]>()
  .contract_outer::<m![H%120,Aa/32],m![Aa%32],_,_,_>(&x)
  .contract_packet::<m![1]>().contract_time::<m![H%120]>()
  .contract_lane::<m![H%120],m![B%8]>(LaneMode::Interleaved)
  .cast::<bf16,m![B%8#16]>().commit_trim::<m![B%8]>().commit();
 let result:DmTensor<bf16,Chip,Cluster,HiddenRows,m![B%8,H%120]>=device.main.begin(dot.view())
  .fetch::<m![H%120],m![B%8#16]>().collect::<m![H%120],m![B%8#16]>()
  .transpose::<m![H/4%30,B%8],m![H%4#16]>().commit_trim::<m![H%4]>().commit();
 result.view().to_hbm_view(&mut device.tdma,out.view_mut());
}
#[device(chip=1)]
pub fn audio_project_b16(device:&mut Device,input:&HbmTensor<bf16,Chip,m![B,Aa]>,
 weight:&HbmTensor<bf16,Chip,m![H,Aa]>) -> HbmTensor<bf16,Chip,m![B,H]> {
 let mut out=HbmTensor::new();project_b16(device,input,weight,&mut out);out
}
#[device(chip=1)]
pub fn audio_project_b16_into(device:&mut Device,input:&HbmTensor<bf16,Chip,m![B,Aa]>,
 weight:&HbmTensor<bf16,Chip,m![H,Aa]>,out:&mut HbmTensor<bf16,Chip,m![B,H]>) {
 project_b16(device,input,weight,out)
}
#[device(chip=1)]
pub fn normalize_b16(device:&mut Device,input:&HbmTensor<bf16,Chip,m![B,Aa]>,
 out:&mut HbmTensor<bf16,Chip,m![B,Aa]>) {
 let input:DmTensor<bf16,Chip,m![B/8],m![B%8,1#32],m![Aa]>=input.to_dm(&mut device.tdma);
 let normalized=rmsnorm_without_weight(device,&input);
 normalized.view().to_hbm_view(&mut device.tdma,out.view_mut());
}

fn project_p128(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,Aa]>,
 weight:&HbmTensor<bf16,Chip,m![H,Aa]>,out:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
 type Cluster=m![P128/64];
 type NormRows=m![P128%64,1#4];
 let input:DmTensor<bf16,Chip,Cluster,NormRows,m![Aa]>=input.to_dm(&mut device.tdma);
 let normalized=rmsnorm_without_weight(device,&input);
 let input:DmTensor<bf16,Chip,Cluster,HiddenRows,m![P128%64,Aa]>=normalized.to_dm(&mut device.tdma);
 let weight:DmTensor<bf16,Chip,Cluster,HiddenRows,m![H%120,Aa]>=weight.to_dm(&mut device.tdma);
 let mut dot:DmTensor<bf16,Chip,Cluster,HiddenRows,m![H%120,P128%64]>=DmTensor::new();
 for chunk in 0..2 {
  let x:TrfTensor<bf16,Chip,Cluster,HiddenRows,m![P128%8],m![P128/8%4,Aa]>=device.sub
   .begin(input.view().tile::<m![P128/32%2],1,m![1#2,P128%32,Aa]>(chunk))
   .fetch::<m![P128%8,P128/8%4,Aa/16],m![Aa%16]>()
   .collect::<m![P128%8,P128/8%4,Aa/16],m![Aa%16]>().to_trf();
  device.main.begin(weight.view())
   .fetch::<m![H%120,Aa/16],m![Aa%16]>().collect::<m![H%120,Aa/16],m![Aa%16]>()
   .contract_outer::<m![H%120,Aa/32,P128/8%4],m![Aa%32],_,_,_>(&x)
   .contract_packet::<m![1]>().contract_time::<m![H%120,P128/8%4]>()
   .contract_lane::<m![H%120,P128/8%4],m![P128%8]>(LaneMode::Interleaved)
   .cast::<bf16,m![P128%8#16]>().commit_trim::<m![P128%8]>()
   .commit_view(dot.view_mut().tile::<m![P128/32%2],1,m![H%120,1#{!}2,P128%32]>(chunk));
 }
 let result:DmTensor<bf16,Chip,Cluster,HiddenRows,m![P128%64,H%120]>=device.main.begin(dot.view())
  .fetch::<m![P128/16%4,H%120],m![P128%16]>().collect::<m![P128/16%4,H%120],m![P128%16]>()
  .transpose::<m![P128/16%4,H/4%30,P128%16],m![H%4#16]>().commit_trim::<m![H%4]>().commit();
 result.view().to_hbm_view(&mut device.tdma,out.view_mut());
}
#[device(chip=1)]
pub fn audio_project_p128(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,Aa]>,
 weight:&HbmTensor<bf16,Chip,m![H,Aa]>) -> HbmTensor<bf16,Chip,m![P128,H]> {
 let mut out=HbmTensor::new();project_p128(device,input,weight,&mut out);out
}
#[device(chip=1)]
pub fn audio_project_p128_into(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,Aa]>,
 weight:&HbmTensor<bf16,Chip,m![H,Aa]>,out:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
 project_p128(device,input,weight,out)
}
#[device(chip=1)]
pub fn normalize_p128(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,Aa]>,
 out:&mut HbmTensor<bf16,Chip,m![P128,Aa]>) {
 let input:DmTensor<bf16,Chip,m![P128/64],m![P128%64,1#4],m![Aa]>=input.to_dm(&mut device.tdma);
 let normalized=rmsnorm_without_weight(device,&input);
 normalized.view().to_hbm_view(&mut device.tdma,out.view_mut());
}
