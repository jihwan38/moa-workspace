//! B8-TRF vision projections; preserve each K768 BF16 partial and balanced sum tree.
use furiosa_opt_std::prelude::*;
use crate::axes::{B,H,Mv,Ov,Rv};
use crate::ops_vision_trf::{Cluster,Slice};
use crate::{Chip,EPS};
axes![V32=32];
type Replicated=m![V32,1#8];
const OV_F32:f32=Ov::SIZE as f32;
type PatchRows=m![Mv/120,1#8];
fn patch_broadcast(device:&mut Device,x:&DmTensor<bf16,Chip,Cluster,Slice,m![Rv]>)
 ->DmTensor<bf16,Chip,Cluster,Replicated,m![B%8,Rv]> {
    // Move the eight independently owned rows into the TRF lane axis and
    // broadcast to32 actual output groups. V32 is live;1#8 remains padding.
    x.to_dm(&mut device.tdma)
}
fn patch_partial(device:&mut Device,x:&DmTensor<bf16,Chip,Cluster,Replicated,m![B%8,Rv]>,
 weight:&HbmTensor<bf16,Chip,m![Mv,Rv]>,offset:usize)
 ->DmTensor<bf16,Chip,Cluster,PatchRows,m![Mv%120,B%8]> {
    let x:DmTensorView<'_,bf16,Chip,Cluster,PatchRows,m![B%8,Rv]>=unsafe{x.view().reshape()};
    let x=x.tile::<m![Rv],768,m![B%8,Rv=768#6912]>(offset);
    // The SDK TRF fetch cannot retain the outer token stride of a cropped Kview.
    // Materialize exactly this K768 tile before the shared-lane contraction.
    let mut compact:DmTensor<bf16,Chip,Cluster,PatchRows,m![B%8,Rv=768]>=DmTensor::new();
    x.to_dm_view(&mut device.tdma,compact.view_mut());
    let x:TrfTensor<bf16,Chip,Cluster,PatchRows,m![B%8],m![Rv=768]>=device.sub.begin(compact.view())
      .fetch::<m![B%8],m![Rv=768]>()
      .collect::<m![B%8,Rv=768/16],m![Rv=768%16]>().to_trf();
    let weight:DmTensor<bf16,Chip,Cluster,PatchRows,m![Mv%120,Rv=768]>=weight.view()
      .tile::<m![Rv],768,m![Mv,Rv=768#6912]>(offset).to_dm(&mut device.tdma);
    device.main.begin(weight.view())
      .fetch::<m![Mv%120,Rv=768/16],m![Rv=768%16]>()
      .collect::<m![Mv%120,Rv=768/16],m![Rv=768%16]>()
      .contract_outer::<m![Mv%120,Rv=768/32],m![Rv=768%32],_,_,_>(&x)
      .contract_packet::<m![1]>().contract_time::<m![Mv%120]>()
      .contract_lane::<m![Mv%120],m![B%8]>(LaneMode::Interleaved)
      .cast::<bf16,m![B%8#16]>().commit_trim::<m![B%8]>().commit()
}
fn patch_add(device:&mut Device,a:DmTensor<bf16,Chip,Cluster,PatchRows,m![Mv%120,B%8]>,
 b:DmTensor<bf16,Chip,Cluster,PatchRows,m![Mv%120,B%8]>)
 ->DmTensor<bf16,Chip,Cluster,PatchRows,m![Mv%120,B%8]> {
    let a:VrfTensor<f32,Chip,Cluster,PatchRows,m![Mv%120,B%8]>=device.sub.begin(a.view())
      .fetch::<m![Mv%120],m![B%8]>().fetch_cast::<f32>().collect::<m![Mv%120],m![B%8]>().to_vrf();
    device.main.begin(b.view()).fetch::<m![Mv%120],m![B%8]>().fetch_cast::<f32>()
      .collect::<m![Mv%120],m![B%8]>().vector_init().vector_intra_slice_tag(TagMode::Zero)
      .vector_clip(ClipBinaryOpF32::Add,&a).vector_final()
      .cast::<bf16,m![B%8#16]>().commit_trim::<m![B%8]>().commit()
}
pub(crate) fn patch_projection(device:&mut Device,x:&DmTensor<bf16,Chip,Cluster,Slice,m![Rv]>,
 weight:&HbmTensor<bf16,Chip,m![Mv,Rv]>,bias:&HbmTensor<bf16,Chip,m![Mv]>)
 ->DmTensor<bf16,Chip,Cluster,Slice,m![Mv]> {
    let x=patch_broadcast(device,x);
    let p0=patch_partial(device,&x,weight,0*768);
    let p1=patch_partial(device,&x,weight,1*768);
    let p2=patch_partial(device,&x,weight,2*768);
    let p3=patch_partial(device,&x,weight,3*768);
    let p4=patch_partial(device,&x,weight,4*768);
    let p5=patch_partial(device,&x,weight,5*768);
    let p6=patch_partial(device,&x,weight,6*768);
    let p7=patch_partial(device,&x,weight,7*768);
    let p8=patch_partial(device,&x,weight,8*768);
    let p01=patch_add(device,p0,p1);let p23=patch_add(device,p2,p3);
    let p0123=patch_add(device,p01,p23);
    let p45=patch_add(device,p4,p5);let p67=patch_add(device,p6,p7);
    let p4567=patch_add(device,p45,p67);let p07=patch_add(device,p0123,p4567);
    let sum=patch_add(device,p07,p8);
    let bias:DmTensor<bf16,Chip,Cluster,PatchRows,m![Mv%120]>=bias.to_dm(&mut device.tdma);
    let bias:VrfTensor<f32,Chip,Cluster,PatchRows,m![Mv%120]>=device.sub.begin(bias.view())
      .fetch::<m![Mv/8%15],m![Mv%8]>().fetch_cast::<f32>()
      .collect::<m![Mv/8%15],m![Mv%8]>().to_vrf();
    let sum:DmTensor<bf16,Chip,Cluster,PatchRows,m![Mv%120,B%8]>=device.main.begin(sum.view())
      .fetch::<m![Mv%120],m![B%8]>().fetch_cast::<f32>().collect::<m![Mv%120],m![B%8]>()
      .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_clip(ClipBinaryOpF32::Add,&bias)
      .vector_final().cast::<bf16,m![B%8#16]>().commit_trim::<m![B%8]>().commit();
    // Gather complete contiguous token rows, never2-byte strided token lanes.
    let sum:DmTensor<bf16,Chip,Cluster,PatchRows,m![B%8,Mv%120]>=device.main.begin(sum.view())
      .fetch::<m![Mv%120],m![B%8#16]>().collect::<m![Mv%120],m![B%8#16]>()
      .transpose::<m![Mv/4%30,B%8],m![Mv%4#16]>().commit_trim::<m![Mv%4]>().commit();
    sum.to_dm(&mut device.tdma)
}
fn mean_square_ov(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, Slice, m![Ov]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> {
    device
        .main
        .begin(input.view())
        .fetch::<m![Ov / 16], m![Ov % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ov / 8], m![Ov % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ov / 4], m![Ov % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ov, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(OV_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn rms_ov(
    device: &mut Device,
    mean_square: &DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> {
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

pub(super) fn rmsnorm_without_weight(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ov]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ov]> {
    let mean_square = mean_square_ov(device, x);
    let rms = rms_ov(device, &mean_square);
    let rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    device
        .main
        .begin(x.view())
        .fetch::<m![Ov / 16], m![Ov % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ov / 8], m![Ov % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ov / 4], m![Ov % 4]>()
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Ov / 8], m![Ov % 8]>()
        .vector_final()
        .cast::<bf16, m![Ov % 8 # 16]>()
        .commit_trim::<m![Ov % 8]>()
        .commit()
}

type TextRows=m![H/120,1#8];
fn text_broadcast(device:&mut Device,x:&DmTensor<bf16,Chip,Cluster,Slice,m![Ov]>)
 ->DmTensor<bf16,Chip,Cluster,Replicated,m![B%8,Ov]> {
    // Move the eight independently owned rows into the TRF lane axis and
    // broadcast to32 actual output groups. V32 is live;1#8 remains padding.
    x.to_dm(&mut device.tdma)
}
fn text_partial(device:&mut Device,x:&DmTensor<bf16,Chip,Cluster,Replicated,m![B%8,Ov]>,
 weight:&HbmTensor<bf16,Chip,m![H,Ov]>,offset:usize)
 ->DmTensor<bf16,Chip,Cluster,TextRows,m![H%120,B%8]> {
    let x:DmTensorView<'_,bf16,Chip,Cluster,TextRows,m![B%8,Ov]>=unsafe{x.view().reshape()};
    let x=x.tile::<m![Ov],768,m![B%8,Ov=768#3840]>(offset);
    // The SDK TRF fetch cannot retain the outer token stride of a cropped Kview.
    // Materialize exactly this K768 tile before the shared-lane contraction.
    let mut compact:DmTensor<bf16,Chip,Cluster,TextRows,m![B%8,Ov=768]>=DmTensor::new();
    x.to_dm_view(&mut device.tdma,compact.view_mut());
    let x:TrfTensor<bf16,Chip,Cluster,TextRows,m![B%8],m![Ov=768]>=device.sub.begin(compact.view())
      .fetch::<m![B%8],m![Ov=768]>()
      .collect::<m![B%8,Ov=768/16],m![Ov=768%16]>().to_trf();
    let weight:DmTensor<bf16,Chip,Cluster,TextRows,m![H%120,Ov=768]>=weight.view()
      .tile::<m![Ov],768,m![H,Ov=768#3840]>(offset).to_dm(&mut device.tdma);
    device.main.begin(weight.view())
      .fetch::<m![H%120,Ov=768/16],m![Ov=768%16]>()
      .collect::<m![H%120,Ov=768/16],m![Ov=768%16]>()
      .contract_outer::<m![H%120,Ov=768/32],m![Ov=768%32],_,_,_>(&x)
      .contract_packet::<m![1]>().contract_time::<m![H%120]>()
      .contract_lane::<m![H%120],m![B%8]>(LaneMode::Interleaved)
      .cast::<bf16,m![B%8#16]>().commit_trim::<m![B%8]>().commit()
}
fn text_add(device:&mut Device,a:DmTensor<bf16,Chip,Cluster,TextRows,m![H%120,B%8]>,
 b:DmTensor<bf16,Chip,Cluster,TextRows,m![H%120,B%8]>)
 ->DmTensor<bf16,Chip,Cluster,TextRows,m![H%120,B%8]> {
    let a:VrfTensor<f32,Chip,Cluster,TextRows,m![H%120,B%8]>=device.sub.begin(a.view())
      .fetch::<m![H%120],m![B%8]>().fetch_cast::<f32>().collect::<m![H%120],m![B%8]>().to_vrf();
    device.main.begin(b.view()).fetch::<m![H%120],m![B%8]>().fetch_cast::<f32>()
      .collect::<m![H%120],m![B%8]>().vector_init().vector_intra_slice_tag(TagMode::Zero)
      .vector_clip(ClipBinaryOpF32::Add,&a).vector_final()
      .cast::<bf16,m![B%8#16]>().commit_trim::<m![B%8]>().commit()
}
pub(crate) fn project_to_text_hidden(device:&mut Device,x:&DmTensor<bf16,Chip,Cluster,Slice,m![Ov]>,
 weight:&HbmTensor<bf16,Chip,m![H,Ov]>)
 ->DmTensor<bf16,Chip,Cluster,Slice,m![H]> {
    let x=rmsnorm_without_weight(device,x);
    let x=text_broadcast(device,&x);
    let p0=text_partial(device,&x,weight,0*768);
    let p1=text_partial(device,&x,weight,1*768);
    let p2=text_partial(device,&x,weight,2*768);
    let p3=text_partial(device,&x,weight,3*768);
    let p4=text_partial(device,&x,weight,4*768);
    let p01=text_add(device,p0,p1);let p23=text_add(device,p2,p3);
    let p0123=text_add(device,p01,p23);
    let sum=text_add(device,p0123,p4);
    // Gather complete contiguous token rows, never2-byte strided token lanes.
    let sum:DmTensor<bf16,Chip,Cluster,TextRows,m![B%8,H%120]>=device.main.begin(sum.view())
      .fetch::<m![H%120],m![B%8#16]>().collect::<m![H%120],m![B%8#16]>()
      .transpose::<m![H/4%30,B%8],m![H%4#16]>().commit_trim::<m![H%4]>().commit();
    sum.to_dm(&mut device.tdma)
}
