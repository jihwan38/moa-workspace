//! Native P128 head RMSNorm and RoPE, four independent rows per slice.
use crate::axes::{P128, Df, Ds, E, Gf, Gs, Ns, Pf, Ps, Qf, Qs};
type Cluster=m![P128/64];
type RopeRows=m![P128/4%16,1#16];
use crate::{Chip, EPS};
use furiosa_opt_std::prelude::*;

pub(crate) fn rms_ds<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![P128%4,Ds]>,
) -> VrfTensor<f32, Chip, Cluster, S, m![P128%4,1 # 8]> {
    let mean: DmTensor<f32, Chip, Cluster, S, m![P128%4,1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![P128%4,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%4,Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![P128%4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(256.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, S, m![P128%4,1 # 8]> = device
        .main
        .begin(mean.view())
        .fetch::<m![P128%4], m![1 # 8]>()
        .collect::<m![P128%4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    device
        .sub
        .begin(rms.view())
        .fetch::<m![P128%4], m![1 # 8]>()
        .collect::<m![P128%4], m![1 # 8]>()
        .to_vrf()
}
pub(crate) fn weight_ds<S: M>(
    device: &mut Device,
    w: &HbmTensor<bf16, Chip, m![Ds]>,
) -> VrfTensor<f32, Chip, Cluster, S, m![Ds]> {
    let w: DmTensor<bf16, Chip, Cluster, S, m![Ds]> = w.to_dm(&mut device.tdma);
    device
        .sub
        .begin(w.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}
pub(crate) fn norm_ds<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![P128%4,Ds]>,
    rms: &VrfTensor<f32, Chip, Cluster, S, m![P128%4,1 # 8]>,
    w: &VrfTensor<f32, Chip, Cluster, S, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![P128%4,Ds]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![P128%4,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%4,Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, rms)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), w)
        .vector_widen_concat::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
pub(crate) fn value_ds<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![P128%4,Ds]>,
    rms: &VrfTensor<f32, Chip, Cluster, S, m![P128%4,1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![P128%4,Ds]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![P128%4,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%4,Ds / 4], m![Ds % 4]>()
        .vector_fp_div(rms)
        .vector_widen_concat::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
pub(crate) fn rope_ds<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![P128%4,Ds]>,
    cos: &DmTensor<bf16, Chip, Cluster, S, m![P128%4,Ds]>,
    sin: &DmTensor<bf16, Chip, Cluster, S, m![P128%4,Ds]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![P128%4,Ds]> {
    let cos: VrfTensor<f32, Chip, Cluster, S, m![P128%4,Ds]> = device
        .sub
        .begin(cos.view())
        .fetch::<m![P128%4,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let sin: VrfTensor<f32, Chip, Cluster, S, m![P128%4,Ds]> = device
        .sub
        .begin(sin.view())
        .fetch::<m![P128%4,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mut rotated: DmTensor<bf16, Chip, Cluster, S, m![P128%4,Ds]> = DmTensor::new();
    device
        .main
        .begin(x.view().tile::<m![Ds], 128, m![P128%4,Ds=128 # 256]>(0))
        .fetch::<m![P128%4], m![Ds = 128]>()
        .collect::<m![P128%4,Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotated
                .view_mut()
                .tile::<m![Ds], 128, m![P128%4,Ds=128 #{!} 256]>(128),
        );
    device
        .main
        .begin(x.view().tile::<m![Ds], 128, m![P128%4,Ds=128 # 256]>(128))
        .fetch::<m![P128%4], m![Ds = 128]>()
        .collect::<m![P128%4,Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotated
                .view_mut()
                .tile::<m![Ds], 128, m![P128%4,Ds=128 #{!} 256]>(0),
        );
    let xc: DmTensor<f32, Chip, Cluster, S, m![P128%4,Ds]> = device
        .main
        .begin(x.view())
        .fetch::<m![P128%4,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%4,Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos)
        .vector_widen_concat::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let xs: DmTensor<f32, Chip, Cluster, S, m![P128%4,Ds]> = device
        .main
        .begin(rotated.view())
        .fetch::<m![P128%4,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%4,Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin)
        .vector_widen_concat::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let xs: VrfTensor<f32, Chip, Cluster, S, m![P128%4,Ds]> = device
        .sub
        .begin(xs.view())
        .fetch::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .collect::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .to_vrf();
    device
        .main
        .begin(xc.view())
        .fetch::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .collect::<m![P128%4,Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &xs)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn rms_df<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![P128%4,Df]>,
) -> VrfTensor<f32, Chip, Cluster, S, m![P128%4,1 # 8]> {
    let mean: DmTensor<f32, Chip, Cluster, S, m![P128%4,1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![P128%4,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%4,Df / 4], m![Df % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Df, m![P128%4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(512.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, S, m![P128%4,1 # 8]> = device
        .main
        .begin(mean.view())
        .fetch::<m![P128%4], m![1 # 8]>()
        .collect::<m![P128%4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    device
        .sub
        .begin(rms.view())
        .fetch::<m![P128%4], m![1 # 8]>()
        .collect::<m![P128%4], m![1 # 8]>()
        .to_vrf()
}
pub(crate) fn weight_df<S: M>(
    device: &mut Device,
    w: &HbmTensor<bf16, Chip, m![Df]>,
) -> VrfTensor<f32, Chip, Cluster, S, m![Df]> {
    let w: DmTensor<bf16, Chip, Cluster, S, m![Df]> = w.to_dm(&mut device.tdma);
    device
        .sub
        .begin(w.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .to_vrf()
}
pub(crate) fn norm_df<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![P128%4,Df]>,
    rms: &VrfTensor<f32, Chip, Cluster, S, m![P128%4,1 # 8]>,
    w: &VrfTensor<f32, Chip, Cluster, S, m![Df]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![P128%4,Df]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![P128%4,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%4,Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), w)
        .vector_fp_div(rms)
        .vector_widen_concat::<m![P128%4,Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}
pub(crate) fn value_df<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![P128%4,Df]>,
    rms: &VrfTensor<f32, Chip, Cluster, S, m![P128%4,1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![P128%4,Df]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![P128%4,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%4,Df / 4], m![Df % 4]>()
        .vector_fp_div(rms)
        .vector_widen_concat::<m![P128%4,Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}
pub(crate) fn rope_df<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![P128%4,Df]>,
    cos: &DmTensor<bf16, Chip, Cluster, S, m![P128%4,Df]>,
    sin: &DmTensor<bf16, Chip, Cluster, S, m![P128%4,Df]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![P128%4,Df]> {
    let cos: VrfTensor<f32, Chip, Cluster, S, m![P128%4,Df]> = device
        .sub
        .begin(cos.view())
        .fetch::<m![P128%4,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Df / 8], m![Df % 8]>()
        .to_vrf();
    let sin: VrfTensor<f32, Chip, Cluster, S, m![P128%4,Df]> = device
        .sub
        .begin(sin.view())
        .fetch::<m![P128%4,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Df / 8], m![Df % 8]>()
        .to_vrf();
    let mut rotated: DmTensor<bf16, Chip, Cluster, S, m![P128%4,Df]> = DmTensor::new();
    device
        .main
        .begin(x.view().tile::<m![Df], 256, m![P128%4,Df=256 # 512]>(0))
        .fetch::<m![P128%4], m![Df = 256]>()
        .collect::<m![P128%4,Df = 256 / 16], m![Df = 256 % 16]>()
        .commit_trim::<m![Df = 256 % 16]>()
        .commit_view(
            rotated
                .view_mut()
                .tile::<m![Df], 256, m![P128%4,Df=256 #{!} 512]>(256),
        );
    device
        .main
        .begin(x.view().tile::<m![Df], 256, m![P128%4,Df=256 # 512]>(256))
        .fetch::<m![P128%4], m![Df = 256]>()
        .collect::<m![P128%4,Df = 256 / 16], m![Df = 256 % 16]>()
        .commit_trim::<m![Df = 256 % 16]>()
        .commit_view(
            rotated
                .view_mut()
                .tile::<m![Df], 256, m![P128%4,Df=256 #{!} 512]>(0),
        );
    let xc: DmTensor<f32, Chip, Cluster, S, m![P128%4,Df]> = device
        .main
        .begin(x.view())
        .fetch::<m![P128%4,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%4,Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos)
        .vector_widen_concat::<m![P128%4,Df / 8], m![Df % 8]>()
        .vector_final()
        .commit_trim::<m![Df % 8]>()
        .commit();
    let xs: DmTensor<f32, Chip, Cluster, S, m![P128%4,Df]> = device
        .main
        .begin(rotated.view())
        .fetch::<m![P128%4,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128%4,Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128%4,Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin)
        .vector_widen_concat::<m![P128%4,Df / 8], m![Df % 8]>()
        .vector_final()
        .commit_trim::<m![Df % 8]>()
        .commit();
    let xs: VrfTensor<f32, Chip, Cluster, S, m![P128%4,Df]> = device
        .sub
        .begin(xs.view())
        .fetch::<m![P128%4,Df / 8], m![Df % 8]>()
        .collect::<m![P128%4,Df / 8], m![Df % 8]>()
        .to_vrf();
    device
        .main
        .begin(xc.view())
        .fetch::<m![P128%4,Df / 8], m![Df % 8]>()
        .collect::<m![P128%4,Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &xs)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

#[device(chip = 1)]
pub fn sliding_heads_p128(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![P128, Qs]>,
    k_raw: &HbmTensor<bf16, Chip, m![P128, Ps]>,
    v_raw: &HbmTensor<bf16, Chip, m![P128, Ps]>,
    q_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![P128]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![P128, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
) {
    type QRows = m![P128/4%16,Ns,Gs];
    type KRows = m![P128/4%16,Ns,1#2];
    let q = unsafe { q_raw.view().reshape::<Chip, m![P128, Ns, Gs, Ds]>() };
    let k = unsafe { k_raw.view().reshape::<Chip, m![P128, Ns, Ds]>() };
    let q: DmTensor<bf16, Chip, Cluster, QRows, m![P128%4,Ds]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Ds]> = k.to_dm(&mut device.tdma);
    let qr = rms_ds(device, &q);
    let kr = rms_ds(device, &k);
    let v = unsafe { v_raw.view().reshape::<Chip, m![P128, Ns, Ds]>() };
    let v: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Ds]> = v.to_dm(&mut device.tdma);
    let vr = rms_ds(device, &v);
    let v = value_ds(device, &v, &vr);
    let qw = weight_ds(device, q_weight);
    let kw = weight_ds(device, k_weight);
    let q = norm_ds(device, &q, &qr, &qw);
    let k = norm_ds(device, &k, &kr, &kw);
    let cos: DmTensor<bf16, Chip, Cluster, RopeRows, m![P128%4,Ds]> = cos
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let cos: DmTensor<bf16, Chip, Cluster, QRows, m![P128%4,Ds]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, RopeRows, m![P128%4,Ds]> = sin
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, QRows, m![P128%4,Ds]> = sin.to_dm(&mut device.tdma);
    let q = rope_ds(device, &q, &cos, &sin);
    let cos: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Ds]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Ds]> = sin.to_dm(&mut device.tdma);
    let k = rope_ds(device, &k, &cos, &sin);
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.view().to_hbm_view(&mut device.tdma, k_out.view_mut());
    v.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}

#[device(chip = 1)]
pub fn full_heads_p128(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![P128, Qf]>,
    k_raw: &HbmTensor<bf16, Chip, m![P128, Pf]>,
    q_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_weight: &HbmTensor<bf16, Chip, m![Df]>,
    rope_offsets: &HbmTensor<i32, Chip, m![P128]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![P128, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![P128, Df]>,
) {
    type QRows = m![P128/4%16,Gf];
    type KRows = m![P128/4%16,1#16];
    let q = unsafe { q_raw.view().reshape::<Chip, m![P128, Gf, Df]>() };
    let k = unsafe { k_raw.view().reshape::<Chip, m![P128, Df]>() };
    let q: DmTensor<bf16, Chip, Cluster, QRows, m![P128%4,Df]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Df]> = k.to_dm(&mut device.tdma);
    let qr = rms_df(device, &q);
    let kr = rms_df(device, &k);
    let v = value_df(device, &k, &kr);
    let qw = weight_df(device, q_weight);
    let kw = weight_df(device, k_weight);
    let q = norm_df(device, &q, &qr, &qw);
    let k = norm_df(device, &k, &kr, &kw);
    let cos: DmTensor<bf16, Chip, Cluster, RopeRows, m![P128%4,Df]> = cos
        .gather::<m![E], m![Df]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let cos: DmTensor<bf16, Chip, Cluster, QRows, m![P128%4,Df]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, RopeRows, m![P128%4,Df]> = sin
        .gather::<m![E], m![Df]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, QRows, m![P128%4,Df]> = sin.to_dm(&mut device.tdma);
    let q = rope_df(device, &q, &cos, &sin);
    let cos: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Df]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KRows, m![P128%4,Df]> = sin.to_dm(&mut device.tdma);
    let k = rope_df(device, &k, &cos, &sin);
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.view().to_hbm_view(&mut device.tdma, k_out.view_mut());
    v.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}

