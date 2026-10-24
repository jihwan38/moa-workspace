//! True B16 head RMSNorm and RoPE: one independent head per slice.
use crate::axes::{B, Df, Ds, E, Gf, Gs, Ns, Pf, Ps, Qf, Qs};
use crate::device::layout::{Cluster, Slice};
use crate::device::{full, sliding};
use crate::{Chip, EPS};
use furiosa_opt_std::prelude::*;

fn rms_ds<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![Ds]>,
) -> VrfTensor<f32, Chip, Cluster, S, m![1 # 8]> {
    let mean: DmTensor<f32, Chip, Cluster, S, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(256.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, S, m![1 # 8]> = device
        .main
        .begin(mean.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
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
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}
fn weight_ds<S: M>(
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
fn norm_ds<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![Ds]>,
    rms: &VrfTensor<f32, Chip, Cluster, S, m![1 # 8]>,
    w: &VrfTensor<f32, Chip, Cluster, S, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![Ds]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, rms)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), w)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
fn value_ds<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![Ds]>,
    rms: &VrfTensor<f32, Chip, Cluster, S, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![Ds]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_div(rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
fn rope_ds<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![Ds]>,
    cos: &DmTensor<bf16, Chip, Cluster, S, m![Ds]>,
    sin: &DmTensor<bf16, Chip, Cluster, S, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![Ds]> {
    let cos: VrfTensor<f32, Chip, Cluster, S, m![Ds]> = device
        .sub
        .begin(cos.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let sin: VrfTensor<f32, Chip, Cluster, S, m![Ds]> = device
        .sub
        .begin(sin.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mut rotated: DmTensor<bf16, Chip, Cluster, S, m![Ds]> = DmTensor::new();
    device
        .main
        .begin(x.view().tile::<m![Ds], 128, m![Ds=128 # 256]>(0))
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotated
                .view_mut()
                .tile::<m![Ds], 128, m![Ds=128 #{!} 256]>(128),
        );
    device
        .main
        .begin(x.view().tile::<m![Ds], 128, m![Ds=128 # 256]>(128))
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotated
                .view_mut()
                .tile::<m![Ds], 128, m![Ds=128 #{!} 256]>(0),
        );
    let xc: DmTensor<f32, Chip, Cluster, S, m![Ds]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let xs: DmTensor<f32, Chip, Cluster, S, m![Ds]> = device
        .main
        .begin(rotated.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let xs: VrfTensor<f32, Chip, Cluster, S, m![Ds]> = device
        .sub
        .begin(xs.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    device
        .main
        .begin(xc.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &xs)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn rms_df<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![Df]>,
) -> VrfTensor<f32, Chip, Cluster, S, m![1 # 8]> {
    let mean: DmTensor<f32, Chip, Cluster, S, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Df, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(512.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, S, m![1 # 8]> = device
        .main
        .begin(mean.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
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
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}
fn weight_df<S: M>(
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
fn norm_df<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![Df]>,
    rms: &VrfTensor<f32, Chip, Cluster, S, m![1 # 8]>,
    w: &VrfTensor<f32, Chip, Cluster, S, m![Df]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![Df]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), w)
        .vector_fp_div(rms)
        .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}
fn value_df<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![Df]>,
    rms: &VrfTensor<f32, Chip, Cluster, S, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![Df]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_fp_div(rms)
        .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}
fn rope_df<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![Df]>,
    cos: &DmTensor<bf16, Chip, Cluster, S, m![Df]>,
    sin: &DmTensor<bf16, Chip, Cluster, S, m![Df]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![Df]> {
    let cos: VrfTensor<f32, Chip, Cluster, S, m![Df]> = device
        .sub
        .begin(cos.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .to_vrf();
    let sin: VrfTensor<f32, Chip, Cluster, S, m![Df]> = device
        .sub
        .begin(sin.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .to_vrf();
    let mut rotated: DmTensor<bf16, Chip, Cluster, S, m![Df]> = DmTensor::new();
    device
        .main
        .begin(x.view().tile::<m![Df], 256, m![Df=256 # 512]>(0))
        .fetch::<m![1], m![Df = 256]>()
        .collect::<m![Df = 256 / 16], m![Df = 256 % 16]>()
        .commit_trim::<m![Df = 256 % 16]>()
        .commit_view(
            rotated
                .view_mut()
                .tile::<m![Df], 256, m![Df=256 #{!} 512]>(256),
        );
    device
        .main
        .begin(x.view().tile::<m![Df], 256, m![Df=256 # 512]>(256))
        .fetch::<m![1], m![Df = 256]>()
        .collect::<m![Df = 256 / 16], m![Df = 256 % 16]>()
        .commit_trim::<m![Df = 256 % 16]>()
        .commit_view(
            rotated
                .view_mut()
                .tile::<m![Df], 256, m![Df=256 #{!} 512]>(0),
        );
    let xc: DmTensor<f32, Chip, Cluster, S, m![Df]> = device
        .main
        .begin(x.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos)
        .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
        .vector_final()
        .commit_trim::<m![Df % 8]>()
        .commit();
    let xs: DmTensor<f32, Chip, Cluster, S, m![Df]> = device
        .main
        .begin(rotated.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin)
        .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
        .vector_final()
        .commit_trim::<m![Df % 8]>()
        .commit();
    let xs: VrfTensor<f32, Chip, Cluster, S, m![Df]> = device
        .sub
        .begin(xs.view())
        .fetch::<m![Df / 8], m![Df % 8]>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .to_vrf();
    device
        .main
        .begin(xc.view())
        .fetch::<m![Df / 8], m![Df % 8]>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &xs)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

#[device(chip = 1)]
pub fn sliding_heads_b16(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![B, Qs]>,
    k_raw: &HbmTensor<bf16, Chip, m![B, Ps]>,
    v_raw: &HbmTensor<bf16, Chip, m![B, Ps]>,
    q_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![B]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
) {
    type QRows = m![B, Ns, Gs];
    type KRows = m![B,Ns,1 # 2];
    let q = unsafe { q_raw.view().reshape::<Chip, m![B, Ns, Gs, Ds]>() };
    let k = unsafe { k_raw.view().reshape::<Chip, m![B, Ns, Ds]>() };
    let q: DmTensor<bf16, Chip, Cluster, QRows, m![Ds]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KRows, m![Ds]> = k.to_dm(&mut device.tdma);
    let qr = rms_ds(device, &q);
    let kr = rms_ds(device, &k);
    let v = unsafe { v_raw.view().reshape::<Chip, m![B, Ns, Ds]>() };
    let v: DmTensor<bf16, Chip, Cluster, KRows, m![Ds]> = v.to_dm(&mut device.tdma);
    let vr = rms_ds(device, &v);
    let v = value_ds(device, &v, &vr);
    let qw = weight_ds(device, q_weight);
    let kw = weight_ds(device, k_weight);
    let q = norm_ds(device, &q, &qr, &qw);
    let k = norm_ds(device, &k, &kr, &kw);
    let cos: DmTensor<bf16, Chip, Cluster, Slice, m![B, Ds]> = cos
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let cos: DmTensor<bf16, Chip, Cluster, QRows, m![Ds]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, Slice, m![B, Ds]> = sin
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, QRows, m![Ds]> = sin.to_dm(&mut device.tdma);
    let q = rope_ds(device, &q, &cos, &sin);
    let cos: DmTensor<bf16, Chip, Cluster, KRows, m![Ds]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KRows, m![Ds]> = sin.to_dm(&mut device.tdma);
    let k = rope_ds(device, &k, &cos, &sin);
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.view().to_hbm_view(&mut device.tdma, k_out.view_mut());
    v.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}

#[device(chip = 1)]
pub fn full_heads_b16(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![B, Qf]>,
    k_raw: &HbmTensor<bf16, Chip, m![B, Pf]>,
    q_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_weight: &HbmTensor<bf16, Chip, m![Df]>,
    rope_offsets: &HbmTensor<i32, Chip, m![B]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![B, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![B, Df]>,
) {
    type QRows = m![B, Gf];
    type KRows = m![B,1 # 16];
    let q = unsafe { q_raw.view().reshape::<Chip, m![B, Gf, Df]>() };
    let k = unsafe { k_raw.view().reshape::<Chip, m![B, Df]>() };
    let q: DmTensor<bf16, Chip, Cluster, QRows, m![Df]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KRows, m![Df]> = k.to_dm(&mut device.tdma);
    let qr = rms_df(device, &q);
    let kr = rms_df(device, &k);
    let v = value_df(device, &k, &kr);
    let qw = weight_df(device, q_weight);
    let kw = weight_df(device, k_weight);
    let q = norm_df(device, &q, &qr, &qw);
    let k = norm_df(device, &k, &kr, &kw);
    let cos: DmTensor<bf16, Chip, Cluster, Slice, m![B, Df]> = cos
        .gather::<m![E], m![Df]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let cos: DmTensor<bf16, Chip, Cluster, QRows, m![Df]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, Slice, m![B, Df]> = sin
        .gather::<m![E], m![Df]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, QRows, m![Df]> = sin.to_dm(&mut device.tdma);
    let q = rope_df(device, &q, &cos, &sin);
    let cos: DmTensor<bf16, Chip, Cluster, KRows, m![Df]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KRows, m![Df]> = sin.to_dm(&mut device.tdma);
    let k = rope_df(device, &k, &cos, &sin);
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.view().to_hbm_view(&mut device.tdma, k_out.view_mut());
    v.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}

/// Nonzero probe control: original scalar normalization and RoPE, untouched arithmetic.
#[device(chip = 1)]
pub fn reference_sliding_head_row(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![Qs]>,
    k_raw: &HbmTensor<bf16, Chip, m![Ps]>,
    v_raw: &HbmTensor<bf16, Chip, m![Ps]>,
    qw: &HbmTensor<bf16, Chip, m![Ds]>,
    kw: &HbmTensor<bf16, Chip, m![Ds]>,
    offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    qo: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    ko: &mut HbmTensor<bf16, Chip, m![Ns, Ds]>,
    vo: &mut HbmTensor<bf16, Chip, m![Ns, Ds]>,
) {
    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Qs]> = q_raw.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = k_raw.to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = v_raw.to_dm(&mut device.tdma);
    let q = unsafe { q.reshape::<Chip, Cluster, Slice, m![Ns, Gs, Ds]>() };
    let k = unsafe { k.reshape::<Chip, Cluster, Slice, m![Ns, Ds]>() };
    let v = unsafe { v.reshape::<Chip, Cluster, Slice, m![Ns, Ds]>() };
    let q = sliding::rmsnorm::normalize_query(device, &q, qw);
    let k = sliding::rmsnorm::normalize_key(device, &k, kw);
    let v = sliding::rmsnorm::normalize_value(device, &v);
    let (q, k) = sliding::rope::apply_rope(device, &q, &k, offset, cos, sin);
    q.view().to_hbm_view(&mut device.tdma, qo.view_mut());
    k.view().to_hbm_view(&mut device.tdma, ko.view_mut());
    v.view().to_hbm_view(&mut device.tdma, vo.view_mut());
}
#[device(chip = 1)]
pub fn reference_full_head_row(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![Qf]>,
    k_raw: &HbmTensor<bf16, Chip, m![Pf]>,
    qw: &HbmTensor<bf16, Chip, m![Df]>,
    kw: &HbmTensor<bf16, Chip, m![Df]>,
    offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    qo: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
    ko: &mut HbmTensor<bf16, Chip, m![Df]>,
    vo: &mut HbmTensor<bf16, Chip, m![Df]>,
) {
    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Qf]> = q_raw.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Pf]> = k_raw.to_dm(&mut device.tdma);
    let q = unsafe { q.reshape::<Chip, Cluster, Slice, m![Gf, Df]>() };
    let k = unsafe { k.reshape::<Chip, Cluster, Slice, m![Df]>() };
    let v = full::rmsnorm::normalize_value(device, &k);
    let q = full::rmsnorm::normalize_query(device, &q, qw);
    let k = full::rmsnorm::normalize_key(device, &k, kw);
    let (q, k) = full::rope::apply_rope(device, &q, &k, offset, cos, sin);
    q.view().to_hbm_view(&mut device.tdma, qo.view_mut());
    k.view().to_hbm_view(&mut device.tdma, ko.view_mut());
    v.view().to_hbm_view(&mut device.tdma, vo.view_mut());
}
