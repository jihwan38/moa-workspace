//! Native R64 head RMSNorm and RoPE, two independent rows per slice.
use crate::axes::{R64, Df, Ds, E, Gf, Gs, Ns, Pf, Ps, Qf, Qs};
type Cluster=m![R64/32];
type RopeRows=m![R64/2%16,1#16];
use crate::{Chip, EPS};
use furiosa_opt_std::prelude::*;

pub(crate) fn rms_ds<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![R64%2,Ds]>,
) -> VrfTensor<f32, Chip, Cluster, S, m![R64%2,1 # 8]> {
    let mean: DmTensor<f32, Chip, Cluster, S, m![R64%2,1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![R64%2,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R64%2,Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![R64%2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(256.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, S, m![R64%2,1 # 8]> = device
        .main
        .begin(mean.view())
        .fetch::<m![R64%2], m![1 # 8]>()
        .collect::<m![R64%2], m![1 # 8]>()
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
        .fetch::<m![R64%2], m![1 # 8]>()
        .collect::<m![R64%2], m![1 # 8]>()
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
    x: &DmTensor<bf16, Chip, Cluster, S, m![R64%2,Ds]>,
    rms: &VrfTensor<f32, Chip, Cluster, S, m![R64%2,1 # 8]>,
    w: &VrfTensor<f32, Chip, Cluster, S, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![R64%2,Ds]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![R64%2,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R64%2,Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, rms)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), w)
        .vector_widen_concat::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
pub(crate) fn value_ds<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![R64%2,Ds]>,
    rms: &VrfTensor<f32, Chip, Cluster, S, m![R64%2,1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![R64%2,Ds]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![R64%2,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R64%2,Ds / 4], m![Ds % 4]>()
        .vector_fp_div(rms)
        .vector_widen_concat::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
pub(crate) fn rope_ds<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![R64%2,Ds]>,
    cos: &DmTensor<bf16, Chip, Cluster, S, m![R64%2,Ds]>,
    sin: &DmTensor<bf16, Chip, Cluster, S, m![R64%2,Ds]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![R64%2,Ds]> {
    let cos: VrfTensor<f32, Chip, Cluster, S, m![R64%2,Ds]> = device
        .sub
        .begin(cos.view())
        .fetch::<m![R64%2,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let sin: VrfTensor<f32, Chip, Cluster, S, m![R64%2,Ds]> = device
        .sub
        .begin(sin.view())
        .fetch::<m![R64%2,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mut rotated: DmTensor<bf16, Chip, Cluster, S, m![R64%2,Ds]> = DmTensor::new();
    device
        .main
        .begin(x.view().tile::<m![Ds], 128, m![R64%2,Ds=128 # 256]>(0))
        .fetch::<m![R64%2], m![Ds = 128]>()
        .collect::<m![R64%2,Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotated
                .view_mut()
                .tile::<m![Ds], 128, m![R64%2,Ds=128 #{!} 256]>(128),
        );
    device
        .main
        .begin(x.view().tile::<m![Ds], 128, m![R64%2,Ds=128 # 256]>(128))
        .fetch::<m![R64%2], m![Ds = 128]>()
        .collect::<m![R64%2,Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotated
                .view_mut()
                .tile::<m![Ds], 128, m![R64%2,Ds=128 #{!} 256]>(0),
        );
    let xc: DmTensor<f32, Chip, Cluster, S, m![R64%2,Ds]> = device
        .main
        .begin(x.view())
        .fetch::<m![R64%2,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R64%2,Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos)
        .vector_widen_concat::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let xs: DmTensor<f32, Chip, Cluster, S, m![R64%2,Ds]> = device
        .main
        .begin(rotated.view())
        .fetch::<m![R64%2,Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R64%2,Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin)
        .vector_widen_concat::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let xs: VrfTensor<f32, Chip, Cluster, S, m![R64%2,Ds]> = device
        .sub
        .begin(xs.view())
        .fetch::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .collect::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .to_vrf();
    device
        .main
        .begin(xc.view())
        .fetch::<m![R64%2,Ds / 8], m![Ds % 8]>()
        .collect::<m![R64%2,Ds / 8], m![Ds % 8]>()
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
    x: &DmTensor<bf16, Chip, Cluster, S, m![R64%2,Df]>,
) -> VrfTensor<f32, Chip, Cluster, S, m![R64%2,1 # 8]> {
    let mean: DmTensor<f32, Chip, Cluster, S, m![R64%2,1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![R64%2,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R64%2,Df / 4], m![Df % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Df, m![R64%2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(512.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, S, m![R64%2,1 # 8]> = device
        .main
        .begin(mean.view())
        .fetch::<m![R64%2], m![1 # 8]>()
        .collect::<m![R64%2], m![1 # 8]>()
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
        .fetch::<m![R64%2], m![1 # 8]>()
        .collect::<m![R64%2], m![1 # 8]>()
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
    x: &DmTensor<bf16, Chip, Cluster, S, m![R64%2,Df]>,
    rms: &VrfTensor<f32, Chip, Cluster, S, m![R64%2,1 # 8]>,
    w: &VrfTensor<f32, Chip, Cluster, S, m![Df]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![R64%2,Df]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![R64%2,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R64%2,Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), w)
        .vector_fp_div(rms)
        .vector_widen_concat::<m![R64%2,Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}
pub(crate) fn value_df<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![R64%2,Df]>,
    rms: &VrfTensor<f32, Chip, Cluster, S, m![R64%2,1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![R64%2,Df]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![R64%2,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R64%2,Df / 4], m![Df % 4]>()
        .vector_fp_div(rms)
        .vector_widen_concat::<m![R64%2,Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}
pub(crate) fn rope_df<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, S, m![R64%2,Df]>,
    cos: &DmTensor<bf16, Chip, Cluster, S, m![R64%2,Df]>,
    sin: &DmTensor<bf16, Chip, Cluster, S, m![R64%2,Df]>,
) -> DmTensor<bf16, Chip, Cluster, S, m![R64%2,Df]> {
    let cos: VrfTensor<f32, Chip, Cluster, S, m![R64%2,Df]> = device
        .sub
        .begin(cos.view())
        .fetch::<m![R64%2,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Df / 8], m![Df % 8]>()
        .to_vrf();
    let sin: VrfTensor<f32, Chip, Cluster, S, m![R64%2,Df]> = device
        .sub
        .begin(sin.view())
        .fetch::<m![R64%2,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Df / 8], m![Df % 8]>()
        .to_vrf();
    let mut rotated: DmTensor<bf16, Chip, Cluster, S, m![R64%2,Df]> = DmTensor::new();
    device
        .main
        .begin(x.view().tile::<m![Df], 256, m![R64%2,Df=256 # 512]>(0))
        .fetch::<m![R64%2], m![Df = 256]>()
        .collect::<m![R64%2,Df = 256 / 16], m![Df = 256 % 16]>()
        .commit_trim::<m![Df = 256 % 16]>()
        .commit_view(
            rotated
                .view_mut()
                .tile::<m![Df], 256, m![R64%2,Df=256 #{!} 512]>(256),
        );
    device
        .main
        .begin(x.view().tile::<m![Df], 256, m![R64%2,Df=256 # 512]>(256))
        .fetch::<m![R64%2], m![Df = 256]>()
        .collect::<m![R64%2,Df = 256 / 16], m![Df = 256 % 16]>()
        .commit_trim::<m![Df = 256 % 16]>()
        .commit_view(
            rotated
                .view_mut()
                .tile::<m![Df], 256, m![R64%2,Df=256 #{!} 512]>(0),
        );
    let xc: DmTensor<f32, Chip, Cluster, S, m![R64%2,Df]> = device
        .main
        .begin(x.view())
        .fetch::<m![R64%2,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R64%2,Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos)
        .vector_widen_concat::<m![R64%2,Df / 8], m![Df % 8]>()
        .vector_final()
        .commit_trim::<m![Df % 8]>()
        .commit();
    let xs: DmTensor<f32, Chip, Cluster, S, m![R64%2,Df]> = device
        .main
        .begin(rotated.view())
        .fetch::<m![R64%2,Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![R64%2,Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R64%2,Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin)
        .vector_widen_concat::<m![R64%2,Df / 8], m![Df % 8]>()
        .vector_final()
        .commit_trim::<m![Df % 8]>()
        .commit();
    let xs: VrfTensor<f32, Chip, Cluster, S, m![R64%2,Df]> = device
        .sub
        .begin(xs.view())
        .fetch::<m![R64%2,Df / 8], m![Df % 8]>()
        .collect::<m![R64%2,Df / 8], m![Df % 8]>()
        .to_vrf();
    device
        .main
        .begin(xc.view())
        .fetch::<m![R64%2,Df / 8], m![Df % 8]>()
        .collect::<m![R64%2,Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &xs)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

#[device(chip = 1)]
pub fn sliding_heads_r64(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![R64, Qs]>,
    k_raw: &HbmTensor<bf16, Chip, m![R64, Ps]>,
    v_raw: &HbmTensor<bf16, Chip, m![R64, Ps]>,
    q_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![R64]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![R64, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![R64, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![R64, Ns, Ds]>,
) {
    type QRows = m![R64/2%16,Ns,Gs];
    type KRows = m![R64/2%16,Ns,1#2];
    let q = unsafe { q_raw.view().reshape::<Chip, m![R64, Ns, Gs, Ds]>() };
    let k = unsafe { k_raw.view().reshape::<Chip, m![R64, Ns, Ds]>() };
    let q: DmTensor<bf16, Chip, Cluster, QRows, m![R64%2,Ds]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KRows, m![R64%2,Ds]> = k.to_dm(&mut device.tdma);
    let qr = rms_ds(device, &q);
    let kr = rms_ds(device, &k);
    let v = unsafe { v_raw.view().reshape::<Chip, m![R64, Ns, Ds]>() };
    let v: DmTensor<bf16, Chip, Cluster, KRows, m![R64%2,Ds]> = v.to_dm(&mut device.tdma);
    let vr = rms_ds(device, &v);
    let v = value_ds(device, &v, &vr);
    let qw = weight_ds(device, q_weight);
    let kw = weight_ds(device, k_weight);
    let q = norm_ds(device, &q, &qr, &qw);
    let k = norm_ds(device, &k, &kr, &kw);
    let cos: DmTensor<bf16, Chip, Cluster, RopeRows, m![R64%2,Ds]> = cos
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let cos: DmTensor<bf16, Chip, Cluster, QRows, m![R64%2,Ds]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, RopeRows, m![R64%2,Ds]> = sin
        .gather::<m![E], m![Ds]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, QRows, m![R64%2,Ds]> = sin.to_dm(&mut device.tdma);
    let q = rope_ds(device, &q, &cos, &sin);
    let cos: DmTensor<bf16, Chip, Cluster, KRows, m![R64%2,Ds]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KRows, m![R64%2,Ds]> = sin.to_dm(&mut device.tdma);
    let k = rope_ds(device, &k, &cos, &sin);
    let k = crate::ops_fixed_e4_quality::maybe_rows::<_,_,m![R64%2]>(device, k);
    let v = crate::ops_fixed_e4_quality::maybe_rows::<_,_,m![R64%2]>(device, v);
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.view().to_hbm_view(&mut device.tdma, k_out.view_mut());
    v.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}

#[device(chip = 1)]
pub fn full_heads_r64(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![R64, Qf]>,
    k_raw: &HbmTensor<bf16, Chip, m![R64, Pf]>,
    q_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_weight: &HbmTensor<bf16, Chip, m![Df]>,
    rope_offsets: &HbmTensor<i32, Chip, m![R64]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![R64, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![R64, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![R64, Df]>,
) {
    type QRows = m![R64/2%16,Gf];
    type KRows = m![R64/2%16,1#16];
    let q = unsafe { q_raw.view().reshape::<Chip, m![R64, Gf, Df]>() };
    let k = unsafe { k_raw.view().reshape::<Chip, m![R64, Df]>() };
    let q: DmTensor<bf16, Chip, Cluster, QRows, m![R64%2,Df]> = q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KRows, m![R64%2,Df]> = k.to_dm(&mut device.tdma);
    let qr = rms_df(device, &q);
    let kr = rms_df(device, &k);
    let v = value_df(device, &k, &kr);
    let qw = weight_df(device, q_weight);
    let kw = weight_df(device, k_weight);
    let q = norm_df(device, &q, &qr, &qw);
    let k = norm_df(device, &k, &kr, &kw);
    let cos: DmTensor<bf16, Chip, Cluster, RopeRows, m![R64%2,Df]> = cos
        .gather::<m![E], m![Df]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let cos: DmTensor<bf16, Chip, Cluster, QRows, m![R64%2,Df]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, RopeRows, m![R64%2,Df]> = sin
        .gather::<m![E], m![Df]>()
        .by_byte_offsets(rope_offsets)
        .to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, QRows, m![R64%2,Df]> = sin.to_dm(&mut device.tdma);
    let q = rope_df(device, &q, &cos, &sin);
    let cos: DmTensor<bf16, Chip, Cluster, KRows, m![R64%2,Df]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KRows, m![R64%2,Df]> = sin.to_dm(&mut device.tdma);
    let k = rope_df(device, &k, &cos, &sin);
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.view().to_hbm_view(&mut device.tdma, k_out.view_mut());
    v.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}

