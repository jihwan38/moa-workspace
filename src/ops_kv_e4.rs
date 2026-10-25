//! Per-token/head E4 quantization and BF16 reconstruction. Q remains unchanged.
use crate::ops_request_sliding_r64_fp8_scales::Scale8;
use crate::{
    Chip,
    axes::{B, Ds, Ns, P128, R64, Ts},
    ops_pool_views::Rc64,
};
use furiosa_opt_std::prelude::*;
pub(crate) fn quantize<Cl: M, Rows: M, RowTime: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cl, Rows, m![{ RowTime }, Ds]>,
) -> (
    DmTensor<f8e4m3, Chip, Cl, Rows, m![{ RowTime }, Ds]>,
    DmTensor<f32, Chip, Cl, Rows, m![{RowTime},1 # 8]>,
) {
    let amax: DmTensor<f32, Chip, Cl, Rows, m![{RowTime},1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![{ RowTime }, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ RowTime }, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![{ RowTime }, Ds / 4], m![Ds % 4]>()
        .vector_intra_slice_reduce::<Ds, m![{ RowTime }], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.925929944387236e-34f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let scale: DmTensor<f32, Chip, Cl, Rows, m![{RowTime},1 # 8]> = device
        .main
        .begin(amax.view())
        .fetch::<m![{ RowTime }], m![1 # 8]>()
        .collect::<m![{ RowTime }], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7f80_0000i32)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0078125f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let sv: VrfTensor<f32, Chip, Cl, Rows, m![{RowTime},1 # 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![{ RowTime }], m![1 # 8]>()
        .collect::<m![{ RowTime }], m![1 # 8]>()
        .to_vrf();
    let q: DmTensor<f8e4m3, Chip, Cl, Rows, m![{ RowTime }, Ds]> = device
        .main
        .begin(x.view())
        .fetch::<m![{ RowTime }, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ RowTime }, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ RowTime }, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &sv)
        .vector_widen_concat::<m![{ RowTime }, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Ds%8 # 32]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    (q, scale)
}
pub(crate) fn reconstruct<Cl: M, Rows: M, RowTime: M>(
    device: &mut Device,
    q: &DmTensor<f8e4m3, Chip, Cl, Rows, m![{ RowTime }, Ds]>,
    scale: &DmTensor<f32, Chip, Cl, Rows, m![{RowTime},1#8]>,
) -> DmTensor<bf16, Chip, Cl, Rows, m![{ RowTime }, Ds]> {
    let scale: VrfTensor<f32, Chip, Cl, Rows, m![{RowTime},1 # 8]> = device
        .sub
        .begin(scale.view())
        .fetch::<RowTime, m![1 # 8]>()
        .collect::<RowTime, m![1 # 8]>()
        .to_vrf();
    device
        .main
        .begin(q.view())
        .fetch::<m![{ RowTime }, Ds / 32], m![Ds % 32]>()
        .fetch_table_lookup::<bf16>()
        .fetch_cast::<f32>()
        .collect::<m![{ RowTime }, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ RowTime }, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![{ RowTime }, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds%8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
pub(crate) fn roundtrip<Cl: M, Rows: M, RowTime: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cl, Rows, m![{ RowTime }, Ds]>,
) -> DmTensor<bf16, Chip, Cl, Rows, m![{ RowTime }, Ds]> {
    let (q, scale) = quantize(device, x);
    reconstruct(device, &q, &scale)
}

fn repeat_scale<Cl: M, Rows: M>(
    device: &mut Device,
    s: &DmTensor<f32, Chip, Cl, Rows, m![1,1 # 8]>,
) -> DmTensor<f32, Chip, Cl, Rows, m![Scale8]> {
    let sv: VrfTensor<f32, Chip, Cl, Rows, m![1 # 8]> = device
        .sub
        .begin(s.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let mut zero: DmTensor<f32, Chip, Cl, Rows, m![Scale8]> = DmTensor::new();
    zero.view_mut().memset(0.0f32, &mut device.sub);
    device
        .main
        .begin(zero.view())
        .fetch::<m![1], m![Scale8]>()
        .collect::<m![1], m![Scale8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &sv)
        .vector_final()
        .commit_trim::<m![Scale8]>()
        .commit()
}
#[device(chip = 1)]
pub fn quantize_commit_r64(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![R64, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![R64, Ns, Ds]>,
    data_offsets: &HbmTensor<i32, Chip, m![R64]>,
    scale_offsets: &HbmTensor<i32, Chip, m![R64, Ns]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    scales_k: &mut HbmTensor<f32, Chip, m![Rc64, Ns, Ts, Scale8]>,
    scales_v: &mut HbmTensor<f32, Chip, m![Rc64, Ns, Ts, Scale8]>,
) {
    type Cl = m![R64 / 32];
    type Rows = m![R64 % 32, Ns];
    let k: DmTensor<bf16, Chip, Cl, Rows, m![1, Ds]> = k.to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cl, Rows, m![1, Ds]> = v.to_dm(&mut device.tdma);
    let (k, ks) = quantize::<Cl, Rows, m![1]>(device, &k);
    let (v, vs) = quantize::<Cl, Rows, m![1]>(device, &v);
    let ks = repeat_scale(device, &ks);
    let vs = repeat_scale(device, &vs);
    pool_k
        .view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(data_offsets.view())
        .from_dm(&mut device.tdma, k);
    pool_v
        .view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(data_offsets.view())
        .from_dm(&mut device.tdma, v);
    scales_k
        .view_mut()
        .scatter::<m![Rc64, Ns, Ts], m![Scale8]>()
        .by_byte_offsets(scale_offsets.view())
        .from_dm(&mut device.tdma, ks);
    scales_v
        .view_mut()
        .scatter::<m![Rc64, Ns, Ts], m![Scale8]>()
        .by_byte_offsets(scale_offsets.view())
        .from_dm(&mut device.tdma, vs);
}

#[device(chip = 1)]
pub fn roundtrip_r64(
    device: &mut Device,
    k: &mut HbmTensor<bf16, Chip, m![R64, Ns, Ds]>,
    v: &mut HbmTensor<bf16, Chip, m![R64, Ns, Ds]>,
) {
    type Cl = m![R64 / 32];
    type Rows = m![R64 % 32, Ns];
    type Time = m![1];
    let kd: DmTensor<bf16, Chip, Cl, Rows, m![{ Time }, Ds]> = k.to_dm(&mut device.tdma);
    let vd: DmTensor<bf16, Chip, Cl, Rows, m![{ Time }, Ds]> = v.to_dm(&mut device.tdma);
    let kd = roundtrip::<Cl, Rows, Time>(device, &kd);
    let vd = roundtrip::<Cl, Rows, Time>(device, &vd);
    kd.view().to_hbm_view(&mut device.tdma, k.view_mut());
    vd.view().to_hbm_view(&mut device.tdma, v.view_mut());
}

#[device(chip = 1)]
pub fn roundtrip_b16(
    device: &mut Device,
    k: &mut HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &mut HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
) {
    type Cl = m![B / 8];
    type Rows = m![B%8,Ns,1 # 4];
    type Time = m![1];
    let kd: DmTensor<bf16, Chip, Cl, Rows, m![{ Time }, Ds]> = k.to_dm(&mut device.tdma);
    let vd: DmTensor<bf16, Chip, Cl, Rows, m![{ Time }, Ds]> = v.to_dm(&mut device.tdma);
    let kd = roundtrip::<Cl, Rows, Time>(device, &kd);
    let vd = roundtrip::<Cl, Rows, Time>(device, &vd);
    kd.view().to_hbm_view(&mut device.tdma, k.view_mut());
    vd.view().to_hbm_view(&mut device.tdma, v.view_mut());
}

#[device(chip = 1)]
pub fn roundtrip_p128(
    device: &mut Device,
    k: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
) {
    type Cl = m![P128 / 64];
    type Rows = m![P128 / 2 % 32, Ns];
    type Time = m![P128 % 2];
    let kd: DmTensor<bf16, Chip, Cl, Rows, m![{ Time }, Ds]> = k.to_dm(&mut device.tdma);
    let vd: DmTensor<bf16, Chip, Cl, Rows, m![{ Time }, Ds]> = v.to_dm(&mut device.tdma);
    let kd = roundtrip::<Cl, Rows, Time>(device, &kd);
    let vd = roundtrip::<Cl, Rows, Time>(device, &vd);
    kd.view().to_hbm_view(&mut device.tdma, k.view_mut());
    vd.view().to_hbm_view(&mut device.tdma, v.view_mut());
}

#[device(chip = 1)]
pub fn roundtrip_scalar(
    device: &mut Device,
    k: &mut HbmTensor<bf16, Chip, m![Ns, Ds]>,
    v: &mut HbmTensor<bf16, Chip, m![Ns, Ds]>,
) {
    type Cl = m![1 # 2];
    type Rows = m![Ns,1 # 32];
    type Time = m![1];
    let kd: DmTensor<bf16, Chip, Cl, Rows, m![{ Time }, Ds]> = k.to_dm(&mut device.tdma);
    let vd: DmTensor<bf16, Chip, Cl, Rows, m![{ Time }, Ds]> = v.to_dm(&mut device.tdma);
    let kd = roundtrip::<Cl, Rows, Time>(device, &kd);
    let vd = roundtrip::<Cl, Rows, Time>(device, &vd);
    kd.view().to_hbm_view(&mut device.tdma, k.view_mut());
    vd.view().to_hbm_view(&mut device.tdma, v.view_mut());
}
