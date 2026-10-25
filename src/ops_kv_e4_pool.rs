//! Sole canonical E4 storage helpers. Prompt capture retains codes/scales once;
//! commit never quantizes already reconstructed BF16. No padded row is committed.
use crate::ops_request_sliding_r64_fp8_scales::Scale8;
use crate::{Chip, axes::*, ops_pool_views::Rc64};
use furiosa_opt_std::prelude::*;
pub type DataPool = m![Rc64, Ts, Ns, Ds];
pub type ScalePool = m![Rc64, Ns, Ts, Scale8];
pub(crate) fn repeat<Cl: M, Rows: M, Time: M>(
    device: &mut Device,
    s: &DmTensor<f32, Chip, Cl, Rows, m![{Time},1#8]>,
) -> DmTensor<f32, Chip, Cl, Rows, m![{ Time }, Scale8]> {
    let sv: VrfTensor<f32, Chip, Cl, Rows, m![{Time},1#8]> = device
        .sub
        .begin(s.view())
        .fetch::<Time, m![1#8]>()
        .collect::<Time, m![1#8]>()
        .to_vrf();
    let mut zero: DmTensor<f32, Chip, Cl, Rows, m![{ Time }, Scale8]> = DmTensor::new();
    zero.view_mut().memset(0.0f32, &mut device.sub);
    device
        .main
        .begin(zero.view())
        .fetch::<Time, m![Scale8]>()
        .collect::<Time, m![Scale8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &sv)
        .vector_final()
        .commit_trim::<m![Scale8]>()
        .commit()
}
#[device(chip = 1)]
pub fn zero_data_pool(device: &mut Device) -> HbmTensor<f8e4m3, Chip, DataPool> {
    type Cl = m![1#2];
    type Rows = m![Ns, Ts / 32];
    let mut zero: DmTensor<f8e4m3, Chip, Cl, Rows, m![Ts % 32, Ds]> = DmTensor::new();
    zero.view_mut()
        .memset(const { f8e4m3::from_f32(0.0) }, &mut device.sub);
    let mut out = HbmTensor::new();
    for r in 0..64 {
        zero.view().to_hbm_view(
            &mut device.tdma,
            out.view_mut()
                .tile::<m![Rc64], 1, m![Rc64=1#{!}64,Ts,Ns,Ds]>(r),
        );
    }
    out
}
#[device(chip = 1)]
pub fn zero_scale_pool(device: &mut Device) -> HbmTensor<f32, Chip, ScalePool> {
    type Cl = m![1#2];
    type Rows = m![Ns, Ts / 32];
    let mut zero: DmTensor<f32, Chip, Cl, Rows, m![Ts % 32, Scale8]> = DmTensor::new();
    zero.view_mut().memset(0.0f32, &mut device.sub);
    let mut out = HbmTensor::new();
    for r in 0..64 {
        zero.view().to_hbm_view(
            &mut device.tdma,
            out.view_mut()
                .tile::<m![Rc64], 1, m![Rc64=1#{!}64,Ns,Ts,Scale8]>(r),
        );
    }
    out
}
fn dequant_one(
    device: &mut Device,
    q: &HbmTensor<f8e4m3, Chip, DataPool>,
    s: &HbmTensor<f32, Chip, ScalePool>,
    data_base: &HbmTensor<i32, Chip, m![1]>,
    scale_base: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    type Cl = m![1#2];
    type Rows = m![Ns, Ts / 32];
    type Time = m![Ts % 32];
    let q: DmTensor<f8e4m3, Chip, Cl, Rows, m![{ Time }, Ds]> = q
        .gather::<m![Rc64], m![Ts, Ns, Ds]>()
        .by_byte_offsets(data_base)
        .to_dm(&mut device.tdma);
    let s: DmTensor<f32, Chip, Cl, Rows, m![{ Time }, Scale8]> = s
        .gather::<m![Rc64], m![Ns, Ts, Scale8]>()
        .by_byte_offsets(scale_base)
        .to_dm(&mut device.tdma);
    let s = unsafe { s.reshape::<Chip, Cl, Rows, m![{Time},1#8]>() };
    let x = crate::ops_kv_e4::reconstruct(device, &q, &s);
    x.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
#[device(chip = 1)]
pub fn dequant_ring(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, DataPool>,
    ks: &HbmTensor<f32, Chip, ScalePool>,
    v: &HbmTensor<f8e4m3, Chip, DataPool>,
    vs: &HbmTensor<f32, Chip, ScalePool>,
    data_base: &HbmTensor<i32, Chip, m![1]>,
    scale_base: &HbmTensor<i32, Chip, m![1]>,
    out_k: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    out_v: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    dequant_one(device, k, ks, data_base, scale_base, out_k);
    dequant_one(device, v, vs, data_base, scale_base, out_v);
}

#[device(chip = 1)]
pub fn capture_b16(
    device: &mut Device,
    k: &mut HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &mut HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    qk: &mut HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    qv: &mut HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &mut HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &mut HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
) {
    type Cl = m![B / 8];
    type Rows = m![B%8,Ns,1#4];
    type Time = m![1];
    let x: DmTensor<bf16, Chip, Cl, Rows, m![{ Time }, Ds]> = k.to_dm(&mut device.tdma);
    let (q, scale) = crate::ops_kv_e4::quantize(device, &x);
    let reconstructed = crate::ops_kv_e4::reconstruct(device, &q, &scale);
    reconstructed
        .view()
        .to_hbm_view(&mut device.tdma, k.view_mut());
    q.view().to_hbm_view(&mut device.tdma, qk.view_mut());
    let scale = repeat(device, &scale);
    scale.view().to_hbm_view(&mut device.tdma, ks.view_mut());
    let x: DmTensor<bf16, Chip, Cl, Rows, m![{ Time }, Ds]> = v.to_dm(&mut device.tdma);
    let (q, scale) = crate::ops_kv_e4::quantize(device, &x);
    let reconstructed = crate::ops_kv_e4::reconstruct(device, &q, &scale);
    reconstructed
        .view()
        .to_hbm_view(&mut device.tdma, v.view_mut());
    q.view().to_hbm_view(&mut device.tdma, qv.view_mut());
    let scale = repeat(device, &scale);
    scale.view().to_hbm_view(&mut device.tdma, vs.view_mut());
}

#[device(chip = 1)]
pub fn capture_p128(
    device: &mut Device,
    k: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    qk: &mut HbmTensor<f8e4m3, Chip, m![P128, Ns, Ds]>,
    qv: &mut HbmTensor<f8e4m3, Chip, m![P128, Ns, Ds]>,
    ks: &mut HbmTensor<f32, Chip, m![P128, Ns, Scale8]>,
    vs: &mut HbmTensor<f32, Chip, m![P128, Ns, Scale8]>,
) {
    type Cl = m![P128 / 64];
    type Rows = m![P128 / 2 % 32, Ns];
    type Time = m![P128 % 2];
    let x: DmTensor<bf16, Chip, Cl, Rows, m![{ Time }, Ds]> = k.to_dm(&mut device.tdma);
    let (q, scale) = crate::ops_kv_e4::quantize(device, &x);
    let reconstructed = crate::ops_kv_e4::reconstruct(device, &q, &scale);
    reconstructed
        .view()
        .to_hbm_view(&mut device.tdma, k.view_mut());
    q.view().to_hbm_view(&mut device.tdma, qk.view_mut());
    let scale = repeat(device, &scale);
    scale.view().to_hbm_view(&mut device.tdma, ks.view_mut());
    let x: DmTensor<bf16, Chip, Cl, Rows, m![{ Time }, Ds]> = v.to_dm(&mut device.tdma);
    let (q, scale) = crate::ops_kv_e4::quantize(device, &x);
    let reconstructed = crate::ops_kv_e4::reconstruct(device, &q, &scale);
    reconstructed
        .view()
        .to_hbm_view(&mut device.tdma, v.view_mut());
    q.view().to_hbm_view(&mut device.tdma, qv.view_mut());
    let scale = repeat(device, &scale);
    scale.view().to_hbm_view(&mut device.tdma, vs.view_mut());
}

#[device(chip = 1)]
pub fn commit_p128(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![P128, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![P128, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![P128, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![P128, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![P128]>,
    scale_offsets: &HbmTensor<i32, Chip, m![P128, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    type Cl = m![P128 / 64];
    type Rows = m![P128 % 64, Ns / 2];
    let k: DmTensor<f8e4m3, Chip, Cl, Rows, m![Ns % 2, Ds]> = k.to_dm(&mut device.tdma);
    let v: DmTensor<f8e4m3, Chip, Cl, Rows, m![Ns % 2, Ds]> = v.to_dm(&mut device.tdma);
    let ks: DmTensor<f32, Chip, Cl, Rows, m![Ns % 2, Scale8]> = ks.to_dm(&mut device.tdma);
    let vs: DmTensor<f32, Chip, Cl, Rows, m![Ns % 2, Scale8]> = vs.to_dm(&mut device.tdma);
    pk.view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(data_offsets.view())
        .from_dm(&mut device.tdma, k);
    pv.view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(data_offsets.view())
        .from_dm(&mut device.tdma, v);
    pks.view_mut()
        .scatter::<m![Rc64, Ns, Ts], m![Scale8]>()
        .by_byte_offsets(scale_offsets.view())
        .from_dm(&mut device.tdma, ks);
    pvs.view_mut()
        .scatter::<m![Rc64, Ns, Ts], m![Scale8]>()
        .by_byte_offsets(scale_offsets.view())
        .from_dm(&mut device.tdma, vs);
}
fn commit_b16<const VALID: usize>(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = { VALID }]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = { VALID }, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    type Cl = m![1#2];
    type Rows = m![1#256];
    let k: DmTensor<f8e4m3, Chip, Cl, Rows, m![B = { VALID }, Ns, Ds]> = k
        .view()
        .tile::<m![B], VALID, m![B={VALID}#16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    let v: DmTensor<f8e4m3, Chip, Cl, Rows, m![B = { VALID }, Ns, Ds]> = v
        .view()
        .tile::<m![B], VALID, m![B={VALID}#16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    let ks: DmTensor<f32, Chip, Cl, Rows, m![B = { VALID }, Ns, Scale8]> = ks
        .view()
        .tile::<m![B], VALID, m![B={VALID}#16,Ns,Scale8]>(0)
        .to_dm(&mut device.tdma);
    let vs: DmTensor<f32, Chip, Cl, Rows, m![B = { VALID }, Ns, Scale8]> = vs
        .view()
        .tile::<m![B], VALID, m![B={VALID}#16,Ns,Scale8]>(0)
        .to_dm(&mut device.tdma);
    pk.view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(data_offsets.view())
        .from_dm(&mut device.tdma, k);
    pv.view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(data_offsets.view())
        .from_dm(&mut device.tdma, v);
    pks.view_mut()
        .scatter::<m![Rc64, Ns, Ts], m![Scale8]>()
        .by_byte_offsets(scale_offsets.view())
        .from_dm(&mut device.tdma, ks);
    pvs.view_mut()
        .scatter::<m![Rc64, Ns, Ts], m![Scale8]>()
        .by_byte_offsets(scale_offsets.view())
        .from_dm(&mut device.tdma, vs);
}
#[device(chip = 1)]
pub fn commit_b16_1(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 1]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 1, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<1>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_2(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 2]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 2, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<2>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_3(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 3]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 3, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<3>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_4(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 4]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 4, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<4>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_5(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 5]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 5, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<5>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_6(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 6]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 6, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<6>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_7(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 7]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 7, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<7>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_8(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 8]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 8, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<8>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_9(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 9]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 9, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<9>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_10(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 10]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 10, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<10>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_11(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 11]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 11, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<11>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_12(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 12]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 12, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<12>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_13(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 13]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 13, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<13>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_14(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 14]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 14, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<14>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_15(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 15]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 15, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<15>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
#[device(chip = 1)]
pub fn commit_b16_16(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    ks: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    vs: &HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    data_offsets: &HbmTensor<i32, Chip, m![B = 16]>,
    scale_offsets: &HbmTensor<i32, Chip, m![B = 16, Ns]>,
    pk: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, DataPool>,
    pks: &mut HbmTensor<f32, Chip, ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, ScalePool>,
) {
    commit_b16::<16>(
        device,
        k,
        v,
        ks,
        vs,
        data_offsets,
        scale_offsets,
        pk,
        pv,
        pks,
        pvs,
    );
}
