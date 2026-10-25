//! Approximate full-attention O: per-K1024 FP8 activation quantization, original E4 weights.
//! Eight original BF16 partials and balanced BF16 tree/channel scales retained.
//! Canonical P128xH output, unchanged full-attention computation; no serving dispatch.
use crate::{
    Chip,
    axes::{Df, Gf, H, P128, Qf},
};
use furiosa_opt_std::prelude::*;
pub type Output = m![P128, H];
type Cluster = m![P128 / 64];
type Parts = m![Qf / 1024, H / 120];
type DownRows = m![H/120,1#8];
type QuantRows = m![Qf / 1024, P128 % 4, P128 / 8 % 8];

fn add_partials_wide(
    device: &mut Device,
    left: &DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, P128 % 64]>,
    right: &DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, P128 % 64]>,
) -> DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, P128 % 64]> {
    let mut result: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, P128 % 64]> =
        DmTensor::new();
    for chunk in 0..4 {
        let left: VrfTensor<f32, Chip, Cluster, DownRows, m![H % 120, P128 % 64 = 16]> = device
            .sub
            .begin(
                left.view()
                    .tile::<m![P128 % 64], 16, m![H % 120,P128%64 = 16 # 64]>(16 * chunk),
            )
            .fetch::<m![H % 120], m![P128 % 64 = 16]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 120, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
            .to_vrf();
        device
            .main
            .begin(
                right
                    .view()
                    .tile::<m![P128 % 64], 16, m![H % 120,P128%64 = 16 # 64]>(16 * chunk),
            )
            .fetch::<m![H % 120], m![P128 % 64 = 16]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 120, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &left)
            .vector_final()
            .cast::<bf16, m![P128%64 = 16 % 8 # 16]>()
            .commit_trim::<m![P128 % 64 = 16 % 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![P128 % 64], 16, m![H % 120,P128%64 = 16 #{!} 64]>(16 * chunk),
            );
    }
    result
}

fn add_gathered_pair_full(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, DownRows, m![Qf / 1024, H % 120, P128 % 64]>,
    offset: usize,
) -> DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, P128 % 64]> {
    let mut result: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, P128 % 64]> =
        DmTensor::new();
    for chunk in 0..4 {
        let left: VrfTensor<f32, Chip, Cluster, DownRows, m![H % 120, P128 % 64 = 16]> = device
            .sub
            .begin(
                input
                    .view()
                    .tile::<m![Qf / 1024], 1, m![1 # 8,H%120,P128%64]>(offset)
                    .tile::<m![P128 % 64], 16, m![H % 120 # 960,P128%64 = 16 # 64]>(16 * chunk),
            )
            .fetch::<m![H % 120], m![P128 % 64 = 16]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 120, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
            .to_vrf();
        device
            .main
            .begin(
                input
                    .view()
                    .tile::<m![Qf / 1024], 1, m![1 # 8,H%120,P128%64]>(offset + 1)
                    .tile::<m![P128 % 64], 16, m![H % 120 # 960,P128%64 = 16 # 64]>(16 * chunk),
            )
            .fetch::<m![H % 120], m![P128 % 64 = 16]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 120, P128 % 64 = 16 / 8], m![P128 % 64 = 16 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &left)
            .vector_final()
            .cast::<bf16, m![P128%64 = 16 % 8 # 16]>()
            .commit_trim::<m![P128 % 64 = 16 % 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![P128 % 64], 16, m![H % 120,P128%64 = 16 #{!} 64]>(16 * chunk),
            );
    }
    result
}

fn scale_output_wide(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, P128 % 64]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, H]>,
) {
    let scale: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120]> = scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, DownRows, m![H % 120]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120, P128 % 64]> = device
        .main
        .begin(input.view())
        .fetch::<m![H % 120, P128 / 8 % 8], m![P128 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120, P128 / 8 % 8], m![P128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
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
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}
fn quantize_local(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, QuantRows, m![P128 / 4 % 2, Qf % 1024]>,
) -> (
    DmTensor<f8e4m3, Chip, Cluster, QuantRows, m![P128 / 4 % 2, Qf % 1024]>,
    DmTensor<f32, Chip, Cluster, QuantRows, m![P128/4%2,1 # 8]>,
) {
    let amax: DmTensor<f32, Chip, Cluster, QuantRows, m![P128/4%2,1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![P128 / 4 % 2, Qf / 16 % 64], m![Qf % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 / 4 % 2, Qf / 8 % 128], m![Qf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![P128 / 4 % 2, Qf / 4 % 256], m![Qf % 4]>()
        .vector_intra_slice_reduce::<Qf, m![P128 / 4 % 2], m![1 #4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 #8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.925929944387236e-34f32)
        .vector_final()
        .commit_trim::<m![1 #8]>()
        .commit();
    let scale: DmTensor<f32, Chip, Cluster, QuantRows, m![P128/4%2,1 #8]> = device
        .main
        .begin(amax.view())
        .fetch::<m![P128 / 4 % 2, 1], m![1 #8]>()
        .collect::<m![P128 / 4 % 2, 1], m![1 #8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7f80_0000i32)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 #4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0078125f32)
        .vector_widen_pad::<m![1 #8]>()
        .vector_final()
        .commit_trim::<m![1 #8]>()
        .commit();
    let sv: VrfTensor<f32, Chip, Cluster, QuantRows, m![P128/4%2,1 #8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![P128 / 4 % 2, 1], m![1 #8]>()
        .collect::<m![P128 / 4 % 2, 1], m![1 #8]>()
        .to_vrf();
    let q: DmTensor<f8e4m3, Chip, Cluster, QuantRows, m![P128 / 4 % 2, Qf % 1024]> = device
        .main
        .begin(x.view())
        .fetch::<m![P128 / 4 % 2, Qf / 16 % 64], m![Qf % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![P128 / 4 % 2, Qf / 8 % 128], m![Qf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128 / 4 % 2, Qf / 4 % 256], m![Qf % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &sv)
        .vector_widen_concat::<m![P128 / 4 % 2, Qf / 8 % 128], m![Qf % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qf %8 #32]>()
        .commit_trim::<m![Qf % 8]>()
        .commit();
    (q, scale)
}
fn partials(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    weight: &DmTensor<f8e4m3, Chip, Cluster, Parts, m![H % 120, Qf % 1024]>,
) -> DmTensor<bf16, Chip, Cluster, Parts, m![H % 120, P128 % 64]> {
    let flat: HbmTensorView<'_, bf16, Chip, m![P128, Qf]> = unsafe { input.view().reshape() };
    let input: DmTensor<bf16, Chip, Cluster, QuantRows, m![P128 / 4 % 2, Qf % 1024]> =
        flat.to_dm(&mut device.tdma);
    let (q, scale) = quantize_local(device, &input);
    let scales: DmTensor<f32, Chip, Cluster, Parts, m![P128%64,1#8]> =
        scale.to_dm(&mut device.tdma);
    let scales: DmTensor<f32, Chip, Cluster, Parts, m![P128 % 64]> = device
        .sub
        .begin(scales.view())
        .fetch::<m![P128 % 64], m![1#8]>()
        .collect::<m![P128 % 64], m![1#8]>()
        .transpose::<m![P128 % 64 / 2], m![P128%64%2#8]>()
        .commit_trim::<m![P128 % 64 % 2]>()
        .commit();
    let scales: VrfTensor<f32, Chip, Cluster, Parts, m![P128 % 64]> = device
        .sub
        .begin(scales.view())
        .fetch::<m![P128 / 8 % 8], m![P128 % 8]>()
        .collect::<m![P128 / 8 % 8], m![P128 % 8]>()
        .to_vrf();
    let x: TrfTensor<f8e4m3, Chip, Cluster, Parts, m![P128 % 8], m![P128 / 8 % 8, Qf % 1024]> =
        device
            .sub
            .begin(q.view())
            .fetch::<m![P128 / 4 % 2], m![Qf % 1024]>()
            .switch::<Parts, m![P128 / 4 % 2, P128 % 4, P128 / 8 % 8]>(
                SwitchConfig::CustomBroadcast { ring_size: 32 },
            )
            .collect::<m![P128 % 8, P128 / 8 % 8, Qf / 32 % 32], m![Qf % 32]>()
            .to_trf();
    device
        .main
        .begin(weight.view())
        .fetch::<m![H % 120, Qf / 32 % 32], m![Qf % 32]>()
        .collect::<m![H % 120, Qf / 32 % 32], m![Qf % 32]>()
        .contract_outer::<m![H % 120, Qf / 64 % 16, P128 / 8 % 8], m![Qf % 64], _, _, _>(&x)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120, P128 / 8 % 8]>()
        .contract_lane::<m![H % 120, P128 / 8 % 8], m![P128 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120, P128 / 4 % 16], m![P128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scales)
        .vector_widen_concat::<m![H % 120, P128 / 8 % 8], m![P128 % 8]>()
        .vector_final()
        .cast::<bf16, m![P128%8#16]>()
        .commit_trim::<m![P128 % 8]>()
        .commit()
}
fn project_one(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    weight: &DmTensor<f8e4m3, Chip, Cluster, Parts, m![H % 120, Qf % 1024]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    output: &mut HbmTensor<bf16, Chip, Output>,
) {
    let partial = partials(device, input, weight);
    let gathered: DmTensor<bf16, Chip, Cluster, DownRows, m![Qf / 1024, H % 120, P128 % 64]> =
        partial.to_dm(&mut device.tdma);
    let p01 = add_gathered_pair_full(device, &gathered, 0);
    let p23 = add_gathered_pair_full(device, &gathered, 2);
    let p45 = add_gathered_pair_full(device, &gathered, 4);
    let p67 = add_gathered_pair_full(device, &gathered, 6);
    let p0123 = add_partials_wide(device, &p01, &p23);
    let p4567 = add_partials_wide(device, &p45, &p67);
    let result = add_partials_wide(device, &p0123, &p4567);
    scale_output_wide(device, &result, scale, output);
}

#[device(chip = 1)]
pub fn project_full_output_four(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    b: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    c: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    d: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    ao: &mut HbmTensor<bf16, Chip, Output>,
    bo: &mut HbmTensor<bf16, Chip, Output>,
    co: &mut HbmTensor<bf16, Chip, Output>,
    od: &mut HbmTensor<bf16, Chip, Output>,
) {
    let weight: DmTensor<f8e4m3, Chip, Cluster, Parts, m![H % 120, Qf % 1024]> =
        weight.to_dm(&mut device.tdma);
    for _once in 0..1 {
        project_one(device, a, &weight, scale, ao);
    }
    for _once in 0..1 {
        project_one(device, b, &weight, scale, bo);
    }
    for _once in 0..1 {
        project_one(device, c, &weight, scale, co);
    }
    for _once in 0..1 {
        project_one(device, d, &weight, scale, od);
    }
}
#[device(chip = 1)]
pub fn quantize_full_input(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    qout: &mut HbmTensor<f8e4m3, Chip, m![P128, Qf]>,
    sout: &mut HbmTensor<f32, Chip, m![P128,Qf/1024,1#8]>,
) {
    let flat: HbmTensorView<'_, bf16, Chip, m![P128, Qf]> = unsafe { input.view().reshape() };
    let x: DmTensor<bf16, Chip, Cluster, QuantRows, m![P128 / 4 % 2, Qf % 1024]> =
        flat.to_dm(&mut device.tdma);
    let (q, s) = quantize_local(device, &x);
    q.view().to_hbm_view(&mut device.tdma, qout.view_mut());
    s.view().to_hbm_view(&mut device.tdma, sout.view_mut());
}
#[device(chip = 1)]
pub fn project_full_partials(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    out: &mut HbmTensor<bf16, Chip, m![P128, Qf / 1024, H]>,
) {
    let weight: DmTensor<f8e4m3, Chip, Cluster, Parts, m![H % 120, Qf % 1024]> =
        weight.to_dm(&mut device.tdma);
    let partial = partials(device, input, &weight);
    partial.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
