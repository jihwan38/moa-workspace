//! Experimental native-FP8 sliding Q projection with original E4 weights.
//! The established row quantizer precedes an H960 F32 partial reduction.
//! Apply input dequant before the original BF16-dot then BF16 channel-scale boundary.
use crate::{
    Chip,
    axes::{H, B, Qs},
};
use furiosa_opt_std::prelude::*;
type Cluster = m![B / 8];
type QuantRows = m![B % 8 # 256];
type Parts = m![Qs / 64, H / 960];
type Rows = m![Qs/64,1 # 4];
fn quantize_hidden(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, QuantRows, m![H]>,
) -> (
    DmTensor<f8e4m3, Chip, Cluster, QuantRows, m![H]>,
    DmTensor<f32, Chip, Cluster, QuantRows, m![1 # 8]>,
) {
    let amax: DmTensor<f32, Chip, Cluster, QuantRows, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 #4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 #8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.925929944387236e-34f32)
        .vector_final()
        .commit_trim::<m![1 #8]>()
        .commit();
    let scale: DmTensor<f32, Chip, Cluster, QuantRows, m![1 #8]> = device
        .main
        .begin(amax.view())
        .fetch::<m![1], m![1 #8]>()
        .collect::<m![1], m![1 #8]>()
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
    let sv: VrfTensor<f32, Chip, Cluster, QuantRows, m![1 #8]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![1 #8]>()
        .collect::<m![1], m![1 #8]>()
        .to_vrf();
    let q: DmTensor<f8e4m3, Chip, Cluster, QuantRows, m![H]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &sv)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H %8 #32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    (q, scale)
}

fn pack_scales(
    device: &mut Device,
    xscale: &DmTensor<f32, Chip, Cluster, Rows, m![B % 8,1 # 8]>,
) -> DmTensor<f32, Chip, Cluster, Rows, m![B % 8]> {
    // Keep the 32-byte scalar slot for aligned inter-slice DMA, then pack by TU.
    let xscale: DmTensor<f32, Chip, Cluster, Rows, m![B % 8]> = device
        .sub
        .begin(xscale.view())
        .fetch::<m![B % 8], m![1 # 8]>()
        .collect::<m![B % 8], m![1 # 8]>()
        .transpose::<m![B % 8 / 2], m![B % 8 % 2 # 8]>()
        .commit_trim::<m![B % 8 % 2]>()
        .commit();
    xscale
}
fn matrix_query(
    device: &mut Device,
    input: &DmTensor<f8e4m3, Chip, Cluster, Parts, m![B % 8, H % 960]>,
    xscale: &DmTensor<f32, Chip, Cluster, Rows, m![B % 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
) -> DmTensor<bf16, Chip, Cluster, Rows, m![Qs % 64, B % 8]> {
    let weight: DmTensor<f8e4m3, Chip, Cluster, Parts, m![Qs % 64, H % 960]> =
        weight.to_dm(&mut device.tdma);
    let xs: VrfTensor<f32, Chip, Cluster, Rows, m![B % 8]> = device
        .sub
        .begin(xscale.view())
        .fetch::<m![B / 8 % 1], m![B % 8]>()
        .collect::<m![B / 8 % 1], m![B % 8]>()
        .to_vrf();
    let x: TrfTensor<f8e4m3, Chip, Cluster, Parts, m![B % 8], m![B / 8 % 1, H % 960]> =
        device
            .sub
            .begin(input.view())
            .fetch::<m![B % 8, B / 8 % 1, H / 32 % 30], m![H % 32]>()
            .collect::<m![B % 8, B / 8 % 1, H / 32 % 30], m![H % 32]>()
            .to_trf();
    device
        .main
        .begin(weight.view())
        .fetch::<m![Qs % 64, H / 32 % 30], m![H % 32]>()
        .collect::<m![Qs % 64, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![Qs % 64, H / 64 % 15, B / 8 % 1], m![H % 64], _, _, _>(&x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 64, B / 8 % 1]>()
        .contract_lane::<m![Qs % 64, B / 8 % 1], m![B % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Rows, m![Qs % 64, B / 8 % 1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs % 64, B / 4 % 2], m![B % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &xs)
        .vector_widen_concat::<m![Qs % 64, B / 8 % 1], m![B % 8]>()
        .vector_final()
        .cast::<bf16, m![B%8#16]>()
        .commit_trim::<m![B % 8]>()
        .commit()
}

fn finish_query(
    device: &mut Device,
    dot: &DmTensor<bf16, Chip, Cluster, Rows, m![Qs % 64, B % 8]>,
    scale: &HbmTensor<bf16, Chip, m![Qs]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Qs]>,
) {
    let scale: DmTensor<bf16, Chip, Cluster, Rows, m![Qs % 64]> = scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, Rows, m![Qs % 64]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Qs % 64]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 8], m![Qs % 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, Rows, m![Qs % 64, B % 8]> = device
        .main
        .begin(dot.view())
        .fetch::<m![Qs % 64, B / 8 % 1], m![B % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs % 64, B / 8 % 1], m![B % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs % 64, B / 4 % 2], m![B % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Qs % 64, B / 8 % 1], m![B % 8]>()
        .vector_final()
        .cast::<bf16, m![B%8#16]>()
        .commit_trim::<m![B % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, Rows, m![B % 8, Qs % 64]> = device
        .main
        .begin(result.view())
        .fetch::<m![B / 8 % 1, Qs % 64], m![B % 8 # 16]>()
        .collect::<m![B / 8 % 1, Qs % 64], m![B % 8 # 16]>()
        .transpose::<m![B / 8 % 1, Qs / 4 % 16, B % 8], m![Qs%4#16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();
    result
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn project_query_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    scale: &HbmTensor<bf16, Chip, m![Qs]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Qs]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, QuantRows, m![H]> = input.to_dm(&mut device.tdma);
    let (input, xscale) = quantize_hidden(device, &input);
    let input: DmTensor<f8e4m3, Chip, Cluster, Parts, m![B % 8, H % 960]> =
        input.to_dm(&mut device.tdma);
    let xscale: DmTensor<f32, Chip, Cluster, Rows, m![B % 8,1#8]> =
        xscale.to_dm(&mut device.tdma);
    let xscale = pack_scales(device, &xscale);
    let dot = matrix_query(device, &input, &xscale, weight);
    finish_query(device, &dot, scale, output);
}

/// Diagnostic row quantization output; only slot0 of each padded scale is logical.
#[device(chip = 1)]
pub fn quantize_input(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    output: &mut HbmTensor<f8e4m3, Chip, m![B, H]>,
    scale: &mut HbmTensor<f32, Chip, m![B,1#8]>,
) {
    quantize_input_stage(device,input,output,scale);
}

pub(crate) fn quantize_input_stage(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    output: &mut HbmTensor<f8e4m3, Chip, m![B, H]>,
    scale: &mut HbmTensor<f32, Chip, m![B,1#8]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, QuantRows, m![H]> = input.to_dm(&mut device.tdma);
    let (q, s) = quantize_hidden(device, &input);
    q.view().to_hbm_view(&mut device.tdma, output.view_mut());
    s.view().to_hbm_view(&mut device.tdma, scale.view_mut());
}

/// Diagnostic core with externally supplied quantized rows and padded scales.
#[device(chip = 1)]
pub fn project_query_quantized(
    device: &mut Device,
    input: &HbmTensor<f8e4m3, Chip, m![B, H]>,
    xscale: &HbmTensor<f32, Chip, m![B,1#8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    scale: &HbmTensor<bf16, Chip, m![Qs]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Qs]>,
) {
    let input: DmTensor<f8e4m3, Chip, Cluster, Parts, m![B % 8, H % 960]> =
        input.to_dm(&mut device.tdma);
    let xscale: DmTensor<f32, Chip, Cluster, Rows, m![B % 8,1#8]> =
        xscale.to_dm(&mut device.tdma);
    let xscale = pack_scales(device, &xscale);
    let dot = matrix_query(device, &input, &xscale, weight);
    finish_query(device, &dot, scale, output);
}
