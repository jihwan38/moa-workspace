//! Exact-rational finite-BF16 P512 hidden15360 compact-scale resource foundation using a generated128-entry table.
//! No model caller. NPU equivalence remains a separate gate from the CPU proof.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{P512, L}};

axes![Mantissa = 128, Record = 8];
type Cluster = m![P512 / 256];
type Rows = m![P512 % 256];

/// The live lane of table[m-128] is F32bits(7*ceil(2^20/m)), m=128..255.
/// All seven remaining physical lanes must also be initialized by the caller.
#[device(chip = 1)]
pub fn quantize_row(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P512, L]>,
    table: &HbmTensor<i32, Chip, m![Mantissa, Record]>,
    output: &mut HbmTensor<i32, Chip, m![P512, L]>,
    scales: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Rows, m![L]> = input.to_dm(&mut device.tdma);
    let amax: DmTensor<f32, Chip, Cluster, Rows, m![1 # 8]> = device.main.begin(x.view())
        .fetch::<m![L / 16], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![L / 8], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![L / 4], m![L % 4]>()
        .vector_intra_slice_reduce::<L, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.925929944387236e-34f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale: DmTensor<f32, Chip, Cluster, Rows, m![1 # 8]> = device.main.begin(amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.1428571492433548f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let indices: DmTensor<i32, Chip, Cluster, Rows, m![1 # 8]> = device.main.begin(amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::LogicRightShift, 16i32)
        .vector_logic(LogicBinaryOpI32::BitAnd, 127i32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // All512 rows participate:256 packed indices(1024B) percluster onPE0.
    // This is a real DM redistribution, followed by SPM placement on PE0.
    // The row indices are generated on device; there is no host index readback.
    // Transfer complete 32-byte records first: direct four-byte row packing
    // fails the public DMA sequencer's eight-byte stride requirement.
    let records: DmTensor<i32, Chip, Cluster, m![1 # 256], m![P512 % 256, 1 # 8]> =
        indices.to_dm(&mut device.tdma);
    let packed: DmTensor<i32, Chip, Cluster, m![1 # 256], m![P512 % 256]> = device.main.begin(records.view())
        .fetch::<m![P512 % 256], m![1 # 8]>().collect::<m![P512 % 256], m![1 # 8]>()
        .transpose::<m![P512 % 256 / 2], m![P512 % 2 # 8]>().commit_trim::<m![P512 % 2]>().commit();
    let positions: SpmTensor<i32, Chip, Cluster, m![1 # 4], m![P512 % 256]> = packed.to_spm(&mut device.tdma);
    let selected: DmTensor<i32, Chip, Cluster, Rows, m![Record]> = table
        .gather::<m![Mantissa], m![Record]>()
        .by_positions::<m![P512]>(&positions)
        .to_dm(&mut device.tdma);
    let u_bits: VrfTensor<i32, Chip, Cluster, Rows, m![1 # 8]> = device.sub.begin(selected.view().tile::<m![Record], 1, m![Record = 1 # 8]>(0))
        .fetch::<m![1], m![Record = 1 # 8]>().collect::<m![1], m![Record = 1 # 8]>().to_vrf();
    // The proof bounds the constructed positive normal T exponent to1..241.
    // Keep this multiplier separate from the reconstruction scale A*RN32(1/7).
    let multiplier: VrfTensor<f32, Chip, Cluster, Rows, m![1 # 8]> = device.sub.begin(amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::LogicRightShift, 23i32)
        .vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, 114i32)
        .vector_fxp(FxpBinaryOp::LeftShift, 23i32)
        .vector_clip(ClipBinaryOpI32::AddFxp, &u_bits)
        .vector_reinterpret::<f32>().vector_final().to_vrf();
    let codes: DmTensor<i32, Chip, Cluster, Rows, m![L]> = device.main.begin(x.view())
        .fetch::<m![L / 16], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![L / 8], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &multiplier)
        .vector_widen_concat::<m![L / 8], m![L % 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::Max, -7i32)
        .vector_clip(ClipBinaryOpI32::Min, 7i32)
        .vector_final().commit_trim::<m![L % 8]>().commit();
    codes.view().to_hbm_view(&mut device.tdma, output.view_mut());
    // Exact scale bits only: real full-record movement, then two-row TU
    // compaction removes padding before the contiguous canonical HBM store.
    let scale_records: DmTensor<f32, Chip, Cluster, m![1 # 256], m![P512 % 256, 1 # 8]> =
        scale.to_dm(&mut device.tdma);
    let scale_dense: DmTensor<f32, Chip, Cluster, m![1 # 256], m![P512 % 256]> = device.main.begin(scale_records.view())
        .fetch::<m![P512 % 256], m![1 # 8]>().collect::<m![P512 % 256], m![1 # 8]>()
        .transpose::<m![P512 % 256 / 2], m![P512 % 2 # 8]>().commit_trim::<m![P512 % 2]>().commit();
    scale_dense.view().to_hbm_view(&mut device.tdma, scales.view_mut());
}
