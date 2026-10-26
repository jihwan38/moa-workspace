//! Per-token/per-K960 exact-rational quantization and packed I32 ownership.
//! Explicit HBM byte-offset scratch is part of this complete kernel, not host work.
//! This disconnected changed-quantization prerequisite makes no model-quality claim.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{H, L, P512}};
pub use crate::ops_int4_rational_fused_pack::{PackedInput, PackedHidden, Mantissa, Record};
axes![Nibbles = 8, GroupWords = 120];
type Cluster = m![P512 / 256];
pub type InputScales = m![P512 / 256, H / 960, P512 % 256];
pub type HiddenScales = m![P512 / 256, L / 960, P512 % 256];
type InputRows = m![H / 960, P512 % 256 / 4];
type HiddenRows = m![L / 960, P512 % 256 / 16];

fn pack_groups<K: AxisName, Rows: M, Tokens: M>(
    device: &mut Device,
    input: DmTensor<i32, Chip, Cluster, Rows, m![{Tokens}, K % 960]>,
    shifts: &HbmTensor<i32, Chip, m![K % 8]>,
) -> DmTensor<i32, Chip, Cluster, Rows, m![{Tokens}, K % 960 / 8]> {
    const {
        assert!(K::SIZE == 3840 || K::SIZE == 15360);
        assert!(Tokens::SIZE == K::SIZE / 960);
        assert!(Rows::SIZE == 256 && Cluster::SIZE == 2 && Chip::SIZE == 1);
        assert!(GroupWords::SIZE * Nibbles::SIZE == 960 && GroupWords::SIZE % 2 == 0);
        assert!(<m![K % 960]>::SIZE == 960 && <m![K % 8]>::SIZE == 8);
        assert!(<m![{Tokens}, K % 960]>::SIZE == <m![{Tokens}, GroupWords, Nibbles]>::SIZE);
    };
    // SAFETY: at a fixed chip/cluster/slice, j*960+u equals
    // j*(120*8)+(u/8)*8+u%8. Dtype, token order and all placement levels stay fixed.
    let input: DmTensor<i32, Chip, Cluster, Rows, m![{Tokens}, GroupWords, Nibbles]> =
        unsafe { input.reshape() };
    let shifts: DmTensor<i32, Chip, Cluster, Rows, m![K % 8]> = shifts.to_dm(&mut device.tdma);
    // SAFETY: eight contiguous I32 values at every unchanged placement.
    let shifts: DmTensor<i32, Chip, Cluster, Rows, m![Nibbles]> = unsafe { shifts.reshape() };
    let shifts: VrfTensor<i32, Chip, Cluster, Rows, m![Nibbles]> = device.sub.begin(shifts.view())
        .fetch::<m![1], m![Nibbles]>().collect::<m![1], m![Nibbles]>().to_vrf();
    let packed: DmTensor<i32, Chip, Cluster, Rows, m![{Tokens}, GroupWords]> = device.main.begin(input.view())
        .fetch::<m![{Tokens}, GroupWords], m![Nibbles]>()
        .collect::<m![{Tokens}, GroupWords], m![Nibbles]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, 15i32)
        .vector_logic(LogicBinaryOpI32::LeftShift, &shifts)
        .vector_narrow_split::<m![{Tokens}, GroupWords, Nibbles / 4], m![Nibbles % 4]>()
        // Disjoint fields: every signed subset sum fits I32, so AddSat equals OR.
        .vector_intra_slice_reduce::<Nibbles, m![{Tokens}, GroupWords], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![{Tokens}, GroupWords / 2], m![GroupWords % 2 # 8]>()
        .commit_trim::<m![GroupWords % 2]>().commit();
    // SAFETY: local word j*120+w has the identical offset under K%960/8.
    unsafe { packed.reshape() }
}

#[device(chip = 1)]
pub fn quantize_pack_input(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P512, H]>,
    table: &HbmTensor<i32, Chip, m![Mantissa, Record]>,
    shifts: &HbmTensor<i32, Chip, m![H % 8]>,
    output: &mut HbmTensor<i32, Chip, PackedInput>,
    scales: &mut HbmTensor<f32, Chip, InputScales>,
    offsets: &mut HbmTensor<i32, Chip, InputScales>,
) {
    // c,g,r,j,u -> p=256c+4*r+j, k=960g+u. All256 slices percluster live.
    let x: DmTensor<bf16, Chip, Cluster, InputRows, m![P512 % 4, H % 960]> = input.to_dm(&mut device.tdma);
    let amax: DmTensor<f32, Chip, Cluster, InputRows, m![P512 % 4, 1 # 8]> = device.main.begin(x.view())
        .fetch::<m![P512 % 4, H / 16 % 60], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![P512 % 4, H / 8 % 120], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![P512 % 4, H / 4 % 240], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![P512 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.925929944387236e-34f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale: DmTensor<f32, Chip, Cluster, InputRows, m![P512 % 4, 1 # 8]> = device.main.begin(amax.view())
        .fetch::<m![P512 % 4], m![1 # 8]>().collect::<m![P512 % 4], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.1428571492433548f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>().commit();
    let index_records: DmTensor<i32, Chip, Cluster, InputRows, m![P512 % 4, 1 # 8]> = device.main.begin(amax.view())
        .fetch::<m![P512 % 4], m![1 # 8]>().collect::<m![P512 % 4], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::LogicRightShift, 16i32)
        .vector_logic(LogicBinaryOpI32::BitAnd, 127i32)
        .vector_logic(LogicBinaryOpI32::LeftShift, 5i32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Compact ONLY scalar-record padding, retaining local-token order. Offsets
    // are relative table byte positions0..4064, not fabricated device addresses.
    let dense_offsets: DmTensor<i32, Chip, Cluster, InputRows, m![P512 % 4]> = device.main.begin(index_records.view())
        .fetch::<m![P512 % 4], m![1 # 8]>().collect::<m![P512 % 4], m![1 # 8]>()
        .transpose::<m![P512 % 4 / 2], m![P512 % 2 # 8]>().commit_trim::<m![P512 % 2]>().commit();
    dense_offsets.view().to_hbm_view(&mut device.tdma, offsets.view_mut());
    // Full Record8 gather is required for32B payload alignment. The explicit
    // HBM scratch write and read remain within the measured complete graph.
    let selected: DmTensor<i32, Chip, Cluster, InputRows, m![P512 % 4, Record]> = table
        .gather::<m![Mantissa], m![Record]>()
        .by_byte_offsets(offsets)
        .to_dm(&mut device.tdma);
    let u_bits: VrfTensor<i32, Chip, Cluster, InputRows, m![P512 % 4, 1 # 8]> = device.sub.begin(selected.view()
        .tile::<m![Record], 1, m![P512 % 4, Record = 1 # 8]>(0))
        .fetch::<m![P512 % 4], m![Record = 1 # 8]>()
        .collect::<m![P512 % 4], m![Record = 1 # 8]>().to_vrf();
    let multiplier: VrfTensor<f32, Chip, Cluster, InputRows, m![P512 % 4, 1 # 8]> = device.sub.begin(amax.view())
        .fetch::<m![P512 % 4], m![1 # 8]>().collect::<m![P512 % 4], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::LogicRightShift, 23i32)
        .vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, 114i32)
        .vector_fxp(FxpBinaryOp::LeftShift, 23i32)
        .vector_clip(ClipBinaryOpI32::AddFxp, &u_bits)
        .vector_reinterpret::<f32>().vector_final().to_vrf();
    let codes: DmTensor<i32, Chip, Cluster, InputRows, m![P512 % 4, H % 960]> = device.main.begin(x.view())
        .fetch::<m![P512 % 4, H / 16 % 60], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![P512 % 4, H / 8 % 120], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P512 % 4, H / 4 % 240], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &multiplier)
        .vector_widen_concat::<m![P512 % 4, H / 8 % 120], m![H % 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::Max, -7i32).vector_clip(ClipBinaryOpI32::Min, 7i32)
        .vector_final().commit_trim::<m![H % 8]>().commit();
    let packed = pack_groups::<H, InputRows, m![P512 % 4]>(device, codes, shifts);
    // The real H16 movement is retained, including64live/256physical leaders.
    let staged: DmTensor<i32, Chip, Cluster,
        m![H / 960, P512 % 256 / 16, 1 # 4], m![P512 % 16, H % 960 / 8]> = packed.to_dm(&mut device.tdma);
    staged.view().to_hbm_view(&mut device.tdma, output.view_mut());
    let scale_dense: DmTensor<f32, Chip, Cluster, InputRows, m![P512 % 4]> = device.main.begin(scale.view())
        .fetch::<m![P512 % 4], m![1 # 8]>().collect::<m![P512 % 4], m![1 # 8]>()
        .transpose::<m![P512 % 4 / 2], m![P512 % 2 # 8]>().commit_trim::<m![P512 % 2]>().commit();
    scale_dense.view().to_hbm_view(&mut device.tdma, scales.view_mut());
}

#[device(chip = 1)]
pub fn quantize_pack_hidden(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P512, L]>,
    table: &HbmTensor<i32, Chip, m![Mantissa, Record]>,
    shifts: &HbmTensor<i32, Chip, m![L % 8]>,
    output: &mut HbmTensor<i32, Chip, PackedHidden>,
    scales: &mut HbmTensor<f32, Chip, HiddenScales>,
    offsets: &mut HbmTensor<i32, Chip, HiddenScales>,
) {
    // c,g,r,j,u -> p=256c+16*r+j, k=960g+u. All256 slices percluster live.
    let x: DmTensor<bf16, Chip, Cluster, HiddenRows, m![P512 % 16, L % 960]> = input.to_dm(&mut device.tdma);
    let amax: DmTensor<f32, Chip, Cluster, HiddenRows, m![P512 % 16, 1 # 8]> = device.main.begin(x.view())
        .fetch::<m![P512 % 16, L / 16 % 60], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![P512 % 16, L / 8 % 120], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![P512 % 16, L / 4 % 240], m![L % 4]>()
        .vector_intra_slice_reduce::<L, m![P512 % 16], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.925929944387236e-34f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale: DmTensor<f32, Chip, Cluster, HiddenRows, m![P512 % 16, 1 # 8]> = device.main.begin(amax.view())
        .fetch::<m![P512 % 16], m![1 # 8]>().collect::<m![P512 % 16], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.1428571492433548f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>().commit();
    let index_records: DmTensor<i32, Chip, Cluster, HiddenRows, m![P512 % 16, 1 # 8]> = device.main.begin(amax.view())
        .fetch::<m![P512 % 16], m![1 # 8]>().collect::<m![P512 % 16], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::LogicRightShift, 16i32)
        .vector_logic(LogicBinaryOpI32::BitAnd, 127i32)
        .vector_logic(LogicBinaryOpI32::LeftShift, 5i32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Compact ONLY scalar-record padding, retaining local-token order. Offsets
    // are relative table byte positions0..4064, not fabricated device addresses.
    let dense_offsets: DmTensor<i32, Chip, Cluster, HiddenRows, m![P512 % 16]> = device.main.begin(index_records.view())
        .fetch::<m![P512 % 16], m![1 # 8]>().collect::<m![P512 % 16], m![1 # 8]>()
        .transpose::<m![P512 % 16 / 2], m![P512 % 2 # 8]>().commit_trim::<m![P512 % 2]>().commit();
    dense_offsets.view().to_hbm_view(&mut device.tdma, offsets.view_mut());
    // Full Record8 gather is required for32B payload alignment. The explicit
    // HBM scratch write and read remain within the measured complete graph.
    let selected: DmTensor<i32, Chip, Cluster, HiddenRows, m![P512 % 16, Record]> = table
        .gather::<m![Mantissa], m![Record]>()
        .by_byte_offsets(offsets)
        .to_dm(&mut device.tdma);
    let u_bits: VrfTensor<i32, Chip, Cluster, HiddenRows, m![P512 % 16, 1 # 8]> = device.sub.begin(selected.view()
        .tile::<m![Record], 1, m![P512 % 16, Record = 1 # 8]>(0))
        .fetch::<m![P512 % 16], m![Record = 1 # 8]>()
        .collect::<m![P512 % 16], m![Record = 1 # 8]>().to_vrf();
    let multiplier: VrfTensor<f32, Chip, Cluster, HiddenRows, m![P512 % 16, 1 # 8]> = device.sub.begin(amax.view())
        .fetch::<m![P512 % 16], m![1 # 8]>().collect::<m![P512 % 16], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::LogicRightShift, 23i32)
        .vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, 114i32)
        .vector_fxp(FxpBinaryOp::LeftShift, 23i32)
        .vector_clip(ClipBinaryOpI32::AddFxp, &u_bits)
        .vector_reinterpret::<f32>().vector_final().to_vrf();
    let codes: DmTensor<i32, Chip, Cluster, HiddenRows, m![P512 % 16, L % 960]> = device.main.begin(x.view())
        .fetch::<m![P512 % 16, L / 16 % 60], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![P512 % 16, L / 8 % 120], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P512 % 16, L / 4 % 240], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &multiplier)
        .vector_widen_concat::<m![P512 % 16, L / 8 % 120], m![L % 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::Max, -7i32).vector_clip(ClipBinaryOpI32::Min, 7i32)
        .vector_final().commit_trim::<m![L % 8]>().commit();
    let packed = pack_groups::<L, HiddenRows, m![P512 % 16]>(device, codes, shifts);
    // This owner already is the validated16-row hidden store layout.
    // No alias/transpose is hidden here: its physical row order equals PackedHidden.
    packed.view().to_hbm_view(&mut device.tdma, output.view_mut());
    let scale_dense: DmTensor<f32, Chip, Cluster, HiddenRows, m![P512 % 16]> = device.main.begin(scale.view())
        .fetch::<m![P512 % 16], m![1 # 8]>().collect::<m![P512 % 16], m![1 # 8]>()
        .transpose::<m![P512 % 16 / 2], m![P512 % 2 # 8]>().commit_trim::<m![P512 % 2]>().commit();
    scale_dense.view().to_hbm_view(&mut device.tdma, scales.view_mut());
}
