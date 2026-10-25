//! Exact-rational BF16 quantization directly into blocked packed I32 words.
//! Original arithmetic is retained; only the canonical I32 HBM roundtrip is removed.
//! This is a disconnected quantize-plus-pack prerequisite, not a model kernel.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{H, L, P512}, ops_int4_pack_p512 as original};
pub use original::{PackedInput, PackedHidden};
pub use crate::ops_int4_rational_lut_p512_compact::{Mantissa, Record};
axes![Nibbles = 8];
type Cluster = m![P512 / 256];
type Rows = m![P512 % 256];

fn quantize_codes<K: AxisName>(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P512, K]>,
    table: &HbmTensor<i32, Chip, m![Mantissa, Record]>,
) -> (DmTensor<i32, Chip, Cluster, Rows, m![K]>,
      DmTensor<f32, Chip, Cluster, Rows, m![1 # 8]>) {
    let x: DmTensor<bf16, Chip, Cluster, Rows, m![K]> = input.to_dm(&mut device.tdma);
    let amax: DmTensor<f32, Chip, Cluster, Rows, m![1 # 8]> = device.main.begin(x.view())
        .fetch::<m![K / 16], m![K % 16]>().fetch_cast::<f32>()
        .collect::<m![K / 8], m![K % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![K / 4], m![K % 4]>()
        .vector_intra_slice_reduce::<K, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
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
    let codes: DmTensor<i32, Chip, Cluster, Rows, m![K]> = device.main.begin(x.view())
        .fetch::<m![K / 16], m![K % 16]>().fetch_cast::<f32>()
        .collect::<m![K / 8], m![K % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![K / 4], m![K % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &multiplier)
        .vector_widen_concat::<m![K / 8], m![K % 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::Max, -7i32)
        .vector_clip(ClipBinaryOpI32::Min, 7i32)
        .vector_final().commit_trim::<m![K % 8]>().commit();
    (codes, scale)
}

fn pack_codes<K: AxisName, W: AxisName>(
    device: &mut Device,
    input: DmTensor<i32, Chip, Cluster, Rows, m![K]>,
    shifts: &HbmTensor<i32, Chip, m![K % 8]>,
) -> DmTensor<i32, Chip, Cluster, Rows, m![K / 8]> {
    // Public AxisName/M size traits make these preconditions compile-time
    // assertions for each private instantiation, before any unsafe boundary.
    // Every reshape below fixes Chip/Cluster/Rows to these identical types;
    // the SDK independently checks all per-memory-level physical extents.
    const {
        assert!(K::SIZE > 0 && K::SIZE % 960 == 0);
        assert!(Nibbles::SIZE == 8);
        assert!(W::SIZE <= usize::MAX / Nibbles::SIZE);
        assert!(W::SIZE * Nibbles::SIZE == K::SIZE);
        assert!(W::SIZE % 2 == 0);
        assert!(<m![K / 8]>::SIZE == W::SIZE);
        assert!(<m![K % 8]>::SIZE == Nibbles::SIZE);
        assert!(<m![W, Nibbles]>::SIZE == <m![K]>::SIZE);
        assert!(Chip::SIZE == 1 && Cluster::SIZE == 2 && Rows::SIZE == 256);
        assert!(P512::SIZE == Cluster::SIZE * Rows::SIZE);
        assert!(core::mem::size_of::<i32>() == 4);
    };
    // SAFETY: at every fixed (chip,cluster,slice), canonical local offset k
    // becomes (word=k/8,nibble=k%8), whose row-major offset is 8*word+nibble=k.
    // No padding, dtype, cluster, slice, or byte order changes. Consuming DM
    // reshape is the SDK's explicit same-wire axis regrouping, not a transpose.
    let input: DmTensor<i32, Chip, Cluster, Rows, m![W, Nibbles]> =
        unsafe { input.reshape() };
    let shifts: DmTensor<i32, Chip, Cluster, Rows, m![K % 8]> = shifts.to_dm(&mut device.tdma);
    // SAFETY: both local mappings are precisely eight contiguous I32 values.
    let shifts: DmTensor<i32, Chip, Cluster, Rows, m![Nibbles]> =
        unsafe { shifts.reshape() };
    let shifts: VrfTensor<i32, Chip, Cluster, Rows, m![Nibbles]> = device
        .sub
        .begin(shifts.view())
        .fetch::<m![1], m![Nibbles]>()
        .collect::<m![1], m![Nibbles]>()
        .to_vrf();
    // Masked fields occupy disjoint bits. Any subset sum fits signed I32,
    // including the negative top nibble, so AddSat equals the intended OR.
    let packed: DmTensor<i32, Chip, Cluster, Rows, m![W]> = device
        .main
        .begin(input.view())
        .fetch::<m![W], m![Nibbles]>()
        .collect::<m![W], m![Nibbles]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, 15i32)
        .vector_logic(LogicBinaryOpI32::LeftShift, &shifts)
        .vector_narrow_split::<m![W, Nibbles / 4], m![Nibbles % 4]>()
        .vector_intra_slice_reduce::<Nibbles, m![W], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![W / 2], m![W % 2 # 8]>()
        .commit_trim::<m![W % 2]>()
        .commit();
    // SAFETY: packed local offset w remains w under K/8; both contain K::SIZE/8
    // contiguous I32 words. The subsequent DMA, not this rename, changes the
    // physical HBM order to K960 blocks across rows.
    unsafe { packed.reshape() }
}

fn store_scales(
    device: &mut Device,
    scale: DmTensor<f32, Chip, Cluster, Rows, m![1 # 8]>,
    scales: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    // Exact scale bits only: real full-record movement, then two-row TU
    // compaction removes padding before the contiguous canonical HBM store.
    let scale_records: DmTensor<f32, Chip, Cluster, m![1 # 256], m![P512 % 256, 1 # 8]> =
        scale.to_dm(&mut device.tdma);
    let scale_dense: DmTensor<f32, Chip, Cluster, m![1 # 256], m![P512 % 256]> = device.main.begin(scale_records.view())
        .fetch::<m![P512 % 256], m![1 # 8]>().collect::<m![P512 % 256], m![1 # 8]>()
        .transpose::<m![P512 % 256 / 2], m![P512 % 2 # 8]>().commit_trim::<m![P512 % 2]>().commit();
    scale_dense.view().to_hbm_view(&mut device.tdma, scales.view_mut());
}

#[device(chip = 1)]
pub fn quantize_pack_input(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P512, H]>,
    table: &HbmTensor<i32, Chip, m![Mantissa, Record]>,
    shifts: &HbmTensor<i32, Chip, m![H % 8]>,
    output: &mut HbmTensor<i32, Chip, PackedInput>,
    scales: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    let (codes, scale) = quantize_codes::<H>(device, input, table);
    let packed = pack_codes::<H, original::InputWords>(device, codes, shifts);
    // Original H blocked store. The failed four-row H staging is excluded.
    packed.view().to_hbm_view(&mut device.tdma, output.view_mut());
    store_scales(device, scale, scales);
}

#[device(chip = 1)]
pub fn quantize_pack_hidden(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P512, L]>,
    table: &HbmTensor<i32, Chip, m![Mantissa, Record]>,
    shifts: &HbmTensor<i32, Chip, m![L % 8]>,
    output: &mut HbmTensor<i32, Chip, PackedHidden>,
    scales: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    let (codes, scale) = quantize_codes::<L>(device, input, table);
    let dense = pack_codes::<L, original::HiddenWords>(device, codes, shifts);
    // Identical real DM staging to the separately validated hidden pack.
    let staged: DmTensor<i32, Chip, Cluster,
        m![L / 960, P512 % 256 / 16], m![P512 % 16, L % 960 / 8]> =
        dense.to_dm(&mut device.tdma);
    staged.view().to_hbm_view(&mut device.tdma, output.view_mut());
    store_scales(device, scale, scales);
}

/// Current separately validated hidden packing control for a future paired probe.
#[device(chip = 1)]
pub fn pack_hidden_control(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![P512, L]>,
    shifts: &HbmTensor<i32, Chip, m![L % 8]>,
    output: &mut HbmTensor<i32, Chip, PackedHidden>,
) {
    let dense = original::pack_dm::<L, original::HiddenWords>(device, input, shifts);
    let staged: DmTensor<i32, Chip, Cluster,
        m![L / 960, P512 % 256 / 16], m![P512 % 16, L % 960 / 8]> =
        dense.to_dm(&mut device.tdma);
    staged.view().to_hbm_view(&mut device.tdma, output.view_mut());
}

/// Distinct aligned-H variant; original H implementation above remains frozen.
#[device(chip = 1)]
pub fn quantize_pack_input_aligned(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P512, H]>,
    table: &HbmTensor<i32, Chip, m![Mantissa, Record]>,
    shifts: &HbmTensor<i32, Chip, m![H % 8]>,
    output: &mut HbmTensor<i32, Chip, PackedInput>,
    scales: &mut HbmTensor<f32, Chip, m![P512]>,
) {
    let (codes, scale) = quantize_codes::<H>(device, input, table);
    let dense = pack_codes::<H, original::InputWords>(device, codes, shifts);
    // Only the reviewed real staging differs from quantize_pack_input.
    let staged: DmTensor<i32, Chip, Cluster,
        m![H / 960, P512 % 256 / 16, 1 # 4], m![P512 % 16, H % 960 / 8]> =
        dense.to_dm(&mut device.tdma);
    staged.view().to_hbm_view(&mut device.tdma, output.view_mut());
    store_scales(device, scale, scales);
}
