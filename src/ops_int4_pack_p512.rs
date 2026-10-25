//! Full-P512 signed-nibble packing, independent of activation quantization.
//! The final DMA performs the real canonical-DM to K960-blocked-HBM conversion.
use crate::{
    Chip,
    axes::{H, L, P512},
};
use furiosa_opt_std::prelude::*;

axes![InputWords = 480, HiddenWords = 1920, Nibbles = 8];

type Cluster = m![P512 / 256];
type Rows = m![P512 % 256];
pub type PackedInput = m![P512 / 256, H / 960, P512 % 256, H % 960 / 8];
pub type PackedHidden = m![P512 / 256, L / 960, P512 % 256, L % 960 / 8];
pub type I4Input = crate::ops_batch_layouts::P512H960;
pub type I4Hidden = crate::ops_batch_layouts::P512L960;

pub(crate) fn pack_dm<K: AxisName, W: AxisName>(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![P512, K]>,
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
    let input: DmTensor<i32, Chip, Cluster, Rows, m![K]> = input.to_dm(&mut device.tdma);
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

#[device(chip = 1)]
pub fn pack_input(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![P512, H]>,
    shifts: &HbmTensor<i32, Chip, m![H % 8]>,
    output: &mut HbmTensor<i32, Chip, PackedInput>,
) {
    let dense = pack_dm::<H, InputWords>(device, input, shifts);
    // Physical bytes become [P/256,H/960,P%256,H%960/8]. This is DMA,
    // not a reshape/bitcast; every layout-transfer cost belongs to this kernel.
    dense
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn pack_hidden(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![P512, L]>,
    shifts: &HbmTensor<i32, Chip, m![L % 8]>,
    output: &mut HbmTensor<i32, Chip, PackedHidden>,
) {
    let dense = pack_dm::<L, HiddenWords>(device, input, shifts);
    dense
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

// Typed byte-layout readers for the future raw-buffer probe. They share the
// actual UP/DOWN blocked input ABI, but do not claim to verify a GEMM.
#[device(chip = 1)]
pub fn read_input(
    device: &mut Device,
    input: &HbmTensor<i4, Chip, I4Input>,
    output: &mut HbmTensor<i4, Chip, m![P512, H]>,
) {
    let canonical: DmTensor<i4, Chip, Cluster, Rows, m![H]> = input.to_dm(&mut device.tdma);
    canonical
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn read_hidden(
    device: &mut Device,
    input: &HbmTensor<i4, Chip, I4Hidden>,
    output: &mut HbmTensor<i4, Chip, m![P512, L]>,
) {
    let canonical: DmTensor<i4, Chip, Cluster, Rows, m![L]> = input.to_dm(&mut device.tdma);
    canonical
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}
