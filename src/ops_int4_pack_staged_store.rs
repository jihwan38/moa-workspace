//! Packed-word output staging: real DM redistribution, unchanged blocked ABI.
//! The added movement must be counted together with the final HBM store.
use crate::{Chip, axes::{H, L, P512}, ops_int4_pack_p512 as original};
use furiosa_opt_std::prelude::*;
pub use original::{PackedInput, PackedHidden, I4Input, I4Hidden, read_input, read_hidden};
type Cluster = m![P512 / 256];

#[device(chip = 1)]
pub fn pack_input(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![P512, H]>,
    shifts: &HbmTensor<i32, Chip, m![H % 8]>,
    output: &mut HbmTensor<i32, Chip, PackedInput>,
) {
    let dense = original::pack_dm::<H, original::InputWords>(device, input, shifts);
    // All 256 slices remain distinct: four K groups times 64 four-row groups.
    // Each local [four rows, 120 words] interval is contiguous in output HBM.
    // to_dm performs the row-slice to block-slice redistribution; no reshape.
    let staged: DmTensor<i32, Chip, Cluster,
        m![H / 960, P512 % 256 / 4], m![P512 % 4, H % 960 / 8]> =
        dense.to_dm(&mut device.tdma);
    staged.view().to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn pack_hidden(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![P512, L]>,
    shifts: &HbmTensor<i32, Chip, m![L % 8]>,
    output: &mut HbmTensor<i32, Chip, PackedHidden>,
) {
    let dense = original::pack_dm::<L, original::HiddenWords>(device, input, shifts);
    // Sixteen K groups times sixteen row groups also use exactly 256 slices.
    // Each local interval holds sixteen contiguous rows in one K960 block.
    let staged: DmTensor<i32, Chip, Cluster,
        m![L / 960, P512 % 256 / 16], m![P512 % 16, L % 960 / 8]> =
        dense.to_dm(&mut device.tdma);
    staged.view().to_hbm_view(&mut device.tdma, output.view_mut());
}
