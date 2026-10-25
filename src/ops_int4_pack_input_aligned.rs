//! H3840-only packed-word staging with sixteen contiguous rows per live slice.
//! The real redistribution and physical slice padding are part of the resource gate.
use crate::{axes::{H, P512}, ops_int4_pack_p512 as original, Chip};
use furiosa_opt_std::prelude::*;

type Cluster = m![P512 / 256];
type StoreParts = m![H / 960, P512 % 256 / 16, 1 # 4];
type StoreLocal = m![P512 % 16, H % 960 / 8];

#[device(chip = 1)]
pub fn pack_input(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![P512, H]>,
    shifts: &HbmTensor<i32, Chip, m![H % 8]>,
    output: &mut HbmTensor<i32, Chip, original::PackedInput>,
) {
    const {
        assert!(H::SIZE == 3840 && P512::SIZE == 512);
        assert!(Cluster::SIZE == 2 && StoreParts::SIZE == 256);
        assert!(StoreLocal::SIZE == 16 * 120);
    };
    let dense = original::pack_dm::<H, original::InputWords>(device, input, shifts);
    // A live slice s=4*(16*h+r) owns p=256*cluster+16*r+j and local word w.
    // h=0..4,r=0..16,j=0..16,w=0..120 cover every output word exactly once.
    // The other three physical slice positions are padding, not extra rows.
    // The blocked output word offset is (((cluster*4+h)*256+16*r+j)*120+w).
    // Real DMA moves row-owned words into 7680-byte local intervals; no reshape.
    let staged: DmTensor<i32, Chip, Cluster, StoreParts, StoreLocal> =
        dense.to_dm(&mut device.tdma);
    staged.view().to_hbm_view(&mut device.tdma, output.view_mut());
}
