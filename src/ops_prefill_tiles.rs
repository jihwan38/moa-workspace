//! Exact transfers between P128 prompt rows and validated B16 epilogues.
//! `tile_index` selects one of the eight contiguous groups of sixteen rows.
use crate::device::layout::Slice;
type Cluster=m![1#2];
type SourceCluster=m![1#2];
type IndexCluster=m![1#2];
use crate::{Chip, axes::*};
use furiosa_opt_std::prelude::*;

fn load_tile<Payload: M, PayloadSlices:M, Elements:M>(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![P128, { Payload }]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![B, { Payload }]>,
) {
    let index: DmTensor<i32, Chip, IndexCluster, Slice, m![1 # 8]> = tile_index.to_dm(&mut device.tdma);
    let index: SpmTensor<i32, Chip, IndexCluster, m![1 # 4], m![1]> = index.to_spm(&mut device.tdma);
    let tile: DmTensor<bf16, Chip, Cluster, m![P128%16,{PayloadSlices}], Elements> = source
        .gather::<m![P128 / 16], m![P128 % 16, { Payload }]>()
        .by_positions::<m![1]>(&index)
        .to_dm(&mut device.tdma);
    let tile: DmTensor<bf16, Chip, SourceCluster, m![B,{PayloadSlices}], Elements> = unsafe { tile.reshape() };
    tile.view().to_hbm_view(&mut device.tdma, output.view_mut());
}

fn store_tile<Payload: M, PayloadSlices:M, Elements:M>(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![B, { Payload }]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, { Payload }]>,
) {
    let index: DmTensor<i32, Chip, IndexCluster, Slice, m![1 # 8]> = tile_index.to_dm(&mut device.tdma);
    let index: SpmTensor<i32, Chip, IndexCluster, m![1 # 4], m![1]> = index.to_spm(&mut device.tdma);
    let tile: DmTensor<bf16, Chip, SourceCluster, m![B,{PayloadSlices}], Elements> =
        source.to_dm(&mut device.tdma);
    let tile: DmTensor<bf16, Chip, Cluster, m![P128%16,{PayloadSlices}], Elements> =
        unsafe { tile.reshape() };
    output
        .view_mut()
        .scatter::<m![P128 / 16], m![P128 % 16, { Payload }]>()
        .by_positions::<m![1]>(&index)
        .from_dm(&mut device.tdma, tile);
}

#[device(chip = 1)]
pub fn load_hidden_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![P128, H]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![B, H]>,
) {
    load_tile::<m![H],m![H/240],m![H%240]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn store_hidden_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![B, H]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, H]>,
) {
    store_tile::<m![H],m![H/240],m![H%240]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn load_query_sliding_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![P128, Qs]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Qs]>,
) {
    load_tile::<m![Qs],m![Qs/256],m![Qs%256]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn store_query_sliding_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![B, Qs]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, Qs]>,
) {
    store_tile::<m![Qs],m![Qs/256],m![Qs%256]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn load_key_sliding_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![P128, Ps]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Ps]>,
) {
    load_tile::<m![Ps],m![Ps/128],m![Ps%128]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn store_key_sliding_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![B, Ps]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, Ps]>,
) {
    store_tile::<m![Ps],m![Ps/128],m![Ps%128]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn load_query_full_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![P128, Qf]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Qf]>,
) {
    load_tile::<m![Qf],m![Qf/512],m![Qf%512]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn store_query_full_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![B, Qf]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, Qf]>,
) {
    store_tile::<m![Qf],m![Qf/512],m![Qf%512]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn load_key_full_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![P128, Pf]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Pf]>,
) {
    load_tile::<m![Pf],m![Pf/32],m![Pf%32]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn store_key_full_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![B, Pf]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, Pf]>,
) {
    store_tile::<m![Pf],m![Pf/32],m![Pf%32]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn load_heads_sliding_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![P128, Ns, Gs, Ds]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
) {
    load_tile::<m![Ns, Gs, Ds],m![Ns,Gs],m![Ds]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn store_heads_sliding_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, Ns, Gs, Ds]>,
) {
    store_tile::<m![Ns, Gs, Ds],m![Ns,Gs],m![Ds]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn load_heads_full_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![B, Gf, Df]>,
) {
    load_tile::<m![Gf, Df],m![Gf],m![Df]>(device, source, tile_index, output);
}

#[device(chip = 1)]
pub fn store_heads_full_tile(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
) {
    store_tile::<m![Gf, Df],m![Gf],m![Df]>(device, source, tile_index, output);
}
