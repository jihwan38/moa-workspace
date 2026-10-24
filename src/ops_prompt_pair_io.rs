//! Exact joins/splits for two independent P128 hidden matrices sharing a P256 FFN.
use crate::{
    Chip,
    axes::{H, P128, P256},
};
use furiosa_opt_std::prelude::*;
type Cluster = m![1#2];
type Shards = m![H / 32];
fn store(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![P128, H]>,
    base: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![P256, H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Shards, m![P128, H % 32]> = source.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, Cluster, Shards, m![P256 % 128, H % 32]> = unsafe { x.reshape() };
    out.view_mut()
        .scatter::<m![P256 / 128], m![P256 % 128, H]>()
        .by_byte_offsets(base.view())
        .from_dm(&mut device.tdma, x);
}
fn load(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![P256, H]>,
    base: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![P128, H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Shards, m![P256 % 128, H % 32]> = source
        .gather::<m![P256 / 128], m![P256 % 128, H]>()
        .by_byte_offsets(base)
        .to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, Cluster, Shards, m![P128, H % 32]> = unsafe { x.reshape() };
    x.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
#[device(chip = 1)]
pub fn pack_hidden_pair(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![P128, H]>,
    b: &HbmTensor<bf16, Chip, m![P128, H]>,
    zero: &HbmTensor<i32, Chip, m![1]>,
    second: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![P256, H]>,
) {
    store(device, a, zero, out);
    store(device, b, second, out);
}
#[device(chip = 1)]
pub fn unpack_hidden_pair(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P256, H]>,
    zero: &HbmTensor<i32, Chip, m![1]>,
    second: &HbmTensor<i32, Chip, m![1]>,
    a: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    b: &mut HbmTensor<bf16, Chip, m![P128, H]>,
) {
    load(device, input, zero, a);
    load(device, input, second, b);
}
