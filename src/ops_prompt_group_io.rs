//! Four independent P128 row groups join/split one P512 FFN buffer.
use crate::{
    Chip,
    axes::{H, P128, P512},
};
use furiosa_opt_std::prelude::*;
type Cluster = m![1#2];
fn store(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![P128, H]>,
    base: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![P512, H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, m![P128, 1 # 2], m![H]> = source.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, Cluster, m![P512 % 128, 1 # 2], m![H]> = unsafe { x.reshape() };
    out.view_mut()
        .scatter::<m![P512 / 128], m![P512 % 128, H]>()
        .by_byte_offsets(base.view())
        .from_dm(&mut device.tdma, x);
}
fn load(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![P512, H]>,
    base: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![P128, H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, m![P512 % 128, 1 # 2], m![H]> = source
        .gather::<m![P512 / 128], m![P512 % 128, H]>()
        .by_byte_offsets(base)
        .to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, Cluster, m![P128, 1 # 2], m![H]> = unsafe { x.reshape() };
    x.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
#[device(chip = 1)]
pub fn pack_hidden_group(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![P128, H]>,
    b: &HbmTensor<bf16, Chip, m![P128, H]>,
    c: &HbmTensor<bf16, Chip, m![P128, H]>,
    d: &HbmTensor<bf16, Chip, m![P128, H]>,
    zero: &HbmTensor<i32, Chip, m![1]>,
    second: &HbmTensor<i32, Chip, m![1]>,
    third: &HbmTensor<i32, Chip, m![1]>,
    fourth: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![P512, H]>,
) {
    store(device, a, zero, out);
    store(device, b, second, out);
    store(device, c, third, out);
    store(device, d, fourth, out);
}
#[device(chip = 1)]
pub fn unpack_hidden_group(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P512, H]>,
    zero: &HbmTensor<i32, Chip, m![1]>,
    second: &HbmTensor<i32, Chip, m![1]>,
    third: &HbmTensor<i32, Chip, m![1]>,
    fourth: &HbmTensor<i32, Chip, m![1]>,
    a: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    b: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    c: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    d: &mut HbmTensor<bf16, Chip, m![P128, H]>,
) {
    load(device, input, zero, a);
    load(device, input, second, b);
    load(device, input, third, c);
    load(device, input, fourth, d);
}
