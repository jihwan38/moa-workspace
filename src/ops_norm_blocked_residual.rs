//! Persistent residual placement matching the proven Q4/H480 normalization DM.
//! This module does not change any model/worker dispatch or arithmetic.
use crate::ops_native_fp8_p512_downblocked::Output;
use crate::ops_norm_quarters::{Q4, load_residual, normalize, residual_add, store_residual};
use crate::{
    Chip,
    axes::{H, P128, P512},
};
use furiosa_opt_std::prelude::*;
/// Physical slice owns all four quarters and two token rows:7680 BF16 bytes.
pub type Residual = m![
    P512 / 64 % 2,
    P512 / 2 % 32,
    H / 480,
    P512 / 128,
    P512 % 2,
    H % 480
];
type Cluster = m![P128 / 64];
type Shards = m![P128 / 2 % 32, H / 480];
type Local = m![Q4, P128 % 2, H % 480];
pub(crate) fn load(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, Residual>,
) -> DmTensor<bf16, Chip, Cluster, Shards, Local> {
    let input: DmTensor<
        bf16,
        Chip,
        m![P512 / 64 % 2],
        m![P512 / 2 % 32, H / 480],
        m![P512 / 128, P512 % 2, H % 480],
    > = input.to_dm(&mut device.tdma);
    unsafe { input.reshape() }
}
pub(crate) fn store(
    device: &mut Device,
    input: DmTensor<bf16, Chip, Cluster, Shards, Local>,
    output: &mut HbmTensor<bf16, Chip, Residual>,
) {
    let input: DmTensor<
        bf16,
        Chip,
        m![P512 / 64 % 2],
        m![P512 / 2 % 32, H / 480],
        m![P512 / 128, P512 % 2, H % 480],
    > = unsafe { input.reshape() };
    input
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}
#[device(chip = 1)]
pub fn pack_four(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![P128, H]>,
    b: &HbmTensor<bf16, Chip, m![P128, H]>,
    c: &HbmTensor<bf16, Chip, m![P128, H]>,
    d: &HbmTensor<bf16, Chip, m![P128, H]>,
    output: &mut HbmTensor<bf16, Chip, Residual>,
) {
    let mut input: DmTensor<bf16, Chip, Cluster, Shards, Local> = DmTensor::new();
    load_residual(device, a, 0, &mut input);
    load_residual(device, b, 1, &mut input);
    load_residual(device, c, 2, &mut input);
    load_residual(device, d, 3, &mut input);
    store(device, input, output);
}
#[device(chip = 1)]
pub fn unpack_four(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, Residual>,
    a: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    b: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    c: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    d: &mut HbmTensor<bf16, Chip, m![P128, H]>,
) {
    let input = load(device, input);
    store_residual(device, &input, 0, a);
    store_residual(device, &input, 1, b);
    store_residual(device, &input, 2, c);
    store_residual(device, &input, 3, d);
}
pub(crate) fn post_feedforward_stage(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, Output>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_gate: &HbmTensor<bf16, Chip, m![1#8]>,
    residual: &mut HbmTensor<bf16, Chip, Residual>,
) {
    let input: DmTensor<
        bf16,
        Chip,
        m![P512 / 64 % 2],
        m![P512 / 2 % 32, H / 480],
        m![P512 / 128, P512 % 2, H % 480],
    > = input.to_dm(&mut device.tdma);
    let input: DmTensor<bf16, Chip, Cluster, Shards, Local> = unsafe { input.reshape() };
    let normalized = normalize(device, &input, weight);
    let previous = load(device, residual);
    let residual_dm = residual_add(device, &normalized, &previous);
    let output = crate::ops_norm_quarter_post::gate(device, &residual_dm, layer_gate);
    store(device, output, residual);
}

#[device(chip = 1)]
pub fn post_feedforward(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, Output>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_gate: &HbmTensor<bf16, Chip, m![1#8]>,
    residual: &mut HbmTensor<bf16, Chip, Residual>,
) {
    post_feedforward_stage(device, input, weight, layer_gate, residual);
}
