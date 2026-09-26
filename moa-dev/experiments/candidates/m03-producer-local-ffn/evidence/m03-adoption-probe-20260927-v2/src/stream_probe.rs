//! Root device entry points avoid alias-dependent image-registration paths.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{H, L}, device::mlp::validation_probe};

#[device(chip = 1)]
pub fn upgate(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    g: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    s: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    t: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    y: &mut HbmTensor<bf16, Chip, m![L]>,
    z: &mut HbmTensor<bf16, Chip, m![L]>,
) {
    validation_probe::upgate(device, x, w, g, s, t, y, z);
}

#[device(chip = 1)]
pub fn down(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![L]>,
    w: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    s: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    y: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    validation_probe::down(device, x, w, s, y);
}
