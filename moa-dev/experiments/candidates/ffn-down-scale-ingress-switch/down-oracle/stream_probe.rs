use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{H, L}, device::mlp::validation_probe};
#[device(chip = 1)]
pub fn down(device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![L]>,
    w: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    s: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    y: &mut HbmTensor<bf16, Chip, m![H]>) {
    validation_probe::down(device, x, w, s, y);
}
