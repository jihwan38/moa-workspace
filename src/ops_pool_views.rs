//! Canonical KV storage axis and explicit zero-copy view ABI probes.
//! Query batch axes remain separate from physical request slots.
use crate::{Chip, axes::*};
use furiosa_opt_std::prelude::*;
axes![Rc64 = 64];

#[device(chip = 1)]
pub fn read_ring(
    device: &mut Device,
    input: HbmTensorView<'_, bf16, Chip, m![Ts, Ns, Ds]>,
    output: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    let x: DmTensor<bf16, Chip, m![1], m![Ts / 4], m![Ts % 4, Ns, Ds]> =
        input.to_dm(&mut device.tdma);
    x.view().to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn write_ring(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, m![Ts, Ns, Ds]>,
) {
    let x: DmTensor<bf16, Chip, m![1], m![Ts / 4], m![Ts % 4, Ns, Ds]> =
        input.to_dm(&mut device.tdma);
    x.view().to_hbm_view(&mut device.tdma, output);
}

#[device(chip = 1)]
pub fn read_page(
    device: &mut Device,
    input: HbmTensorView<'_, bf16, Chip, m![Tf, Df]>,
    output: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    let x: DmTensor<bf16, Chip, m![1], m![Tf / 2], m![Tf % 2, Df]> = input.to_dm(&mut device.tdma);
    x.view().to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn write_page(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    output: HbmTensorViewMut<'_, bf16, Chip, m![Tf, Df]>,
) {
    let x: DmTensor<bf16, Chip, m![1], m![Tf / 2], m![Tf % 2, Df]> = input.to_dm(&mut device.tdma);
    x.view().to_hbm_view(&mut device.tdma, output);
}
