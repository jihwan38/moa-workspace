//! Explicit whole-pool byte addressing; no host buffer slicing or padded view ABI.
use crate::{
    Chip, axes::*, device::layout::Cluster, ops_pool_views::Rc64, ops_request_attention::Fc,
};
use furiosa_opt_std::prelude::*;
#[device(chip = 1)]
pub fn read_ring(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    offset: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, m![Ts / 4], m![Ts % 4, Ns, Ds]> = input
        .gather::<m![Rc64], m![Ts, Ns, Ds]>()
        .by_byte_offsets(offset)
        .to_dm(&mut device.tdma);
    x.view().to_hbm_view(&mut device.tdma, output.view_mut());
}
#[device(chip = 1)]
pub fn write_ring(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    offset: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, m![Ts / 4], m![Ts % 4, Ns, Ds]> =
        input.to_dm(&mut device.tdma);
    output
        .view_mut()
        .scatter::<m![Rc64], m![Ts, Ns, Ds]>()
        .by_byte_offsets(offset.view())
        .from_dm(&mut device.tdma, x);
}
#[device(chip = 1)]
pub fn read_page(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    offset: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, m![Tf / 2], m![Tf % 2, Df]> = input
        .gather::<m![Fc], m![Tf, Df]>()
        .by_byte_offsets(offset)
        .to_dm(&mut device.tdma);
    x.view().to_hbm_view(&mut device.tdma, output.view_mut());
}
#[device(chip = 1)]
pub fn write_page(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    offset: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, m![Tf / 2], m![Tf % 2, Df]> =
        input.to_dm(&mut device.tdma);
    output
        .view_mut()
        .scatter::<m![Fc], m![Tf, Df]>()
        .by_byte_offsets(offset.view())
        .from_dm(&mut device.tdma, x);
}
