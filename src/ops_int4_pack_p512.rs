//! Full-P512 signed-nibble packing, independent of activation quantization.
//! The final DMA performs the real canonical-DM to K960-blocked-HBM conversion.
use crate::{
    Chip,
    axes::{H, L, P512},
};
use furiosa_opt_std::prelude::*;

type Cluster = m![P512 / 256];
type Rows = m![P512 % 256];
pub type PackedInput = m![P512 / 256, H / 960, P512 % 256, H % 960 / 8];
pub type PackedHidden = m![P512 / 256, L / 960, P512 % 256, L % 960 / 8];
pub type I4Input = crate::ops_batch_layouts::P512H960;
pub type I4Hidden = crate::ops_batch_layouts::P512L960;

fn pack_dm<K: AxisName>(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![P512, K]>,
    shifts: &HbmTensor<i32, Chip, m![K % 8]>,
) -> DmTensor<i32, Chip, Cluster, Rows, m![K / 8]> {
    let input: DmTensor<i32, Chip, Cluster, Rows, m![K]> = input.to_dm(&mut device.tdma);
    let shifts: DmTensor<i32, Chip, Cluster, Rows, m![K % 8]> = shifts.to_dm(&mut device.tdma);
    let shifts: VrfTensor<i32, Chip, Cluster, Rows, m![K % 8]> = device
        .sub
        .begin(shifts.view())
        .fetch::<m![1], m![K % 8]>()
        .collect::<m![1], m![K % 8]>()
        .to_vrf();
    // Masked fields occupy disjoint bits. Any subset sum fits signed I32,
    // including the negative top nibble, so AddSat equals the intended OR.
    device
        .main
        .begin(input.view())
        .fetch::<m![K / 8], m![K % 8]>()
        .collect::<m![K / 8], m![K % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, 15i32)
        .vector_logic(LogicBinaryOpI32::LeftShift, &shifts)
        .vector_narrow_split::<m![K / 4], m![K % 4]>()
        .vector_intra_slice_reduce::<K, m![K / 8], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![K / 16], m![K / 8 % 2 # 8]>()
        .commit_trim::<m![K / 8 % 2]>()
        .commit()
}

#[device(chip = 1)]
pub fn pack_input(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![P512, H]>,
    shifts: &HbmTensor<i32, Chip, m![H % 8]>,
    output: &mut HbmTensor<i32, Chip, PackedInput>,
) {
    let dense = pack_dm::<H>(device, input, shifts);
    // Physical bytes become [P/256,H/960,P%256,H%960/8]. This is DMA,
    // not a reshape/bitcast; every layout-transfer cost belongs to this kernel.
    dense
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn pack_hidden(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![P512, L]>,
    shifts: &HbmTensor<i32, Chip, m![L % 8]>,
    output: &mut HbmTensor<i32, Chip, PackedHidden>,
) {
    let dense = pack_dm::<L>(device, input, shifts);
    dense
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

// Typed byte-layout readers for the future raw-buffer probe. They share the
// actual UP/DOWN blocked input ABI, but do not claim to verify a GEMM.
#[device(chip = 1)]
pub fn read_input(
    device: &mut Device,
    input: &HbmTensor<i4, Chip, I4Input>,
    output: &mut HbmTensor<i4, Chip, m![P512, H]>,
) {
    let canonical: DmTensor<i4, Chip, Cluster, Rows, m![H]> = input.to_dm(&mut device.tdma);
    canonical
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn read_hidden(
    device: &mut Device,
    input: &HbmTensor<i4, Chip, I4Hidden>,
    output: &mut HbmTensor<i4, Chip, m![P512, L]>,
) {
    let canonical: DmTensor<i4, Chip, Cluster, Rows, m![L]> = input.to_dm(&mut device.tdma);
    canonical
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}
