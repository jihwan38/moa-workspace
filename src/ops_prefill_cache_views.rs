//! Live-prefix commits into an explicitly borrowed cache window.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::*};
use crate::device::layout::{Cluster, Slice};
/// Commit only the live prefix, with an unpadded index vector required by scatter.
fn commit_sliding_kv<const VALID: usize>(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = { VALID }]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![B = { VALID }, Ns, Ds]> = k
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![B = { VALID }, Ns, Ds]> = v
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    k_cache
        .scatter::<m![Ts], m![Ns, Ds]>()
        .by_byte_offsets(offsets.view())
        .from_dm(&mut device.tdma, k);
    v_cache
        .scatter::<m![Ts], m![Ns, Ds]>()
        .by_byte_offsets(offsets.view())
        .from_dm(&mut device.tdma, v);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_1(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 1]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<1>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_2(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 2]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<2>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_3(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 3]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<3>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_4(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 4]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<4>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_5(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 5]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<5>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_6(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 6]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<6>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_7(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 7]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<7>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_8(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 8]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<8>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_9(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 9]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<9>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_10(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 10]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<10>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_11(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 11]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<11>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_12(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 12]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<12>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_13(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 13]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<13>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_14(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 14]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<14>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_15(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 15]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<15>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_sliding_kv_16(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    offsets: &HbmTensor<i32, Chip, m![B = 16]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    commit_sliding_kv::<16>(device, k, v, offsets, k_cache, v_cache);
}

/// Commit only the live prefix, with an unpadded index vector required by scatter.
fn commit_full_kv<const VALID: usize>(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = { VALID }]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![B = { VALID }, Df]> = k
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Df]>(0)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![B = { VALID }, Df]> = v
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Df]>(0)
        .to_dm(&mut device.tdma);
    k_cache
        .scatter::<m![Tf], m![Df]>()
        .by_byte_offsets(offsets.view())
        .from_dm(&mut device.tdma, k);
    v_cache
        .scatter::<m![Tf], m![Df]>()
        .by_byte_offsets(offsets.view())
        .from_dm(&mut device.tdma, v);
}

#[device(chip = 1)]
pub fn commit_full_kv_1(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 1]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<1>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_2(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 2]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<2>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_3(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 3]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<3>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_4(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 4]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<4>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_5(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 5]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<5>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_6(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 6]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<6>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_7(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 7]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<7>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_8(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 8]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<8>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_9(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 9]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<9>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_10(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 10]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<10>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_11(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 11]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<11>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_12(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 12]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<12>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_13(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 13]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<13>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_14(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 14]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<14>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_15(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 15]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<15>(device, k, v, offsets, k_cache, v_cache);
}

#[device(chip = 1)]
pub fn commit_full_kv_16(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Df]>,
    v: &HbmTensor<bf16, Chip, m![B, Df]>,
    offsets: &HbmTensor<i32, Chip, m![B = 16]>,
    k_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
    v_cache: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    commit_full_kv::<16>(device, k, v, offsets, k_cache, v_cache);
}
