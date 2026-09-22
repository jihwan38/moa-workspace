//! Diagnostic wrappers call the frozen candidate projection implementations.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{H, L, Dummy256}, device::{layout::{Cluster, Slice, Replicated}, mlp}};
use mlp::{UpGateCluster, UpGateSlices, UpGateRowsPaired};

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
    // Same activation path as feedforward_sharded after pre-RMS.
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, UpGateCluster, m![Dummy256 / 16, 1 # 16], m![H]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, UpGateCluster, m![Dummy256 / 16, 16], m![H]> = device.main.begin(x.view())
        .fetch::<m![1], m![H]>()
        .switch::<m![Dummy256 / 16, 16], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 16 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>().commit();
    let x: DmTensor<bf16, Chip, UpGateCluster, Replicated, m![H]> = unsafe {x.reshape()};
    let x: DmTensorView<'_, bf16, Chip, UpGateCluster, UpGateSlices, m![H]> = unsafe {x.view().reshape()};
    let x: TrfTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![1], m![H]> = device.sub.begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>().to_trf();
    let (up, gate) = mlp::project_up_and_gate(device, &x, w, g, s, t);
    up.view().to_hbm_view(&mut device.tdma, y.view_mut());
    gate.view().to_hbm_view(&mut device.tdma, z.view_mut());
}

#[device(chip = 1)]
pub fn down(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![L]>,
    w: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    s: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    y: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = x.to_dm(&mut device.tdma);
    let down = mlp::project_down_sharded(device, &x, w, s);
    down.view().to_hbm_view(&mut device.tdma, y.view_mut());
}
