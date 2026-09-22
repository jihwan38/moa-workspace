//! Diagnostic wrappers call the frozen candidate projection implementations.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{H, L, Dummy256}, device::{layout::{Cluster, Slice, Replicated}, mlp}};
use mlp::{UpGateCluster, UpGateSlices, UpGateRowsPaired, FfnTerm};

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
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x.to_dm(&mut device.tdma);
    // Quantize the actual runtime BF16 activation; both projections reuse it.
    let x: DmTensor<bf16, Chip, m![2], Slice, m![H]> = x.to_dm(&mut device.tdma);
    let (x, x_scale) = mlp::prepare_ffn_fp8_split(device, &x);
    let x: DmTensorView<'_, f8e4m3, Chip, UpGateCluster, UpGateSlices, m![FfnTerm, H]> = unsafe { x.view().reshape() };
    let x_scale: DmTensorView<'_, f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> = unsafe { x_scale.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnTerm, H]> = device
        .sub.begin(x)
        .fetch::<m![FfnTerm], m![H]>()
        .collect::<m![FfnTerm, H / 32], m![H % 32]>().to_trf();
    let x_scale_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> = device
        .sub.begin(x_scale)
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    let (up, gate) = mlp::project_up_and_gate(
        device,
        &x_trf,
        &x_scale_vrf,
        w,
        g,
        s,
        t,
    );
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
    let result = mlp::project_down_sharded(device, &x, w, s);
    result.view().to_hbm_view(&mut device.tdma, y.view_mut());
}
