//! Startup-only byte-preserving joins; old consumers retain their original tensors.
use crate::{Chip, axes::{H, L}, ops_native_fp8_blocked::UpWeight,
    ops_ffn_projection_clusters::{FfnProjection, JoinedWeight, JoinedScale}};
use furiosa_opt_std::prelude::*;
type Cluster = m![1 # 2];

#[device(chip=1)]
pub fn join(device: &mut Device,
    up: &HbmTensor<f8e4m3, Chip, UpWeight>,
    gate: &HbmTensor<f8e4m3, Chip, UpWeight>,
    up_scale: &HbmTensor<f32, Chip, m![L]>,
    gate_scale: &HbmTensor<f32, Chip, m![L]>,
) -> (HbmTensor<f8e4m3, Chip, JoinedWeight>, HbmTensor<f32, Chip, JoinedScale>) {
    let mut weight = HbmTensor::new();
    let mut scale = HbmTensor::new();
    {
        let value: DmTensor<f8e4m3, Chip, Cluster, m![L/240,H/960], m![L%240,H%960]> = up.to_dm(&mut device.tdma);
        value.view().to_hbm_view(&mut device.tdma, weight.view_mut()
            .tile::<m![FfnProjection],1,m![FfnProjection=1#{!}2,L/240,H/960,L%240,H%960]>(0));
    }
    {
        let value: DmTensor<f8e4m3, Chip, Cluster, m![L/240,H/960], m![L%240,H%960]> = gate.to_dm(&mut device.tdma);
        value.view().to_hbm_view(&mut device.tdma, weight.view_mut()
            .tile::<m![FfnProjection],1,m![FfnProjection=1#{!}2,L/240,H/960,L%240,H%960]>(1));
    }
    {
        let value: DmTensor<f32, Chip, Cluster, m![L/60], m![L%60]> = up_scale.to_dm(&mut device.tdma);
        value.view().to_hbm_view(&mut device.tdma, scale.view_mut()
            .tile::<m![FfnProjection],1,m![FfnProjection=1#{!}2,L]>(0));
    }
    {
        let value: DmTensor<f32, Chip, Cluster, m![L/60], m![L%60]> = gate_scale.to_dm(&mut device.tdma);
        value.view().to_hbm_view(&mut device.tdma, scale.view_mut()
            .tile::<m![FfnProjection],1,m![FfnProjection=1#{!}2,L]>(1));
    }
    (weight, scale)
}
