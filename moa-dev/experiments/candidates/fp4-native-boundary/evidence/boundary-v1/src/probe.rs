use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{H, L, Dummy2}};
axes![Term = 3, Row = 30, SliceIndex = 256];
type Clusters = m![Dummy2];
type Slices = m![SliceIndex];

#[device(chip = 1)]
pub fn materialized(
    device: &mut Device,
    x: &HbmTensor<f8e5m2, Chip, m![Term, H]>,
    w: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    y: &mut HbmTensor<f32, Chip, m![L, H / 16]>,
) {
    let x: DmTensor<f8e5m2, Chip, Clusters, Slices, m![Term, H]> = x.to_dm(&mut device.tdma);
    let x: TrfTensor<f8e5m2, Chip, Clusters, Slices, m![1], m![Term, H]> = device.sub
        .begin(x.view()).fetch::<m![Term], m![H]>()
        .collect::<m![Term, H / 32], m![H % 32]>().to_trf();
    let w: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, SliceIndex, Row, H]> = unsafe { w.view().reshape() };
    let w: DmTensor<f4e2m1, Chip, Clusters, Slices, m![Row, H]> = w.to_dm(&mut device.tdma);
    let w: DmTensor<f8e5m2, Chip, Clusters, Slices, m![Row, H]> = device.main
        .begin(w.view()).fetch::<m![Row], m![H]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![Row, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>().commit();
    let partials: DmTensor<f32, Chip, Clusters, Slices, m![Row, H / 16]> = device.main
        .begin(w.view()).fetch::<m![Row], m![H]>()
        .collect::<m![Row, H / 32], m![H % 32]>()
        .contract_outer::<m![Row, H / 64, Term], m![H % 64], _, _, _>(&x)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![Row, H / 64]>()
        .contract_lane::<m![Row, H / 64], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .commit_trim::<m![H / 16 % 4]>().commit();
    let partials: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30, H / 16]> = unsafe { partials.reshape() };
    partials.view().to_hbm_view(&mut device.tdma, y.view_mut());
}

#[device(chip = 1)]
pub fn fused(
    device: &mut Device,
    x: &HbmTensor<f8e5m2, Chip, m![Term, H]>,
    w: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    y: &mut HbmTensor<f32, Chip, m![L, H / 16]>,
) {
    let x: DmTensor<f8e5m2, Chip, Clusters, Slices, m![Term, H]> = x.to_dm(&mut device.tdma);
    let x: TrfTensor<f8e5m2, Chip, Clusters, Slices, m![1], m![Term, H]> = device.sub
        .begin(x.view()).fetch::<m![Term], m![H]>()
        .collect::<m![Term, H / 32], m![H % 32]>().to_trf();
    let w: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, SliceIndex, Row, H]> = unsafe { w.view().reshape() };
    let w: DmTensor<f4e2m1, Chip, Clusters, Slices, m![Row, H]> = w.to_dm(&mut device.tdma);
    let partials: DmTensor<f32, Chip, Clusters, Slices, m![Row, H / 16]> = device.main
        .begin(w.view()).fetch::<m![Row], m![H]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![Row, H / 32], m![H % 32]>()
        .contract_outer::<m![Row, H / 64, Term], m![H % 64], _, _, _>(&x)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![Row, H / 64]>()
        .contract_lane::<m![Row, H / 64], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .commit_trim::<m![H / 16 % 4]>().commit();
    let partials: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30, H / 16]> = unsafe { partials.reshape() };
    partials.view().to_hbm_view(&mut device.tdma, y.view_mut());
}

#[device(chip = 1)]
pub fn decode(
    device: &mut Device,
    w: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    y: &mut HbmTensor<f8e5m2, Chip, m![L, H]>,
) {
    let w: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, SliceIndex, Row, H]> = unsafe { w.view().reshape() };
    let w: DmTensor<f4e2m1, Chip, Clusters, Slices, m![Row, H]> = w.to_dm(&mut device.tdma);
    let w: DmTensor<f8e5m2, Chip, Clusters, Slices, m![Row, H]> = device.main
        .begin(w.view()).fetch::<m![Row], m![H]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![Row, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>().commit();
    let w: DmTensor<f8e5m2, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30, H]> = unsafe { w.reshape() };
    w.view().to_hbm_view(&mut device.tdma, y.view_mut());
}
