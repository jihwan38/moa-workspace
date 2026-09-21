//! Diagnostic kernels only; never an official score candidate.
use furiosa_opt_std::prelude::*;
use crate::Chip;

axes![ProbeK = 64, ProbeR = 4, ProbeS = 256];
type OneCluster = m![1 # 2];
type AllSlices = m![ProbeS];

#[device(chip = 1)]
pub fn fp8_scalar(
    device: &mut Device,
    x: &HbmTensor<f8e4m3, Chip, m![ProbeK]>,
    weights: &HbmTensor<f8e4m3, Chip, m![ProbeR, ProbeK]>,
    out: &mut HbmTensor<f32, Chip, m![ProbeS, ProbeR]>,
) {
    let x: DmTensor<f8e4m3, Chip, OneCluster, AllSlices, m![ProbeK]> = x.to_dm(&mut device.tdma);
    let w: DmTensor<f8e4m3, Chip, OneCluster, AllSlices, m![ProbeR, ProbeK]> = weights.to_dm(&mut device.tdma);
    let trf: TrfTensor<f8e4m3, Chip, OneCluster, AllSlices, m![1], m![ProbeK]> = device.sub
        .begin(x.view()).fetch::<m![ProbeK / 32], m![ProbeK % 32]>()
        .collect::<m![ProbeK / 32], m![ProbeK % 32]>().to_trf();
    let result: DmTensor<f32, Chip, OneCluster, AllSlices, m![ProbeR]> = device.main
        .begin(w.view()).fetch::<m![ProbeR, ProbeK / 32], m![ProbeK % 32]>()
        .collect::<m![ProbeR, ProbeK / 32], m![ProbeK % 32]>()
        .contract_outer::<m![ProbeR, ProbeK / 64], m![ProbeK % 64], _, _, _>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![ProbeR]>()
        .contract_lane::<m![ProbeR], m![1 # 8]>(LaneMode::Interleaved)
        .transpose::<m![ProbeR / 2], m![ProbeR % 2 # 8]>()
        .commit_trim::<m![ProbeR % 2]>().commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn fp8_blocks(
    device: &mut Device,
    x: &HbmTensor<f8e4m3, Chip, m![ProbeK]>,
    weights: &HbmTensor<f8e4m3, Chip, m![ProbeR, ProbeK]>,
    out: &mut HbmTensor<f32, Chip, m![ProbeS, ProbeR, ProbeK / 16]>,
) {
    let x: DmTensor<f8e4m3, Chip, OneCluster, AllSlices, m![ProbeK]> = x.to_dm(&mut device.tdma);
    let w: DmTensor<f8e4m3, Chip, OneCluster, AllSlices, m![ProbeR, ProbeK]> = weights.to_dm(&mut device.tdma);
    let trf: TrfTensor<f8e4m3, Chip, OneCluster, AllSlices, m![1], m![ProbeK]> = device.sub
        .begin(x.view()).fetch::<m![ProbeK / 32], m![ProbeK % 32]>()
        .collect::<m![ProbeK / 32], m![ProbeK % 32]>().to_trf();
    let result: DmTensor<f32, Chip, OneCluster, AllSlices, m![ProbeR, ProbeK / 16]> = device.main
        .begin(w.view()).fetch::<m![ProbeR, ProbeK / 32], m![ProbeK % 32]>()
        .collect::<m![ProbeR, ProbeK / 32], m![ProbeK % 32]>()
        .contract_outer::<m![ProbeR, ProbeK / 64], m![ProbeK % 64], _, _, _>(&trf)
        .contract_packet::<m![ProbeK / 16 % 4]>()
        .contract_time::<m![ProbeR, ProbeK / 64]>()
        .contract_lane::<m![ProbeR, ProbeK / 16], m![1 # 8]>(LaneMode::Interleaved)
        .transpose::<m![ProbeR, ProbeK / 16 / 2], m![ProbeK / 16 % 2 # 8]>()
        .commit_trim::<m![ProbeK / 16 % 2]>().commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
