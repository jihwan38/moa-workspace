//! FP4 decode diagnostic only: no DPE, scale, quantizer, or official kernel edits.
use furiosa_opt_std::prelude::*;
use crate::Chip;

axes![DecodeR = 32, DecodeK = 512, DecodeS = 256];
type OneCluster = m![1 # 2];
type AllSlices = m![DecodeS];

#[device(chip = 1)]
pub fn legacy_decode(
    device: &mut Device,
    input: &HbmTensor<f4e2m1, Chip, m![DecodeR, DecodeK]>,
    output: &mut HbmTensor<bf16, Chip, m![DecodeS, DecodeR, DecodeK]>,
) {
    let packed: DmTensor<f4e2m1, Chip, OneCluster, AllSlices, m![DecodeR, DecodeK]> = input.to_dm(&mut device.tdma);
    let fp8: DmTensor<f8e4m3, Chip, OneCluster, AllSlices, m![DecodeR, DecodeK]> = device.main
        .begin(packed.view())
        .fetch::<m![DecodeR], m![DecodeK]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![DecodeR, DecodeK / 32], m![DecodeK % 32]>()
        .commit_trim::<m![DecodeK % 32]>().commit();
    let decoded: DmTensor<bf16, Chip, OneCluster, AllSlices, m![DecodeR, DecodeK]> = device.main
        .begin(fp8.view())
        .fetch::<m![DecodeR], m![DecodeK]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![DecodeR, DecodeK / 16], m![DecodeK % 16]>()
        .commit_trim::<m![DecodeK % 16]>().commit();
    decoded.view().to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn fused_decode(
    device: &mut Device,
    input: &HbmTensor<f4e2m1, Chip, m![DecodeR, DecodeK]>,
    output: &mut HbmTensor<bf16, Chip, m![DecodeS, DecodeR, DecodeK]>,
) {
    let packed: DmTensor<f4e2m1, Chip, OneCluster, AllSlices, m![DecodeR, DecodeK]> = input.to_dm(&mut device.tdma);
    let decoded: DmTensor<bf16, Chip, OneCluster, AllSlices, m![DecodeR, DecodeK]> = device.main
        .begin(packed.view())
        .fetch::<m![DecodeR], m![DecodeK]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![DecodeR, DecodeK / 8], m![DecodeK % 8]>()
        .cast::<bf16, m![DecodeK % 8 # 16]>()
        .commit_trim::<m![DecodeK % 8]>().commit();
    decoded.view().to_hbm_view(&mut device.tdma, output.view_mut());
}
