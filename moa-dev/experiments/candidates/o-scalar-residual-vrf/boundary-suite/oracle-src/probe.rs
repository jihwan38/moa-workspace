//! Diagnostic boundary only: production helper including F32 scale and BF16 cast.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{H, Qs, Ns, Gs, Ds}};
use crate::device::layout::Cluster;

#[device(chip = 1)]
pub fn output_before_rms(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    output: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x: HbmTensorView<'_, bf16, Chip, m![Qs]> = unsafe { x.view().reshape() };
    let result: DmTensor<bf16, Chip, Cluster, m![1 # 32, H / 480], m![H % 480]> =
        crate::device::sliding::projection::project_output(device, x, weight, scale);
    result.view().to_hbm_view(&mut device.tdma, output.view_mut());
}
