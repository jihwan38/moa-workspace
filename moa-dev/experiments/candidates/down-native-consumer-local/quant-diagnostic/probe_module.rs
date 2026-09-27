// Source-only diagnostic append. Do not substitute this file for production mlp.rs.
// A future builder must prepend the complete frozen local-v4/mlp.rs unchanged.
pub mod quant_diagnostic {
    use super::*;
    // Preserve the quantizer's physical cluster/slice/element ordering in HBM.
    pub type PackedShape = m![Dummy2, H / 60, TileL / 1920, FfnTerm, TileL % 1920];
    pub type CoefficientShape = m![Dummy2, H / 60, TileL / 1920, FfnTerm, 1 # 8];

    pub fn expose(
        device: &mut Device,
        input: &HbmTensor<bf16, Chip, m![L]>,
        output_terms: &mut HbmTensor<f8e5m2, Chip, PackedShape>,
        output_coefficients: &mut HbmTensor<f32, Chip, CoefficientShape>,
    ) {
        // The supplied actual post-GeGLU BF16 vector has cluster-local L halves.
        let input: HbmTensorView<'_, bf16, Chip, m![Dummy2, TileL]> =
            unsafe { input.view().reshape() };
        let input: DmTensor<bf16, Chip, UpGateCluster, Slice, m![TileL]> =
            input.to_dm(&mut device.tdma);
        let (terms, coefficients) = prepare_down_fp8_split(device, &input);
        terms.view().to_hbm_view(&mut device.tdma, output_terms.view_mut());
        coefficients.view().to_hbm_view(&mut device.tdma, output_coefficients.view_mut());
    }
}
