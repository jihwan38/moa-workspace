
// Diagnostic entry points only. The production M03 prefix above is byte-identical.
pub mod validation_probe {
    use super::*;

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
        let x: DmTensor<bf16, Chip, m![2], Slice, m![H]> = x.to_dm(&mut device.tdma);
        let (x, scale) = prepare_ffn_fp8_split(device, &x);
        let x: DmTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![FfnTerm, H]> = unsafe { x.reshape() };
        let scale: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> = unsafe { scale.reshape() };
        let (up, gate) = project_up_and_gate(device, &x, &scale, w, g, s, t);
        // Rename logical L-half/local axes; preserve all physical owner/padding axes.
        let up: DmTensor<bf16, Chip, m![L / 7680], m![L / 120 % 64, 1 # 4], m![L % 120]> = unsafe { up.reshape() };
        let gate: DmTensor<bf16, Chip, m![L / 7680], m![L / 120 % 64, 1 # 4], m![L % 120]> = unsafe { gate.reshape() };
        up.view().to_hbm_view(&mut device.tdma, y.view_mut());
        gate.view().to_hbm_view(&mut device.tdma, z.view_mut());
    }

    pub fn down(
        device: &mut Device,
        x: &HbmTensor<bf16, Chip, m![L]>,
        w: &HbmTensor<f4e2m1, Chip, m![H, L]>,
        s: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
        y: &mut HbmTensor<bf16, Chip, m![H]>,
    ) {
        let x: HbmTensorView<'_, bf16, Chip, m![Dummy2, TileL]> = unsafe { x.view().reshape() };
        let x: DmTensor<bf16, Chip, UpGateCluster, LocalRows, m![TileL % 120]> = x.to_dm(&mut device.tdma);
        let partial = project_down_tile(device, &x, w, s);
        let result = finish_down(device, &partial);
        result.view().to_hbm_view(&mut device.tdma, y.view_mut());
    }
}
