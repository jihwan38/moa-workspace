
// Diagnostic entry only: the complete frozen production source remains above.
pub mod validation_probe {
    use super::*;
    pub fn down(device: &mut Device,
        x: &HbmTensor<bf16, Chip, m![L]>,
        w: &HbmTensor<f4e2m1, Chip, m![H, L]>,
        s: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
        y: &mut HbmTensor<bf16, Chip, m![H]>) {
        let x: DmTensor<bf16, Chip, m![L / 7680], m![L / 120 % 64, 1 # 4], m![L % 120]> = x.to_dm(&mut device.tdma);
        let x: DmTensor<bf16, Chip, UpGateCluster, LocalRows, m![TileL % 120]> = unsafe { x.reshape() };
        let partial = project_down_tile(device, &x, w, s);
        let down = finish_down(device, &partial);
        down.view().to_hbm_view(&mut device.tdma, y.view_mut());
    }
}
