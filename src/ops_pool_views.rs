//! Canonical KV storage axis and explicit zero-copy view ABI probes.
//! Query batch axes remain separate from physical request slots.
use crate::device::layout::Cluster;
use crate::{Chip, axes::*};
use furiosa_opt_std::prelude::*;
// Physical request storage is independent of the unchanged R64 query axis.
axes![Rc = 32];
// Compatibility spelling for existing canonical-pool kernel imports.
pub use Rc as Rc64;

#[device(chip = 1)]
pub fn read_ring(
    device: &mut Device,
    input: HbmTensorView<'_, bf16, Chip, m![Ts # 32768, Ns, Ds]>,
    output: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, m![Ts / 4], m![Ts % 4, Ns, Ds]> =
        input.to_dm(&mut device.tdma);
    x.view().to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn write_ring(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    output: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, m![Ts / 4], m![Ts % 4, Ns, Ds]> =
        input.to_dm(&mut device.tdma);
    x.view().to_hbm_view(&mut device.tdma, output);
}

#[device(chip = 1)]
pub fn read_page(
    device: &mut Device,
    input: HbmTensorView<'_, bf16, Chip, m![Tf # 131072, Df]>,
    output: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, m![Tf / 2], m![Tf % 2, Df]> =
        input.to_dm(&mut device.tdma);
    x.view().to_hbm_view(&mut device.tdma, output.view_mut());
}

#[device(chip = 1)]
pub fn write_page(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    output: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, m![Tf / 2], m![Tf % 2, Df]> =
        input.to_dm(&mut device.tdma);
    x.view().to_hbm_view(&mut device.tdma, output);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops_request_attention::Fc;
    #[test]
    fn physical_pool_capacity_does_not_change_compute_or_full_context() {
        assert_eq!(Rc64::SIZE, 32);
        assert_eq!(R64::SIZE, 64);
        assert_eq!(Fc::SIZE, 256);
        assert_eq!(Ts::SIZE, 1024);
        assert_eq!(E::SIZE, 32768);
        assert_eq!(HbmTensor::<bf16, Chip, m![Rc64,Ts,Ns,Ds]>::size(), 134217728);
    }
    #[test]
    fn host_windows_preserve_pool_extent_and_borrow_kind() {
        // Unplaced tensors are sufficient to validate host mapping operations;
        // launch binding and actual byte ranges are covered by test_pool_views.
        let mut rings: HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]> = HbmTensor::new();
        for slot in [0, 15, Rc64::SIZE - 1] {
            let _: HbmTensorView<'_, bf16, Chip, m![Ts # 32768, Ns, Ds]> = unsafe {
                rings.view().tile::<m![Rc64], 1, m![Rc64 = 1 # 32, Ts, Ns, Ds]>(slot).reshape()
            };
            let _: HbmTensorViewMut<'_, bf16, Chip, m![Ts #{!} 32768, Ns, Ds]> = unsafe {
                rings.view_mut().tile::<m![Rc64], 1, m![Rc64 = 1 #{!} 32, Ts, Ns, Ds]>(slot).reshape()
            };
        }
        let mut pages: HbmTensor<bf16, Chip, m![Fc, Tf, Df]> = HbmTensor::new();
        for page in [0, 127, 255] {
            let _: HbmTensorView<'_, bf16, Chip, m![Tf # 131072, Df]> = unsafe {
                pages.view().tile::<m![Fc], 1, m![Fc = 1 # 256, Tf, Df]>(page).reshape()
            };
            let _: HbmTensorViewMut<'_, bf16, Chip, m![Tf #{!} 131072, Df]> = unsafe {
                pages.view_mut().tile::<m![Fc], 1, m![Fc = 1 #{!} 256, Tf, Df]>(page).reshape()
            };
        }
    }
}
