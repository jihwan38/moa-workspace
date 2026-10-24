//! Allocate canonical KV pools on device and write every logical cell to +0.
//! Returned HBM storage is placed by the runtime; a small DM tile is reused.
use crate::{Chip, axes::*, ops_pool_views::Rc64, ops_request_attention::Fc};
use furiosa_opt_std::prelude::*;
type Cluster = m![1 # 2];

#[device(chip = 1)]
pub fn zero_sliding_pool(device: &mut Device) -> HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]> {
    let mut zero: DmTensor<bf16, Chip, Cluster, m![Ts / 4], m![Ts % 4, Ns, Ds]> = DmTensor::new();
    zero.view_mut()
        .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    let mut output = HbmTensor::new();
    for ring in 0..Rc64::SIZE {
        zero.view().to_hbm_view(
            &mut device.tdma,
            output
                .view_mut()
                .tile::<m![Rc64], 1, m![Rc64 = 1 #{!} 64, Ts, Ns, Ds]>(ring),
        );
    }
    output
}

#[device(chip = 1)]
pub fn zero_full_pool(device: &mut Device) -> HbmTensor<bf16, Chip, m![Fc, Tf, Df]> {
    let mut zero: DmTensor<bf16, Chip, Cluster, m![Tf / 2], m![Tf % 2, Df]> = DmTensor::new();
    zero.view_mut()
        .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    let mut output = HbmTensor::new();
    for page in 0..Fc::SIZE {
        zero.view().to_hbm_view(
            &mut device.tdma,
            output
                .view_mut()
                .tile::<m![Fc], 1, m![Fc = 1 #{!} 256, Tf, Df]>(page),
        );
    }
    output
}
