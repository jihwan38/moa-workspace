//! Untimed independent P128 GeGLU oracle. Uses the original P128 consumer,
//! not the new joined-output bridge, and exports canonical BF16 hidden rows.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{P128,L}};

#[device(chip=1)]
pub fn geglu_p128(device:&mut Device,
    up:&HbmTensor<bf16,Chip,m![P128,L]>, gate:&HbmTensor<bf16,Chip,m![P128,L]>,
    ug:&HbmTensor<f32,Chip,m![1]>, gg:&HbmTensor<f32,Chip,m![1]>,
    out:&mut HbmTensor<bf16,Chip,m![P128,L]>) {
    let up:DmTensor<bf16,Chip,m![P128/64],m![L/240,1#4],m![L%240,P128%64]>=up.to_dm(&mut device.tdma);
    let gate:DmTensor<bf16,Chip,m![P128/64],m![L/240,1#4],m![L%240,P128%64]>=gate.to_dm(&mut device.tdma);
    let hidden=crate::ops_batch_kpart::geglu_k4(device,&up,&gate,ug,gg);
    hidden.view().to_hbm_view(&mut device.tdma,out.view_mut());
}
