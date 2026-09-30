//! Diagnostic post-pre-RMS to pre-head-RMS boundary using unchanged production helpers.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{H, Qs, Ps, Ns, Gs, Ds}};

#[device(chip = 1)]
pub fn qkv_before_head_rms(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    qw: &HbmTensor<f8e4m3, Chip, m![Qs,H]>,
    kw: &HbmTensor<f8e4m3, Chip, m![Ps,H]>,
    vw: &HbmTensor<f8e4m3, Chip, m![Ps,H]>,
    qs: &HbmTensor<bf16, Chip, m![Qs]>,
    ks: &HbmTensor<bf16, Chip, m![Ps]>,
    vs: &HbmTensor<bf16, Chip, m![Ps]>,
    qo: &mut HbmTensor<bf16, Chip, m![Ns,Gs,Ds]>,
    ko: &mut HbmTensor<bf16, Chip, m![Ns,Ds]>,
    vo: &mut HbmTensor<bf16, Chip, m![Ns,Ds]>,
) {
    let x: DmTensor<bf16,Chip,m![2],m![1 # 32,H/480],m![H%480]> = x.to_dm(&mut device.tdma);
    let (q,k,v) = crate::device::sliding::projection::project_qkv_native(device,&x,qw,kw,vw,qs,ks,vs);
    q.view().to_hbm_view(&mut device.tdma,qo.view_mut());
    k.view().to_hbm_view(&mut device.tdma,ko.view_mut());
    v.view().to_hbm_view(&mut device.tdma,vo.view_mut());
}
