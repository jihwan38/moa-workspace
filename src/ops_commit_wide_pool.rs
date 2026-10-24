//! All128 rows must be live. Sliding commits happen only after every query has attended.
use crate::{Chip, axes::{P128, Ns, Ds, Df, Ts, Tf}};
use furiosa_opt_std::prelude::*;
use crate::{ops_pool_views::Rc64,ops_request_attention::Fc};
type Cluster=m![P128/64];
#[device(chip=1)]
pub fn commit_sliding_kv_p128(device:&mut Device,k:&HbmTensor<bf16,Chip,m![P128,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![P128,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![P128]>,k_cache:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,v_cache:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){
 type Rows=m![P128%64,Ns/2];
 let k:DmTensor<bf16,Chip,Cluster,Rows,m![Ns%2,Ds]>=k.to_dm(&mut device.tdma);
 let v:DmTensor<bf16,Chip,Cluster,Rows,m![Ns%2,Ds]>=v.to_dm(&mut device.tdma);
 k_cache.view_mut().scatter::<m![Rc64,Ts],m![Ns,Ds]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,k);
 v_cache.view_mut().scatter::<m![Rc64,Ts],m![Ns,Ds]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,v);
}
#[device(chip=1)]
pub fn commit_full_kv_p128(device:&mut Device,k:&HbmTensor<bf16,Chip,m![P128,Df]>,v:&HbmTensor<bf16,Chip,m![P128,Df]>,offsets:&HbmTensor<i32,Chip,m![P128]>,k_cache:&mut HbmTensor<bf16,Chip,m![Fc,Tf,Df]>,v_cache:&mut HbmTensor<bf16,Chip,m![Fc,Tf,Df]>){
 type Rows=m![P128%64,Df/128];
 let k:DmTensor<bf16,Chip,Cluster,Rows,m![Df%128]>=k.to_dm(&mut device.tdma);
 let v:DmTensor<bf16,Chip,Cluster,Rows,m![Df%128]>=v.to_dm(&mut device.tdma);
 k_cache.view_mut().scatter::<m![Fc,Tf],m![Df]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,k);
 v_cache.view_mut().scatter::<m![Fc,Tf],m![Df]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,v);
}

#[device(chip = 1)]
pub fn copy_sliding_pool_to_cache_rc64(
    device: &mut Device,
    pool_k: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    slot_byte_offset: &HbmTensor<i32, Chip, m![1]>,
    k: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    type Source = m![Ts/8,1 # 2];
    let rk: DmTensor<bf16, Chip, crate::device::layout::Cluster, Source, m![Ts % 8, Ns, Ds]> = pool_k
        .gather::<m![Rc64], m![Ts, Ns, Ds]>()
        .by_byte_offsets(slot_byte_offset)
        .to_dm(&mut device.tdma);
    let rv: DmTensor<bf16, Chip, crate::device::layout::Cluster, Source, m![Ts % 8, Ns, Ds]> = pool_v
        .gather::<m![Rc64], m![Ts, Ns, Ds]>()
        .by_byte_offsets(slot_byte_offset)
        .to_dm(&mut device.tdma);
    rk.view().to_hbm_view(&mut device.tdma, k.view_mut());
    rv.view().to_hbm_view(&mut device.tdma, v.view_mut());
}
