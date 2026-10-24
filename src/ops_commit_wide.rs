//! All128 rows must be live. Sliding commits happen only after every query has attended.
use crate::{Chip, axes::{P128, Ns, Ds, Df, Ts, Tf}};
use furiosa_opt_std::prelude::*;
type Cluster=m![P128/64];
#[device(chip=1)]
pub fn commit_sliding_kv_p128(device:&mut Device,k:&HbmTensor<bf16,Chip,m![P128,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![P128,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![P128]>,k_cache:&mut HbmTensor<bf16,Chip,m![Ts,Ns,Ds]>,v_cache:&mut HbmTensor<bf16,Chip,m![Ts,Ns,Ds]>){
 type Rows=m![P128%64,Ns/2];
 let k:DmTensor<bf16,Chip,Cluster,Rows,m![Ns%2,Ds]>=k.to_dm(&mut device.tdma);
 let v:DmTensor<bf16,Chip,Cluster,Rows,m![Ns%2,Ds]>=v.to_dm(&mut device.tdma);
 k_cache.view_mut().scatter::<m![Ts],m![Ns,Ds]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,k);
 v_cache.view_mut().scatter::<m![Ts],m![Ns,Ds]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,v);
}
#[device(chip=1)]
pub fn commit_full_kv_p128(device:&mut Device,k:&HbmTensor<bf16,Chip,m![P128,Df]>,v:&HbmTensor<bf16,Chip,m![P128,Df]>,offsets:&HbmTensor<i32,Chip,m![P128]>,k_cache:&mut HbmTensor<bf16,Chip,m![Tf,Df]>,v_cache:&mut HbmTensor<bf16,Chip,m![Tf,Df]>){
 type Rows=m![P128%64,Df/128];
 let k:DmTensor<bf16,Chip,Cluster,Rows,m![Df%128]>=k.to_dm(&mut device.tdma);
 let v:DmTensor<bf16,Chip,Cluster,Rows,m![Df%128]>=v.to_dm(&mut device.tdma);
 k_cache.view_mut().scatter::<m![Tf],m![Df]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,k);
 v_cache.view_mut().scatter::<m![Tf],m![Df]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,v);
}
