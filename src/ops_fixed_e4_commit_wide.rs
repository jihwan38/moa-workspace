//! E4-byte commit offsets; input is already q022-rounded BF16.
//! All128 rows must be live. Sliding commits happen only after every query has attended.
use crate::{Chip, axes::{P128, Ns, Ds, Ts}};
use furiosa_opt_std::prelude::*;
use crate::ops_pool_views::Rc64;
type Cluster=m![P128/64];
#[device(chip=1)]
pub fn commit_sliding_kv_p128(device:&mut Device,k:&HbmTensor<bf16,Chip,m![P128,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![P128,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![P128]>,k_cache:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,v_cache:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){
 type Rows=m![P128%64,Ns/2];
 let k:DmTensor<bf16,Chip,Cluster,Rows,m![Ns%2,Ds]>=k.to_dm(&mut device.tdma);
 let v:DmTensor<bf16,Chip,Cluster,Rows,m![Ns%2,Ds]>=v.to_dm(&mut device.tdma);
 let k=crate::ops_fixed_e4_prompt_codec::encode_rows::<_,_,m![Ns%2]>(device,&k);
 let v=crate::ops_fixed_e4_prompt_codec::encode_rows::<_,_,m![Ns%2]>(device,&v);
 k_cache.view_mut().scatter::<m![Rc64,Ts],m![Ns,Ds]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,k);
 v_cache.view_mut().scatter::<m![Rc64,Ts],m![Ns,Ds]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,v);
}
