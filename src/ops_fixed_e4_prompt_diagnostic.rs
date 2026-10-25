//! Standalone oracle setup only; no serving callers.
use crate::{Chip,axes::{Ts,Ns,Ds},ops_pool_views::Rc64};
use furiosa_opt_std::prelude::*;
#[device(chip=1)]
pub fn copy_ring(device:&mut Device,k:&HbmTensor<f8e4m3,Chip,m![Ts,Ns,Ds]>,v:&HbmTensor<f8e4m3,Chip,m![Ts,Ns,Ds]>,base:&HbmTensor<i32,Chip,m![1]>,pk:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pv:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>) {
 type Rows=m![Ts/8,1#2];
 let k:DmTensor<f8e4m3,Chip,crate::device::layout::Cluster,Rows,m![Ts%8,Ns,Ds]>=k.to_dm(&mut device.tdma);
 let v:DmTensor<f8e4m3,Chip,crate::device::layout::Cluster,Rows,m![Ts%8,Ns,Ds]>=v.to_dm(&mut device.tdma);
 pk.view_mut().scatter::<m![Rc64],m![Ts,Ns,Ds]>().by_byte_offsets(base.view()).from_dm(&mut device.tdma,k);
 pv.view_mut().scatter::<m![Rc64],m![Ts,Ns,Ds]>().by_byte_offsets(base.view()).from_dm(&mut device.tdma,v);
}
