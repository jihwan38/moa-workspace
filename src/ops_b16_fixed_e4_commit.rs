//! Fixed scale1 E4 append. Caller supplies byte offsets for the E4 pools.
use crate::{Chip, axes::{B,Ns,Ds,Ts},ops_pool_views::Rc64};
use furiosa_opt_std::prelude::*;
fn encode<const VALID: usize>(device:&mut Device,x:&DmTensor<bf16,Chip,crate::device::layout::Cluster,crate::device::layout::Slice,m![B={VALID},Ns,Ds]>)->DmTensor<f8e4m3,Chip,crate::device::layout::Cluster,crate::device::layout::Slice,m![B={VALID},Ns,Ds]>{
    device.main.begin(x.view())
        .fetch::<m![B={VALID},Ns,Ds/16],m![Ds%16]>()
        .fetch_cast::<f32>()
        .collect::<m![B={VALID},Ns,Ds/8],m![Ds%8]>()
        .cast::<f8e4m3,m![Ds%8 #32]>()
        .commit_trim::<m![Ds%8]>().commit()
}
pub(crate) fn commit_sliding_pool<const VALID: usize>(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    byte_offsets: &HbmTensor<i32, Chip, m![B = { VALID }]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
) {
    let k: DmTensor<bf16, Chip, crate::device::layout::Cluster, crate::device::layout::Slice, m![B = { VALID }, Ns, Ds]> = k
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, crate::device::layout::Cluster, crate::device::layout::Slice, m![B = { VALID }, Ns, Ds]> = v
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    let k = encode::<VALID>(device, &k);
    let v = encode::<VALID>(device, &v);
    pool_k
        .view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(byte_offsets.view())
        .from_dm(&mut device.tdma, k);
    pool_v
        .view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(byte_offsets.view())
        .from_dm(&mut device.tdma, v);
}
