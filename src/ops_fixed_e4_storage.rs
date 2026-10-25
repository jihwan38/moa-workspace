//! Fixed scale-one E4 storage utilities; q022 BF16 head values remain the input.
use crate::{Chip, axes::*, ops_pool_views::Rc64};
use furiosa_opt_std::prelude::*;

pub(crate) fn encode_rows<C:M,S:M,T:M>(device:&mut Device,
    x:&DmTensor<bf16,Chip,C,S,m![{T},Ds]>) -> DmTensor<f8e4m3,Chip,C,S,m![{T},Ds]> {
    device.main.begin(x.view())
        .fetch::<m![{T},Ds/16],m![Ds%16]>().fetch_cast::<f32>()
        .collect::<m![{T},Ds/8],m![Ds%8]>()
        .cast::<f8e4m3,m![Ds%8#32]>().commit_trim::<m![Ds%8]>().commit()
}
pub(crate) fn decode_rows<C:M,S:M,T:M>(device:&mut Device,
    x:&DmTensor<f8e4m3,Chip,C,S,m![{T},Ds]>) -> DmTensor<bf16,Chip,C,S,m![{T},Ds]> {
    device.main.begin(x.view())
        .fetch::<m![{T},Ds/32],m![Ds%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![{T},Ds/16],m![Ds%16]>().commit_trim::<m![Ds%16]>().commit()
}
pub(crate) fn encode_head_rows<C:M,S:M,T:M>(device:&mut Device,
    x:&DmTensor<bf16,Chip,C,S,m![{T},Ns,Ds]>) -> DmTensor<f8e4m3,Chip,C,S,m![{T},Ns,Ds]> {
    device.main.begin(x.view())
        .fetch::<m![{T},Ns,Ds/16],m![Ds%16]>().fetch_cast::<f32>()
        .collect::<m![{T},Ns,Ds/8],m![Ds%8]>()
        .cast::<f8e4m3,m![Ds%8#32]>().commit_trim::<m![Ds%8]>().commit()
}
pub(crate) fn decode_head_rows<C:M,S:M,T:M>(device:&mut Device,
    x:&DmTensor<f8e4m3,Chip,C,S,m![{T},Ns,Ds]>) -> DmTensor<bf16,Chip,C,S,m![{T},Ns,Ds]> {
    device.main.begin(x.view())
        .fetch::<m![{T},Ns,Ds/32],m![Ds%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![{T},Ns,Ds/16],m![Ds%16]>().commit_trim::<m![Ds%16]>().commit()
}

#[device(chip=1)]
pub fn zero_sliding_pool(device:&mut Device) -> HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]> {
    type C=m![1#2];
    type S=m![Ts/4];
    let mut zero:DmTensor<bf16,Chip,C,S,m![Ts%4,Ns,Ds]>=DmTensor::new();
    zero.view_mut().memset(const {bf16::from_f32(0.0)}, &mut device.sub);
    let zero=encode_head_rows::<C,S,m![Ts%4]>(device,&zero);
    let mut output=HbmTensor::new();
    for ring in 0..Rc64::SIZE {
        zero.view().to_hbm_view(&mut device.tdma,
            output.view_mut().tile::<m![Rc64],1,m![Rc64=1#{!}32,Ts,Ns,Ds]>(ring));
    }
    output
}

/// Diagnostic ring read only. Both output buffers are private to the caller.
#[device(chip=1)]
pub fn copy_sliding_pool_to_cache(device:&mut Device,
    k:&HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,
    v:&HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,
    base:&HbmTensor<i32,Chip,m![1]>,
    out_k:&mut HbmTensor<bf16,Chip,m![Ts,Ns,Ds]>,
    out_v:&mut HbmTensor<bf16,Chip,m![Ts,Ns,Ds]>) {
    type C=m![1#2]; type S=m![Ts/4];
    let k:DmTensor<f8e4m3,Chip,C,S,m![Ts%4,Ns,Ds]>=k.gather::<m![Rc64],m![Ts,Ns,Ds]>().by_byte_offsets(base).to_dm(&mut device.tdma);
    let k=decode_head_rows::<C,S,m![Ts%4]>(device,&k);
    k.view().to_hbm_view(&mut device.tdma,out_k.view_mut());
    let v:DmTensor<f8e4m3,Chip,C,S,m![Ts%4,Ns,Ds]>=v.gather::<m![Rc64],m![Ts,Ns,Ds]>().by_byte_offsets(base).to_dm(&mut device.tdma);
    let v=decode_head_rows::<C,S,m![Ts%4]>(device,&v);
    v.view().to_hbm_view(&mut device.tdma,out_v.view_mut());
}
