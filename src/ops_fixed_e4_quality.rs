//! Fixed scale1 sliding-KV numerical surrogate. Pools remain BF16.
//! This source switch applies from the first token across every active producer.
use crate::{Chip, axes::Ds};
use furiosa_opt_std::prelude::*;

pub const ENABLED: bool = false;

pub(crate) fn roundtrip_ds<C:M,S:M>(device:&mut Device,
    x:&DmTensor<bf16,Chip,C,S,m![Ds]>) -> DmTensor<bf16,Chip,C,S,m![Ds]> {
    let code:DmTensor<f8e4m3,Chip,C,S,m![Ds]>=device.main.begin(x.view())
        .fetch::<m![Ds/16],m![Ds%16]>().fetch_cast::<f32>()
        .collect::<m![Ds/8],m![Ds%8]>()
        .cast::<f8e4m3,m![Ds%8#32]>().commit_trim::<m![Ds%8]>().commit();
    device.main.begin(code.view())
        .fetch::<m![Ds/32],m![Ds%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![Ds/16],m![Ds%16]>().commit_trim::<m![Ds%16]>().commit()
}
pub(crate) fn roundtrip_rows<C:M,S:M,T:M>(device:&mut Device,
    x:&DmTensor<bf16,Chip,C,S,m![{T},Ds]>) -> DmTensor<bf16,Chip,C,S,m![{T},Ds]> {
    let code:DmTensor<f8e4m3,Chip,C,S,m![{T},Ds]>=device.main.begin(x.view())
        .fetch::<m![{T},Ds/16],m![Ds%16]>().fetch_cast::<f32>()
        .collect::<m![{T},Ds/8],m![Ds%8]>()
        .cast::<f8e4m3,m![Ds%8#32]>().commit_trim::<m![Ds%8]>().commit();
    device.main.begin(code.view())
        .fetch::<m![{T},Ds/32],m![Ds%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![{T},Ds/16],m![Ds%16]>().commit_trim::<m![Ds%16]>().commit()
}
pub(crate) fn maybe_ds<C:M,S:M>(device:&mut Device,
    x:DmTensor<bf16,Chip,C,S,m![Ds]>) -> DmTensor<bf16,Chip,C,S,m![Ds]> {
    if ENABLED {roundtrip_ds(device,&x)} else {x}
}
pub(crate) fn maybe_rows<C:M,S:M,T:M>(device:&mut Device,
    x:DmTensor<bf16,Chip,C,S,m![{T},Ds]>) -> DmTensor<bf16,Chip,C,S,m![{T},Ds]> {
    if ENABLED {roundtrip_rows::<C,S,T>(device,&x)} else {x}
}
