//! Copy one explicitly indexed media row, leaving every other prompt row intact.
//! Source/destination byte offsets are validated and owned by the host caller.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{B,H,P128}};
use crate::device::layout::{Cluster,Slice};

fn insert<Rows:M>(device:&mut Device,source:&HbmTensor<bf16,Chip,m![{Rows},H]>,
    source_offset:&HbmTensor<i32,Chip,m![1]>,target_offset:&HbmTensor<i32,Chip,m![1]>,
    target:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    let row:DmTensor<bf16,Chip,Cluster,Slice,m![H]>=source
        .gather::<Rows,m![H]>().by_byte_offsets(source_offset).to_dm(&mut device.tdma);
    target.view_mut().scatter::<m![P128],m![H]>()
        .by_byte_offsets(target_offset.view()).from_dm(&mut device.tdma,row);
}

#[device(chip=1)]
pub fn insert_b16_in_p128(device:&mut Device,source:&HbmTensor<bf16,Chip,m![B,H]>,
    source_offset:&HbmTensor<i32,Chip,m![1]>,target_offset:&HbmTensor<i32,Chip,m![1]>,
    target:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    insert::<m![B]>(device,source,source_offset,target_offset,target);
}

#[device(chip=1)]
pub fn insert_p128_in_p128(device:&mut Device,source:&HbmTensor<bf16,Chip,m![P128,H]>,
    source_offset:&HbmTensor<i32,Chip,m![1]>,target_offset:&HbmTensor<i32,Chip,m![1]>,
    target:&mut HbmTensor<bf16,Chip,m![P128,H]>) {
    insert::<m![P128]>(device,source,source_offset,target_offset,target);
}
