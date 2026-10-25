//! Normal typed signed-I4 consumer for a standalone checked raw-ABI probe.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{B,H}};
axes![RawM32=32];
pub type DenseInput=m![B,H];
pub type PackedWords=m![B,H/8];
pub type Weight=m![RawM32,H];
pub type Output=m![RawM32,B];
type Cluster=m![1#2];
type Slice=m![1#256];
#[device(chip=1)]
pub fn dot(device:&mut Device,input:&HbmTensor<i4,Chip,DenseInput>,
    weight:&HbmTensor<i4,Chip,Weight>,out:&mut HbmTensor<i32,Chip,Output>) {
    let input:DmTensor<i4,Chip,Cluster,Slice,DenseInput>=input.to_dm(&mut device.tdma);
    let weight:DmTensor<i4,Chip,Cluster,Slice,Weight>=weight.to_dm(&mut device.tdma);
    let trf:TrfTensor<i4,Chip,Cluster,Slice,m![B%8],m![B/8,H]>=device.sub.begin(input.view())
        .fetch::<m![B%8,B/8],m![H]>()
        .collect::<m![B%8,B/8,H/64],m![H%64]>()
        .to_trf();
    let result:DmTensor<i32,Chip,Cluster,Slice,Output>=device.main.begin(weight.view())
        .fetch::<m![RawM32,H/64],m![H%64]>()
        .collect::<m![RawM32,H/64],m![H%64]>()
        .contract_outer::<m![RawM32,H/64,B/8],m![H%64],_,_,_>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![RawM32,B/8]>()
        .contract_lane::<m![RawM32,B/8],m![B%8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_final()
        .commit_trim::<m![B%8]>().commit();
    result.view().to_hbm_view(&mut device.tdma,out.view_mut());
}

