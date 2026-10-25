//! Integer-domain signed4-bit packing. No activation quantization is performed.
//! Eight masked nibble fields have disjoint bits; every partial signed sum fits
//! I32, so the SDK's saturating-add tree is identical to bitwise OR here.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::B};
axes![PackWords=480,PackNibbles=8];
pub type Codes=m![B,PackWords,PackNibbles];
pub type PackedWords=m![B,PackWords];
type Cluster=m![B/8];
type Rows=m![B%8,PackWords/16#32];
#[device(chip=1)]
pub fn pack_i32_codes(device:&mut Device,input:&HbmTensor<i32,Chip,Codes>,
    shifts:&HbmTensor<i32,Chip,m![PackNibbles]>,output:&mut HbmTensor<i32,Chip,PackedWords>) {
    let input:DmTensor<i32,Chip,Cluster,Rows,m![PackWords%16,PackNibbles]>=input.to_dm(&mut device.tdma);
    let shifts:DmTensor<i32,Chip,Cluster,Rows,m![PackNibbles]>=shifts.to_dm(&mut device.tdma);
    let shifts:VrfTensor<i32,Chip,Cluster,Rows,m![PackNibbles]>=device.sub.begin(shifts.view())
        .fetch::<m![1],m![PackNibbles]>().collect::<m![1],m![PackNibbles]>().to_vrf();
    let packed:DmTensor<i32,Chip,Cluster,Rows,m![PackWords%16]>=device.main.begin(input.view())
        .fetch::<m![PackWords%16],m![PackNibbles]>()
        .collect::<m![PackWords%16],m![PackNibbles]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd,15i32)
        .vector_logic(LogicBinaryOpI32::LeftShift,&shifts)
        .vector_narrow_split::<m![PackWords%16,PackNibbles/4],m![PackNibbles%4]>()
        .vector_intra_slice_reduce::<PackNibbles,m![PackWords%16],m![1#4]>(IntraSliceReduceOpI32::AddSat)
        .vector_widen_pad::<m![1#8]>().vector_final()
        .transpose::<m![PackWords/2%8],m![PackWords%2#8]>()
        .commit_trim::<m![PackWords%2]>().commit();
    packed.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
