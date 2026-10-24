//! Generic greedy selection; nonfinite or zero/subnormal maxima request CPU fallback.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{R64,W},ops_r64_io_blocked::BlockedLogits};
axes![GreedyField=8];
pub type GreedyResult=m![R64,GreedyField];
pub type BanWords=m![W/32,R64];
type Cluster=m![R64/32];
type Rows=m![W/32%256];
type Reduced=m![1#256];
type Keys=DmTensor<i32,Chip,Cluster,Rows,m![R64%32=16,W/8192,W%32]>;
type Stats=DmTensor<i32,Chip,Cluster,Reduced,m![R64%32=16,1#8]>;
fn max_per_row(device:&mut Device,input:&Keys)->Stats {
    let partial:DmTensor<i32,Chip,Cluster,Rows,m![R64%32=16,1#8]>=device.main.begin(input.view())
        .fetch::<m![R64%32=16,W/8192,W/8%4],m![W%8]>()
        .collect::<m![R64%32=16,W/8192,W/8%4],m![W%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R64%32=16,W/8192,W/4%8],m![W%4]>()
        .vector_intra_slice_reduce::<W,m![R64%32=16],m![1#4]>(IntraSliceReduceOpI32::Max)
        .vector_widen_pad::<m![1#8]>()
        .vector_final().commit_trim::<m![1#8]>().commit();
    device.main.begin(partial.view()).fetch::<m![R64%32=16],m![1#8]>()
        .collect::<m![R64%32=16],m![1#8]>().vector_init()
        .vector_inter_slice_reduce::<Reduced,m![R64%32=16]>(InterSliceReduceOpI32::Max)
        .vector_final().commit_trim::<m![1#8]>().commit()
}
fn winning_ids(device:&mut Device,keys:&Keys,max:&Stats,negative_ids:&DmTensor<i32,Chip,Cluster,Rows,m![W/8192,W%32]>)->Stats {
    let max:DmTensor<i32,Chip,Cluster,Rows,m![R64%32=16,1#8]>=max.to_dm(&mut device.tdma);
    let mut candidates:Keys=DmTensor::new();
    for tile in 0..16 {
        let max:VrfTensor<i32,Chip,Cluster,Rows,m![R64%32=1,1#8]>=device.sub
            .begin(max.view().tile::<m![R64%32=16],1,m![R64%32=1#16,1#8]>(tile))
            .fetch::<m![R64%32=1],m![1#8]>().collect::<m![R64%32=1],m![1#8]>().to_vrf();
        // Difference is in[-98303,98303], so AbsMin never sees i32::MIN.
        let mismatch:DmTensor<i32,Chip,Cluster,Rows,m![R64%32=1,W/8192,W%32]>=device.main
            .begin(keys.view().tile::<m![R64%32=16],1,m![R64%32=1#16,W/8192,W%32]>(tile))
            .fetch::<m![R64%32=1,W/8192,W/8%4],m![W%8]>()
            .collect::<m![R64%32=1,W/8192,W/8%4],m![W%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp,&max)
            .vector_clip(ClipBinaryOpI32::AbsMin,1i32)
            .vector_final().commit_trim::<m![W%8]>().commit();
        let index:VrfTensor<i32,Chip,Cluster,Rows,m![W/8192,W%32]>=device.sub.begin(negative_ids.view())
            .fetch::<m![W/8192,W/8%4],m![W%8]>().collect::<m![W/8192,W/8%4],m![W%8]>().to_vrf();
        device.main.begin(mismatch.view())
            .fetch::<m![R64%32=1,W/8192,W/8%4],m![W%8]>()
            .collect::<m![R64%32=1,W/8192,W/8%4],m![W%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpI32::LeftShift,31i32)
            .vector_clip(ClipBinaryOpI32::AddFxpSat,&index)
            .vector_final().commit_trim::<m![W%8]>()
            .commit_view(candidates.view_mut().tile::<m![R64%32=16],1,m![R64%32=1#{!}16,W/8192,W%32]>(tile));
    }
    max_per_row(device,&candidates)
}
fn write_stat(device:&mut Device,value:&Stats,out:&mut HbmTensor<i32,Chip,GreedyResult>,field:usize,group:usize){
    value.view().to_hbm_view(&mut device.tdma,out.view_mut()
        .tile::<m![R64%32],16,m![R64/32,R64%32=16#{!}32,GreedyField]>(group*16)
        .tile::<m![GreedyField],1,m![R64/32,R64%32=16#{!}32,GreedyField=1#{!}8]>(field));
}
fn greedy_group(
    device:&mut Device,
    logits:&HbmTensor<bf16,Chip,BlockedLogits>,
    ban_words:&HbmTensor<i32,Chip,BanWords>,
    negative_token_ids:&HbmTensor<i32,Chip,m![W]>,
    bit_positions:&HbmTensor<i32,Chip,m![W%32]>,
    out:&mut HbmTensor<i32,Chip,GreedyResult>,
    group:usize,
){
    let values:DmTensor<bf16,Chip,Cluster,Rows,m![R64%32=16,W/8192,W%32]>=logits.view().tile::<m![R64%32],16,m![W/8192,W/32%256,R64/32,R64%32=16#32,W%32]>(group*16).to_dm(&mut device.tdma);
    let raw:Keys=device.main.begin(values.view())
        .fetch::<m![R64%32=16,W/8192,W/16%2],m![W%16]>().fetch_cast::<f32>()
        .collect::<m![R64%32=16,W/8192,W/8%4],m![W%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::ArithRightShift,16i32)
        .vector_final().commit_trim::<m![W%8]>().commit();
    let absolute:Keys=device.main.begin(raw.view()).fetch::<m![R64%32=16,W/8192,W/8%4],m![W%8]>()
        .collect::<m![R64%32=16,W/8192,W/8%4],m![W%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_logic(LogicBinaryOpI32::BitAnd,0x7fffi32)
        .vector_final().commit_trim::<m![W%8]>().commit();
    let abs_max=max_per_row(device,&absolute);write_stat(device,&abs_max,out,3,group);
    let keys:Keys=device.main.begin(raw.view()).fetch::<m![R64%32=16,W/8192,W/8%4],m![W%8]>()
        .collect::<m![R64%32=16,W/8192,W/8%4],m![W%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_stash()
        .vector_logic(LogicBinaryOpI32::ArithRightShift,15i32)
        .vector_logic(LogicBinaryOpI32::BitAnd,0x7fffi32)
        .vector_logic(LogicBinaryOpI32::BitXor,Stash)
        .vector_final().commit_trim::<m![W%8]>().commit();
    let unmasked_max=max_per_row(device,&keys);write_stat(device,&unmasked_max,out,4,group);
    let negative_ids:DmTensor<i32,Chip,Cluster,Rows,m![W/8192,W%32]>=negative_token_ids.to_dm(&mut device.tdma);
    let unmasked_ids=winning_ids(device,&keys,&unmasked_max,&negative_ids);write_stat(device,&unmasked_ids,out,2,group);
    let words:DmTensor<i32,Chip,Cluster,Rows,m![W/8192,R64%32=16]>=ban_words.view().tile::<m![R64%32],16,m![W/32,R64/32,R64%32=16#32]>(group*16).to_dm(&mut device.tdma);
    let words:DmTensor<i32,Chip,Cluster,Rows,m![R64%32=16,W/8192,1#8]>=device.main.begin(words.view())
        .fetch::<m![W/8192,R64/8%4=2],m![R64%8]>().collect::<m![W/8192,R64/8%4=2],m![R64%8]>()
        .transpose::<m![W/8192,R64%32=16],m![1#8]>().commit_trim::<m![1#8]>().commit();
    let mut masked:Keys=DmTensor::new();
    let shifts:DmTensor<i32,Chip,Cluster,Rows,m![W%32]>=bit_positions.to_dm(&mut device.tdma);

    for tile in 0..16 {
        let word:VrfTensor<i32,Chip,Cluster,Rows,m![R64%32=1,W/8192,1#8]>=device.sub
            .begin(words.view().tile::<m![R64%32=16],1,m![R64%32=1#16,W/8192,1#8]>(tile))
            .fetch::<m![R64%32=1,W/8192],m![1#8]>().collect::<m![R64%32=1,W/8192],m![1#8]>().to_vrf();
        let shift:VrfTensor<i32,Chip,Cluster,Rows,m![W%32]>=device.sub.begin(shifts.view())
            .fetch::<m![W/8%4],m![W%8]>().collect::<m![W/8%4],m![W%8]>().to_vrf();
        let mask:DmTensor<i32,Chip,Cluster,Rows,m![R64%32=1,W/8192,W%32]>=device.main.begin(keys.view().tile::<m![R64%32=16],1,m![R64%32=1#16,W/8192,W%32]>(tile))
            .fetch::<m![R64%32=1,W/8192,W/8%4],m![W%8]>()
            .collect::<m![R64%32=1,W/8192,W/8%4],m![W%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_logic_with_mode(LogicBinaryOpI32::BitOr,BinaryArgMode::Mode11,&word)
            .vector_logic(LogicBinaryOpI32::LogicRightShift,&shift)
            .vector_logic(LogicBinaryOpI32::BitAnd,1i32)
            .vector_fxp(FxpBinaryOp::MulInt,-131072i32)
            .vector_clip(ClipBinaryOpI32::AddFxp,65536i32)
            .vector_final().commit_trim::<m![W%8]>().commit();
        let mask:VrfTensor<i32,Chip,Cluster,Rows,m![R64%32=1,W/8192,W%32]>=device.sub.begin(mask.view())
            .fetch::<m![R64%32=1,W/8192,W/8%4],m![W%8]>()
            .collect::<m![R64%32=1,W/8192,W/8%4],m![W%8]>().to_vrf();
        device.main.begin(keys.view().tile::<m![R64%32=16],1,m![R64%32=1#16,W/8192,W%32]>(tile))
            .fetch::<m![R64%32=1,W/8192,W/8%4],m![W%8]>()
            .collect::<m![R64%32=1,W/8192,W/8%4],m![W%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_clip(ClipBinaryOpI32::Min,&mask)
            .vector_final().commit_trim::<m![W%8]>()
            .commit_view(masked.view_mut().tile::<m![R64%32=16],1,m![R64%32=1#{!}16,W/8192,W%32]>(tile));
    }
    let masked_max=max_per_row(device,&masked);write_stat(device,&masked_max,out,0,group);
    let masked_ids=winning_ids(device,&masked,&masked_max,&negative_ids);write_stat(device,&masked_ids,out,1,group);
}

#[device(chip=1)]
pub fn greedy_argmax_r64(
    device:&mut Device,
    logits:&HbmTensor<bf16,Chip,BlockedLogits>,
    ban_words:&HbmTensor<i32,Chip,BanWords>,
    negative_token_ids:&HbmTensor<i32,Chip,m![W]>,
    bit_positions:&HbmTensor<i32,Chip,m![W%32]>,
    out:&mut HbmTensor<i32,Chip,GreedyResult>,
){
    for group in 0..2 {greedy_group(device,logits,ban_words,negative_token_ids,bit_positions,out,group);}
}
