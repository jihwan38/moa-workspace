//! Explicit finite-BF16 symmetric A4 arithmetic, output as ordinary I32 codes.
//! s is the smallest normal power of two with 7*s >= max(amax,2^-112).
//! Integer exponent arithmetic forms s and its exact normal reciprocal; no DivF.
//! Codes use SDK FpToFxp(31) nearest/ties-away rounding and clamp[-7,7].
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{B,H}};
type Cluster=m![B/8];
type Rows=m![B%8#256];
#[device(chip=1)]
pub fn quantize_row(device:&mut Device,input:&HbmTensor<bf16,Chip,m![B,H]>,
    output:&mut HbmTensor<i32,Chip,m![B,H]>,scales:&mut HbmTensor<f32,Chip,m![B]>) {
    let x:DmTensor<bf16,Chip,Cluster,Rows,m![H]>=input.to_dm(&mut device.tdma);
    let amax:DmTensor<f32,Chip,Cluster,Rows,m![1#8]>=device.main.begin(x.view())
        .fetch::<m![H/16],m![H%16]>().fetch_cast::<f32>()
        .collect::<m![H/8],m![H%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H/4],m![H%4]>()
        .vector_intra_slice_reduce::<H,m![1],m![1#4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1#8]>()
        .vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
        .vector_final().commit_trim::<m![1#8]>().commit();
    let scale:DmTensor<f32,Chip,Cluster,Rows,m![1#8]>=device.main.begin(amax.view())
        .fetch::<m![1],m![1#8]>().collect::<m![1],m![1#8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_fxp(FxpBinaryOp::AddFxp,0x001f_ffffi32)
        .vector_fxp(FxpBinaryOp::LogicRightShift,23i32)
        .vector_fxp(FxpBinaryOp::LeftShift,23i32)
        .vector_clip(ClipBinaryOpI32::AddFxp,-0x0100_0000i32)
        .vector_reinterpret::<f32>()
        .vector_final().commit_trim::<m![1#8]>().commit();
    let reciprocal:VrfTensor<f32,Chip,Cluster,Rows,m![1#8]>=device.sub.begin(scale.view())
        .fetch::<m![1],m![1#8]>().collect::<m![1],m![1#8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitXor,-1i32)
        .vector_clip(ClipBinaryOpI32::AddFxp,0x7f00_0001i32)
        .vector_reinterpret::<f32>().vector_final().to_vrf();
    let q:DmTensor<i32,Chip,Cluster,Rows,m![H]>=device.main.begin(x.view())
        .fetch::<m![H/16],m![H%16]>().fetch_cast::<f32>()
        .collect::<m![H/8],m![H%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H/4],m![H%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&reciprocal)
        .vector_widen_concat::<m![H/8],m![H%8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::Max,-7i32)
        .vector_clip(ClipBinaryOpI32::Min,7i32)
        .vector_final().commit_trim::<m![H%8]>().commit();
    q.view().to_hbm_view(&mut device.tdma,output.view_mut());
    scale.view().to_hbm_view(&mut device.tdma,scales.view_mut());
}
