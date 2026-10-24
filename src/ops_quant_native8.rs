//! Per-token dynamic FP8 activation quantization for isolated native-FP4 experiments.
//! Scale is a power of two: 2^floor(log2(max(abs(x),2^-112))) /128.
//! Finite BF16 rows map into [-256,256], without saturation; zero rows stay zero.
//! RNGD vector arithmetic flushes F32 subnormal inputs to signed zero. The
//! quantized activation contract includes this behavior for |x| < 2^-126.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{P128,H,L}};
type Cluster=m![P128 / 64];
type Slice=m![P128 % 64 # 256];
macro_rules! quantizer {
    ($name:ident,$axis:ident) => {
        #[device(chip = 1)]
        pub fn $name(device:&mut Device,input:&HbmTensor<bf16,Chip,m![P128,$axis]>,
            output:&mut HbmTensor<f8e4m3,Chip,m![P128,$axis]>,
            scales:&mut HbmTensor<f32,Chip,m![P128]>) {
            let x:DmTensor<bf16,Chip,Cluster,Slice,m![$axis]>=input.to_dm(&mut device.tdma);
            let amax:DmTensor<f32,Chip,Cluster,Slice,m![1 # 8]>=device.main.begin(x.view())
                .fetch::<m![$axis /16],m![$axis %16]>().fetch_cast::<f32>()
                .collect::<m![$axis /8],m![$axis %8]>()
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7fff_ffffi32)
                .vector_reinterpret::<f32>()
                .vector_narrow_split::<m![$axis /4],m![$axis %4]>()
                .vector_intra_slice_reduce::<$axis,m![1],m![1 #4]>(IntraSliceReduceOpF32::Max)
                .vector_widen_pad::<m![1 #8]>()
                .vector_clip(ClipBinaryOpF32::Max,1.925929944387236e-34f32)
                .vector_final().commit_trim::<m![1 #8]>().commit();
            let scale:DmTensor<f32,Chip,Cluster,Slice,m![1 #8]>=device.main.begin(amax.view())
                .fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>()
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd,0x7f80_0000i32)
                .vector_reinterpret::<f32>().vector_narrow_trim::<m![1 #4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),0.0078125f32)
                .vector_widen_pad::<m![1 #8]>().vector_final()
                .commit_trim::<m![1 #8]>().commit();
            let sv:VrfTensor<f32,Chip,Cluster,Slice,m![1 #8]>=device.sub.begin(scale.view())
                .fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
            let q:DmTensor<f8e4m3,Chip,Cluster,Slice,m![$axis]>=device.main.begin(x.view())
                .fetch::<m![$axis /16],m![$axis %16]>().fetch_cast::<f32>()
                .collect::<m![$axis /8],m![$axis %8]>()
                .vector_init().vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$axis /4],m![$axis %4]>()
                .vector_fp_binary(FpBinaryOp::DivF,&sv)
                .vector_widen_concat::<m![$axis /8],m![$axis %8]>().vector_final()
                .cast::<f8e4m3,m![$axis %8 #32]>().commit_trim::<m![$axis %8]>().commit();
            q.view().to_hbm_view(&mut device.tdma,output.view_mut());
            scale.view().to_hbm_view(&mut device.tdma,scales.view_mut());
        }
    }
}
quantizer!(quantize_hidden,H);
quantizer!(quantize_ffn,L);
