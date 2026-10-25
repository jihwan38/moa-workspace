//! Exact four-call B16 prefix composition with every HBM stage retained.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{B,H,Qs,Ps,Qf,Pf,Ns,Gs,Ds,Gf,Df,E},
    ops_qkv_blocked_b16 as qkv,ops_head_b16_blocked as heads};

#[device(chip=1)]
pub fn sliding(
    device:&mut Device,
    input:&HbmTensor<bf16,Chip,m![B,H]>,
    input_norm:&HbmTensor<bf16,Chip,m![H]>,
    qw:&HbmTensor<f8e4m3,Chip,m![Qs,H]>,
    kw:&HbmTensor<f8e4m3,Chip,m![Ps,H]>,
    vw:&HbmTensor<f8e4m3,Chip,m![Ps,H]>,
    qs:&HbmTensor<bf16,Chip,m![Qs]>,
    ks:&HbmTensor<bf16,Chip,m![Ps]>,
    vs:&HbmTensor<bf16,Chip,m![Ps]>,
    qn:&HbmTensor<bf16,Chip,m![Ds]>,
    kn:&HbmTensor<bf16,Chip,m![Ds]>,
    rope:&HbmTensor<i32,Chip,m![B]>,
    cos:&HbmTensor<bf16,Chip,m![E,Ds]>,
    sin:&HbmTensor<bf16,Chip,m![E,Ds]>,
    normalized:&mut HbmTensor<bf16,Chip,m![B,H]>,
    quantized:&mut HbmTensor<f8e4m3,Chip,m![B,H]>,
    xscale:&mut HbmTensor<f32,Chip,m![B,1#8]>,
    qr:&mut HbmTensor<bf16,Chip,qkv::QueryOutput>,
    kr:&mut HbmTensor<bf16,Chip,qkv::KeyOutput>,
    vr:&mut HbmTensor<bf16,Chip,qkv::KeyOutput>,
    q:&mut HbmTensor<bf16,Chip,m![B,Ns,Gs,Ds]>,
    k:&mut HbmTensor<bf16,Chip,m![B,Ns,Ds]>,
    v:&mut HbmTensor<bf16,Chip,m![B,Ns,Ds]>,
) {
    crate::ops_norm_batch::rmsnorm_b16_stage(device,input,input_norm,normalized);
    crate::ops_query_b16_native_fp8::quantize_input_stage(device,normalized,quantized,xscale);
    qkv::project_sliding_qkv_stage(device,quantized,xscale,qw,kw,vw,qs,ks,vs,qr,kr,vr);
    heads::sliding_heads_b16_stage(device,qr,kr,vr,qn,kn,rope,cos,sin,q,k,v);
}

#[device(chip=1)]
pub fn full(
    device:&mut Device,
    input:&HbmTensor<bf16,Chip,m![B,H]>,
    input_norm:&HbmTensor<bf16,Chip,m![H]>,
    qw:&HbmTensor<f8e4m3,Chip,m![Qf,H]>,
    kw:&HbmTensor<f8e4m3,Chip,m![Pf,H]>,
    qs:&HbmTensor<bf16,Chip,m![Qf]>,
    ks:&HbmTensor<bf16,Chip,m![Pf]>,
    qn:&HbmTensor<bf16,Chip,m![Df]>,
    kn:&HbmTensor<bf16,Chip,m![Df]>,
    rope:&HbmTensor<i32,Chip,m![B]>,
    cos:&HbmTensor<bf16,Chip,m![E,Df]>,
    sin:&HbmTensor<bf16,Chip,m![E,Df]>,
    normalized:&mut HbmTensor<bf16,Chip,m![B,H]>,
    quantized:&mut HbmTensor<f8e4m3,Chip,m![B,H]>,
    xscale:&mut HbmTensor<f32,Chip,m![B,1#8]>,
    qr:&mut HbmTensor<bf16,Chip,qkv::FullQueryOutput>,
    kr:&mut HbmTensor<bf16,Chip,qkv::FullKeyOutput>,
    q:&mut HbmTensor<bf16,Chip,m![B,Gf,Df]>,
    k:&mut HbmTensor<bf16,Chip,m![B,Df]>,
    v:&mut HbmTensor<bf16,Chip,m![B,Df]>,
) {
    crate::ops_norm_batch::rmsnorm_b16_stage(device,input,input_norm,normalized);
    crate::ops_query_b16_native_fp8::quantize_input_stage(device,normalized,quantized,xscale);
    qkv::project_full_qk_stage(device,quantized,xscale,qw,kw,qs,ks,qr,kr);
    heads::full_heads_b16_stage(device,qr,kr,qn,kn,rope,cos,sin,q,k,v);
}
