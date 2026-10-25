//! Exact three-to-one grouped FFN composition, retaining every HBM boundary.
//! Original and validated global-preload cores share unchanged norm stages.
use crate::{Chip, axes::{H,L,P128,P512}};
use crate::ops_batch_layouts::{P512H960,P512L240,P512L960};
use crate::ops_native_fp8_blocked::{UpWeight,DownWeight};
use crate::ops_norm_blocked_residual::Residual;
use furiosa_opt_std::prelude::*;

#[device(chip=1)]
pub fn sliding(
    device:&mut Device,
    projected:(&HbmTensor<bf16,Chip,crate::ops_output_p128_h64_blocked::Output>,&HbmTensor<bf16,Chip,crate::ops_output_p128_h64_blocked::Output>,&HbmTensor<bf16,Chip,crate::ops_output_p128_h64_blocked::Output>,&HbmTensor<bf16,Chip,crate::ops_output_p128_h64_blocked::Output>),
    norms:(&HbmTensor<bf16,Chip,m![H]>,&HbmTensor<bf16,Chip,m![H]>,
        &HbmTensor<bf16,Chip,m![H]>,&HbmTensor<bf16,Chip,m![1#8]>),
    residual:&mut HbmTensor<bf16,Chip,Residual>,
    up_weight:&HbmTensor<f8e4m3,Chip,UpWeight>,
    gate_weight:&HbmTensor<f8e4m3,Chip,UpWeight>,
    down_weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
    us:&HbmTensor<f32,Chip,m![L]>,gs:&HbmTensor<f32,Chip,m![L]>,ds:&HbmTensor<f32,Chip,m![H]>,
    ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
    qinput:&mut HbmTensor<f8e4m3,Chip,P512H960>,
    xscale:&mut HbmTensor<f32,Chip,m![P512]>,
    up_output:&mut HbmTensor<bf16,Chip,P512L240>,
    gate_output:&mut HbmTensor<bf16,Chip,P512L240>,
    qhidden:&mut HbmTensor<f8e4m3,Chip,P512L960>,
    hscale:&mut HbmTensor<f32,Chip,m![P512]>,
    output:&mut HbmTensor<bf16,Chip,crate::ops_native_fp8_p512_downblocked::Output>,
) {
    let (a,b,c,d)=projected;
    let (post_attention,pre_ffn,post_ffn,layer_gate)=norms;
    crate::ops_norm_blocked_chain::post_attention_sliding_stage(
        device,a,b,c,d,post_attention,pre_ffn,residual,qinput,xscale);
    crate::ops_native_fp8_p512_core_fused::ffn_core_stage(
        device,qinput,xscale,up_weight,gate_weight,down_weight,us,gs,ds,ug,gg,dg,
        up_output,gate_output,qhidden,hscale,output);
    crate::ops_norm_blocked_residual::post_feedforward_stage(
        device,output,post_ffn,layer_gate,residual);
}

#[device(chip=1)]
pub fn full(
    device:&mut Device,
    projected:(&HbmTensor<bf16,Chip,m![P128,H]>,&HbmTensor<bf16,Chip,m![P128,H]>,&HbmTensor<bf16,Chip,m![P128,H]>,&HbmTensor<bf16,Chip,m![P128,H]>),
    norms:(&HbmTensor<bf16,Chip,m![H]>,&HbmTensor<bf16,Chip,m![H]>,
        &HbmTensor<bf16,Chip,m![H]>,&HbmTensor<bf16,Chip,m![1#8]>),
    residual:&mut HbmTensor<bf16,Chip,Residual>,
    up_weight:&HbmTensor<f8e4m3,Chip,UpWeight>,
    gate_weight:&HbmTensor<f8e4m3,Chip,UpWeight>,
    down_weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
    us:&HbmTensor<f32,Chip,m![L]>,gs:&HbmTensor<f32,Chip,m![L]>,ds:&HbmTensor<f32,Chip,m![H]>,
    ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
    qinput:&mut HbmTensor<f8e4m3,Chip,P512H960>,
    xscale:&mut HbmTensor<f32,Chip,m![P512]>,
    up_output:&mut HbmTensor<bf16,Chip,P512L240>,
    gate_output:&mut HbmTensor<bf16,Chip,P512L240>,
    qhidden:&mut HbmTensor<f8e4m3,Chip,P512L960>,
    hscale:&mut HbmTensor<f32,Chip,m![P512]>,
    output:&mut HbmTensor<bf16,Chip,crate::ops_native_fp8_p512_downblocked::Output>,
) {
    let (a,b,c,d)=projected;
    let (post_attention,pre_ffn,post_ffn,layer_gate)=norms;
    crate::ops_norm_blocked_chain::post_attention_full_stage(
        device,a,b,c,d,post_attention,pre_ffn,residual,qinput,xscale);
    crate::ops_native_fp8_p512_core_fused::ffn_core_stage(
        device,qinput,xscale,up_weight,gate_weight,down_weight,us,gs,ds,ug,gg,dg,
        up_output,gate_output,qhidden,hscale,output);
    crate::ops_norm_blocked_residual::post_feedforward_stage(
        device,output,post_ffn,layer_gate,residual);
}


/// Same composition with separately validated global-DM preload.
#[device(chip=1)]
pub fn sliding_globals(
    device:&mut Device,
    projected:(&HbmTensor<bf16,Chip,crate::ops_output_p128_h64_blocked::Output>,&HbmTensor<bf16,Chip,crate::ops_output_p128_h64_blocked::Output>,&HbmTensor<bf16,Chip,crate::ops_output_p128_h64_blocked::Output>,&HbmTensor<bf16,Chip,crate::ops_output_p128_h64_blocked::Output>),
    norms:(&HbmTensor<bf16,Chip,m![H]>,&HbmTensor<bf16,Chip,m![H]>,
        &HbmTensor<bf16,Chip,m![H]>,&HbmTensor<bf16,Chip,m![1#8]>),
    residual:&mut HbmTensor<bf16,Chip,Residual>,
    up_weight:&HbmTensor<f8e4m3,Chip,UpWeight>,
    gate_weight:&HbmTensor<f8e4m3,Chip,UpWeight>,
    down_weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
    us:&HbmTensor<f32,Chip,m![L]>,gs:&HbmTensor<f32,Chip,m![L]>,ds:&HbmTensor<f32,Chip,m![H]>,
    ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
    qinput:&mut HbmTensor<f8e4m3,Chip,P512H960>,
    xscale:&mut HbmTensor<f32,Chip,m![P512]>,
    up_output:&mut HbmTensor<bf16,Chip,P512L240>,
    gate_output:&mut HbmTensor<bf16,Chip,P512L240>,
    qhidden:&mut HbmTensor<f8e4m3,Chip,P512L960>,
    hscale:&mut HbmTensor<f32,Chip,m![P512]>,
    output:&mut HbmTensor<bf16,Chip,crate::ops_native_fp8_p512_downblocked::Output>,
) {
    let (a,b,c,d)=projected;
    let (post_attention,pre_ffn,post_ffn,layer_gate)=norms;
    crate::ops_norm_blocked_chain::post_attention_sliding_stage(
        device,a,b,c,d,post_attention,pre_ffn,residual,qinput,xscale);
    crate::ops_native_fp8_p512_core_globals::ffn_core_stage(
        device,qinput,xscale,up_weight,gate_weight,down_weight,us,gs,ds,ug,gg,dg,
        up_output,gate_output,qhidden,hscale,output);
    crate::ops_norm_blocked_residual::post_feedforward_stage(
        device,output,post_ffn,layer_gate,residual);
}


/// Same composition with separately validated global-DM preload.
#[device(chip=1)]
pub fn full_globals(
    device:&mut Device,
    projected:(&HbmTensor<bf16,Chip,m![P128,H]>,&HbmTensor<bf16,Chip,m![P128,H]>,&HbmTensor<bf16,Chip,m![P128,H]>,&HbmTensor<bf16,Chip,m![P128,H]>),
    norms:(&HbmTensor<bf16,Chip,m![H]>,&HbmTensor<bf16,Chip,m![H]>,
        &HbmTensor<bf16,Chip,m![H]>,&HbmTensor<bf16,Chip,m![1#8]>),
    residual:&mut HbmTensor<bf16,Chip,Residual>,
    up_weight:&HbmTensor<f8e4m3,Chip,UpWeight>,
    gate_weight:&HbmTensor<f8e4m3,Chip,UpWeight>,
    down_weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
    us:&HbmTensor<f32,Chip,m![L]>,gs:&HbmTensor<f32,Chip,m![L]>,ds:&HbmTensor<f32,Chip,m![H]>,
    ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
    qinput:&mut HbmTensor<f8e4m3,Chip,P512H960>,
    xscale:&mut HbmTensor<f32,Chip,m![P512]>,
    up_output:&mut HbmTensor<bf16,Chip,P512L240>,
    gate_output:&mut HbmTensor<bf16,Chip,P512L240>,
    qhidden:&mut HbmTensor<f8e4m3,Chip,P512L960>,
    hscale:&mut HbmTensor<f32,Chip,m![P512]>,
    output:&mut HbmTensor<bf16,Chip,crate::ops_native_fp8_p512_downblocked::Output>,
) {
    let (a,b,c,d)=projected;
    let (post_attention,pre_ffn,post_ffn,layer_gate)=norms;
    crate::ops_norm_blocked_chain::post_attention_full_stage(
        device,a,b,c,d,post_attention,pre_ffn,residual,qinput,xscale);
    crate::ops_native_fp8_p512_core_globals::ffn_core_stage(
        device,qinput,xscale,up_weight,gate_weight,down_weight,us,gs,ds,ug,gg,dg,
        up_output,gate_output,qhidden,hscale,output);
    crate::ops_norm_blocked_residual::post_feedforward_stage(
        device,output,post_ffn,layer_gate,residual);
}
