//! Same current norm stages and one launch, using validated projection-owned UP/GATE.
use crate::{Chip,axes::{H,P128,P512}};
use crate::ops_batch_layouts::{P512H960,P512L960};
use crate::ops_native_fp8_blocked::DownWeight;
use crate::ops_ffn_projection_clusters::{JoinedWeight,JoinedScale,JoinedOutput};
use crate::ops_norm_blocked_residual::Residual;
use furiosa_opt_std::prelude::*;
/// Same composition with separately validated global-DM preload.
#[device(chip=1)]
pub fn sliding_projection_owned(
    device:&mut Device,
    projected:(&HbmTensor<bf16,Chip,crate::ops_output_p128_h64_blocked::Output>,&HbmTensor<bf16,Chip,crate::ops_output_p128_h64_blocked::Output>,&HbmTensor<bf16,Chip,crate::ops_output_p128_h64_blocked::Output>,&HbmTensor<bf16,Chip,crate::ops_output_p128_h64_blocked::Output>),
    norms:(&HbmTensor<bf16,Chip,m![H]>,&HbmTensor<bf16,Chip,m![H]>,
        &HbmTensor<bf16,Chip,m![H]>,&HbmTensor<bf16,Chip,m![1#8]>),
    residual:&mut HbmTensor<bf16,Chip,Residual>,
    joined_weight:&HbmTensor<f8e4m3,Chip,JoinedWeight>,
    joined_scale:&HbmTensor<f32,Chip,JoinedScale>,
    down_weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
    ds:&HbmTensor<f32,Chip,m![H]>,
    ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
    qinput:&mut HbmTensor<f8e4m3,Chip,P512H960>,
    xscale:&mut HbmTensor<f32,Chip,m![P512]>,
    joined_output:&mut HbmTensor<bf16,Chip,JoinedOutput>,
    qhidden:&mut HbmTensor<f8e4m3,Chip,P512L960>,
    hscale:&mut HbmTensor<f32,Chip,m![P512]>,
    output:&mut HbmTensor<bf16,Chip,crate::ops_native_fp8_p512_downblocked::Output>,
) {
    let (a,b,c,d)=projected;
    let (post_attention,pre_ffn,post_ffn,layer_gate)=norms;
    crate::ops_norm_blocked_chain_storegroup::post_attention_sliding_stage(
        device,a,b,c,d,post_attention,pre_ffn,residual,qinput,xscale);
    crate::ops_ffn_projection_clusters::ffn_core_stage(
        device,qinput,xscale,joined_weight,joined_scale,down_weight,ds,ug,gg,dg,
        joined_output,qhidden,hscale,output);
    crate::ops_norm_blocked_residual::post_feedforward_stage(
        device,output,post_ffn,layer_gate,residual);
}


/// Same composition with separately validated global-DM preload.
#[device(chip=1)]
pub fn full_projection_owned(
    device:&mut Device,
    projected:(&HbmTensor<bf16,Chip,m![P128,H]>,&HbmTensor<bf16,Chip,m![P128,H]>,&HbmTensor<bf16,Chip,m![P128,H]>,&HbmTensor<bf16,Chip,m![P128,H]>),
    norms:(&HbmTensor<bf16,Chip,m![H]>,&HbmTensor<bf16,Chip,m![H]>,
        &HbmTensor<bf16,Chip,m![H]>,&HbmTensor<bf16,Chip,m![1#8]>),
    residual:&mut HbmTensor<bf16,Chip,Residual>,
    joined_weight:&HbmTensor<f8e4m3,Chip,JoinedWeight>,
    joined_scale:&HbmTensor<f32,Chip,JoinedScale>,
    down_weight:&HbmTensor<f8e4m3,Chip,DownWeight>,
    ds:&HbmTensor<f32,Chip,m![H]>,
    ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,dg:&HbmTensor<f32,Chip,m![1]>,
    qinput:&mut HbmTensor<f8e4m3,Chip,P512H960>,
    xscale:&mut HbmTensor<f32,Chip,m![P512]>,
    joined_output:&mut HbmTensor<bf16,Chip,JoinedOutput>,
    qhidden:&mut HbmTensor<f8e4m3,Chip,P512L960>,
    hscale:&mut HbmTensor<f32,Chip,m![P512]>,
    output:&mut HbmTensor<bf16,Chip,crate::ops_native_fp8_p512_downblocked::Output>,
) {
    let (a,b,c,d)=projected;
    let (post_attention,pre_ffn,post_ffn,layer_gate)=norms;
    crate::ops_norm_blocked_chain_storegroup::post_attention_full_stage(
        device,a,b,c,d,post_attention,pre_ffn,residual,qinput,xscale);
    crate::ops_ffn_projection_clusters::ffn_core_stage(
        device,qinput,xscale,joined_weight,joined_scale,down_weight,ds,ug,gg,dg,
        joined_output,qhidden,hscale,output);
    crate::ops_norm_blocked_residual::post_feedforward_stage(
        device,output,post_ffn,layer_gate,residual);
}
