//! Approximate per960 partial-scaled F4 resource epilogue; no host caller.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{B,H,L}, ops_b16_f4_partialscale_dense as ffn};

#[device(chip=1)]
pub fn post_attention_ffn(
    device:&mut Device,
    projected:&mut HbmTensor<bf16,Chip,m![B,H]>,
    residual:&mut HbmTensor<bf16,Chip,m![B,H]>,
    post_attention:&HbmTensor<bf16,Chip,m![H]>,
    pre_ffn:&HbmTensor<bf16,Chip,m![H]>,
    post_ffn:&HbmTensor<bf16,Chip,m![H]>,
    layer_gate:&HbmTensor<bf16,Chip,m![1#8]>,
    norm:&mut HbmTensor<bf16,Chip,m![B,H]>,
    up:&HbmTensor<f8e4m3,Chip,ffn::UpWeight>,
    gate:&HbmTensor<f8e4m3,Chip,ffn::UpWeight>,
    down:&HbmTensor<f8e4m3,Chip,ffn::DownWeight>,
    block_scales:(&HbmTensor<f32,Chip,ffn::UpBlockScale>,
                  &HbmTensor<f32,Chip,ffn::UpBlockScale>,
                  &HbmTensor<f32,Chip,ffn::DownBlockScale>),
    us:&HbmTensor<f32,Chip,m![L]>,gs:&HbmTensor<f32,Chip,m![L]>,
    ds:&HbmTensor<f32,Chip,m![H]>,
    ug:&HbmTensor<f32,Chip,m![1]>,gg:&HbmTensor<f32,Chip,m![1]>,
    dg:&HbmTensor<f32,Chip,m![1]>,
    qinput:&mut HbmTensor<f8e4m3,Chip,ffn::Input>,
    xscale:&mut HbmTensor<f32,Chip,m![B]>,
    hidden:&mut HbmTensor<bf16,Chip,ffn::Hidden>,
    qhidden:&mut HbmTensor<f8e4m3,Chip,ffn::QuantHidden>,
    hscale:&mut HbmTensor<f32,Chip,m![B]>,
) {
    let (up_block,gate_block,down_block)=block_scales;
    crate::ops_norm_batch::post_attention_b16_stage(device,projected,post_attention,residual);
    crate::ops_norm_batch::rmsnorm_b16_stage(device,residual,pre_ffn,norm);
    ffn::batch_ffn_stages(device,norm,up,gate,down,up_block,gate_block,down_block,us,gs,ds,ug,gg,dg,
        qinput,xscale,hidden,qhidden,hscale,projected);
    crate::ops_norm_batch::post_feedforward_b16_stage(device,projected,post_ffn,layer_gate,residual);
}
