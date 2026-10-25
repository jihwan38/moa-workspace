//! Sixteen independent vision patches using B8-TRF projections; unchanged BF16 boundaries.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::*};
use crate::device::{shared, vision};
pub(crate) type Cluster = m![B/8];
pub(crate) type Slice = m![B%8,1#32];
#[path = "device/vision/projection_trf.rs"]
mod projection;

#[device(chip=1)]
pub fn normalized_patch(device:&mut Device, x:&HbmTensor<bf16,Chip,m![B,Rv]>,
    weight:&HbmTensor<bf16,Chip,m![Rv]>,bias:&HbmTensor<bf16,Chip,m![Rv]>,
    out:&mut HbmTensor<bf16,Chip,m![B,Rv]>) {
    let x:DmTensor<bf16,Chip,Cluster,Slice,m![Rv]>=x.to_dm(&mut device.tdma);
    let x=vision::layernorm::normalize_patch(device,&x,weight,bias);
    x.view().to_hbm_view(&mut device.tdma,out.view_mut());
}

#[device(chip=1)]
pub fn patch_dense_before_ln2(device:&mut Device,x:&HbmTensor<bf16,Chip,m![B,Rv]>,
    weight:&HbmTensor<bf16,Chip,m![Mv,Rv]>,bias:&HbmTensor<bf16,Chip,m![Mv]>,
    out:&mut HbmTensor<bf16,Chip,m![B,Mv]>) {
    let x:DmTensor<bf16,Chip,Cluster,Slice,m![Rv]>=x.to_dm(&mut device.tdma);
    let x=projection::patch_projection(device,&x,weight,bias);
    x.view().to_hbm_view(&mut device.tdma,out.view_mut());
}

#[device(chip=1)]
pub fn text_rms_input(device:&mut Device,x:&HbmTensor<bf16,Chip,m![B,Ov]>,
    out:&mut HbmTensor<bf16,Chip,m![B,Ov]>) {
    let x:DmTensor<bf16,Chip,Cluster,Slice,m![Ov]>=x.to_dm(&mut device.tdma);
    let x=projection::rmsnorm_without_weight(device,&x);
    x.view().to_hbm_view(&mut device.tdma,out.view_mut());
}

#[device(chip=1)]
pub fn patch_embed(
    device:&mut Device, x:&HbmTensor<bf16,Chip,m![B,Rv]>,
    ln1_weight:&HbmTensor<bf16,Chip,m![Rv]>, ln1_bias:&HbmTensor<bf16,Chip,m![Rv]>,
    dense_weight:&HbmTensor<bf16,Chip,m![Mv,Rv]>, dense_bias:&HbmTensor<bf16,Chip,m![Mv]>,
    ln2_weight:&HbmTensor<bf16,Chip,m![Mv]>, ln2_bias:&HbmTensor<bf16,Chip,m![Mv]>,
    out:&mut HbmTensor<bf16,Chip,m![B,Mv]>,
) {
    let x:DmTensor<bf16,Chip,Cluster,Slice,m![Rv]>=x.to_dm(&mut device.tdma);
    let x=vision::layernorm::normalize_patch(device,&x,ln1_weight,ln1_bias);
    let x=projection::patch_projection(device,&x,dense_weight,dense_bias);
    let x=vision::layernorm::normalize(device,&x,ln2_weight,ln2_bias);
    x.view().to_hbm_view(&mut device.tdma,out.view_mut());
}

#[device(chip=1)]
pub fn add_position_and_norm(
    device:&mut Device,x:&HbmTensor<bf16,Chip,m![B,Mv]>,pos:&HbmTensor<bf16,Chip,m![B,Mv]>,
    weight:&HbmTensor<bf16,Chip,m![Mv]>,bias:&HbmTensor<bf16,Chip,m![Mv]>,
    out:&mut HbmTensor<bf16,Chip,m![B,Mv]>,
) {
    let x:DmTensor<bf16,Chip,Cluster,Slice,m![Mv]>=x.to_dm(&mut device.tdma);
    let pos:DmTensor<bf16,Chip,Cluster,Slice,m![Mv]>=pos.to_dm(&mut device.tdma);
    let x=shared::residual::add_vision(device,&x,&pos);
    let x=vision::layernorm::normalize(device,&x,weight,bias);
    x.view().to_hbm_view(&mut device.tdma,out.view_mut());
}

#[device(chip=1)]
pub fn project_to_text_embedding(
    device:&mut Device,x:&HbmTensor<bf16,Chip,m![B,Ov]>,
    weight:&HbmTensor<bf16,Chip,m![H,Ov]>,out:&mut HbmTensor<bf16,Chip,m![B,H]>,
) {
    let x:DmTensor<bf16,Chip,Cluster,Slice,m![Ov]>=x.to_dm(&mut device.tdma);
    let x=projection::project_to_text_hidden(device,&x,weight);
    x.view().to_hbm_view(&mut device.tdma,out.view_mut());
}

#[device(chip=1)]
pub fn encode_patches(
    device:&mut Device,pixels:&HbmTensor<bf16,Chip,m![B,Rv]>,pos:&HbmTensor<bf16,Chip,m![B,Mv]>,
    ln1_weight:&HbmTensor<bf16,Chip,m![Rv]>,ln1_bias:&HbmTensor<bf16,Chip,m![Rv]>,
    dense_weight:&HbmTensor<bf16,Chip,m![Mv,Rv]>,dense_bias:&HbmTensor<bf16,Chip,m![Mv]>,
    ln2_weight:&HbmTensor<bf16,Chip,m![Mv]>,ln2_bias:&HbmTensor<bf16,Chip,m![Mv]>,
    pos_weight:&HbmTensor<bf16,Chip,m![Mv]>,pos_bias:&HbmTensor<bf16,Chip,m![Mv]>,
    embedding_weight:&HbmTensor<bf16,Chip,m![H,Ov]>,out:&mut HbmTensor<bf16,Chip,m![B,H]>,
) {
    let x:DmTensor<bf16,Chip,Cluster,Slice,m![Rv]>=pixels.to_dm(&mut device.tdma);
    let x=vision::layernorm::normalize_patch(device,&x,ln1_weight,ln1_bias);
    let x=projection::patch_projection(device,&x,dense_weight,dense_bias);
    let x=vision::layernorm::normalize(device,&x,ln2_weight,ln2_bias);
    let pos:DmTensor<bf16,Chip,Cluster,Slice,m![Mv]>=pos.to_dm(&mut device.tdma);
    let x=shared::residual::add_vision(device,&x,&pos);
    let x=vision::layernorm::normalize(device,&x,pos_weight,pos_bias);
    let x:DmTensor<bf16,Chip,Cluster,Slice,m![Ov]>=unsafe{x.reshape()};
    let x=projection::project_to_text_hidden(device,&x,embedding_weight);
    x.view().to_hbm_view(&mut device.tdma,out.view_mut());
}
