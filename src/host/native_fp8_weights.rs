//! Startup-only FP8 repacking of the provided FP4 matrices and their block scales.
//! Original tensors remain available for the scalar and exact B16 execution paths.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{H,L}};

pub(crate) struct NativeFp8Weights {
    pub up: HbmTensor<f8e4m3,Chip,m![L,H]>,
    pub gate: HbmTensor<f8e4m3,Chip,m![L,H]>,
    pub down: HbmTensor<f8e4m3,Chip,m![H,L]>,
    pub up_dequant: HbmTensor<f32,Chip,m![L]>,
    pub gate_dequant: HbmTensor<f32,Chip,m![L]>,
    pub down_dequant: HbmTensor<f32,Chip,m![H]>,
}

impl NativeFp8Weights {
    pub async fn from_packed(
        device:&mut Device,
        up:&HbmTensor<f4e2m1,Chip,m![L,H]>,
        gate:&HbmTensor<f4e2m1,Chip,m![L,H]>,
        down:&HbmTensor<f4e2m1,Chip,m![H,L]>,
        up_block:&HbmTensor<f8e4m3,Chip,m![L,H/16]>,
        gate_block:&HbmTensor<f8e4m3,Chip,m![L,H/16]>,
        down_block:&HbmTensor<f8e4m3,Chip,m![H,L/16]>,
    )->Result<Self,Box<dyn std::error::Error>> {
        // Allocate and initialize one tensor at a time, bounding host peak memory.
        let mut quantized_up=HostTensor::<f8e4m3,m![L,H]>::zero().to_hbm(&mut device.pdma).await?;
        let mut up_dequant=HostTensor::<f32,m![L]>::zero().to_hbm(&mut device.pdma).await?;
        launch(crate::ops_batch_prepack8::prepack_up,(device,up,up_block,&mut quantized_up,&mut up_dequant)).await?;
        let mut quantized_gate=HostTensor::<f8e4m3,m![L,H]>::zero().to_hbm(&mut device.pdma).await?;
        let mut gate_dequant=HostTensor::<f32,m![L]>::zero().to_hbm(&mut device.pdma).await?;
        launch(crate::ops_batch_prepack8::prepack_up,(device,gate,gate_block,&mut quantized_gate,&mut gate_dequant)).await?;
        let mut quantized_down=HostTensor::<f8e4m3,m![H,L]>::zero().to_hbm(&mut device.pdma).await?;
        let mut down_dequant=HostTensor::<f32,m![H]>::zero().to_hbm(&mut device.pdma).await?;
        launch(crate::ops_batch_prepack8::prepack_down,(device,down,down_block,&mut quantized_down,&mut down_dequant)).await?;
        Ok(Self {up:quantized_up,gate:quantized_gate,down:quantized_down,up_dequant,gate_dequant,down_dequant})
    }
}
