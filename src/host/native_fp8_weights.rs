//! Startup-only FP8 repacking of the provided FP4 matrices and their block scales.
//! Only the blocked FP8 copy is retained; temporary row-major tensors are dropped.
//! Original tensors remain available for the scalar and exact B16 execution paths.
use crate::ops_native_fp8_blocked::{DownWeight, UpWeight};
use crate::{
    Chip,
    axes::{H, L},
};
use furiosa_opt_std::prelude::*;

pub(crate) struct NativeFp8Weights {
    pub up: HbmTensor<f8e4m3, Chip, UpWeight>,
    pub gate: HbmTensor<f8e4m3, Chip, UpWeight>,
    pub down: HbmTensor<f8e4m3, Chip, DownWeight>,
    pub up_dequant: HbmTensor<f32, Chip, m![L]>,
    pub gate_dequant: HbmTensor<f32, Chip, m![L]>,
    pub down_dequant: HbmTensor<f32, Chip, m![H]>,
}

impl NativeFp8Weights {
    pub async fn from_packed(
        device: &mut Device,
        up: &HbmTensor<f4e2m1, Chip, m![L, H]>,
        gate: &HbmTensor<f4e2m1, Chip, m![L, H]>,
        down: &HbmTensor<f4e2m1, Chip, m![H, L]>,
        up_block: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
        gate_block: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
        down_block: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Allocate and initialize one tensor at a time, bounding host peak memory.
        let mut quantized_up = HostTensor::<f8e4m3, m![L, H]>::zero()
            .to_hbm(&mut device.pdma)
            .await?;
        let mut up_dequant = HostTensor::<f32, m![L]>::zero()
            .to_hbm(&mut device.pdma)
            .await?;
        launch(
            crate::ops_batch_prepack8::prepack_up,
            (device, up, up_block, &mut quantized_up, &mut up_dequant),
        )
        .await?;
        let mut blocked_up = HostTensor::<f8e4m3, UpWeight>::zero()
            .to_hbm(&mut device.pdma)
            .await?;
        launch(
            crate::ops_native_fp8_blocked::reblock_up,
            (device, &quantized_up, &mut blocked_up),
        )
        .await?;
        drop(quantized_up);
        let mut quantized_gate = HostTensor::<f8e4m3, m![L, H]>::zero()
            .to_hbm(&mut device.pdma)
            .await?;
        let mut gate_dequant = HostTensor::<f32, m![L]>::zero()
            .to_hbm(&mut device.pdma)
            .await?;
        launch(
            crate::ops_batch_prepack8::prepack_up,
            (
                device,
                gate,
                gate_block,
                &mut quantized_gate,
                &mut gate_dequant,
            ),
        )
        .await?;
        let mut blocked_gate = HostTensor::<f8e4m3, UpWeight>::zero()
            .to_hbm(&mut device.pdma)
            .await?;
        launch(
            crate::ops_native_fp8_blocked::reblock_up,
            (device, &quantized_gate, &mut blocked_gate),
        )
        .await?;
        drop(quantized_gate);
        let mut quantized_down = HostTensor::<f8e4m3, m![H, L]>::zero()
            .to_hbm(&mut device.pdma)
            .await?;
        let mut down_dequant = HostTensor::<f32, m![H]>::zero()
            .to_hbm(&mut device.pdma)
            .await?;
        launch(
            crate::ops_batch_prepack8::prepack_down,
            (
                device,
                down,
                down_block,
                &mut quantized_down,
                &mut down_dequant,
            ),
        )
        .await?;
        let mut blocked_down = HostTensor::<f8e4m3, DownWeight>::zero()
            .to_hbm(&mut device.pdma)
            .await?;
        launch(
            crate::ops_native_fp8_blocked::reblock_down,
            (device, &quantized_down, &mut blocked_down),
        )
        .await?;
        drop(quantized_down);
        Ok(Self {
            up: blocked_up,
            gate: blocked_gate,
            down: blocked_down,
            up_dequant,
            gate_dequant,
            down_dequant,
        })
    }
}
