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
        // Each device kernel returns fully written storage. Drop each temporary
        // row-major weight immediately after blocking; retain only one FP8 copy.
        let (quantized_up, up_dequant) = launch(
            crate::ops_prepack_alloc::prepack_up,
            (&mut *device, up, up_block),
        )
        .await?;
        let blocked_up = launch(
            crate::ops_prepack_alloc::reblock_up,
            (&mut *device, &quantized_up),
        )
        .await?;
        drop(quantized_up);
        let (quantized_gate, gate_dequant) = launch(
            crate::ops_prepack_alloc::prepack_up,
            (&mut *device, gate, gate_block),
        )
        .await?;
        let blocked_gate = launch(
            crate::ops_prepack_alloc::reblock_up,
            (&mut *device, &quantized_gate),
        )
        .await?;
        drop(quantized_gate);
        let (quantized_down, down_dequant) = launch(
            crate::ops_prepack_alloc::prepack_down,
            (&mut *device, down, down_block),
        )
        .await?;
        let blocked_down = launch(
            crate::ops_prepack_alloc::reblock_down,
            (&mut *device, &quantized_down),
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
