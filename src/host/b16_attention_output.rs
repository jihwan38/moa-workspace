//! Per-call host adapter for exact original live-prefix commit/attention/O.
//! No persistent mode; callers retain the existing independent-owner checks.
use super::{
    load::SlidingLayer,
    request_pool::{PoolOffsets, PooledStep},
};
use crate::{
    Chip,
    axes::{B, Ds, Gs, H, Ns, Ts},
    ops_b16_attention_output as op,
    ops_pool_views::Rc64,
};
use furiosa_opt_std::prelude::*;
impl PooledStep {
    pub(crate) async fn sliding_attention_output(
        &self,
        device: &mut Device,
        q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
        k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
        pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
        layer: &SlidingLayer,
        attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
        output: &mut HbmTensor<bf16, Chip, m![B, H]>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match &self.commits {
            PoolOffsets::V1 { sliding, .. } => {
                launch(
                    op::sliding_1,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V2 { sliding, .. } => {
                launch(
                    op::sliding_2,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V3 { sliding, .. } => {
                launch(
                    op::sliding_3,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V4 { sliding, .. } => {
                launch(
                    op::sliding_4,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V5 { sliding, .. } => {
                launch(
                    op::sliding_5,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V6 { sliding, .. } => {
                launch(
                    op::sliding_6,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V7 { sliding, .. } => {
                launch(
                    op::sliding_7,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V8 { sliding, .. } => {
                launch(
                    op::sliding_8,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V9 { sliding, .. } => {
                launch(
                    op::sliding_9,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V10 { sliding, .. } => {
                launch(
                    op::sliding_10,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V11 { sliding, .. } => {
                launch(
                    op::sliding_11,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V12 { sliding, .. } => {
                launch(
                    op::sliding_12,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V13 { sliding, .. } => {
                launch(
                    op::sliding_13,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V14 { sliding, .. } => {
                launch(
                    op::sliding_14,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V15 { sliding, .. } => {
                launch(
                    op::sliding_15,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V16 { sliding, .. } => {
                launch(
                    op::sliding_16,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
        }
        Ok(())
    }
    pub(crate) async fn sliding_attention_output_fixed_e4(
        &self,
        device: &mut Device,
        q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
        k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
        pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
        layer: &SlidingLayer,
        attention: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
        output: &mut HbmTensor<bf16, Chip, m![B, H]>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match &self.commits {
            PoolOffsets::V1 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_1,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V2 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_2,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V3 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_3,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V4 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_4,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V5 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_5,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V6 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_6,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V7 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_7,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V8 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_8,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V9 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_9,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V10 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_10,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V11 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_11,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V12 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_12,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V13 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_13,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V14 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_14,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V15 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_15,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            PoolOffsets::V16 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_b16::sliding_16,
                    (
                        device,
                        q,
                        k,
                        v,
                        sliding,
                        &self.ring_offsets,
                        &self.sliding_mask,
                        pool_k,
                        pool_v,
                        &layer.o_weight,
                        &layer.o_weight_scale,
                        attention,
                        unsafe { output.view_mut().reshape() },
                    ),
                )
                .await?
            }
        }
        Ok(())
    }
}
