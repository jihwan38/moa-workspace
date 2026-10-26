//! Retained-HBM composition only. Every stage delegates to an unchanged body.
//! Packed I32/I4 interpretation belongs to the host boundary between graphs.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{H, L}, ops_batch_layouts::{P512H960, P512L960}};
use crate::{ops_int4_ffn_bridges as bridge, ops_int4_grouped_activation_store as quant,
    ops_int4_grouped_activation_projection as projection, ops_norm_blocked_residual as tail};
use crate::ops_output_p128_h64_blocked::Output as Sliding;
use quant::{Mantissa, Record};

#[device(chip = 1)]
pub fn prefix_input(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, Sliding>,
    b: &HbmTensor<bf16, Chip, Sliding>,
    c: &HbmTensor<bf16, Chip, Sliding>,
    d: &HbmTensor<bf16, Chip, Sliding>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    table: &HbmTensor<i32, Chip, m![Mantissa, Record]>,
    shifts: &HbmTensor<i32, Chip, m![H % 8]>,
    residual: &mut HbmTensor<bf16, Chip, tail::Residual>,
    prefix: &mut HbmTensor<bf16, Chip, bridge::InputBf16>,
    packed: &mut HbmTensor<i32, Chip, quant::PackedInput>,
    scales: &mut HbmTensor<f32, Chip, quant::InputScales>,
    offsets: &mut HbmTensor<i32, Chip, quant::InputScales>,
) {
    bridge::prefix_sliding_bf16_stage(device, a, b, c, d, post, pre, residual, prefix);
    quant::quantize_pack_input_stage(device, prefix, table, shifts, packed, scales, offsets);
}

#[device(chip = 1)]
pub fn upgate_hidden(
    device: &mut Device,
    input: &HbmTensor<i4, Chip, P512H960>,
    input_scales: &HbmTensor<f32, Chip, projection::InputScale>,
    weight: &HbmTensor<i4, Chip, projection::JoinedWeight>,
    weight_scales: &HbmTensor<f32, Chip, projection::JoinedBlockScale>,
    up_global: &HbmTensor<f32, Chip, m![1]>,
    gate_global: &HbmTensor<f32, Chip, m![1]>,
    table: &HbmTensor<i32, Chip, m![Mantissa, Record]>,
    shifts: &HbmTensor<i32, Chip, m![L % 8]>,
    joined: &mut HbmTensor<bf16, Chip, projection::JoinedOutput>,
    hidden: &mut HbmTensor<bf16, Chip, bridge::HiddenBf16>,
    packed: &mut HbmTensor<i32, Chip, quant::PackedHidden>,
    scales: &mut HbmTensor<f32, Chip, quant::HiddenScales>,
    offsets: &mut HbmTensor<i32, Chip, quant::HiddenScales>,
) {
    projection::project_upgate_stage(device, input, input_scales, weight, weight_scales, joined);
    bridge::geglu_bf16_stage(device, joined, up_global, gate_global, hidden);
    quant::quantize_pack_hidden_stage(device, hidden, table, shifts, packed, scales, offsets);
}

#[device(chip = 1)]
pub fn down_tail(
    device: &mut Device,
    input: &HbmTensor<i4, Chip, P512L960>,
    input_scales: &HbmTensor<f32, Chip, projection::HiddenScale>,
    weight: &HbmTensor<i4, Chip, projection::DownWeight>,
    weight_scales: &HbmTensor<f32, Chip, projection::DownBlockScale>,
    global: &HbmTensor<f32, Chip, m![1]>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    layer_gate: &HbmTensor<bf16, Chip, m![1#8]>,
    down: &mut HbmTensor<bf16, Chip, projection::Output>,
    residual: &mut HbmTensor<bf16, Chip, tail::Residual>,
) {
    projection::project_down_stage(device, input, input_scales, weight, weight_scales, global, down);
    tail::post_feedforward_stage(device, down, post, layer_gate, residual);
}
