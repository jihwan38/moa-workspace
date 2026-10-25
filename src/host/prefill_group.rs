//! Four P128 row groups share QKV weight loads and a blocked P512 FFN pass.
//! Separate constructors preserve either independent request allocations or four
//! chronological chunks of one request; attention always visits rows in order.
use super::*;
use crate::ops_batch_layouts::{P512H960, P512L240, P512L960};
use crate::ops_prompt_group_io as group_io;
/// Raw QKV belongs to one group row and remains in the producer's physical
/// layout until the matching head kernel consumes it. Ordinary Wide keeps its
/// canonical buffers and never receives these values.
pub(super) struct BlockedQkv {
    pub(super) q_s: HbmTensor<bf16, Chip, crate::ops_qkv_shared_blocked::QueryOutput>,
    pub(super) k_s: HbmTensor<bf16, Chip, crate::ops_qkv_shared_blocked::KeyOutput>,
    pub(super) v_s: HbmTensor<bf16, Chip, crate::ops_qkv_shared_blocked::KeyOutput>,
    pub(super) q_f: HbmTensor<bf16, Chip, crate::ops_qk_shared_blocked::QueryOutput>,
    pub(super) k_f: HbmTensor<bf16, Chip, crate::ops_qk_shared_blocked::KeyOutput>,
}
impl BlockedQkv {
    async fn new(device: &mut Device) -> Result<Self> {
        macro_rules! zero {
            ($layout:ty) => {
                HostTensor::<bf16, $layout>::zero().to_hbm(&mut device.pdma).await?
            };
        }
        Ok(Self {
            q_s: zero!(crate::ops_qkv_shared_blocked::QueryOutput),
            k_s: zero!(crate::ops_qkv_shared_blocked::KeyOutput),
            v_s: zero!(crate::ops_qkv_shared_blocked::KeyOutput),
            q_f: zero!(crate::ops_qk_shared_blocked::QueryOutput),
            k_f: zero!(crate::ops_qk_shared_blocked::KeyOutput),
        })
    }
}
pub(crate) struct GroupPrefill {
    blocked: [BlockedQkv; 4],
    output_s: [HbmTensor<bf16, Chip, crate::ops_output_p128_h64_blocked::Output>; 4],
    rows: [WidePrefill; 4],
    residual_blocked: HbmTensor<bf16, Chip, crate::ops_norm_blocked_residual::Residual>,
    norm: HbmTensor<bf16, Chip, m![P512, H]>,
    projected: HbmTensor<bf16, Chip, crate::ops_native_fp8_p512_downblocked::Output>,
    zero: HbmTensor<i32, Chip, m![1]>,
    second: HbmTensor<i32, Chip, m![1]>,
    third: HbmTensor<i32, Chip, m![1]>,
    fourth: HbmTensor<i32, Chip, m![1]>,
    q_input: HbmTensor<f8e4m3, Chip, P512H960>,
    input_scale: HbmTensor<f32, Chip, m![P512]>,
    up: HbmTensor<bf16, Chip, P512L240>,
    gate: HbmTensor<bf16, Chip, P512L240>,
    q_hidden: HbmTensor<f8e4m3, Chip, P512L960>,
    hidden_scale: HbmTensor<f32, Chip, m![P512]>,
}
pub(crate) struct PreparedGroup {
    states: [PreparedWide; 4],
    // Owned by this prepared execution; no persistent Model mode survives cancellation.
    fused_norms: bool,
    blocked_output: bool,
    shard_quant: bool,
    quarter_norm: bool,
    fused_ffn_core: bool,
    quarter_postnorm: bool,
    blocked_residual: bool,
}
/// Four chronological chunks of one request. Only its single allocation may be
/// exposed to the scheduler; the four copies below are execution metadata.
pub(crate) struct PreparedPrompt512 {
    chunks: PreparedGroup,
}
/// Two independent owners contribute four chronological P128 quarters in total.
/// The historical name remains for compatibility with the 384+128 diagnostic.
/// Repeated metadata is private; only the two real allocations are exposed.
pub(crate) struct PreparedPrompt384128 {
    chunks: PreparedGroup,
    second_owner_row: usize,
    final_rows: [usize; 2],
}
impl PreparedPrompt384128 {
    pub(crate) fn fused_head_attention(&mut self, enabled: bool) {
        self.chunks.fused_head_attention(enabled);
    }
    pub(crate) fn original_blocked_residual(&mut self) { self.chunks.blocked_residual = false; }
    pub(crate) fn blocks(&self) -> [&runtime::Blocks; 2] {
        [&self.chunks.states[0].blocks, &self.chunks.states[self.second_owner_row].blocks]
    }
}

impl PreparedPrompt512 {
    pub(crate) fn fused_head_attention(&mut self, enabled: bool) {
        self.chunks.fused_head_attention(enabled);
    }
    pub(crate) fn prune_empty_history(&mut self) { self.chunks.prune_empty_history(); }
    pub(crate) fn original_empty_history(&mut self) { self.chunks.original_empty_history(); }
    pub(crate) fn original_blocked_residual(&mut self) { self.chunks.blocked_residual = false; }
    pub(crate) fn original_norms(&mut self) { self.chunks.fused_norms = false; }
    pub(crate) fn original_output(&mut self) { self.chunks.blocked_output = false; }
    pub(crate) fn original_shard_quant(&mut self) { self.chunks.shard_quant = false; }
    pub(crate) fn original_quarter_norm(&mut self) { self.chunks.quarter_norm = false; }
    pub(crate) fn original_ffn_core(&mut self) { self.chunks.fused_ffn_core = false; }
    pub(crate) fn original_quarter_postnorm(&mut self) { self.chunks.quarter_postnorm = false; }
    pub(crate) fn blocks(&self) -> &runtime::Blocks {
        &self.chunks.states[0].blocks
    }
}
impl PreparedGroup {
    pub(crate) fn fused_head_attention(&mut self, enabled: bool) {
        for state in &mut self.states { state.fused_head_attention = enabled; }
    }
    pub(crate) fn prune_empty_history(&mut self) {
        for state in &mut self.states { state.prune_empty_history = true; }
    }
    pub(crate) fn original_empty_history(&mut self) {
        for state in &mut self.states { state.prune_empty_history = false; }
    }
    pub(crate) fn original_blocked_residual(&mut self) { self.blocked_residual = false; }
    fn uses_blocked_residual(&self) -> bool {
        // Old diagnostic implementations retain canonical state throughout.
        // Their timings include this storage fallback; the dedicated storage
        // control is the isolated comparison. FFN core selection is independent.
        self.blocked_residual && self.fused_norms && self.blocked_output
            && self.shard_quant && self.quarter_norm && self.quarter_postnorm
    }
    pub(crate) fn original_norms(&mut self) { self.fused_norms = false; }
    pub(crate) fn original_output(&mut self) { self.blocked_output = false; }
    pub(crate) fn original_shard_quant(&mut self) { self.shard_quant = false; }
    pub(crate) fn original_quarter_norm(&mut self) { self.quarter_norm = false; }
    pub(crate) fn original_ffn_core(&mut self) { self.fused_ffn_core = false; }
    pub(crate) fn original_quarter_postnorm(&mut self) { self.quarter_postnorm = false; }
    pub(crate) fn blocks(&self) -> [&runtime::Blocks; 4] {
        [
            &self.states[0].blocks,
            &self.states[1].blocks,
            &self.states[2].blocks,
            &self.states[3].blocks,
        ]
    }
}
impl GroupPrefill {
    pub(crate) async fn new(device: &mut Device) -> Result<Self> {
        macro_rules! zero {
            ($d:ty,$m:ty) => {
                HostTensor::<$d, $m>::zero()
                    .to_hbm(&mut device.pdma)
                    .await?
            };
        }
        macro_rules! offset {
            ($row:expr) => {
                HostTensor::<i32, m![1]>::from_vec([($row * P128::SIZE * H::SIZE * 2) as i32])
                    .to_hbm(&mut device.pdma)
                    .await?
            };
        }
        Ok(Self {
            blocked: [
                BlockedQkv::new(device).await?, BlockedQkv::new(device).await?,
                BlockedQkv::new(device).await?, BlockedQkv::new(device).await?,
            ],
            output_s: [
                zero!(bf16, crate::ops_output_p128_h64_blocked::Output),
                zero!(bf16, crate::ops_output_p128_h64_blocked::Output),
                zero!(bf16, crate::ops_output_p128_h64_blocked::Output),
                zero!(bf16, crate::ops_output_p128_h64_blocked::Output),
            ],
            rows: [
                WidePrefill::new(device).await?,
                WidePrefill::new(device).await?,
                WidePrefill::new(device).await?,
                WidePrefill::new(device).await?,
            ],
            residual_blocked: zero!(bf16, crate::ops_norm_blocked_residual::Residual),
            norm: zero!(bf16, m![P512, H]),
            projected: zero!(bf16, crate::ops_native_fp8_p512_downblocked::Output),
            zero: offset!(0),
            second: offset!(1),
            third: offset!(2),
            fourth: offset!(3),
            q_input: zero!(f8e4m3, P512H960),
            input_scale: zero!(f32, m![P512]),
            up: zero!(bf16, P512L240),
            gate: zero!(bf16, P512L240),
            q_hidden: zero!(f8e4m3, P512L960),
            hidden_scale: zero!(f32, m![P512]),
        })
    }
    pub(crate) fn supports(sequences: &[Sequence<'_>]) -> bool {
        sequences.len() == 4
            && sequences
                .iter()
                .all(|s| s.inputs.len() == P128::SIZE && WidePrefill::supports(s))
            && crate::host::request_pool::independent_sequences(sequences)
    }
    pub(crate) async fn prepare(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        sequences: &[Sequence<'_>],
        scratch: &mut Prefill,
        batched_media: bool,
    ) -> Result<PreparedGroup> {
        if !Self::supports(sequences) {
            return Err("grouped prefill needs four independent page-local128-token chunks".into());
        }
        self.prepare_rows(device, weights, sequences, scratch, batched_media).await
    }

    pub(crate) fn supports_prompt512(sequence: &Sequence<'_>) -> bool {
        sequence.inputs.len() == P512::SIZE
            && sequence.start_position % P128::SIZE == 0
            && sequence.start_position.checked_add(P512::SIZE).is_some()
            && sequence.inputs[..P512::SIZE - 1]
                .iter()
                .all(|input| !input.logits)
    }

    /// This constructor deliberately does not call the independent-request
    /// constructor. All chunks borrow the same allocation and their positions
    /// are derived here, so callers cannot reorder or overlap them.
    pub(crate) async fn prepare_prompt512(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        sequence: &Sequence<'_>,
        scratch: &mut Prefill,
        batched_media: bool,
    ) -> Result<PreparedPrompt512> {
        if !Self::supports_prompt512(sequence) {
            return Err(
                "Prompt512 needs512 consecutive inputs aligned to128 with final-only logits".into(),
            );
        }
        let chunks: Vec<_> = sequence
            .inputs
            .chunks_exact(P128::SIZE)
            .enumerate()
            .map(|(chunk, inputs)| Sequence {
                inputs,
                start_position: sequence.start_position + chunk * P128::SIZE,
                blocks: sequence.blocks,
            })
            .collect();
        Ok(PreparedPrompt512 {
            chunks: self.prepare_rows(device, weights, &chunks, scratch, batched_media).await?,
        })
    }

    pub(crate) fn supports_prompt384128(sequences: &[Sequence<'_>]) -> bool {
        sequences.len() == 2
            && sequences[0].inputs.len() == 3 * P128::SIZE
            && sequences[1].inputs.len() == P128::SIZE
            && Self::supports_prompt_pair512(sequences)
    }

    pub(crate) fn supports_prompt_pair512(sequences: &[Sequence<'_>]) -> bool {
        sequences.len() == 2
            && sequences.iter().all(|s| matches!(s.inputs.len(), 128 | 256 | 384))
            && sequences[0].inputs.len() + sequences[1].inputs.len() == P512::SIZE
            && sequences.iter().all(|sequence| {
                sequence.start_position % P128::SIZE == 0
                    && sequence.start_position.checked_add(sequence.inputs.len()).is_some()
                    && sequence.inputs[..sequence.inputs.len() - 1].iter().all(|input| !input.logits)
            })
            && crate::host::request_pool::independent_sequences(sequences)
    }

    /// Validate the two actual owners before deriving A0/A1/A2/B0 internally.
    /// Ordinary independent-quarter validation remains unchanged.
    pub(crate) async fn prepare_prompt384128(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        sequences: &[Sequence<'_>],
        scratch: &mut Prefill,
        batched_media: bool,
    ) -> Result<PreparedPrompt384128> {
        if !Self::supports_prompt384128(sequences) {
            return Err("Prompt384128 needs independent384/128 ranges aligned to128 with final-only logits".into());
        }
        self.prepare_prompt_pair512(device, weights, sequences, scratch, batched_media).await
    }

    pub(crate) async fn prepare_prompt_pair512(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        sequences: &[Sequence<'_>],
        scratch: &mut Prefill,
        batched_media: bool,
    ) -> Result<PreparedPrompt384128> {
        if !Self::supports_prompt_pair512(sequences) {
            return Err("PromptPair512 needs two independent aligned128/256/384 ranges totaling512, with final-only logits".into());
        }
        let chunks = prompt_pair512_chunks(sequences);
        let second_owner_row = sequences[0].inputs.len() / P128::SIZE;
        debug_assert!(chunks.iter().all(WidePrefill::supports));
        Ok(PreparedPrompt384128 {
            chunks: self.prepare_rows(device, weights, &chunks, scratch, batched_media).await?,
            second_owner_row,
            final_rows: [second_owner_row - 1, 3],
        })
    }

    /// Only validated constructors above call this execution helper.
    async fn prepare_rows(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        sequences: &[Sequence<'_>],
        scratch: &mut Prefill,
        batched_media: bool,
    ) -> Result<PreparedGroup> {
        debug_assert_eq!(sequences.len(), 4);
        Ok(PreparedGroup {
            fused_norms: true,
            blocked_output: true,
            shard_quant: true,
            quarter_norm: true,
            fused_ffn_core: true,
            quarter_postnorm: true,
            blocked_residual: true,
            states: [
                self.rows[0]
                    .prepare_with_media_batch(device, weights, &sequences[0], scratch, true, batched_media)
                    .await?,
                self.rows[1]
                    .prepare_with_media_batch(device, weights, &sequences[1], scratch, true, batched_media)
                    .await?,
                self.rows[2]
                    .prepare_with_media_batch(device, weights, &sequences[2], scratch, true, batched_media)
                    .await?,
                self.rows[3]
                    .prepare_with_media_batch(device, weights, &sequences[3], scratch, true, batched_media)
                    .await?,
            ],
        })
    }

    pub(crate) async fn advance_prompt512(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        state: &mut PreparedPrompt512,
        ws: &mut Workspace,
        scratch: &mut Prefill,
    ) -> Result<bool> {
        // advance_layer visits attention/commit in row order after all QKV
        // projections. Thus each chunk sees all earlier chunks in this layer,
        // and a sliding wrap never evicts history before that chunk attends.
        self.advance_layer(device, weights, &mut state.chunks, ws, scratch)
            .await
    }

    pub(crate) async fn finish_prompt512(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        state: &PreparedPrompt512,
        ws: &mut Workspace,
        scratch: &mut Prefill,
    ) -> Result<()> {
        self.finish(3, device, weights, &state.chunks, ws, scratch)
            .await
    }
    pub(crate) async fn advance_prompt384128(
        &mut self, device: &mut Device, weights: &Weights,
        state: &mut PreparedPrompt384128, ws: &mut Workspace, scratch: &mut Prefill,
    ) -> Result<bool> {
        // All A quarters precede all B quarters; each owner's commits are causal.
        self.advance_layer(device, weights, &mut state.chunks, ws, scratch).await
    }
    pub(crate) async fn finish_prompt384128(
        &mut self, owner: usize, device: &mut Device, weights: &Weights,
        state: &PreparedPrompt384128, ws: &mut Workspace, scratch: &mut Prefill,
    ) -> Result<()> {
        self.finish(state.final_rows[owner], device, weights, &state.chunks, ws, scratch).await
    }

    pub(crate) async fn advance_layer(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        state: &mut PreparedGroup,
        ws: &mut Workspace,
        scratch: &mut Prefill,
    ) -> Result<bool> {
        let index = state.states[0].next_layer;
        if state.states.iter().any(|s| s.next_layer != index) {
            return Err("grouped layer cursors diverged".into());
        }
        let layer = weights
            .layers
            .get(index)
            .ok_or("grouped prefill layers already complete")?;
        let mut mark = scratch.profile_mark();
        let blocked_residual = state.uses_blocked_residual();
        if blocked_residual && index == 0 {
            let [a, b, c, d] = &self.rows;
            launch(crate::ops_norm_blocked_residual::pack_four,
                (device, &a.x, &b.x, &c.x, &d.x, &mut self.residual_blocked)).await?;
            scratch.record_optional_profile("group.residual.pack", &mut mark);
        }
        let input_norm = match layer {
            Layer::Sliding(l) => &l.input_norm,
            Layer::Full(l) => &l.input_norm,
        };
        if blocked_residual {
            let [a, b, c, d] = &mut self.rows;
            launch(crate::ops_norm_blocked_chain::normalize_qkv_four,
                (device, &self.residual_blocked, input_norm,
                 &mut a.qkv_input, &mut a.qkv_scale,
                 &mut b.qkv_input, &mut b.qkv_scale,
                 &mut c.qkv_input, &mut c.qkv_scale,
                 &mut d.qkv_input, &mut d.qkv_scale)).await?;
        } else if state.fused_norms {
            let [a, b, c, d] = &mut self.rows;
            launch(crate::ops_norm_fused::normalize_qkv_four,
                (device, &a.x, &b.x, &c.x, &d.x, input_norm,
                 &mut a.qkv_input, &mut a.qkv_scale,
                 &mut b.qkv_input, &mut b.qkv_scale,
                 &mut c.qkv_input, &mut c.qkv_scale,
                 &mut d.qkv_input, &mut d.qkv_scale)).await?;
        } else {
            for row in &mut self.rows {
                row.normalize(device, input_norm, scratch).await?;
                launch(
                    crate::ops_qkv_native_four::quantize_input,
                    (device, &row.norm, &mut row.qkv_input, &mut row.qkv_scale),
                )
                .await?;
            }
        }
        scratch.record_optional_profile("group.qkv.norm_quantize", &mut mark);
        let [a, b, c, d] = &self.rows;
        let [oa, ob, oc, od] = &mut self.blocked;
        match layer {
            Layer::Sliding(l) => {
                launch(crate::ops_qkv_shared_ingress::project_sliding_qkv_four,
                    (device, &a.qkv_input, &a.qkv_scale, &b.qkv_input, &b.qkv_scale,
                     &c.qkv_input, &c.qkv_scale, &d.qkv_input, &d.qkv_scale,
                     &l.q_weight, &l.k_weight, &l.v_weight,
                     &l.q_weight_scale, &l.k_weight_scale, &l.v_weight_scale,
                     (&mut oa.q_s, &mut oa.k_s, &mut oa.v_s),
                     (&mut ob.q_s, &mut ob.k_s, &mut ob.v_s),
                     (&mut oc.q_s, &mut oc.k_s, &mut oc.v_s),
                     (&mut od.q_s, &mut od.k_s, &mut od.v_s))).await?;
            }
            Layer::Full(l) => {
                launch(crate::ops_qk_shared_ingress::project_full_qk_four,
                    (device, &a.qkv_input, &a.qkv_scale, &b.qkv_input, &b.qkv_scale,
                     &c.qkv_input, &c.qkv_scale, &d.qkv_input, &d.qkv_scale,
                     &l.q_weight, &l.k_weight, &l.q_weight_scale, &l.k_weight_scale,
                     (&mut oa.q_f, &mut oa.k_f), (&mut ob.q_f, &mut ob.k_f),
                     (&mut oc.q_f, &mut oc.k_f), (&mut od.q_f, &mut od.k_f))).await?;
            }
        }
        scratch.record_optional_profile("group.qkv.projections", &mut mark);
        for row in 0..4 {
            let current = &state.states[row];
            ws.blocks.clone_from(&current.blocks);
            match layer {
                Layer::Sliding(layer) => {
                    self.rows[row]
                        .sliding_after_qkv(
                            device,
                            ws,
                            index,
                            layer,
                            current,
                            &current.segments,
                            scratch,
                            false,
                            true,
                            Some(&self.blocked[row]),
                        )
                        .await?
                }
                Layer::Full(layer) => {
                    self.rows[row]
                        .full_after_qkv(
                            device,
                            ws,
                            index,
                            layer,
                            current,
                            &current.segments,
                            scratch,
                            false,
                            true,
                            Some(&self.blocked[row]),
                        )
                        .await?
                }
            }
        }
        // All chronological attention/cache updates finish before the independent
        // O projections. These projections do not read or mutate KV ownership.
        let mut output_mark = scratch.profile_mark();
        let [a, b, c, d] = &mut self.rows;
        match layer {
            Layer::Sliding(l) => {
                if state.blocked_output && state.fused_norms {
                    let [oa, ob, oc, od] = &mut self.output_s;
                    launch(crate::ops_output_p128_h64_blocked::project_sliding_output_four,
                        (device, &a.attn_s, &b.attn_s, &c.attn_s, &d.attn_s,
                         &l.o_weight, &l.o_weight_scale, oa, ob, oc, od)).await?;
                } else {
                launch(crate::ops_output_p128_keyrings_h64::project_sliding_output_four,
                    (device, &a.attn_s, &b.attn_s, &c.attn_s, &d.attn_s,
                     &l.o_weight, &l.o_weight_scale, &mut a.projected,
                     &mut b.projected, &mut c.projected, &mut d.projected)).await?;
                }
            }
            Layer::Full(l) => {
                launch(crate::ops_output_p128_four_keyrings::project_full_output_four,
                    (device, &a.attn_f, &b.attn_f, &c.attn_f, &d.attn_f,
                     &l.o_weight, &l.o_weight_scale, &mut a.projected,
                     &mut b.projected, &mut c.projected, &mut d.projected)).await?;
            }
        }
        scratch.record_optional_profile("group.attention.output", &mut output_mark);
        let post_attention = match layer {
            Layer::Sliding(l) => &l.post_attention_norm,
            Layer::Full(l) => &l.post_attention_norm,
        };
        let (mlp, pre, post, gate) = match layer {
            Layer::Sliding(l) => (
                &l.mlp,
                &l.pre_feedforward_norm,
                &l.post_feedforward_norm,
                &l.layer_scalar,
            ),
            Layer::Full(l) => (
                &l.mlp,
                &l.pre_feedforward_norm,
                &l.post_feedforward_norm,
                &l.layer_scalar,
            ),
        };
        let mut mark = scratch.profile_mark();
        if blocked_residual {
            let [a, b, c, d] = &self.rows;
            match layer {
                Layer::Sliding(_) => launch(crate::ops_norm_blocked_chain::post_attention_sliding,
                    (device, &self.output_s[0], &self.output_s[1], &self.output_s[2], &self.output_s[3],
                     post_attention, pre, &mut self.residual_blocked,
                     &mut self.q_input, &mut self.input_scale)).await?,
                Layer::Full(_) => launch(crate::ops_norm_blocked_chain::post_attention_full,
                    (device, &a.projected, &b.projected, &c.projected, &d.projected,
                     post_attention, pre, &mut self.residual_blocked,
                     &mut self.q_input, &mut self.input_scale)).await?,
            }
            scratch.record_optional_profile("group.ffn.blocked_post_pre_norm_quant", &mut mark);
        } else if state.fused_norms {
            let [a, b, c, d] = &mut self.rows;
            if state.blocked_output && matches!(layer, Layer::Sliding(_)) {
                if state.shard_quant {
                    if state.quarter_norm {
                    launch(crate::ops_norm_quarters::post_attention_ffn_input_four_blocked,
                        (device, &self.output_s[0], &self.output_s[1], &self.output_s[2], &self.output_s[3],
                         post_attention, pre, &mut a.x, &mut b.x, &mut c.x, &mut d.x,
                         &mut self.q_input, &mut self.input_scale)).await?;
                    } else {
                    launch(crate::ops_norm_shardquant::post_attention_ffn_input_four_blocked,
                        (device, &self.output_s[0], &self.output_s[1], &self.output_s[2], &self.output_s[3],
                         post_attention, pre, &mut a.x, &mut b.x, &mut c.x, &mut d.x,
                         &mut self.q_input, &mut self.input_scale)).await?;
                    }
                } else {
                launch(crate::ops_norm_fused::post_attention_ffn_input_four_blocked,
                    (device, &self.output_s[0], &self.output_s[1], &self.output_s[2], &self.output_s[3],
                     post_attention, pre, &mut a.x, &mut b.x, &mut c.x, &mut d.x,
                     &mut self.q_input, &mut self.input_scale)).await?;
                }
            } else {
            launch(crate::ops_norm_fused::post_attention_ffn_input_four,
                (device, &a.projected, &b.projected, &c.projected, &d.projected,
                 post_attention, pre, &mut a.x, &mut b.x, &mut c.x, &mut d.x,
                 &mut self.q_input, &mut self.input_scale)).await?;
            }
            scratch.record_optional_profile("group.ffn.fused_norm_quantize", &mut mark);
        } else {
            for row in &mut self.rows {
                row.post_attention(device, post_attention, scratch).await?;
            }
            scratch.record_optional_profile("group.attention.post_norm", &mut mark);
            for row in &mut self.rows {
                row.normalize(device, pre, scratch).await?;
            }
            scratch.record_optional_profile("group.ffn.pre_norm", &mut mark);
            launch(
                group_io::pack_hidden_group,
                (
                    device,
                    &self.rows[0].norm,
                    &self.rows[1].norm,
                    &self.rows[2].norm,
                    &self.rows[3].norm,
                    &self.zero,
                    &self.second,
                    &self.third,
                    &self.fourth,
                    &mut self.norm,
                ),
            )
            .await?;
            scratch.record_optional_profile("group.ffn.pack", &mut mark);
            launch(
                crate::ops_native_fp8_p512_blocks::quantize_hidden,
                (device, &self.norm, &mut self.q_input, &mut self.input_scale),
            )
            .await?;
            scratch.record_optional_profile("group.ffn.quantize_hidden", &mut mark);
        }
        if state.fused_ffn_core {
            launch(crate::ops_native_fp8_p512_core_fused::ffn_core,
                (device, &self.q_input, &self.input_scale,
                 &mlp.native_fp8.up, &mlp.native_fp8.gate, &mlp.native_fp8.down,
                 &mlp.native_fp8.up_dequant, &mlp.native_fp8.gate_dequant, &mlp.native_fp8.down_dequant,
                 &mlp.up_global_scale, &mlp.gate_global_scale, &mlp.down_global_scale,
                 &mut self.up, &mut self.gate, &mut self.q_hidden, &mut self.hidden_scale,
                 &mut self.projected)).await?;
            scratch.record_optional_profile("group.ffn.core", &mut mark);
        } else {
        launch(
            crate::ops_native_fp8_p512_ingress::project_up_scaled,
            (device, &self.q_input, &self.input_scale, &mlp.native_fp8.up,
             &mlp.native_fp8.up_dequant, &mut self.up),
        ).await?;
        scratch.record_optional_profile("group.ffn.up", &mut mark);
        launch(
            crate::ops_native_fp8_p512_ingress::project_up_scaled,
            (device, &self.q_input, &self.input_scale, &mlp.native_fp8.gate,
             &mlp.native_fp8.gate_dequant, &mut self.gate),
        ).await?;
        scratch.record_optional_profile("group.ffn.gate", &mut mark);
        launch(
            crate::ops_native_fp8_p512_blocks_sharded::geglu_quantize,
            (device, &self.up, &self.gate, &mlp.up_global_scale,
             &mlp.gate_global_scale, &mut self.q_hidden, &mut self.hidden_scale),
        ).await?;
        scratch.record_optional_profile("group.ffn.geglu_quantize", &mut mark);
        launch(
            crate::ops_native_fp8_p512_pipeline::project_down_blocked,
            (
                device,
                &self.q_hidden,
                &self.hidden_scale,
                &mlp.native_fp8.down,
                &mlp.native_fp8.down_dequant,
                &mlp.down_global_scale,
                &mut self.projected,
            ),
        )
        .await?;
        scratch.record_optional_profile("group.ffn.down", &mut mark);
        }
        if blocked_residual {
            launch(crate::ops_norm_blocked_residual::post_feedforward,
                (device, &self.projected, post, gate, &mut self.residual_blocked)).await?;
            scratch.record_optional_profile("group.ffn.blocked_post_norm", &mut mark);
        } else if state.fused_norms {
            let [a, b, c, d] = &mut self.rows;
            if state.quarter_postnorm {
                launch(crate::ops_norm_quarter_post::post_feedforward_four,
                    (device, &self.projected, post, gate,
                     &mut a.x, &mut b.x, &mut c.x, &mut d.x)).await?;
            } else {
            launch(crate::ops_norm_fused::post_feedforward_four,
                (device, &self.projected, post, gate,
                 &mut a.x, &mut b.x, &mut c.x, &mut d.x)).await?;
            }
            scratch.record_optional_profile("group.ffn.fused_post_norm", &mut mark);
        } else {
            let [a, b, c, d] = &mut self.rows;
            launch(
                crate::ops_native_fp8_p512_downblocked::unpack_hidden_group,
                (
                    device,
                    &self.projected,
                    &mut a.projected,
                    &mut b.projected,
                    &mut c.projected,
                    &mut d.projected,
                ),
            )
            .await?;
            scratch.record_optional_profile("group.ffn.unpack", &mut mark);
            for row in &mut self.rows {
                launch(
                    crate::ops_norm_wide::post_feedforward_p128,
                    (device, &row.projected, post, gate, &mut row.x),
                )
                .await?;
            }
            scratch.record_optional_profile("group.ffn.post_norm", &mut mark);
        }
        if blocked_residual && index + 1 == weights.layers.len() {
            // Restore all canonical quarters once before any owner-specific finish.
            let [a, b, c, d] = &mut self.rows;
            launch(crate::ops_norm_blocked_residual::unpack_four,
                (device, &self.residual_blocked, &mut a.x, &mut b.x, &mut c.x, &mut d.x)).await?;
            scratch.record_optional_profile("group.residual.unpack", &mut mark);
        }
        for current in &mut state.states {
            current.next_layer += 1;
        }
        Ok(state.states[0].next_layer == weights.layers.len())
    }
    pub(crate) async fn finish(
        &mut self,
        row: usize,
        device: &mut Device,
        weights: &Weights,
        state: &PreparedGroup,
        ws: &mut Workspace,
        scratch: &mut Prefill,
    ) -> Result<()> {
        ws.blocks.clone_from(&state.states[row].blocks);
        self.rows[row]
            .finish(device, weights, &state.states[row], ws, scratch)
            .await
    }
}

/// Caller has validated exact lengths, alignment, checked ends and independent owners.
fn prompt384128_chunks<'a>(sequences: &[Sequence<'a>]) -> Vec<Sequence<'a>> {
    prompt_pair512_chunks(sequences)
}

fn prompt_pair512_chunks<'a>(sequences: &[Sequence<'a>]) -> Vec<Sequence<'a>> {
    sequences.iter().flat_map(|sequence| {
        sequence.inputs.chunks_exact(P128::SIZE).enumerate().map(move |(quarter, inputs)| Sequence {
            inputs,
            start_position: sequence.start_position + quarter * P128::SIZE,
            blocks: sequence.blocks,
        })
    }).collect()
}

#[cfg(test)]
mod prompt512_tests {
    use super::*;
    use crate::host::model::Input;
    fn input(logits: bool) -> Input<'static> {
        Input {
            token: 2,
            soft: None,
            logits,
            image: None,
            audio: None,
        }
    }
    #[test]
    fn same_owner_chronological_chunks_have_a_separate_validation_contract() {
        let mut inputs: Vec<_> = (0..512).map(|p| input(p == 511)).collect();
        let blocks = runtime::Blocks {
            sliding: 63,
            full: vec![255, 127, 7],
        };
        for start in [0, 384, 896] {
            let sequence = Sequence {
                inputs: &inputs,
                start_position: start,
                blocks: &blocks,
            };
            assert!(GroupPrefill::supports_prompt512(&sequence));
            let chunks: Vec<_> = inputs
                .chunks_exact(128)
                .enumerate()
                .map(|(i, rows)| Sequence {
                    inputs: rows,
                    start_position: start + 128 * i,
                    blocks: &blocks,
                })
                .collect();
            assert!(chunks.iter().all(WidePrefill::supports));
            assert!(
                !GroupPrefill::supports(&chunks),
                "independent-request validation must still reject a shared allocation"
            );
        }
        for (start, count) in [(1, 512), (384, 511), (0, 128), (usize::MAX - 127, 512)] {
            assert!(!GroupPrefill::supports_prompt512(&Sequence {
                inputs: &inputs[..count],
                start_position: start,
                blocks: &blocks,
            }));
        }
        inputs[127].logits = true;
        assert!(!GroupPrefill::supports_prompt512(&Sequence {
            inputs: &inputs,
            start_position: 0,
            blocks: &blocks
        }));
    }
}

#[cfg(test)]
mod prompt384128_tests {
    use super::*;
    use crate::host::model::Input;
    fn inputs(count: usize, base: usize) -> Vec<Input<'static>> {
        (0..count).map(|i| Input {token:base+i,soft:None,logits:i+1==count,image:None,audio:None}).collect()
    }
    fn allocations() -> [runtime::Blocks;2] {
        [runtime::Blocks{sliding:3,full:vec![255,9,13,23,42]},
         runtime::Blocks{sliding:61,full:vec![254,8,12,22,41]}]
    }
    #[test]
    fn all_pair_partitions_preserve_owner_order_and_final_quarters() {
        for count in [128,256,384] {
            let a=inputs(count,100);let b=inputs(512-count,1000);let blocks=allocations();
            let owners=[Sequence{inputs:&a,start_position:2048-count,blocks:&blocks[0]},
                Sequence{inputs:&b,start_position:2048-(512-count),blocks:&blocks[1]}];
            assert!(GroupPrefill::supports_prompt_pair512(&owners));
            assert_eq!(GroupPrefill::supports_prompt384128(&owners),count==384);
            let quarters=prompt_pair512_chunks(&owners);assert_eq!(quarters.len(),4);
            let split=count/128;
            for (row,q) in quarters.iter().enumerate() {
                let owner=usize::from(row>=split);let local=if owner==0{row}else{row-split};
                assert_eq!(q.blocks.sliding,blocks[owner].sliding);
                assert_eq!(q.start_position,owners[owner].start_position+local*128);
                assert_eq!(q.inputs[0].token,if owner==0{100+128*local}else{1000+128*local});
                assert_eq!(q.inputs.last().unwrap().logits,row==split-1||row==3);
            }
            assert!(!GroupPrefill::supports(&quarters));
        }
    }
    #[test]
    fn generic_pair_rejects_incomplete_overfull_and_aliased_owners() {
        let data=inputs(512,100);let blocks=allocations();
        for (a,b) in [(0,512),(128,128),(128,256),(256,384),(384,384),(512,0)] {
            assert!(!GroupPrefill::supports_prompt_pair512(&[
                Sequence{inputs:&data[..a],start_position:0,blocks:&blocks[0]},
                Sequence{inputs:&data[..b],start_position:0,blocks:&blocks[1]}]));
        }
        for count in [128,256,384] {
            let a=inputs(count,100);let b=inputs(512-count,1000);
            assert!(!GroupPrefill::supports_prompt_pair512(&[
                Sequence{inputs:&a,start_position:0,blocks:&blocks[0]},
                Sequence{inputs:&b,start_position:0,blocks:&blocks[0]}]));
        }
    }
    #[test]
    fn two_owners_derive_three_chronological_a_quarters_and_one_b() {
        let a=inputs(384,100);let b=inputs(128,1000);let blocks=allocations();
        for [sa,sb] in [[0,0],[384,128],[896,384],[1664,1920]] {
            let owners=[Sequence{inputs:&a,start_position:sa,blocks:&blocks[0]},
                Sequence{inputs:&b,start_position:sb,blocks:&blocks[1]}];
            assert!(GroupPrefill::supports_prompt384128(&owners));
            let quarters=prompt384128_chunks(&owners);
            assert_eq!(quarters.len(),4);
            assert_eq!(quarters.iter().map(|s|s.start_position).collect::<Vec<_>>(),vec![sa,sa+128,sa+256,sb]);
            assert_eq!(quarters.iter().map(|s|s.inputs[0].token).collect::<Vec<_>>(),vec![100,228,356,1000]);
            assert_eq!(quarters.iter().map(|s|s.blocks.sliding).collect::<Vec<_>>(),vec![3,3,3,61]);
            assert_eq!(quarters.iter().map(|s|s.inputs.last().unwrap().logits).collect::<Vec<_>>(),vec![false,false,true,true]);
            assert!(quarters.iter().all(WidePrefill::supports));
            assert!(!GroupPrefill::supports(&quarters),"ordinary independent-quarter guard must reject repeated A");
        }
    }
    #[test]
    fn rejects_wrong_lengths_alignment_overflow_and_intermediate_outputs() {
        let mut a=inputs(384,100);let mut b=inputs(128,1000);let blocks=allocations();
        for (sa,sb,na,nb) in [(1,0,384,128),(0,1,384,128),(0,0,383,128),(0,0,384,127),
            (0,0,0,128),(usize::MAX-127,0,384,128),(0,usize::MAX-127,384,128)] {
            assert!(!GroupPrefill::supports_prompt384128(&[
                Sequence{inputs:&a[..na],start_position:sa,blocks:&blocks[0]},
                Sequence{inputs:&b[..nb],start_position:sb,blocks:&blocks[1]}]));
        }
        assert!(!GroupPrefill::supports_prompt384128(&[]));
        assert!(!GroupPrefill::supports_prompt384128(&[Sequence{inputs:&a,start_position:0,blocks:&blocks[0]}]));
        a[127].logits=true;
        assert!(!GroupPrefill::supports_prompt384128(&[Sequence{inputs:&a,start_position:0,blocks:&blocks[0]},Sequence{inputs:&b,start_position:0,blocks:&blocks[1]}]));
        a[127].logits=false;b[0].logits=true;
        assert!(!GroupPrefill::supports_prompt384128(&[Sequence{inputs:&a,start_position:0,blocks:&blocks[0]},Sequence{inputs:&b,start_position:0,blocks:&blocks[1]}]));
    }
    #[test]
    fn rejects_shared_rings_pages_and_duplicate_owner_pages() {
        let a=inputs(384,100);let b=inputs(128,1000);
        for mode in 0..3 {
            let mut blocks=allocations();
            match mode {0=>blocks[1].sliding=blocks[0].sliding,1=>blocks[1].full[0]=blocks[0].full[0],_=>blocks[0].full[1]=blocks[0].full[0]}
            assert!(!GroupPrefill::supports_prompt384128(&[Sequence{inputs:&a,start_position:0,blocks:&blocks[0]},Sequence{inputs:&b,start_position:0,blocks:&blocks[1]}]));
        }
    }
}
