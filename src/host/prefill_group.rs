//! Four P128 row groups share QKV weight loads and a blocked P512 FFN pass.
//! Separate constructors preserve either independent request allocations or four
//! chronological chunks of one request; attention always visits rows in order.
use super::*;
use crate::ops_batch_layouts::{P512H960, P512L240, P512L960};
use crate::ops_prompt_group_io as group_io;
pub(crate) struct GroupPrefill {
    rows: [WidePrefill; 4],
    norm: HbmTensor<bf16, Chip, m![P512, H]>,
    projected: HbmTensor<bf16, Chip, crate::ops_native_fp8_p512_downblocked::Output>,
    zero: HbmTensor<i32, Chip, m![1]>,
    second: HbmTensor<i32, Chip, m![1]>,
    third: HbmTensor<i32, Chip, m![1]>,
    fourth: HbmTensor<i32, Chip, m![1]>,
    q_input: HbmTensor<f8e4m3, Chip, P512H960>,
    input_scale: HbmTensor<f32, Chip, m![P512]>,
    up: HbmTensor<bf16, Chip, P512L240>,
    q_hidden: HbmTensor<f8e4m3, Chip, P512L960>,
    hidden_scale: HbmTensor<f32, Chip, m![P512]>,
}
pub(crate) struct PreparedGroup {
    states: [PreparedWide; 4],
}
/// Four chronological chunks of one request. Only its single allocation may be
/// exposed to the scheduler; the four copies below are execution metadata.
pub(crate) struct PreparedPrompt512 {
    chunks: PreparedGroup,
}
impl PreparedPrompt512 {
    pub(crate) fn blocks(&self) -> &runtime::Blocks {
        &self.chunks.states[0].blocks
    }
}
impl PreparedGroup {
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
            rows: [
                WidePrefill::new(device).await?,
                WidePrefill::new(device).await?,
                WidePrefill::new(device).await?,
                WidePrefill::new(device).await?,
            ],
            norm: zero!(bf16, m![P512, H]),
            projected: zero!(bf16, crate::ops_native_fp8_p512_downblocked::Output),
            zero: offset!(0),
            second: offset!(1),
            third: offset!(2),
            fourth: offset!(3),
            q_input: zero!(f8e4m3, P512H960),
            input_scale: zero!(f32, m![P512]),
            up: zero!(bf16, P512L240),
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
    ) -> Result<PreparedGroup> {
        if !Self::supports(sequences) {
            return Err("grouped prefill needs four independent page-local128-token chunks".into());
        }
        self.prepare_rows(device, weights, sequences, scratch).await
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
            chunks: self.prepare_rows(device, weights, &chunks, scratch).await?,
        })
    }

    /// Only the two validated constructors above call this execution helper.
    async fn prepare_rows(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        sequences: &[Sequence<'_>],
        scratch: &mut Prefill,
    ) -> Result<PreparedGroup> {
        debug_assert_eq!(sequences.len(), 4);
        Ok(PreparedGroup {
            states: [
                self.rows[0]
                    .prepare(device, weights, &sequences[0], scratch, true)
                    .await?,
                self.rows[1]
                    .prepare(device, weights, &sequences[1], scratch, true)
                    .await?,
                self.rows[2]
                    .prepare(device, weights, &sequences[2], scratch, true)
                    .await?,
                self.rows[3]
                    .prepare(device, weights, &sequences[3], scratch, true)
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
        let input_norm = match layer {
            Layer::Sliding(l) => &l.input_norm,
            Layer::Full(l) => &l.input_norm,
        };
        for row in &mut self.rows {
            row.normalize(device, input_norm, scratch).await?;
            launch(
                crate::ops_qkv_native_four::quantize_input,
                (device, &row.norm, &mut row.qkv_input, &mut row.qkv_scale),
            )
            .await?;
        }
        scratch.record_optional_profile("group.qkv.norm_quantize", &mut mark);
        let [a, b, c, d] = &mut self.rows;
        macro_rules! project_four {
            ($kernel:path, $weight:expr, $scale:expr, $field:ident) => {
                launch(
                    $kernel,
                    (
                        device,
                        &a.qkv_input,
                        &a.qkv_scale,
                        &b.qkv_input,
                        &b.qkv_scale,
                        &c.qkv_input,
                        &c.qkv_scale,
                        &d.qkv_input,
                        &d.qkv_scale,
                        $weight,
                        $scale,
                        &mut a.$field,
                        &mut b.$field,
                        &mut c.$field,
                        &mut d.$field,
                    ),
                )
                .await?
            };
        }
        match layer {
            Layer::Sliding(l) => {
                project_four!(
                    crate::ops_qkv_native_four::project_query_four,
                    &l.q_weight,
                    &l.q_weight_scale,
                    q_s
                );
                project_four!(
                    crate::ops_qkv_native_four::project_sliding_kv_four,
                    &l.k_weight,
                    &l.k_weight_scale,
                    k_s
                );
                project_four!(
                    crate::ops_qkv_native_four::project_sliding_kv_four,
                    &l.v_weight,
                    &l.v_weight_scale,
                    v_s
                );
            }
            Layer::Full(l) => {
                project_four!(
                    crate::ops_qkv_native_four::project_full_query_four,
                    &l.q_weight,
                    &l.q_weight_scale,
                    q_f
                );
                project_four!(
                    crate::ops_qkv_native_four::project_full_key_four,
                    &l.k_weight,
                    &l.k_weight_scale,
                    k_f
                );
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
                        )
                        .await?
                }
            }
        }
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
        launch(
            crate::ops_native_fp8_p512_blocks_fused::project_upgate_geglu,
            (device, &self.q_input, &self.input_scale, &mlp.native_fp8.up,
             &mlp.native_fp8.gate, &mlp.native_fp8.up_dequant,
             &mlp.native_fp8.gate_dequant, &mlp.up_global_scale,
             &mlp.gate_global_scale, &mut self.up),
        ).await?;
        scratch.record_optional_profile("group.ffn.upgate_geglu", &mut mark);
        launch(
            crate::ops_native_fp8_p512_blocks_sharded::quantize_ffn,
            (device, &self.up, &mut self.q_hidden, &mut self.hidden_scale),
        ).await?;
        scratch.record_optional_profile("group.ffn.quantize_ffn", &mut mark);
        launch(
            crate::ops_native_fp8_p512_downblocked::project_down_scaled,
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
