//! Four independent full P128 prompts share only the FFN matrix pass.
//! Each request keeps its own chronological attention, hidden state, positions and KV.
use super::*;
use crate::ops_prompt_group_io as group_io;
pub(crate) struct GroupPrefill {
    rows: [WidePrefill; 4],
    norm: HbmTensor<bf16, Chip, m![P512, H]>,
    projected: HbmTensor<bf16, Chip, m![P512, H]>,
    zero: HbmTensor<i32, Chip, m![1]>,
    second: HbmTensor<i32, Chip, m![1]>,
    third: HbmTensor<i32, Chip, m![1]>,
    fourth: HbmTensor<i32, Chip, m![1]>,
    q_input: HbmTensor<f8e4m3, Chip, m![P512, H]>,
    input_scale: HbmTensor<f32, Chip, m![P512]>,
    up: HbmTensor<bf16, Chip, m![P512, L]>,
    gate: HbmTensor<bf16, Chip, m![P512, L]>,
    q_hidden: HbmTensor<f8e4m3, Chip, m![P512, L]>,
    hidden_scale: HbmTensor<f32, Chip, m![P512]>,
}
pub(crate) struct PreparedGroup {
    states: [PreparedWide; 4],
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
            projected: zero!(bf16, m![P512, H]),
            zero: offset!(0),
            second: offset!(1),
            third: offset!(2),
            fourth: offset!(3),
            q_input: zero!(f8e4m3, m![P512, H]),
            input_scale: zero!(f32, m![P512]),
            up: zero!(bf16, m![P512, L]),
            gate: zero!(bf16, m![P512, L]),
            q_hidden: zero!(f8e4m3, m![P512, L]),
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
        for row in 0..4 {
            let current = &state.states[row];
            ws.blocks.clone_from(&current.blocks);
            match layer {
                Layer::Sliding(layer) => {
                    self.rows[row]
                        .sliding(
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
                        .full(
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
            crate::ops_native_fp8_p512::quantize_hidden,
            (device, &self.norm, &mut self.q_input, &mut self.input_scale),
        )
        .await?;
        scratch.record_optional_profile("group.ffn.quantize_hidden", &mut mark);
        launch(
            crate::ops_native_fp8_p512::project_up_scaled,
            (
                device,
                &self.q_input,
                &self.input_scale,
                &mlp.native_fp8.up,
                &mlp.native_fp8.up_dequant,
                &mut self.up,
            ),
        )
        .await?;
        scratch.record_optional_profile("group.ffn.up", &mut mark);
        launch(
            crate::ops_native_fp8_p512::project_up_scaled,
            (
                device,
                &self.q_input,
                &self.input_scale,
                &mlp.native_fp8.gate,
                &mlp.native_fp8.gate_dequant,
                &mut self.gate,
            ),
        )
        .await?;
        scratch.record_optional_profile("group.ffn.gate", &mut mark);
        launch(
            crate::ops_native_fp8_p512::geglu_quantize,
            (
                device,
                &self.up,
                &self.gate,
                &mlp.up_global_scale,
                &mlp.gate_global_scale,
                &mut self.q_hidden,
                &mut self.hidden_scale,
            ),
        )
        .await?;
        scratch.record_optional_profile("group.ffn.geglu_quantize", &mut mark);
        launch(
            crate::ops_native_fp8_p512::project_down_scaled,
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
            group_io::unpack_hidden_group,
            (
                device,
                &self.projected,
                &self.zero,
                &self.second,
                &self.third,
                &self.fourth,
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
