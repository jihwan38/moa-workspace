//! Two independent full P128 prompts share only the FFN matrix pass.
//! Each request keeps its own chronological attention, hidden state, positions and KV.
use super::*;
use crate::ops_prompt_pair_io as pair_io;
pub(crate) struct PairPrefill {
    rows: [WidePrefill; 2],
    norm: HbmTensor<bf16, Chip, m![P256, H]>,
    projected: HbmTensor<bf16, Chip, m![P256, H]>,
    zero: HbmTensor<i32, Chip, m![1]>,
    second: HbmTensor<i32, Chip, m![1]>,
}
pub(crate) struct PreparedPair {
    states: [PreparedWide; 2],
}
impl PreparedPair {
    pub(crate) fn blocks(&self) -> [&runtime::Blocks; 2] {
        [&self.states[0].blocks, &self.states[1].blocks]
    }
}
impl PairPrefill {
    pub(crate) async fn new(device: &mut Device) -> Result<Self> {
        Ok(Self {
            rows: [
                WidePrefill::new(device).await?,
                WidePrefill::new(device).await?,
            ],
            norm: HostTensor::<bf16, m![P256, H]>::zero()
                .to_hbm(&mut device.pdma)
                .await?,
            projected: HostTensor::<bf16, m![P256, H]>::zero()
                .to_hbm(&mut device.pdma)
                .await?,
            zero: HostTensor::<i32, m![1]>::from_vec([0])
                .to_hbm(&mut device.pdma)
                .await?,
            second: HostTensor::<i32, m![1]>::from_vec([(P128::SIZE * H::SIZE * 2) as i32])
                .to_hbm(&mut device.pdma)
                .await?,
        })
    }
    pub(crate) fn supports(sequences: &[Sequence<'_>]) -> bool {
        sequences.len() == 2
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
    ) -> Result<PreparedPair> {
        if !Self::supports(sequences) {
            return Err("paired prefill needs two independent page-local128-token chunks".into());
        }
        Ok(PreparedPair {
            states: [
                self.rows[0]
                    .prepare(device, weights, &sequences[0], scratch, true)
                    .await?,
                self.rows[1]
                    .prepare(device, weights, &sequences[1], scratch, true)
                    .await?,
            ],
        })
    }
    pub(crate) async fn advance_layer(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        state: &mut PreparedPair,
        ws: &mut Workspace,
        scratch: &mut Prefill,
    ) -> Result<bool> {
        let index = state.states[0].next_layer;
        if state.states[1].next_layer != index {
            return Err("paired layer cursors diverged".into());
        }
        let layer = weights
            .layers
            .get(index)
            .ok_or("paired prefill layers already complete")?;
        for row in 0..2 {
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
        let mut mark = std::time::Instant::now();
        for row in &mut self.rows {
            row.normalize(device, pre, scratch).await?;
        }
        scratch.record_profile("pair.ffn.pre_norm", &mut mark);
        launch(
            pair_io::pack_hidden_pair,
            (
                device,
                &self.rows[0].norm,
                &self.rows[1].norm,
                &self.zero,
                &self.second,
                &mut self.norm,
            ),
        )
        .await?;
        scratch.record_profile("pair.ffn.pack", &mut mark);
        launch(
            crate::ops_native_fp8_p256::batch_ffn,
            (
                device,
                &self.norm,
                &mlp.native_fp8.up,
                &mlp.native_fp8.gate,
                &mlp.native_fp8.down,
                &mlp.native_fp8.up_dequant,
                &mlp.native_fp8.gate_dequant,
                &mlp.native_fp8.down_dequant,
                &mlp.up_global_scale,
                &mlp.gate_global_scale,
                &mlp.down_global_scale,
                &mut self.projected,
            ),
        )
        .await?;
        scratch.record_profile("pair.ffn.core", &mut mark);
        let [a, b] = &mut self.rows;
        launch(
            pair_io::unpack_hidden_pair,
            (
                device,
                &self.projected,
                &self.zero,
                &self.second,
                &mut a.projected,
                &mut b.projected,
            ),
        )
        .await?;
        scratch.record_profile("pair.ffn.unpack", &mut mark);
        for row in &mut self.rows {
            launch(
                crate::ops_norm_wide::post_feedforward_p128,
                (device, &row.projected, post, gate, &mut row.x),
            )
            .await?;
        }
        scratch.record_profile("pair.ffn.post_norm", &mut mark);
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
        state: &PreparedPair,
        ws: &mut Workspace,
        scratch: &mut Prefill,
    ) -> Result<()> {
        ws.blocks.clone_from(&state.states[row].blocks);
        self.rows[row]
            .finish(device, weights, &state.states[row], ws, scratch)
            .await
    }
}
