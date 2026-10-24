//! P128 prompt matrix work with chronological B16 causal attention segments.
//! Wide and decode batches have independent axes and scratch ownership.
use super::*;
use crate::{ops_batch_kpart as linear, ops_prefill_tiles as tiles};

pub(crate) struct WidePrefill {
    x: HbmTensor<bf16, Chip, m![P128, H]>,
    norm: HbmTensor<bf16, Chip, m![P128, H]>,
    projected: HbmTensor<bf16, Chip, m![P128, H]>,
    q_s: HbmTensor<bf16, Chip, m![P128, Qs]>,
    k_s: HbmTensor<bf16, Chip, m![P128, Ps]>,
    v_s: HbmTensor<bf16, Chip, m![P128, Ps]>,
    attn_s: HbmTensor<bf16, Chip, m![P128, Ns, Gs, Ds]>,
    q_f: HbmTensor<bf16, Chip, m![P128, Qf]>,
    k_f: HbmTensor<bf16, Chip, m![P128, Pf]>,
    attn_f: HbmTensor<bf16, Chip, m![P128, Gf, Df]>,
    tile_indices: Vec<HbmTensor<i32, Chip, m![1]>>,
}
/// Everything needed after embedding preparation is owned; no request/media
/// references survive a scheduling yield. Wide tensors remain in WidePrefill.
pub(crate) struct PreparedWide {
    pub(crate) blocks: runtime::Blocks,
    start_position: usize,
    count: usize,
    logits: bool,
    next_layer: usize,
    segments: Vec<Segment>,
}
struct Segment {
    rope_s: HbmTensor<i32, Chip, m![B]>,
    rope_f: HbmTensor<i32, Chip, m![B]>,
    old_mask: HbmTensor<f32, Chip, m![B, Ts]>,
    chunk_mask: HbmTensor<f32, Chip, m![B, Bk]>,
    full_mask: HbmTensor<f32, Chip, m![B, Tf]>,
    commits: CommitOffsets,
}

impl WidePrefill {
    pub(crate) async fn new(device: &mut Device) -> Result<Self> {
        macro_rules! zero {
            ($mapping:ty) => {
                HostTensor::<bf16, $mapping>::zero()
                    .to_hbm(&mut device.pdma)
                    .await?
            };
        }
        let mut tile_indices = Vec::new();
        for tile in 0..P128::SIZE / B::SIZE {
            tile_indices.push(
                HostTensor::<i32, m![1]>::from_vec([tile as i32])
                    .to_hbm(&mut device.pdma)
                    .await?,
            );
        }
        Ok(Self {
            x: zero!(m![P128, H]),
            norm: zero!(m![P128, H]),
            projected: zero!(m![P128, H]),
            q_s: zero!(m![P128, Qs]),
            k_s: zero!(m![P128, Ps]),
            v_s: zero!(m![P128, Ps]),
            attn_s: zero!(m![P128, Ns, Gs, Ds]),
            q_f: zero!(m![P128, Qf]),
            k_f: zero!(m![P128, Pf]),
            attn_f: zero!(m![P128, Gf, Df]),
            tile_indices,
        })
    }
    pub(crate) fn supports(sequence: &Sequence<'_>) -> bool {
        let count = sequence.inputs.len();
        count > B::SIZE
            && count <= P128::SIZE
            && sequence.start_position / Tf::SIZE
                == (sequence.start_position + count - 1) / Tf::SIZE
    }
    pub(crate) async fn prepare(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        sequence: &Sequence<'_>,
        scratch: &mut Prefill,
    ) -> Result<PreparedWide> {
        if !Self::supports(sequence) {
            return Err("unsupported P128 prompt shape".into());
        }
        let mut mark = std::time::Instant::now();
        let count = sequence.inputs.len();
        for tile in 0..self.tile_indices.len() {
            let offsets = HostTensor::<i32, m![B]>::from_vec(
                (0..B::SIZE)
                    .map(|row| {
                        (sequence.inputs[(tile * B::SIZE + row).min(count - 1)].token * H::SIZE * 2)
                            as i32
                    })
                    .collect::<Vec<_>>(),
            )
            .to_hbm(&mut device.pdma)
            .await?;
            launch(
                ops_batch_io::embed_tokens,
                (device, &weights.embedding_table, &offsets, &mut scratch.x),
            )
            .await?;
            for row in 0..B::SIZE {
                let Some(input) = sequence.inputs.get(tile * B::SIZE + row) else {
                    break;
                };
                let embedding = match input.soft {
                    Some(SoftToken::Image(index)) => {
                        let image = input.image.ok_or("image is missing")?;
                        runtime::encode_vision_patch(
                            device,
                            weights,
                            &image.pixels[index],
                            image.positions[index],
                        )
                        .await?
                    }
                    Some(SoftToken::Audio(index)) => {
                        let audio = input.audio.ok_or("audio is missing")?;
                        runtime::encode_audio_frame(device, weights, &audio.frames[index]).await?
                    }
                    None => continue,
                };
                let offset = HostTensor::<i32, m![1]>::from_vec([(row * H::SIZE * 2) as i32])
                    .to_hbm(&mut device.pdma)
                    .await?;
                launch(
                    ep::insert_hidden_row,
                    (device, &embedding, &offset, &mut scratch.x),
                )
                .await?;
            }
            launch(
                tiles::store_hidden_tile,
                (device, &scratch.x, &self.tile_indices[tile], &mut self.x),
            )
            .await?;
        }
        let mut segments = Vec::new();
        for tile in 0..count.div_ceil(B::SIZE) {
            let start = sequence.start_position + tile * B::SIZE;
            let valid = (count - tile * B::SIZE).min(B::SIZE);
            let (old, chunk) = sliding_masks(start, valid);
            segments.push(Segment {
                rope_s: HostTensor::<i32, m![B]>::from_vec(
                    (0..B::SIZE)
                        .map(|row| ((start + row.min(valid - 1)) * Ds::SIZE * 2) as i32)
                        .collect::<Vec<_>>(),
                )
                .to_hbm(&mut device.pdma)
                .await?,
                rope_f: HostTensor::<i32, m![B]>::from_vec(
                    (0..B::SIZE)
                        .map(|row| ((start + row.min(valid - 1)) * Df::SIZE * 2) as i32)
                        .collect::<Vec<_>>(),
                )
                .to_hbm(&mut device.pdma)
                .await?,
                old_mask: HostTensor::<f32, m![B, Ts]>::from_vec(old)
                    .to_hbm(&mut device.pdma)
                    .await?,
                chunk_mask: HostTensor::<f32, m![B, Bk]>::from_vec(chunk)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full_mask: HostTensor::<f32, m![B, Tf]>::from_vec(full_mask(start, valid))
                    .to_hbm(&mut device.pdma)
                    .await?,
                commits: CommitOffsets::new(device, start, valid).await?,
            });
        }
        scratch.record_profile("wide.prepare", &mut mark);
        Ok(PreparedWide {
            blocks: sequence.blocks.clone(),
            start_position: sequence.start_position,
            count,
            logits: sequence.inputs.last().unwrap().logits,
            next_layer: 0,
            segments,
        })
    }

    /// A complete layer is the scheduling quantum. B16 temporary tensors may be
    /// overwritten by another request after this returns; persistent P128 x and
    /// the request's own KV are sufficient to resume the next layer.
    pub(crate) async fn advance_layer(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        state: &mut PreparedWide,
        ws: &mut Workspace,
        scratch: &mut Prefill,
    ) -> Result<bool> {
        let index = state.next_layer;
        let layer = weights
            .layers
            .get(index)
            .ok_or("prefill layers already complete")?;
        match layer {
            Layer::Sliding(layer) => {
                self.sliding(device, ws, index, layer, state, &state.segments, scratch)
                    .await?
            }
            Layer::Full(layer) => {
                self.full(device, ws, index, layer, state, &state.segments, scratch)
                    .await?
            }
        }
        state.next_layer += 1;
        Ok(state.next_layer == weights.layers.len())
    }

    pub(crate) async fn finish(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        state: &PreparedWide,
        ws: &mut Workspace,
        scratch: &mut Prefill,
    ) -> Result<()> {
        if state.next_layer != weights.layers.len() {
            return Err("prefill logits requested before all layers finished".into());
        }
        let count = state.count;
        let mut mark = std::time::Instant::now();
        if state.logits {
            let tile = (count - 1) / B::SIZE;
            launch(
                tiles::load_hidden_tile,
                (device, &self.x, &self.tile_indices[tile], &mut scratch.x),
            )
            .await?;
            let offset =
                HostTensor::<i32, m![1]>::from_vec(
                    [(((count - 1) % B::SIZE) * H::SIZE * 2) as i32],
                )
                .to_hbm(&mut device.pdma)
                .await?;
            launch(
                ep::extract_hidden_row,
                (device, &scratch.x, &offset, &mut scratch.last),
            )
            .await?;
            launch(
                ops::final_norm_and_logits,
                (
                    device,
                    &scratch.last,
                    &weights.final_norm,
                    &weights.embedding_table,
                    &mut ws.logits,
                ),
            )
            .await?;
        }
        scratch.record_profile("wide.logits", &mut mark);
        Ok(())
    }

    pub(crate) async fn run(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        sequence: &Sequence<'_>,
        ws: &mut Workspace,
        scratch: &mut Prefill,
    ) -> Result<()> {
        let mut state = self.prepare(device, weights, sequence, scratch).await?;
        while !self
            .advance_layer(device, weights, &mut state, ws, scratch)
            .await?
        {}
        self.finish(device, weights, &state, ws, scratch).await
    }

    async fn normalize(
        &mut self,
        device: &mut Device,
        weight: &HbmTensor<bf16, Chip, m![H]>,
        _scratch: &mut Prefill,
    ) -> Result<()> {
        launch(
            crate::ops_norm_wide::rmsnorm_p128,
            (device, &self.x, weight, &mut self.norm),
        )
        .await?;
        Ok(())
    }
    async fn post_attention(
        &mut self,
        device: &mut Device,
        weight: &HbmTensor<bf16, Chip, m![H]>,
        _scratch: &mut Prefill,
    ) -> Result<()> {
        launch(
            crate::ops_norm_wide::post_attention_p128,
            (device, &self.projected, weight, &mut self.x),
        )
        .await?;
        Ok(())
    }
    async fn ffn(
        &mut self,
        device: &mut Device,
        weights: &MlpWeights,
        pre: &HbmTensor<bf16, Chip, m![H]>,
        post: &HbmTensor<bf16, Chip, m![H]>,
        gate: &HbmTensor<bf16, Chip, m![1 # 8]>,
        scratch: &mut Prefill,
    ) -> Result<()> {
        let mut mark = std::time::Instant::now();
        self.normalize(device, pre, scratch).await?;
        scratch.record_profile("wide.ffn.pre_norm", &mut mark);
        launch(
            crate::ops_native_fp8::batch_ffn,
            (
                device,
                &self.norm,
                &weights.native_fp8.up,
                &weights.native_fp8.gate,
                &weights.native_fp8.down,
                &weights.native_fp8.up_dequant,
                &weights.native_fp8.gate_dequant,
                &weights.native_fp8.down_dequant,
                &weights.up_global_scale,
                &weights.gate_global_scale,
                &weights.down_global_scale,
                &mut self.projected,
            ),
        )
        .await?;
        scratch.record_profile("wide.ffn.core", &mut mark);
        launch(
            crate::ops_norm_wide::post_feedforward_p128,
            (device, &self.projected, post, gate, &mut self.x),
        )
        .await?;
        scratch.record_profile("wide.ffn.post_norm", &mut mark);
        Ok(())
    }

    async fn sliding(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        index: usize,
        layer: &SlidingLayer,
        state: &PreparedWide,
        segments: &[Segment],
        scratch: &mut Prefill,
    ) -> Result<()> {
        let mut mark = std::time::Instant::now();
        self.normalize(device, &layer.input_norm, scratch).await?;
        scratch.record_profile("wide.sliding.input_norm", &mut mark);
        launch(
            linear::project_query_batch,
            (
                device,
                &self.norm,
                &layer.q_weight,
                &layer.q_weight_scale,
                &mut self.q_s,
            ),
        )
        .await?;
        launch(
            linear::project_sliding_kv_batch,
            (
                device,
                &self.norm,
                &layer.k_weight,
                &layer.k_weight_scale,
                &mut self.k_s,
            ),
        )
        .await?;
        launch(
            linear::project_sliding_kv_batch,
            (
                device,
                &self.norm,
                &layer.v_weight,
                &layer.v_weight_scale,
                &mut self.v_s,
            ),
        )
        .await?;
        scratch.record_profile("wide.sliding.qkv", &mut mark);
        let ring = state.blocks.sliding;
        let (k, v) = ws.kv_cache[index].as_sliding_mut();
        for (tile, segment) in segments.iter().enumerate() {
            let tile = &self.tile_indices[tile];
            launch(
                crate::ops_head_wide_tile::sliding_heads_from_wide_tile,
                (
                    device,
                    &self.q_s,
                    &self.k_s,
                    &self.v_s,
                    tile,
                    &layer.q_norm,
                    &layer.k_norm,
                    &segment.rope_s,
                    &ws.cos_s,
                    &ws.sin_s,
                    &mut scratch.q_s,
                    &mut scratch.k_s,
                    &mut scratch.v_s,
                ),
            )
            .await?;
            launch(
                crate::ops_sliding_prompt_clusters::sliding_attention_b16,
                (
                    device,
                    &scratch.q_s,
                    &k[ring],
                    &v[ring],
                    &scratch.k_s,
                    &scratch.v_s,
                    &segment.old_mask,
                    &segment.chunk_mask,
                    &mut scratch.attn_s,
                ),
            )
            .await?;
            segment
                .commits
                .sliding(
                    device,
                    &scratch.k_s,
                    &scratch.v_s,
                    &mut k[ring],
                    &mut v[ring],
                )
                .await?;
            launch(
                tiles::store_heads_sliding_tile,
                (device, &scratch.attn_s, tile, &mut self.attn_s),
            )
            .await?;
        }
        scratch.record_profile("wide.sliding.head_attention_tiles", &mut mark);
        launch(
            linear::project_sliding_output_batch,
            (
                device,
                &self.attn_s,
                &layer.o_weight,
                &layer.o_weight_scale,
                &mut self.projected,
            ),
        )
        .await?;
        scratch.record_profile("wide.sliding.output", &mut mark);
        self.post_attention(device, &layer.post_attention_norm, scratch)
            .await?;
        scratch.record_profile("wide.sliding.post_attention", &mut mark);
        self.ffn(
            device,
            &layer.mlp,
            &layer.pre_feedforward_norm,
            &layer.post_feedforward_norm,
            &layer.layer_scalar,
            scratch,
        )
        .await
    }
    async fn full(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        index: usize,
        layer: &FullLayer,
        state: &PreparedWide,
        segments: &[Segment],
        scratch: &mut Prefill,
    ) -> Result<()> {
        let mut mark = std::time::Instant::now();
        self.normalize(device, &layer.input_norm, scratch).await?;
        scratch.record_profile("wide.full.input_norm", &mut mark);
        launch(
            linear::project_full_query_batch,
            (
                device,
                &self.norm,
                &layer.q_weight,
                &layer.q_weight_scale,
                &mut self.q_f,
            ),
        )
        .await?;
        launch(
            linear::project_full_key_batch,
            (
                device,
                &self.norm,
                &layer.k_weight,
                &layer.k_weight_scale,
                &mut self.k_f,
            ),
        )
        .await?;
        scratch.record_profile("wide.full.qkv", &mut mark);
        let page = state.start_position / Tf::SIZE;
        let blocks = &state.blocks.full;
        let (k, v) = ws.kv_cache[index].as_full_mut();
        for (tile, segment) in segments.iter().enumerate() {
            let tile = &self.tile_indices[tile];
            launch(
                crate::ops_head_wide_tile::full_heads_from_wide_tile,
                (
                    device,
                    &self.q_f,
                    &self.k_f,
                    tile,
                    &layer.q_norm,
                    &layer.k_norm,
                    &segment.rope_f,
                    &ws.cos_f,
                    &ws.sin_f,
                    &mut scratch.q_f,
                    &mut scratch.k_f,
                    &mut scratch.v_f,
                ),
            )
            .await?;
            segment
                .commits
                .full(
                    device,
                    &scratch.k_f,
                    &scratch.v_f,
                    &mut k[blocks[page]],
                    &mut v[blocks[page]],
                )
                .await?;
            for current in 0..=page {
                let mask = if current == page {
                    &segment.full_mask
                } else {
                    &scratch.full_line_mask
                };
                let args = (
                    &mut *device,
                    &scratch.q_f,
                    &k[blocks[current]],
                    &v[blocks[current]],
                    mask,
                    &mut scratch.max_f,
                    &mut scratch.sum_f,
                    &mut scratch.attn_f,
                );
                if current == 0 {
                    launch(ops_attention::full_attention_first_page_b16, args).await?;
                } else {
                    launch(ops_attention::full_attention_next_page_b16, args).await?;
                }
            }
            launch(
                ops_attention::full_attention_divide_b16,
                (
                    device,
                    &scratch.attn_f,
                    &scratch.sum_f,
                    &mut scratch.divided_f,
                ),
            )
            .await?;
            launch(
                tiles::store_heads_full_tile,
                (device, &scratch.divided_f, tile, &mut self.attn_f),
            )
            .await?;
        }
        scratch.record_profile("wide.full.head_attention_tiles", &mut mark);
        launch(
            linear::project_full_output_batch,
            (
                device,
                &self.attn_f,
                &layer.o_weight,
                &layer.o_weight_scale,
                &mut self.projected,
            ),
        )
        .await?;
        scratch.record_profile("wide.full.output", &mut mark);
        self.post_attention(device, &layer.post_attention_norm, scratch)
            .await?;
        scratch.record_profile("wide.full.post_attention", &mut mark);
        self.ffn(
            device,
            &layer.mlp,
            &layer.pre_feedforward_norm,
            &layer.post_feedforward_norm,
            &layer.layer_scalar,
            scratch,
        )
        .await
    }
}
