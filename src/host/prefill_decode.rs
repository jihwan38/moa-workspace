//! Independent request rows share matrix-weight reads; attention addresses each
//! request's existing KV allocation and never packs or reuses another request's KV.
use super::super::model::Logits;
use super::*;
use crate::ops_decode;

struct RowOffsets {
    row: HbmTensor<i32, Chip, m![1]>,
    kv_s: HbmTensor<i32, Chip, m![1]>,
    kv_f: HbmTensor<i32, Chip, m![1]>,
    logits: HbmTensor<i32, Chip, m![1]>,
}

impl Prefill {
    pub(crate) async fn decode_batch(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        sequences: &[Sequence<'_>],
        ws: &mut Workspace,
        logits: &mut [Logits],
        output_rows: &[usize],
    ) -> Result<()> {
        let count = sequences.len();
        if count < 2
            || count > B::SIZE
            || output_rows.len() != count
            || output_rows.iter().any(|&row| row >= logits.len())
            || sequences.iter().any(|s| s.inputs.len() != 1)
        {
            return Err("invalid independent-request batch".into());
        }
        let token_offsets = HostTensor::<i32, m![B]>::from_vec(
            (0..B::SIZE)
                .map(|r| (sequences[r.min(count - 1)].inputs[0].token * H::SIZE * 2) as i32)
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await?;
        launch(
            ops_batch_io::embed_tokens,
            (
                device,
                &weights.embedding_table,
                &token_offsets,
                &mut self.x,
            ),
        )
        .await?;
        let mut offsets = Vec::with_capacity(count);
        for (row, sequence) in sequences.iter().enumerate() {
            let input = &sequence.inputs[0];
            let embedding = match input.soft {
                Some(SoftToken::Image(index)) => {
                    let image = input.image.ok_or("image is missing")?;
                    Some(
                        runtime::encode_vision_patch(
                            device,
                            weights,
                            &image.pixels[index],
                            image.positions[index],
                        )
                        .await?,
                    )
                }
                Some(SoftToken::Audio(index)) => {
                    let audio = input.audio.ok_or("audio is missing")?;
                    Some(runtime::encode_audio_frame(device, weights, &audio.frames[index]).await?)
                }
                None => None,
            };
            if let Some(embedding) = embedding {
                let offset = HostTensor::<i32, m![1]>::from_vec([(row * H::SIZE * 2) as i32])
                    .to_hbm(&mut device.pdma)
                    .await?;
                launch(
                    ep::insert_hidden_row,
                    (device, &embedding, &offset, &mut self.x),
                )
                .await?;
            }
            let position = sequence.start_position;
            offsets.push(RowOffsets {
                row: HostTensor::<i32, m![1]>::from_vec([row as i32])
                    .to_hbm(&mut device.pdma)
                    .await?,
                kv_s: HostTensor::<i32, m![1]>::from_vec([((position % Ts::SIZE)
                    * Ns::SIZE
                    * Ds::SIZE
                    * 2) as i32])
                .to_hbm(&mut device.pdma)
                .await?,
                kv_f: HostTensor::<i32, m![1]>::from_vec([
                    ((position % Tf::SIZE) * Df::SIZE * 2) as i32
                ])
                .to_hbm(&mut device.pdma)
                .await?,
                logits: HostTensor::<i32, m![1]>::from_vec([(row * W::SIZE * 2) as i32])
                    .to_hbm(&mut device.pdma)
                    .await?,
            });
        }
        let rope_s = HostTensor::<i32, m![B]>::from_vec(
            (0..B::SIZE)
                .map(|r| (sequences[r.min(count - 1)].start_position * Ds::SIZE * 2) as i32)
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await?;
        let rope_f = HostTensor::<i32, m![B]>::from_vec(
            (0..B::SIZE)
                .map(|r| (sequences[r.min(count - 1)].start_position * Df::SIZE * 2) as i32)
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await?;
        for (index, layer) in weights.layers.iter().enumerate() {
            match layer {
                Layer::Sliding(layer) => {
                    self.decode_sliding(device, ws, index, layer, &rope_s, sequences, &offsets)
                        .await?
                }
                Layer::Full(layer) => {
                    self.decode_full(device, ws, index, layer, &rope_f, sequences, &offsets)
                        .await?
                }
            }
        }
        if sequences.iter().any(|s| s.inputs[0].logits) {
            launch(
                ep::rmsnorm_b16,
                (device, &self.x, &weights.final_norm, &mut self.norm),
            )
            .await?;
            launch(
                ops_batch_io::logits_batch,
                (
                    device,
                    &self.norm,
                    &weights.embedding_table,
                    &mut self.logits_batch,
                ),
            )
            .await?;
            for (row, sequence) in sequences.iter().enumerate() {
                if sequence.inputs[0].logits {
                    launch(
                        ops_batch_io::extract_logits_row,
                        (
                            device,
                            &self.logits_batch,
                            &offsets[row].logits,
                            &mut logits[output_rows[row]],
                        ),
                    )
                    .await?;
                }
            }
        }
        Ok(())
    }

    async fn decode_sliding(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        index: usize,
        layer: &SlidingLayer,
        rope: &HbmTensor<i32, Chip, m![B]>,
        sequences: &[Sequence<'_>],
        offsets: &[RowOffsets],
    ) -> Result<()> {
        launch(
            ep::rmsnorm_b16,
            (device, &self.x, &layer.input_norm, &mut self.norm),
        )
        .await?;
        launch(
            ops_batch::project_query_batch,
            (
                device,
                &self.norm,
                &layer.q_weight,
                &layer.q_weight_scale,
                &mut self.q_raw_s,
            ),
        )
        .await?;
        launch(
            ops_batch::project_sliding_kv_batch,
            (
                device,
                &self.norm,
                &layer.k_weight,
                &layer.k_weight_scale,
                &mut self.k_raw_s,
            ),
        )
        .await?;
        launch(
            ops_batch::project_sliding_kv_batch,
            (
                device,
                &self.norm,
                &layer.v_weight,
                &layer.v_weight_scale,
                &mut self.v_raw_s,
            ),
        )
        .await?;
        launch(
            ep::sliding_heads_b16,
            (
                device,
                &self.q_raw_s,
                &self.k_raw_s,
                &self.v_raw_s,
                &layer.q_norm,
                &layer.k_norm,
                rope,
                &ws.cos_s,
                &ws.sin_s,
                &mut self.q_s,
                &mut self.k_s,
                &mut self.v_s,
            ),
        )
        .await?;
        let (k, v) = ws.kv_cache[index].as_sliding_mut();
        for (row, sequence) in sequences.iter().enumerate() {
            let ring = sequence.blocks.sliding;
            launch(
                ops_decode::sliding_attention_row,
                (
                    device,
                    &self.q_s,
                    &self.k_s,
                    &self.v_s,
                    &offsets[row].row,
                    &offsets[row].kv_s,
                    &mut k[ring],
                    &mut v[ring],
                    &ws.mask_s[sequence.start_position.min(Ts::SIZE - 1)],
                    &mut self.attn_s,
                ),
            )
            .await?;
        }
        launch(
            ops_batch::project_sliding_output_batch,
            (
                device,
                &self.attn_s,
                &layer.o_weight,
                &layer.o_weight_scale,
                &mut self.projected,
            ),
        )
        .await?;
        launch(
            ep::post_attention_b16,
            (
                device,
                &self.projected,
                &layer.post_attention_norm,
                &mut self.x,
            ),
        )
        .await?;
        self.ffn(
            device,
            &layer.mlp,
            &layer.pre_feedforward_norm,
            &layer.post_feedforward_norm,
            &layer.layer_scalar,
        )
        .await
    }

    async fn decode_full(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        index: usize,
        layer: &FullLayer,
        rope: &HbmTensor<i32, Chip, m![B]>,
        sequences: &[Sequence<'_>],
        offsets: &[RowOffsets],
    ) -> Result<()> {
        launch(
            ep::rmsnorm_b16,
            (device, &self.x, &layer.input_norm, &mut self.norm),
        )
        .await?;
        launch(
            ops_batch::project_full_query_batch,
            (
                device,
                &self.norm,
                &layer.q_weight,
                &layer.q_weight_scale,
                &mut self.q_raw_f,
            ),
        )
        .await?;
        launch(
            ops_batch::project_full_key_batch,
            (
                device,
                &self.norm,
                &layer.k_weight,
                &layer.k_weight_scale,
                &mut self.k_raw_f,
            ),
        )
        .await?;
        launch(
            ep::full_heads_b16,
            (
                device,
                &self.q_raw_f,
                &self.k_raw_f,
                &layer.q_norm,
                &layer.k_norm,
                rope,
                &ws.cos_f,
                &ws.sin_f,
                &mut self.q_f,
                &mut self.k_f,
                &mut self.v_f,
            ),
        )
        .await?;
        let (k, v) = ws.kv_cache[index].as_full_mut();
        for (row, sequence) in sequences.iter().enumerate() {
            let position = sequence.start_position;
            let page = position / Tf::SIZE;
            let blocks = &sequence.blocks.full;
            launch(
                ops_decode::full_commit_row,
                (
                    device,
                    &self.q_f,
                    &self.k_f,
                    &self.v_f,
                    &offsets[row].row,
                    &offsets[row].kv_f,
                    &mut k[blocks[page]],
                    &mut v[blocks[page]],
                    &mut self.decode_q_f,
                ),
            )
            .await?;
            for current in 0..=page {
                let mask = if current == page {
                    &ws.mask_f_diag[position % Tf::SIZE]
                } else {
                    &ws.mask_f_line
                };
                let args = (
                    &mut *device,
                    &self.decode_q_f,
                    &k[blocks[current]],
                    &v[blocks[current]],
                    mask,
                    &mut self.decode_max_f,
                    &mut self.decode_sum_f,
                    &mut self.decode_attn_f,
                );
                if current == 0 {
                    launch(ops::full_attention_first_page, args).await?;
                } else {
                    launch(ops::full_attention_page, args).await?;
                }
            }
            launch(
                ops_decode::full_divide_pack_row,
                (
                    device,
                    &self.decode_attn_f,
                    &self.decode_sum_f,
                    &offsets[row].row,
                    &mut self.divided_f,
                ),
            )
            .await?;
        }
        launch(
            ops_batch::project_full_output_batch,
            (
                device,
                &self.divided_f,
                &layer.o_weight,
                &layer.o_weight_scale,
                &mut self.projected,
            ),
        )
        .await?;
        launch(
            ep::post_attention_b16,
            (
                device,
                &self.projected,
                &layer.post_attention_norm,
                &mut self.x,
            ),
        )
        .await?;
        self.ffn(
            device,
            &layer.mlp,
            &layer.pre_feedforward_norm,
            &layer.post_feedforward_norm,
            &layer.layer_scalar,
        )
        .await
    }
}
