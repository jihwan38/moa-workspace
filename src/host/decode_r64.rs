//! R64 independent-request execution over the canonical Rc64/Fc256 caches.
//! Padded rows never scatter KV; soft image/audio embeddings retain their encoders.
use super::{
    generate::SoftToken,
    load::{FullLayer, Layer, MlpWeights, SlidingLayer, Weights},
    model::{Logits, Sequence},
    prefill::StageTiming,
    request_pool_r64::Step64,
    runtime::{self, Workspace},
};
use crate::{
    Chip, axes::*, ops_batch_r64 as linear, ops_commit_r64 as commit, ops_norm_r64 as norm,
    ops_pool_views::Rc64, ops_r64_io as io, ops_request_attention::Fc,
};
use furiosa_opt_std::prelude::*;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(crate) struct Decode64 {
    k_tile_s: HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v_tile_s: HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    k_tile_f: HbmTensor<bf16, Chip, m![B, Df]>,
    v_tile_f: HbmTensor<bf16, Chip, m![B, Df]>,
    tile_indices: Vec<HbmTensor<i32, Chip, m![1]>>,

    profile: Option<std::collections::BTreeMap<&'static str, (usize, std::time::Duration)>>,
    logits_batch: HbmTensor<bf16, Chip, crate::ops_r64_io_blocked::BlockedLogits>,

    x: HbmTensor<bf16, Chip, m![R64, H]>,
    norm: HbmTensor<bf16, Chip, m![R64, H]>,
    projected: HbmTensor<bf16, Chip, m![R64, H]>,
    q_raw_s: HbmTensor<bf16, Chip, m![R64, Qs]>,
    k_raw_s: HbmTensor<bf16, Chip, m![R64, Ps]>,
    v_raw_s: HbmTensor<bf16, Chip, m![R64, Ps]>,
    q_s: HbmTensor<bf16, Chip, m![R64, Ns, Gs, Ds]>,
    k_s: HbmTensor<bf16, Chip, m![R64, Ns, Ds]>,
    v_s: HbmTensor<bf16, Chip, m![R64, Ns, Ds]>,
    attn_s: HbmTensor<bf16, Chip, m![R64, Ns, Gs, Ds]>,
    q_raw_f: HbmTensor<bf16, Chip, m![R64, Qf]>,
    k_raw_f: HbmTensor<bf16, Chip, m![R64, Pf]>,
    q_f: HbmTensor<bf16, Chip, m![R64, Gf, Df]>,
    k_f: HbmTensor<bf16, Chip, m![R64, Df]>,
    v_f: HbmTensor<bf16, Chip, m![R64, Df]>,
    attn_f: HbmTensor<bf16, Chip, m![R64, Gf, Df]>,
    divided_f: HbmTensor<bf16, Chip, m![R64, Gf, Df]>,
    max_f: HbmTensor<f32, Chip, m![R64,Gf,1 # 8]>,
    sum_f: HbmTensor<f32, Chip, m![R64,Gf,1 # 8]>,
}

impl Decode64 {
    pub(crate) fn blocked_logits(&self) -> &HbmTensor<bf16, Chip, crate::ops_r64_io_blocked::BlockedLogits> {
        &self.logits_batch
    }
    /// Diagnostic projection from the final norm retained by the immediately preceding R64 step.
    /// This does not touch KV or choose serving behavior.
    pub(crate) async fn reproject_lm(
        &mut self, device: &mut Device, weights: &Weights, partitioned: bool,
    ) -> Result<&HbmTensor<bf16, Chip, crate::ops_r64_io_blocked::BlockedLogits>> {
        if partitioned {
            launch(crate::ops_r64_lm_partitioned::logits_batch,
                (device, &self.norm, &weights.embedding_table, &mut self.logits_batch)).await?;
        } else {
            launch(crate::ops_r64_io_blocked::logits_batch,
                (device, &self.norm, &weights.embedding_table, &mut self.logits_batch)).await?;
        }
        Ok(&self.logits_batch)
    }
    pub(crate) fn begin_profile(&mut self) {
        self.profile = Some(Default::default());
    }
    pub(crate) fn take_profile(&mut self) -> Vec<StageTiming> {
        self.profile
            .take()
            .unwrap_or_default()
            .into_iter()
            .map(|(stage, (calls, total))| StageTiming {
                stage,
                calls,
                milliseconds: total.as_secs_f64() * 1000.0,
            })
            .collect()
    }
    fn record_profile(&mut self, stage: &'static str, last: &mut std::time::Instant) {
        if let Some(profile) = &mut self.profile {
            let now = std::time::Instant::now();
            let entry = profile.entry(stage).or_default();
            entry.0 += 1;
            entry.1 += now.duration_since(*last);
            *last = now;
        }
    }

    pub(crate) async fn new(device: &mut Device) -> Result<Self> {
        macro_rules! zero {
            ($dtype:ty, $mapping:ty) => {
                HostTensor::<$dtype, $mapping>::zero()
                    .to_hbm(&mut device.pdma)
                    .await?
            };
        }
        let mut tile_indices = Vec::new();
        for tile in 0..R64::SIZE / B::SIZE {
            tile_indices.push(
                HostTensor::<i32, m![1]>::from_vec([tile as i32])
                    .to_hbm(&mut device.pdma)
                    .await?,
            );
        }
        Ok(Self {
            k_tile_s: zero!(bf16, m![B, Ns, Ds]),
            v_tile_s: zero!(bf16, m![B, Ns, Ds]),
            k_tile_f: zero!(bf16, m![B, Df]),
            v_tile_f: zero!(bf16, m![B, Df]),
            tile_indices,
            profile: None,
            logits_batch: zero!(bf16, crate::ops_r64_io_blocked::BlockedLogits),
            x: zero!(bf16, m![R64, H]),
            norm: zero!(bf16, m![R64, H]),
            projected: zero!(bf16, m![R64, H]),
            q_raw_s: zero!(bf16, m![R64, Qs]),
            k_raw_s: zero!(bf16, m![R64, Ps]),
            v_raw_s: zero!(bf16, m![R64, Ps]),
            q_s: zero!(bf16, m![R64, Ns, Gs, Ds]),
            k_s: zero!(bf16, m![R64, Ns, Ds]),
            v_s: zero!(bf16, m![R64, Ns, Ds]),
            attn_s: zero!(bf16, m![R64, Ns, Gs, Ds]),
            q_raw_f: zero!(bf16, m![R64, Qf]),
            k_raw_f: zero!(bf16, m![R64, Pf]),
            q_f: zero!(bf16, m![R64, Gf, Df]),
            k_f: zero!(bf16, m![R64, Df]),
            v_f: zero!(bf16, m![R64, Df]),
            attn_f: zero!(bf16, m![R64, Gf, Df]),
            divided_f: zero!(bf16, m![R64, Gf, Df]),
            max_f: zero!(f32, m![R64,Gf,1 # 8]),
            sum_f: zero!(f32, m![R64,Gf,1 # 8]),
        })
    }

    pub(crate) async fn run(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        sequences: &[Sequence<'_>],
        ws: &mut Workspace,
        logits: &mut [Logits],
        output_rows: &[usize],
        extract_rows: bool,
    ) -> Result<()> {
        let mut profile_mark = std::time::Instant::now();
        let count = sequences.len();
        if count == 0
            || count > R64::SIZE
            || output_rows.len() != count
            || output_rows.iter().any(|&row| row >= logits.len())
            || sequences.iter().any(|s| s.inputs.len() != 1)
        {
            return Err("invalid independent-request batch".into());
        }
        if !ws.is_canonical() || !super::request_pool::independent_sequences(sequences) {
            return Err("R64 execution requires distinct canonical request allocations".into());
        }
        let metadata = Step64::new(device, sequences).await?;
        let token_offsets = HostTensor::<i32, m![R64]>::from_vec(
            (0..R64::SIZE)
                .map(|r| (sequences[r.min(count - 1)].inputs[0].token * H::SIZE * 2) as i32)
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await?;
        launch(
            io::embed_tokens,
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
                    norm::insert_hidden_row,
                    (device, &embedding, &offset, &mut self.x),
                )
                .await?;
            }
            if extract_rows { offsets.push(
                HostTensor::<i32, m![1]>::from_vec([(row * 32 * 2) as i32])
                    .to_hbm(&mut device.pdma)
                    .await?,
            ); }
        }
        let rope_s = HostTensor::<i32, m![R64]>::from_vec(
            (0..R64::SIZE)
                .map(|r| (sequences[r.min(count - 1)].start_position * Ds::SIZE * 2) as i32)
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await?;
        let rope_f = HostTensor::<i32, m![R64]>::from_vec(
            (0..R64::SIZE)
                .map(|r| (sequences[r.min(count - 1)].start_position * Df::SIZE * 2) as i32)
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await?;
        self.record_profile("r64.prepare.decode", &mut profile_mark);
        for (index, layer) in weights.layers.iter().enumerate() {
            match layer {
                Layer::Sliding(layer) => {
                    self.decode_sliding(device, ws, index, layer, &rope_s, &metadata)
                        .await?
                }
                Layer::Full(layer) => {
                    self.decode_full(device, ws, index, layer, &rope_f, &metadata)
                        .await?
                }
            }
        }
        profile_mark = std::time::Instant::now();
        if sequences.iter().any(|s| s.inputs[0].logits) {
            launch(
                norm::rmsnorm_r64,
                (device, &self.x, &weights.final_norm, &mut self.norm),
            )
            .await?;
            self.record_profile("r64.logits.norm", &mut profile_mark);
            launch(
                crate::ops_r64_lm_partitioned::logits_batch,
                (
                    device,
                    &self.norm,
                    &weights.embedding_table,
                    &mut self.logits_batch,
                ),
            )
            .await?;
            self.record_profile("r64.logits.projection", &mut profile_mark);
            for (row, sequence) in sequences.iter().enumerate() {
                if extract_rows && sequence.inputs[0].logits {
                    launch(
                        crate::ops_r64_io_blocked::extract_logits_row,
                        (
                            device,
                            &self.logits_batch,
                            &offsets[row],
                            &mut logits[output_rows[row]],
                        ),
                    )
                    .await?;
                }
            }
        }
        self.record_profile("r64.logits.extract", &mut profile_mark);
        Ok(())
    }

    async fn decode_sliding(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        index: usize,
        layer: &SlidingLayer,
        rope: &HbmTensor<i32, Chip, m![R64]>,
        metadata: &Step64,
    ) -> Result<()> {
        let mut profile_mark = std::time::Instant::now();
        launch(
            norm::rmsnorm_r64,
            (device, &self.x, &layer.input_norm, &mut self.norm),
        )
        .await?;
        self.record_profile("r64.sliding.input_norm", &mut profile_mark);
        launch(
            crate::ops_qkv_r64_trf32::project_query_batch,
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
            crate::ops_qkv_r64_trf32::project_sliding_kv_batch,
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
            crate::ops_qkv_r64_trf32::project_sliding_kv_batch,
            (
                device,
                &self.norm,
                &layer.v_weight,
                &layer.v_weight_scale,
                &mut self.v_raw_s,
            ),
        )
        .await?;
        self.record_profile("r64.sliding.qkv", &mut profile_mark);
        launch(
            crate::ops_head_r64::sliding_heads_r64,
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
        self.record_profile("r64.sliding.heads_rope", &mut profile_mark);
        let runtime::KvCache::SlidingCanonical { k, v } = &mut ws.kv_cache[index] else {
            return Err("sliding layer has wrong pool kind".into());
        };
        self.commit_sliding(device, metadata, k, v).await?;
        if metadata.sliding_short_groups == 255 {
        launch(
            crate::ops_request_sliding_r64_prefix512::sliding_attention_r64,
            (
                device,
                &self.q_s,
                &*k,
                &*v,
                &metadata.ring_offsets,
                &metadata.sliding_mask,
                &mut self.attn_s,
            ),
        )
        .await?;
        } else if let Some(policy) = &metadata.sliding_group_policy {
            launch(
                crate::ops_request_sliding_r64_mixed512::sliding_attention_r64,
                (
                    device,
                    &self.q_s,
                    &*k,
                    &*v,
                    &metadata.ring_offsets,
                    &metadata.sliding_mask,
                    policy,
                    &mut self.attn_s,
                ),
            )
            .await?;
        } else {
        launch(
            crate::ops_request_sliding_r64::sliding_attention_r64,
            (
                device,
                &self.q_s,
                &*k,
                &*v,
                &metadata.ring_offsets,
                &metadata.sliding_mask,
                &mut self.attn_s,
            ),
        )
        .await?;
        }
        self.record_profile("r64.sliding.attention", &mut profile_mark);
        launch(
            crate::ops_output_r64_parallel_gather::project_sliding_output_batch,
            (
                device,
                &self.attn_s,
                &layer.o_weight,
                &layer.o_weight_scale,
                &mut self.projected,
            ),
        )
        .await?;
        self.record_profile("r64.sliding.output", &mut profile_mark);
        launch(
            norm::post_attention_r64,
            (
                device,
                &self.projected,
                &layer.post_attention_norm,
                &mut self.x,
            ),
        )
        .await?;
        self.record_profile("r64.sliding.post_attention", &mut profile_mark);
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
        rope: &HbmTensor<i32, Chip, m![R64]>,
        metadata: &Step64,
    ) -> Result<()> {
        let mut profile_mark = std::time::Instant::now();
        launch(
            norm::rmsnorm_r64,
            (device, &self.x, &layer.input_norm, &mut self.norm),
        )
        .await?;
        self.record_profile("r64.full.input_norm", &mut profile_mark);
        launch(
            crate::ops_qkv_r64_trf32::project_full_query_batch,
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
            linear::project_full_key_batch,
            (
                device,
                &self.norm,
                &layer.k_weight,
                &layer.k_weight_scale,
                &mut self.k_raw_f,
            ),
        )
        .await?;
        self.record_profile("r64.full.qkv", &mut profile_mark);
        launch(
            crate::ops_head_r64::full_heads_r64,
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
        self.record_profile("r64.full.heads_rope", &mut profile_mark);
        let runtime::KvCache::FullCanonical { k, v } = &mut ws.kv_cache[index] else {
            return Err("full layer has wrong pool kind".into());
        };
        self.commit_full(device, metadata, k, v).await?;
        for (page_index, page) in metadata.pages.iter().enumerate() {
            let args = (
                &mut *device,
                &self.q_f,
                &*k,
                &*v,
                &page.offsets,
                &page.mask,
                &page.active,
                &mut self.max_f,
                &mut self.sum_f,
                &mut self.attn_f,
            );
            if page_index == 0 {
                launch(
                    crate::ops_request_full_r64::full_attention_first_page_r64,
                    args,
                )
                .await?;
            } else {
                launch(
                    crate::ops_request_full_r64::full_attention_next_page_r64,
                    args,
                )
                .await?;
            }
        }
        launch(
            crate::ops_request_full_r64::full_attention_divide_r64,
            (device, &self.attn_f, &self.sum_f, &mut self.divided_f),
        )
        .await?;
        self.record_profile("r64.full.attention", &mut profile_mark);
        launch(
            crate::ops_output_r64_parallel_gather::project_full_output_batch,
            (
                device,
                &self.divided_f,
                &layer.o_weight,
                &layer.o_weight_scale,
                &mut self.projected,
            ),
        )
        .await?;
        self.record_profile("r64.full.output", &mut profile_mark);
        launch(
            norm::post_attention_r64,
            (
                device,
                &self.projected,
                &layer.post_attention_norm,
                &mut self.x,
            ),
        )
        .await?;
        self.record_profile("r64.full.post_attention", &mut profile_mark);
        self.ffn(
            device,
            &layer.mlp,
            &layer.pre_feedforward_norm,
            &layer.post_feedforward_norm,
            &layer.layer_scalar,
        )
        .await
    }
    async fn ffn(
        &mut self,
        device: &mut Device,
        weights: &MlpWeights,
        pre: &HbmTensor<bf16, Chip, m![H]>,
        post: &HbmTensor<bf16, Chip, m![H]>,
        gate: &HbmTensor<bf16, Chip, m![1 # 8]>,
    ) -> Result<()> {
        let mut profile_mark = std::time::Instant::now();
        launch(norm::rmsnorm_r64, (device, &self.x, pre, &mut self.norm)).await?;
        self.record_profile("r64.ffn.pre_norm", &mut profile_mark);
        launch(
            crate::ops_native_fp8_r64_local_reuse::batch_ffn_partitioned_input,
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
        self.record_profile("r64.ffn.core", &mut profile_mark);
        launch(
            norm::post_feedforward_r64,
            (device, &self.projected, post, gate, &mut self.x),
        )
        .await?;
        self.record_profile("r64.ffn.post_norm_residual_gate", &mut profile_mark);
        Ok(())
    }
    async fn commit_sliding(
        &mut self,
        device: &mut Device,
        metadata: &Step64,
        k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
        v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    ) -> Result<()> {
        if metadata.live == R64::SIZE {
            launch(
                commit::commit_sliding_kv_r64,
                (
                    device,
                    &self.k_s,
                    &self.v_s,
                    &metadata.sliding_offsets,
                    k,
                    v,
                ),
            )
            .await?;
        } else {
            for (tile, offsets) in metadata.segments.iter().enumerate() {
                launch(
                    commit::load_sliding_kv_tile,
                    (
                        device,
                        &self.k_s,
                        &self.v_s,
                        &self.tile_indices[tile],
                        &mut self.k_tile_s,
                        &mut self.v_tile_s,
                    ),
                )
                .await?;
                offsets
                    .sliding_canonical(device, &self.k_tile_s, &self.v_tile_s, &mut *k, &mut *v)
                    .await?;
            }
        }
        Ok(())
    }

    async fn commit_full(
        &mut self,
        device: &mut Device,
        metadata: &Step64,
        k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
        v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    ) -> Result<()> {
        if metadata.live == R64::SIZE {
            launch(
                commit::commit_full_kv_r64,
                (device, &self.k_f, &self.v_f, &metadata.full_offsets, k, v),
            )
            .await?;
        } else {
            for (tile, offsets) in metadata.segments.iter().enumerate() {
                launch(
                    commit::load_full_kv_tile,
                    (
                        device,
                        &self.k_f,
                        &self.v_f,
                        &self.tile_indices[tile],
                        &mut self.k_tile_f,
                        &mut self.v_tile_f,
                    ),
                )
                .await?;
                offsets
                    .full(device, &self.k_tile_f, &self.v_tile_f, &mut *k, &mut *v)
                    .await?;
            }
        }
        Ok(())
    }
}
