//! Layer-wise causal prompt execution. A chunk belongs to exactly one request.
//! Projection/FFN weights are reused across its B16 rows; KV remains request-owned.

use super::prefill_masks::{full_mask, sliding_masks};
use super::{
    generate::SoftToken,
    load::{FullLayer, Layer, MlpWeights, SlidingLayer, Weights},
    model::Sequence,
    runtime::{self, Workspace},
};
use crate::{
    Chip, axes::*, ops, ops_attention, ops_batch, ops_batch_io, ops_prefill as ep,
    ops_sliding_batch,
};
use furiosa_opt_std::prelude::*;
use ops_sliding_batch::Bk;

#[path = "prefill_wide.rs"]
pub(crate) mod wide;

#[path = "prefill_decode.rs"]
mod decode;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug)]
pub struct StageTiming {
    pub stage: &'static str,
    pub calls: usize,
    pub milliseconds: f64,
}

/// Request-owned metadata for a short prompt chunk. Hidden values are retained
/// in Prefill::prompt_x between layers, independent of decode's x scratch.
pub(crate) struct PreparedNarrow {
    pub(crate) blocks: runtime::Blocks,
    start: usize,
    count: usize,
    logits: bool,
    next_layer: usize,
    rope_s: HbmTensor<i32, Chip, m![B]>,
    rope_f: HbmTensor<i32, Chip, m![B]>,
    old_mask: HbmTensor<f32, Chip, m![B, Ts]>,
    chunk_mask: HbmTensor<f32, Chip, m![B, Bk]>,
    mask_full: HbmTensor<f32, Chip, m![B, Tf]>,
    commits: CommitOffsets,
}

pub(crate) struct Prefill {
    profile: Option<std::collections::BTreeMap<&'static str, (usize, std::time::Duration)>>,
    logits_batch: HbmTensor<bf16, Chip, m![B, W]>,
    decode_q_f: HbmTensor<bf16, Chip, m![Gf, Df]>,
    decode_max_f: HbmTensor<f32, Chip, m![Gf]>,
    decode_sum_f: HbmTensor<f32, Chip, m![Gf]>,
    decode_attn_f: HbmTensor<bf16, Chip, m![Gf, Df]>,

    x: HbmTensor<bf16, Chip, m![B, H]>,
    prompt_x: HbmTensor<bf16, Chip, m![B, H]>,
    norm: HbmTensor<bf16, Chip, m![B, H]>,
    projected: HbmTensor<bf16, Chip, m![B, H]>,
    q_raw_s: HbmTensor<bf16, Chip, m![B, Qs]>,
    k_raw_s: HbmTensor<bf16, Chip, m![B, Ps]>,
    v_raw_s: HbmTensor<bf16, Chip, m![B, Ps]>,
    q_s: HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k_s: HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v_s: HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    attn_s: HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    q_raw_f: HbmTensor<bf16, Chip, m![B, Qf]>,
    k_raw_f: HbmTensor<bf16, Chip, m![B, Pf]>,
    q_f: HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    k_f: HbmTensor<bf16, Chip, m![B, Df]>,
    v_f: HbmTensor<bf16, Chip, m![B, Df]>,
    attn_f: HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    divided_f: HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    max_f: HbmTensor<f32, Chip, m![B,Gf,1 # 8]>,
    sum_f: HbmTensor<f32, Chip, m![B,Gf,1 # 8]>,
    last: HbmTensor<bf16, Chip, m![H]>,
    full_line_mask: HbmTensor<f32, Chip, m![B, Tf]>,
}

impl Prefill {
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
        Ok(Self {
            profile: None,
            logits_batch: zero!(bf16, m![B, W]),
            decode_q_f: zero!(bf16, m![Gf, Df]),
            decode_max_f: zero!(f32, m![Gf]),
            decode_sum_f: zero!(f32, m![Gf]),
            decode_attn_f: zero!(bf16, m![Gf, Df]),
            x: zero!(bf16, m![B, H]),
            prompt_x: zero!(bf16, m![B, H]),
            norm: zero!(bf16, m![B, H]),
            projected: zero!(bf16, m![B, H]),
            q_raw_s: zero!(bf16, m![B, Qs]),
            k_raw_s: zero!(bf16, m![B, Ps]),
            v_raw_s: zero!(bf16, m![B, Ps]),
            q_s: zero!(bf16, m![B, Ns, Gs, Ds]),
            k_s: zero!(bf16, m![B, Ns, Ds]),
            v_s: zero!(bf16, m![B, Ns, Ds]),
            attn_s: zero!(bf16, m![B, Ns, Gs, Ds]),
            q_raw_f: zero!(bf16, m![B, Qf]),
            k_raw_f: zero!(bf16, m![B, Pf]),
            q_f: zero!(bf16, m![B, Gf, Df]),
            k_f: zero!(bf16, m![B, Df]),
            v_f: zero!(bf16, m![B, Df]),
            attn_f: zero!(bf16, m![B, Gf, Df]),
            divided_f: zero!(bf16, m![B, Gf, Df]),
            max_f: zero!(f32, m![B,Gf,1 # 8]),
            sum_f: zero!(f32, m![B,Gf,1 # 8]),
            last: zero!(bf16, m![H]),
            full_line_mask: zero!(f32, m![B, Tf]),
        })
    }

    pub(crate) fn supports(sequence: &Sequence<'_>) -> bool {
        let n = sequence.inputs.len();
        n > 1
            && n <= B::SIZE
            && sequence.start_position / Tf::SIZE == (sequence.start_position + n - 1) / Tf::SIZE
    }

    pub(crate) async fn prepare(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        sequence: &Sequence<'_>,
    ) -> Result<PreparedNarrow> {
        let mut profile_mark = std::time::Instant::now();
        let count = sequence.inputs.len();
        if !Self::supports(sequence) {
            return Err("unsupported prefill chunk shape".into());
        }
        let start = sequence.start_position;
        let token_offsets = HostTensor::<i32, m![B]>::from_vec(
            (0..B::SIZE)
                .map(|r| (sequence.inputs[r.min(count - 1)].token * H::SIZE * 2) as i32)
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
        for (row, input) in sequence.inputs.iter().enumerate() {
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
                (device, &embedding, &offset, &mut self.x),
            )
            .await?;
        }
        let rope_s = HostTensor::<i32, m![B]>::from_vec(
            (0..B::SIZE)
                .map(|r| ((start + r.min(count - 1)) * Ds::SIZE * 2) as i32)
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await?;
        let rope_f = HostTensor::<i32, m![B]>::from_vec(
            (0..B::SIZE)
                .map(|r| ((start + r.min(count - 1)) * Df::SIZE * 2) as i32)
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await?;
        let mask_full = HostTensor::<f32, m![B, Tf]>::from_vec(full_mask(start, count))
            .to_hbm(&mut device.pdma)
            .await?;
        let (old_mask, chunk_mask) = sliding_masks(start, count);
        let old_mask = HostTensor::<f32, m![B, Ts]>::from_vec(old_mask)
            .to_hbm(&mut device.pdma)
            .await?;
        let chunk_mask = HostTensor::<f32, m![B, Bk]>::from_vec(chunk_mask)
            .to_hbm(&mut device.pdma)
            .await?;
        let commits = CommitOffsets::new(device, start, count).await?;
        self.record_profile("prepare.prompt", &mut profile_mark);
        std::mem::swap(&mut self.x, &mut self.prompt_x);
        Ok(PreparedNarrow {
            blocks: sequence.blocks.clone(),
            start,
            count,
            logits: sequence.inputs.last().unwrap().logits,
            next_layer: 0,
            rope_s,
            rope_f,
            old_mask,
            chunk_mask,
            mask_full,
            commits,
        })
    }

    pub(crate) async fn advance_layer(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        state: &mut PreparedNarrow,
        ws: &mut Workspace,
    ) -> Result<bool> {
        let index = state.next_layer;
        let layer = weights
            .layers
            .get(index)
            .ok_or("prefill layers already complete")?;
        // Exchange handles, not tensor contents. Decode can freely overwrite x
        // after this complete layer returns; its saved prompt hidden is separate.
        std::mem::swap(&mut self.x, &mut self.prompt_x);
        let result = match layer {
            Layer::Sliding(layer) => {
                self.sliding(
                    device,
                    ws,
                    index,
                    layer,
                    &state.rope_s,
                    &state.old_mask,
                    &state.chunk_mask,
                    &state.commits,
                    state.blocks.sliding,
                )
                .await
            }
            Layer::Full(layer) => {
                self.full(
                    device,
                    ws,
                    index,
                    layer,
                    &state.rope_f,
                    &state.mask_full,
                    &state.commits,
                    state.start / Tf::SIZE,
                    &state.blocks.full,
                )
                .await
            }
        };
        std::mem::swap(&mut self.x, &mut self.prompt_x);
        result?;
        state.next_layer += 1;
        Ok(state.next_layer == weights.layers.len())
    }

    pub(crate) async fn finish(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        state: &PreparedNarrow,
        ws: &mut Workspace,
    ) -> Result<()> {
        if state.next_layer != weights.layers.len() {
            return Err("prefill logits requested before all layers finished".into());
        }
        let count = state.count;
        let mut profile_mark = std::time::Instant::now();
        if state.logits {
            let offset = HostTensor::<i32, m![1]>::from_vec([((count - 1) * H::SIZE * 2) as i32])
                .to_hbm(&mut device.pdma)
                .await?;
            launch(
                ep::extract_hidden_row,
                (device, &self.prompt_x, &offset, &mut self.last),
            )
            .await?;
            launch(
                ops::final_norm_and_logits,
                (
                    device,
                    &self.last,
                    &weights.final_norm,
                    &weights.embedding_table,
                    &mut ws.logits,
                ),
            )
            .await?;
            self.record_profile("logits.scalar", &mut profile_mark);
        }
        Ok(())
    }

    pub(crate) async fn run(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        sequence: &Sequence<'_>,
        ws: &mut Workspace,
    ) -> Result<()> {
        let mut state = self.prepare(device, weights, sequence).await?;
        while !self.advance_layer(device, weights, &mut state, ws).await? {}
        self.finish(device, weights, &state, ws).await
    }

    async fn sliding(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        index: usize,
        layer: &SlidingLayer,
        rope: &HbmTensor<i32, Chip, m![B]>,
        old_mask: &HbmTensor<f32, Chip, m![B, Ts]>,
        chunk_mask: &HbmTensor<f32, Chip, m![B, Bk]>,
        commits: &CommitOffsets,
        ring: usize,
    ) -> Result<()> {
        let mut profile_mark = std::time::Instant::now();
        launch(
            crate::ops_norm_batch::rmsnorm_b16,
            (device, &self.x, &layer.input_norm, &mut self.norm),
        )
        .await?;
        self.record_profile("sliding.input_norm", &mut profile_mark);
        launch(
            crate::ops_batch_kpart16::project_query_batch,
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
            crate::ops_batch_kpart16::project_sliding_kv_batch,
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
            crate::ops_batch_kpart16::project_sliding_kv_batch,
            (
                device,
                &self.norm,
                &layer.v_weight,
                &layer.v_weight_scale,
                &mut self.v_raw_s,
            ),
        )
        .await?;
        self.record_profile("sliding.qkv", &mut profile_mark);
        launch(
            crate::ops_head_batch::sliding_heads_b16,
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
        self.record_profile("sliding.heads_rope", &mut profile_mark);
        let (k, v) = ws.kv_cache[index].as_sliding_mut();
        launch(
            crate::ops_sliding_prompt_clusters::sliding_attention_b16,
            (
                device,
                &self.q_s,
                &k[ring],
                &v[ring],
                &self.k_s,
                &self.v_s,
                old_mask,
                chunk_mask,
                &mut self.attn_s,
            ),
        )
        .await?;
        commits
            .sliding(device, &self.k_s, &self.v_s, &mut k[ring], &mut v[ring])
            .await?;
        self.record_profile("sliding.attention", &mut profile_mark);
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
        self.record_profile("sliding.output", &mut profile_mark);
        launch(
            crate::ops_norm_batch::post_attention_b16,
            (
                device,
                &self.projected,
                &layer.post_attention_norm,
                &mut self.x,
            ),
        )
        .await?;
        self.record_profile("sliding.post_attention", &mut profile_mark);
        self.ffn(
            device,
            &layer.mlp,
            &layer.pre_feedforward_norm,
            &layer.post_feedforward_norm,
            &layer.layer_scalar,
        )
        .await
    }

    async fn full(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        index: usize,
        layer: &FullLayer,
        rope: &HbmTensor<i32, Chip, m![B]>,
        mask: &HbmTensor<f32, Chip, m![B, Tf]>,
        commits: &CommitOffsets,
        page: usize,
        blocks: &[usize],
    ) -> Result<()> {
        let mut profile_mark = std::time::Instant::now();
        launch(
            crate::ops_norm_batch::rmsnorm_b16,
            (device, &self.x, &layer.input_norm, &mut self.norm),
        )
        .await?;
        self.record_profile("full.input_norm", &mut profile_mark);
        launch(
            crate::ops_batch_kpart16::project_full_query_batch,
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
            crate::ops_batch_kpart16::project_full_key_batch,
            (
                device,
                &self.norm,
                &layer.k_weight,
                &layer.k_weight_scale,
                &mut self.k_raw_f,
            ),
        )
        .await?;
        self.record_profile("full.qkv", &mut profile_mark);
        launch(
            crate::ops_head_batch::full_heads_b16,
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
        self.record_profile("full.heads_rope", &mut profile_mark);
        let (k, v) = ws.kv_cache[index].as_full_mut();
        commits
            .full(
                device,
                &self.k_f,
                &self.v_f,
                &mut k[blocks[page]],
                &mut v[blocks[page]],
            )
            .await?;
        for current in 0..=page {
            let mask = if current == page {
                mask
            } else {
                &self.full_line_mask
            };
            let args = (
                &mut *device,
                &self.q_f,
                &k[blocks[current]],
                &v[blocks[current]],
                mask,
                &mut self.max_f,
                &mut self.sum_f,
                &mut self.attn_f,
            );
            if current == 0 {
                launch(ops_attention::full_attention_first_page_b16, args).await?;
            } else {
                launch(ops_attention::full_attention_next_page_b16, args).await?;
            }
        }
        launch(
            ops_attention::full_attention_divide_b16,
            (device, &self.attn_f, &self.sum_f, &mut self.divided_f),
        )
        .await?;
        self.record_profile("full.attention", &mut profile_mark);
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
        self.record_profile("full.output", &mut profile_mark);
        launch(
            crate::ops_norm_batch::post_attention_b16,
            (
                device,
                &self.projected,
                &layer.post_attention_norm,
                &mut self.x,
            ),
        )
        .await?;
        self.record_profile("full.post_attention", &mut profile_mark);
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
        launch(
            crate::ops_norm_batch::rmsnorm_b16,
            (device, &self.x, pre, &mut self.norm),
        )
        .await?;
        self.record_profile("ffn.pre_norm", &mut profile_mark);
        launch(
            crate::ops_batch_clusters::batch_ffn_clusters,
            (
                device,
                &self.norm,
                &weights.up_weight_packed,
                &weights.gate_weight_packed,
                &weights.down_weight_packed,
                &weights.up_weight_scale,
                &weights.gate_weight_scale,
                &weights.down_weight_scale,
                &weights.up_global_scale,
                &weights.gate_global_scale,
                &weights.down_global_scale,
                &mut self.projected,
            ),
        )
        .await?;
        self.record_profile("ffn.core", &mut profile_mark);
        launch(
            crate::ops_norm_batch::post_feedforward_b16,
            (device, &self.projected, post, gate, &mut self.x),
        )
        .await?;
        self.record_profile("ffn.post_norm_residual_gate", &mut profile_mark);
        Ok(())
    }
}

enum CommitOffsets {
    V1 {
        sliding: HbmTensor<i32, Chip, m![B = 1]>,
        full: HbmTensor<i32, Chip, m![B = 1]>,
    },
    V2 {
        sliding: HbmTensor<i32, Chip, m![B = 2]>,
        full: HbmTensor<i32, Chip, m![B = 2]>,
    },
    V3 {
        sliding: HbmTensor<i32, Chip, m![B = 3]>,
        full: HbmTensor<i32, Chip, m![B = 3]>,
    },
    V4 {
        sliding: HbmTensor<i32, Chip, m![B = 4]>,
        full: HbmTensor<i32, Chip, m![B = 4]>,
    },
    V5 {
        sliding: HbmTensor<i32, Chip, m![B = 5]>,
        full: HbmTensor<i32, Chip, m![B = 5]>,
    },
    V6 {
        sliding: HbmTensor<i32, Chip, m![B = 6]>,
        full: HbmTensor<i32, Chip, m![B = 6]>,
    },
    V7 {
        sliding: HbmTensor<i32, Chip, m![B = 7]>,
        full: HbmTensor<i32, Chip, m![B = 7]>,
    },
    V8 {
        sliding: HbmTensor<i32, Chip, m![B = 8]>,
        full: HbmTensor<i32, Chip, m![B = 8]>,
    },
    V9 {
        sliding: HbmTensor<i32, Chip, m![B = 9]>,
        full: HbmTensor<i32, Chip, m![B = 9]>,
    },
    V10 {
        sliding: HbmTensor<i32, Chip, m![B = 10]>,
        full: HbmTensor<i32, Chip, m![B = 10]>,
    },
    V11 {
        sliding: HbmTensor<i32, Chip, m![B = 11]>,
        full: HbmTensor<i32, Chip, m![B = 11]>,
    },
    V12 {
        sliding: HbmTensor<i32, Chip, m![B = 12]>,
        full: HbmTensor<i32, Chip, m![B = 12]>,
    },
    V13 {
        sliding: HbmTensor<i32, Chip, m![B = 13]>,
        full: HbmTensor<i32, Chip, m![B = 13]>,
    },
    V14 {
        sliding: HbmTensor<i32, Chip, m![B = 14]>,
        full: HbmTensor<i32, Chip, m![B = 14]>,
    },
    V15 {
        sliding: HbmTensor<i32, Chip, m![B = 15]>,
        full: HbmTensor<i32, Chip, m![B = 15]>,
    },
    V16 {
        sliding: HbmTensor<i32, Chip, m![B = 16]>,
        full: HbmTensor<i32, Chip, m![B = 16]>,
    },
}
impl CommitOffsets {
    async fn new(device: &mut Device, start: usize, count: usize) -> Result<Self> {
        let sliding: Vec<_> = (0..count)
            .map(|r| (((start + r) % Ts::SIZE) * Ns::SIZE * Ds::SIZE * 2) as i32)
            .collect();
        let full: Vec<_> = (0..count)
            .map(|r| (((start + r) % Tf::SIZE) * Df::SIZE * 2) as i32)
            .collect();
        Ok(match count {
            1 => Self::V1 {
                sliding: HostTensor::<i32, m![B = 1]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 1]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            2 => Self::V2 {
                sliding: HostTensor::<i32, m![B = 2]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 2]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            3 => Self::V3 {
                sliding: HostTensor::<i32, m![B = 3]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 3]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            4 => Self::V4 {
                sliding: HostTensor::<i32, m![B = 4]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 4]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            5 => Self::V5 {
                sliding: HostTensor::<i32, m![B = 5]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 5]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            6 => Self::V6 {
                sliding: HostTensor::<i32, m![B = 6]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 6]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            7 => Self::V7 {
                sliding: HostTensor::<i32, m![B = 7]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 7]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            8 => Self::V8 {
                sliding: HostTensor::<i32, m![B = 8]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 8]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            9 => Self::V9 {
                sliding: HostTensor::<i32, m![B = 9]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 9]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            10 => Self::V10 {
                sliding: HostTensor::<i32, m![B = 10]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 10]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            11 => Self::V11 {
                sliding: HostTensor::<i32, m![B = 11]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 11]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            12 => Self::V12 {
                sliding: HostTensor::<i32, m![B = 12]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 12]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            13 => Self::V13 {
                sliding: HostTensor::<i32, m![B = 13]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 13]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            14 => Self::V14 {
                sliding: HostTensor::<i32, m![B = 14]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 14]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            15 => Self::V15 {
                sliding: HostTensor::<i32, m![B = 15]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 15]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            16 => Self::V16 {
                sliding: HostTensor::<i32, m![B = 16]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 16]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            _ => return Err("invalid live KV prefix".into()),
        })
    }
    async fn sliding(
        &self,
        device: &mut Device,
        k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        cache_k: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
        cache_v: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    ) -> Result<()> {
        match self {
            Self::V1 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_1,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V2 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_2,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V3 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_3,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V4 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_4,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V5 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_5,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V6 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_6,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V7 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_7,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V8 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_8,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V9 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_9,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V10 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_10,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V11 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_11,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V12 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_12,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V13 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_13,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V14 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_14,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V15 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_15,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
            Self::V16 { sliding, .. } => {
                launch(
                    ep::commit_sliding_kv_16,
                    (device, k, v, sliding, cache_k, cache_v),
                )
                .await?
            }
        }
        Ok(())
    }
    async fn full(
        &self,
        device: &mut Device,
        k: &HbmTensor<bf16, Chip, m![B, Df]>,
        v: &HbmTensor<bf16, Chip, m![B, Df]>,
        cache_k: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
        cache_v: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    ) -> Result<()> {
        match self {
            Self::V1 { full, .. } => {
                launch(ep::commit_full_kv_1, (device, k, v, full, cache_k, cache_v)).await?
            }
            Self::V2 { full, .. } => {
                launch(ep::commit_full_kv_2, (device, k, v, full, cache_k, cache_v)).await?
            }
            Self::V3 { full, .. } => {
                launch(ep::commit_full_kv_3, (device, k, v, full, cache_k, cache_v)).await?
            }
            Self::V4 { full, .. } => {
                launch(ep::commit_full_kv_4, (device, k, v, full, cache_k, cache_v)).await?
            }
            Self::V5 { full, .. } => {
                launch(ep::commit_full_kv_5, (device, k, v, full, cache_k, cache_v)).await?
            }
            Self::V6 { full, .. } => {
                launch(ep::commit_full_kv_6, (device, k, v, full, cache_k, cache_v)).await?
            }
            Self::V7 { full, .. } => {
                launch(ep::commit_full_kv_7, (device, k, v, full, cache_k, cache_v)).await?
            }
            Self::V8 { full, .. } => {
                launch(ep::commit_full_kv_8, (device, k, v, full, cache_k, cache_v)).await?
            }
            Self::V9 { full, .. } => {
                launch(ep::commit_full_kv_9, (device, k, v, full, cache_k, cache_v)).await?
            }
            Self::V10 { full, .. } => {
                launch(
                    ep::commit_full_kv_10,
                    (device, k, v, full, cache_k, cache_v),
                )
                .await?
            }
            Self::V11 { full, .. } => {
                launch(
                    ep::commit_full_kv_11,
                    (device, k, v, full, cache_k, cache_v),
                )
                .await?
            }
            Self::V12 { full, .. } => {
                launch(
                    ep::commit_full_kv_12,
                    (device, k, v, full, cache_k, cache_v),
                )
                .await?
            }
            Self::V13 { full, .. } => {
                launch(
                    ep::commit_full_kv_13,
                    (device, k, v, full, cache_k, cache_v),
                )
                .await?
            }
            Self::V14 { full, .. } => {
                launch(
                    ep::commit_full_kv_14,
                    (device, k, v, full, cache_k, cache_v),
                )
                .await?
            }
            Self::V15 { full, .. } => {
                launch(
                    ep::commit_full_kv_15,
                    (device, k, v, full, cache_k, cache_v),
                )
                .await?
            }
            Self::V16 { full, .. } => {
                launch(
                    ep::commit_full_kv_16,
                    (device, k, v, full, cache_k, cache_v),
                )
                .await?
            }
        }
        Ok(())
    }
}
