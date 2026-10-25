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
struct ProjectionLayerWeights {
    weight: HbmTensor<f8e4m3, Chip, crate::ops_ffn_projection_clusters::JoinedWeight>,
    scale: HbmTensor<f32, Chip, crate::ops_ffn_projection_clusters::JoinedScale>,
}
struct ProjectionStorage {
    layers: Vec<ProjectionLayerWeights>,
    output: HbmTensor<bf16, Chip, crate::ops_ffn_projection_clusters::JoinedOutput>,
}
pub(crate) struct GroupPrefill {
    projection_storage: Option<ProjectionStorage>,
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
    preload_ffn_globals: bool,
    quarter_postnorm: bool,
    blocked_residual: bool,
    composed_ffn_epilogue: bool, projection_owned_ffn: bool,
    weight_stationary_output: bool,
    weight_stationary_qkv: bool,
    grouped_norm_stores: bool,
}
/// Exactly two actual P128 execution states; padding has no request metadata.
pub(crate) struct PreparedTail256 {
    preload_ffn_globals: bool,
    composed_ffn_epilogue: bool, projection_owned_ffn: bool,
    weight_stationary_output: bool,
    weight_stationary_qkv: bool,
    grouped_norm_stores: bool,
    states: [PreparedWide; 2],
    owners: usize,
}
impl PreparedTail256 {
    pub(crate) fn grouped_norm_stores(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("norm stores must be selected before the first layer".into());
        }
        if enabled && !self.projection_owned_ffn {
            return Err("grouped norm stores require the current projection-owned FFN graph".into());
        }
        self.grouped_norm_stores = enabled;
        Ok(())
    }
    pub(crate) fn weight_stationary_output(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("O producer must be selected before the first layer".into());
        }
        self.weight_stationary_output = enabled;
        Ok(())
    }
    pub(crate) fn weight_stationary_qkv(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("sliding QKV producer must be selected before the first layer".into());
        }
        self.weight_stationary_qkv = enabled;
        Ok(())
    }
    pub(crate) fn preload_ffn_globals(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("FFN global preload must be selected before the first layer".into());
        }
        self.preload_ffn_globals = enabled;
        if !enabled { self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
        Ok(())
    }
    pub(crate) fn composed_ffn_epilogue(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("FFN epilogue composition must be selected before the first layer".into());
        }
        self.composed_ffn_epilogue = enabled;
        if !enabled { self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
        Ok(())
    }
    pub(crate) fn projection_owned_ffn(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("Projection-owned FFN must be selected before the first layer".into());
        }
        if enabled && (!self.composed_ffn_epilogue || !self.preload_ffn_globals) {
            return Err("projection-owned FFN requires the current composed preload route".into());
        }
        self.projection_owned_ffn = enabled;
        if !enabled { self.grouped_norm_stores = false; }
        Ok(())
    }
    pub(crate) fn blocks(&self) -> Vec<&runtime::Blocks> {
        if self.owners == 1 { vec![&self.states[0].blocks] }
        else { vec![&self.states[0].blocks, &self.states[1].blocks] }
    }
    pub(crate) fn owners(&self) -> usize { self.owners }
    fn final_row(&self, owner: usize) -> usize {
        assert!(owner < self.owners);
        if self.owners == 1 { 1 } else { owner }
    }
}
/// Three real chronological quarters; the fourth tensor quarter has no owner.
pub(crate) struct PreparedTail384 {
    preload_ffn_globals: bool,
    composed_ffn_epilogue: bool, projection_owned_ffn: bool,
    weight_stationary_output: bool,
    weight_stationary_qkv: bool,
    grouped_norm_stores: bool,
    states: [PreparedWide; 3],
    owner_first_rows: Vec<usize>,
    owner_final_rows: Vec<usize>,
}
impl PreparedTail384 {
    pub(crate) fn grouped_norm_stores(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("norm stores must be selected before the first layer".into());
        }
        if enabled && !self.projection_owned_ffn {
            return Err("grouped norm stores require the current projection-owned FFN graph".into());
        }
        self.grouped_norm_stores = enabled;
        Ok(())
    }
    pub(crate) fn weight_stationary_output(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("O producer must be selected before the first layer".into());
        }
        self.weight_stationary_output = enabled;
        Ok(())
    }
    pub(crate) fn weight_stationary_qkv(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("sliding QKV producer must be selected before the first layer".into());
        }
        self.weight_stationary_qkv = enabled;
        Ok(())
    }
    pub(crate) fn preload_ffn_globals(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("FFN global preload must be selected before the first layer".into());
        }
        self.preload_ffn_globals = enabled;
        if !enabled { self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
        Ok(())
    }
    pub(crate) fn composed_ffn_epilogue(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("FFN epilogue composition must be selected before the first layer".into());
        }
        self.composed_ffn_epilogue = enabled;
        if !enabled { self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
        Ok(())
    }
    pub(crate) fn projection_owned_ffn(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("Projection-owned FFN must be selected before the first layer".into());
        }
        if enabled && (!self.composed_ffn_epilogue || !self.preload_ffn_globals) {
            return Err("projection-owned FFN requires the current composed preload route".into());
        }
        self.projection_owned_ffn = enabled;
        if !enabled { self.grouped_norm_stores = false; }
        Ok(())
    }
    pub(crate) fn blocks(&self) -> Vec<&runtime::Blocks> {
        self.owner_first_rows.iter().map(|&row| &self.states[row].blocks).collect()
    }
    pub(crate) fn owners(&self) -> usize { self.owner_first_rows.len() }
}
/// Three independent reservations contribute four privately derived quarters.
/// Unlike Tail384, every tensor quarter belongs to a real input position.
pub(crate) struct PreparedThreeOwner512 {
    chunks: PreparedGroup,
    owner_first_rows: [usize; 3],
    owner_final_rows: [usize; 3],
}
impl PreparedThreeOwner512 {
    pub(crate) fn grouped_norm_stores(&mut self, enabled: bool) -> Result<()> {
        self.chunks.grouped_norm_stores(enabled)
    }
    pub(crate) fn weight_stationary_output(&mut self, enabled: bool) -> Result<()> {
        self.chunks.weight_stationary_output(enabled)
    }
    pub(crate) fn weight_stationary_qkv(&mut self, enabled: bool) -> Result<()> {
        self.chunks.weight_stationary_qkv(enabled)
    }
    pub(crate) fn preload_ffn_globals(&mut self, enabled: bool) -> Result<()> {
        self.chunks.preload_ffn_globals(enabled)
    }
    pub(crate) fn composed_ffn_epilogue(&mut self, enabled: bool) -> Result<()> {
        self.chunks.composed_ffn_epilogue(enabled)
    }
    pub(crate) fn projection_owned_ffn(&mut self, enabled: bool) -> Result<()> {
        self.chunks.projection_owned_ffn(enabled)
    }
    pub(crate) fn blocks(&self) -> [&runtime::Blocks; 3] {
        self.owner_first_rows.map(|row| &self.chunks.states[row].blocks)
    }
}
#[derive(Clone, Copy)]
struct GroupExecution {
    fused_norms: bool, blocked_output: bool, shard_quant: bool,
    quarter_norm: bool, fused_ffn_core: bool, preload_ffn_globals: bool, quarter_postnorm: bool,
    blocked_residual: bool, composed_ffn_epilogue: bool, projection_owned_ffn: bool,
    weight_stationary_output: bool,
    weight_stationary_qkv: bool,
    grouped_norm_stores: bool,
}
impl GroupExecution {
    fn production() -> Self {
        Self { fused_norms:true, blocked_output:true, shard_quant:true,
            quarter_norm:true, fused_ffn_core:true, preload_ffn_globals:true, quarter_postnorm:true,
            blocked_residual:true, composed_ffn_epilogue:true, projection_owned_ffn:false,
            weight_stationary_output:true, grouped_norm_stores: false, weight_stationary_qkv: true }
    }
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
    pub(crate) fn grouped_norm_stores(&mut self, enabled: bool) -> Result<()> {
        self.chunks.grouped_norm_stores(enabled)
    }
    pub(crate) fn weight_stationary_output(&mut self, enabled: bool) -> Result<()> {
        self.chunks.weight_stationary_output(enabled)
    }
    pub(crate) fn weight_stationary_qkv(&mut self, enabled: bool) -> Result<()> {
        self.chunks.weight_stationary_qkv(enabled)
    }
    pub(crate) fn preload_ffn_globals(&mut self, enabled: bool) -> Result<()> {
        self.chunks.preload_ffn_globals(enabled)
    }
    pub(crate) fn composed_ffn_epilogue(&mut self, enabled: bool) -> Result<()> {
        self.chunks.composed_ffn_epilogue(enabled)
    }
    pub(crate) fn projection_owned_ffn(&mut self, enabled: bool) -> Result<()> {
        self.chunks.projection_owned_ffn(enabled)
    }
    pub(crate) fn fused_head_attention(&mut self, enabled: bool) {
        self.chunks.fused_head_attention(enabled);
    }
    pub(crate) fn original_blocked_residual(&mut self) { self.chunks.original_blocked_residual(); }
    pub(crate) fn blocks(&self) -> [&runtime::Blocks; 2] {
        [&self.chunks.states[0].blocks, &self.chunks.states[self.second_owner_row].blocks]
    }
}

impl PreparedPrompt512 {
    pub(crate) fn grouped_norm_stores(&mut self, enabled: bool) -> Result<()> {
        self.chunks.grouped_norm_stores(enabled)
    }
    pub(crate) fn weight_stationary_output(&mut self, enabled: bool) -> Result<()> {
        self.chunks.weight_stationary_output(enabled)
    }
    pub(crate) fn weight_stationary_qkv(&mut self, enabled: bool) -> Result<()> {
        self.chunks.weight_stationary_qkv(enabled)
    }
    pub(crate) fn preload_ffn_globals(&mut self, enabled: bool) -> Result<()> {
        self.chunks.preload_ffn_globals(enabled)
    }
    pub(crate) fn composed_ffn_epilogue(&mut self, enabled: bool) -> Result<()> {
        self.chunks.composed_ffn_epilogue(enabled)
    }
    pub(crate) fn projection_owned_ffn(&mut self, enabled: bool) -> Result<()> {
        self.chunks.projection_owned_ffn(enabled)
    }
    pub(crate) fn fused_head_attention(&mut self, enabled: bool) {
        self.chunks.fused_head_attention(enabled);
    }
    pub(crate) fn prune_empty_history(&mut self) { self.chunks.prune_empty_history(); }
    pub(crate) fn original_empty_history(&mut self) { self.chunks.original_empty_history(); }
    pub(crate) fn original_blocked_residual(&mut self) { self.chunks.original_blocked_residual(); }
    pub(crate) fn original_norms(&mut self) { self.chunks.original_norms(); }
    pub(crate) fn original_output(&mut self) { self.chunks.original_output(); }
    pub(crate) fn original_shard_quant(&mut self) { self.chunks.original_shard_quant(); }
    pub(crate) fn original_quarter_norm(&mut self) { self.chunks.original_quarter_norm(); }
    pub(crate) fn original_ffn_core(&mut self) { self.chunks.original_ffn_core(); }
    pub(crate) fn original_quarter_postnorm(&mut self) { self.chunks.original_quarter_postnorm(); }
    pub(crate) fn blocks(&self) -> &runtime::Blocks {
        &self.chunks.states[0].blocks
    }
}
impl PreparedGroup {
    pub(crate) fn grouped_norm_stores(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("norm stores must be selected before the first layer".into());
        }
        if enabled && !self.projection_owned_ffn {
            return Err("grouped norm stores require the current projection-owned FFN graph".into());
        }
        self.grouped_norm_stores = enabled;
        Ok(())
    }
    pub(crate) fn weight_stationary_output(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("O producer must be selected before the first layer".into());
        }
        self.weight_stationary_output = enabled;
        Ok(())
    }
    pub(crate) fn weight_stationary_qkv(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("sliding QKV producer must be selected before the first layer".into());
        }
        self.weight_stationary_qkv = enabled;
        Ok(())
    }
    pub(crate) fn preload_ffn_globals(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("FFN global preload must be selected before the first layer".into());
        }
        if enabled && !self.fused_ffn_core {
            return Err("FFN global preload requires the current single-launch core".into());
        }
        self.preload_ffn_globals = enabled;
        if !enabled { self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
        Ok(())
    }
    pub(crate) fn composed_ffn_epilogue(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("FFN epilogue composition must be selected before the first layer".into());
        }
        self.composed_ffn_epilogue = enabled;
        if !enabled { self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
        Ok(())
    }
    pub(crate) fn projection_owned_ffn(&mut self, enabled: bool) -> Result<()> {
        if self.states.iter().any(|state| state.next_layer != 0) {
            return Err("Projection-owned FFN must be selected before the first layer".into());
        }
        if enabled && (!self.composed_ffn_epilogue || !self.preload_ffn_globals) {
            return Err("projection-owned FFN requires the current composed preload route".into());
        }
        self.projection_owned_ffn = enabled;
        if !enabled { self.grouped_norm_stores = false; }
        Ok(())
    }
    pub(crate) fn fused_head_attention(&mut self, enabled: bool) {
        for state in &mut self.states { state.fused_head_attention = enabled; }
    }
    pub(crate) fn prune_empty_history(&mut self) {
        for state in &mut self.states { state.prune_empty_history = true; }
    }
    pub(crate) fn original_empty_history(&mut self) {
        for state in &mut self.states { state.prune_empty_history = false; }
    }
    pub(crate) fn original_blocked_residual(&mut self) { self.blocked_residual = false; self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
    fn uses_blocked_residual(&self) -> bool {
        // Old diagnostic implementations retain canonical state throughout.
        // Their timings include this storage fallback; the dedicated storage
        // control is the isolated comparison. FFN core selection is independent.
        self.blocked_residual && self.fused_norms && self.blocked_output
            && self.shard_quant && self.quarter_norm && self.quarter_postnorm
    }
    pub(crate) fn original_norms(&mut self) { self.fused_norms = false; self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
    pub(crate) fn original_output(&mut self) { self.blocked_output = false; self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
    pub(crate) fn original_shard_quant(&mut self) { self.shard_quant = false; self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
    pub(crate) fn original_quarter_norm(&mut self) { self.quarter_norm = false; self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
    pub(crate) fn original_ffn_core(&mut self) { self.fused_ffn_core = false; self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
    pub(crate) fn original_quarter_postnorm(&mut self) { self.quarter_postnorm = false; self.projection_owned_ffn = false; self.grouped_norm_stores = false; }
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
    pub(crate) fn has_projection_storage(&self) -> bool { self.projection_storage.is_some() }
    pub(crate) async fn prepare_projection_storage(&mut self, device:&mut Device, weights:&Weights) -> Result<()> {
        if self.projection_storage.is_some() { return Ok(()); }
        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            let mlp = match layer { Layer::Sliding(l) => &l.mlp, Layer::Full(l) => &l.mlp };
            let native = &mlp.native_fp8;
            let (weight,scale) = launch(crate::ops_ffn_projection_join::join,
                (&mut *device,&native.up,&native.gate,&native.up_dequant,&native.gate_dequant)).await?;
            layers.push(ProjectionLayerWeights { weight,scale });
        }
        let output = HostTensor::<bf16,crate::ops_ffn_projection_clusters::JoinedOutput>::zero()
            .to_hbm(&mut device.pdma).await?;
        self.projection_storage = Some(ProjectionStorage { layers,output });
        Ok(())
    }
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
            projection_storage: None,
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

    pub(crate) fn supports_three_owner512(sequences: &[Sequence<'_>]) -> bool {
        sequences.len() == 3
            && sequences.iter().all(|s| matches!(s.inputs.len(),128|256)
                && s.start_position % P128::SIZE == 0
                && s.start_position.checked_add(s.inputs.len()).is_some()
                && s.inputs[..s.inputs.len()-1].iter().all(|input| !input.logits))
            && sequences.iter().map(|s|s.inputs.len()).sum::<usize>() == P512::SIZE
            && crate::host::request_pool::independent_sequences(sequences)
    }

    pub(crate) async fn prepare_three_owner512(
        &mut self, device:&mut Device, weights:&Weights, sequences:&[Sequence<'_>],
        scratch:&mut Prefill, batched_media:bool,
    ) -> Result<PreparedThreeOwner512> {
        if !Self::supports_three_owner512(sequences) {
            return Err("ThreeOwner512 needs independent aligned256/128/128 ranges in owner order".into());
        }
        let mut owner_first_rows=[0;3];let mut owner_final_rows=[0;3];
        let mut chunks=Vec::with_capacity(4);
        for (owner,sequence) in sequences.iter().enumerate() {
            owner_first_rows[owner]=chunks.len();
            for (quarter,inputs) in sequence.inputs.chunks_exact(P128::SIZE).enumerate() {
                chunks.push(Sequence{inputs,start_position:sequence.start_position+quarter*P128::SIZE,
                    blocks:sequence.blocks});
            }
            owner_final_rows[owner]=chunks.len()-1;
        }
        debug_assert_eq!(chunks.len(),4);
        debug_assert!(chunks.iter().all(WidePrefill::supports));
        Ok(PreparedThreeOwner512 {
            chunks:self.prepare_rows(device,weights,&chunks,scratch,batched_media).await?,
            owner_first_rows,owner_final_rows,
        })
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
            preload_ffn_globals: true,
            quarter_postnorm: true,
            blocked_residual: true,
            composed_ffn_epilogue: true, projection_owned_ffn: self.has_projection_storage(),
            weight_stationary_output: true,
            grouped_norm_stores: false, weight_stationary_qkv: true,
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

    pub(crate) fn supports_tail256(sequences: &[Sequence<'_>]) -> bool {
        match sequences {
            [one] => one.inputs.len() == 256 && one.start_position % 128 == 0
                && one.start_position.checked_add(256).is_some()
                && one.inputs[..255].iter().all(|input| !input.logits),
            [a, b] => [a, b].iter().all(|s| s.inputs.len() == 128
                && s.start_position % 128 == 0 && s.start_position.checked_add(128).is_some()
                && s.inputs[..127].iter().all(|input| !input.logits) && WidePrefill::supports(s))
                && crate::host::request_pool::independent_sequences(sequences),
            _ => false,
        }
    }

    pub(crate) async fn prepare_tail256(
        &mut self, device:&mut Device, weights:&Weights, sequences:&[Sequence<'_>],
        scratch:&mut Prefill, batched_media:bool, padding:bf16,
    ) -> Result<PreparedTail256> {
        if !Self::supports_tail256(sequences) {
            return Err("Tail256 needs one aligned256 range or two independent128 ranges".into());
        }
        let chunks:Vec<_> = sequences.iter().flat_map(|s| {
            s.inputs.chunks_exact(P128::SIZE).enumerate().map(move |(i, inputs)| Sequence {
                inputs, start_position:s.start_position+i*P128::SIZE, blocks:s.blocks,
            })
        }).collect();
        debug_assert_eq!(chunks.len(),2);
        // Reset tensor-only padding after any previous four-owner execution.
        // These tensors are never given request/KV metadata or attention calls.
        for row in &mut self.rows[2..] {
            row.x = HostTensor::<bf16,m![P128,H]>::from_vec(
                std::iter::repeat_n(padding,P128::SIZE*H::SIZE)).to_hbm(&mut device.pdma).await?;
            row.attn_s = HostTensor::<bf16,m![P128,Ns,Gs,Ds]>::from_vec(
                std::iter::repeat_n(padding,P128::SIZE*Ns::SIZE*Gs::SIZE*Ds::SIZE)).to_hbm(&mut device.pdma).await?;
            row.attn_f = HostTensor::<bf16,m![P128,Gf,Df]>::from_vec(
                std::iter::repeat_n(padding,P128::SIZE*Gf::SIZE*Df::SIZE)).to_hbm(&mut device.pdma).await?;
        }
        Ok(PreparedTail256 { preload_ffn_globals:true, composed_ffn_epilogue:true, projection_owned_ffn:self.has_projection_storage(),
            weight_stationary_output:true, grouped_norm_stores: false, weight_stationary_qkv: true, owners:sequences.len(), states:[
            self.rows[0].prepare_with_media_batch(device,weights,&chunks[0],scratch,true,batched_media).await?,
            self.rows[1].prepare_with_media_batch(device,weights,&chunks[1],scratch,true,batched_media).await?,
        ]})
    }
    pub(crate) async fn advance_tail256(
        &mut self, device:&mut Device, weights:&Weights, state:&mut PreparedTail256,
        ws:&mut Workspace, scratch:&mut Prefill,
    ) -> Result<bool> {
        let mut execution = GroupExecution::production();
        execution.composed_ffn_epilogue = state.composed_ffn_epilogue;
        execution.projection_owned_ffn = state.projection_owned_ffn;
        execution.preload_ffn_globals = state.preload_ffn_globals;
        execution.weight_stationary_output = state.weight_stationary_output;
        execution.weight_stationary_qkv = state.weight_stationary_qkv;
        execution.grouped_norm_stores = state.grouped_norm_stores;
        self.advance_rows(device,weights,&mut state.states,execution,ws,scratch).await
    }
    pub(crate) async fn finish_tail256(
        &mut self, owner:usize, device:&mut Device, weights:&Weights,
        state:&PreparedTail256, ws:&mut Workspace, scratch:&mut Prefill,
    ) -> Result<()> {
        let row=state.final_row(owner);
        ws.blocks.clone_from(&state.states[row].blocks);
        self.rows[row].finish(device,weights,&state.states[row],ws,scratch).await
    }

    pub(crate) fn supports_tail384(sequences: &[Sequence<'_>]) -> bool {
        (1..=3).contains(&sequences.len())
            && sequences.iter().all(|s| matches!(s.inputs.len(),128|256|384)
                && s.start_position % 128 == 0
                && s.start_position.checked_add(s.inputs.len()).is_some()
                && s.inputs[..s.inputs.len()-1].iter().all(|input| !input.logits))
            && sequences.iter().map(|s|s.inputs.len()).sum::<usize>() == 384
            && crate::host::request_pool::independent_sequences(sequences)
    }

    /// Validate real reservations before deriving repeated metadata privately.
    /// The original four-independent-owner constructor keeps its separate guard.
    pub(crate) async fn prepare_tail384(
        &mut self, device:&mut Device, weights:&Weights, sequences:&[Sequence<'_>],
        scratch:&mut Prefill, batched_media:bool, padding:bf16,
    ) -> Result<PreparedTail384> {
        if !Self::supports_tail384(sequences) {
            return Err("Tail384 needs aligned independent owner ranges totaling384".into());
        }
        let mut owner_first_rows=Vec::with_capacity(sequences.len());
        let mut owner_final_rows=Vec::with_capacity(sequences.len());
        let mut chunks=Vec::with_capacity(3);
        for sequence in sequences {
            owner_first_rows.push(chunks.len());
            for (i,inputs) in sequence.inputs.chunks_exact(P128::SIZE).enumerate() {
                chunks.push(Sequence { inputs, start_position:sequence.start_position+i*P128::SIZE,
                    blocks:sequence.blocks });
            }
            owner_final_rows.push(chunks.len()-1);
        }
        debug_assert_eq!(chunks.len(),3);
        let padding_row=&mut self.rows[3];
        padding_row.x=HostTensor::<bf16,m![P128,H]>::from_vec(
            std::iter::repeat_n(padding,P128::SIZE*H::SIZE)).to_hbm(&mut device.pdma).await?;
        padding_row.attn_s=HostTensor::<bf16,m![P128,Ns,Gs,Ds]>::from_vec(
            std::iter::repeat_n(padding,P128::SIZE*Ns::SIZE*Gs::SIZE*Ds::SIZE)).to_hbm(&mut device.pdma).await?;
        padding_row.attn_f=HostTensor::<bf16,m![P128,Gf,Df]>::from_vec(
            std::iter::repeat_n(padding,P128::SIZE*Gf::SIZE*Df::SIZE)).to_hbm(&mut device.pdma).await?;
        Ok(PreparedTail384 { preload_ffn_globals:true, composed_ffn_epilogue:true, projection_owned_ffn:self.has_projection_storage(),
            weight_stationary_output:true, grouped_norm_stores: false, weight_stationary_qkv: true, owner_first_rows, owner_final_rows, states:[
            self.rows[0].prepare_with_media_batch(device,weights,&chunks[0],scratch,true,batched_media).await?,
            self.rows[1].prepare_with_media_batch(device,weights,&chunks[1],scratch,true,batched_media).await?,
            self.rows[2].prepare_with_media_batch(device,weights,&chunks[2],scratch,true,batched_media).await?,
        ]})
    }
    pub(crate) async fn advance_tail384(
        &mut self, device:&mut Device, weights:&Weights, state:&mut PreparedTail384,
        ws:&mut Workspace, scratch:&mut Prefill,
    ) -> Result<bool> {
        let mut execution = GroupExecution::production();
        execution.composed_ffn_epilogue = state.composed_ffn_epilogue;
        execution.projection_owned_ffn = state.projection_owned_ffn;
        execution.preload_ffn_globals = state.preload_ffn_globals;
        execution.weight_stationary_output = state.weight_stationary_output;
        execution.weight_stationary_qkv = state.weight_stationary_qkv;
        execution.grouped_norm_stores = state.grouped_norm_stores;
        self.advance_rows(device,weights,&mut state.states,execution,ws,scratch).await
    }
    pub(crate) async fn finish_tail384(
        &mut self, owner:usize, device:&mut Device, weights:&Weights,
        state:&PreparedTail384, ws:&mut Workspace, scratch:&mut Prefill,
    ) -> Result<()> {
        let row=state.owner_final_rows[owner];
        ws.blocks.clone_from(&state.states[row].blocks);
        self.rows[row].finish(device,weights,&state.states[row],ws,scratch).await
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
    pub(crate) async fn advance_three_owner512(
        &mut self, device:&mut Device, weights:&Weights, state:&mut PreparedThreeOwner512,
        ws:&mut Workspace, scratch:&mut Prefill,
    ) -> Result<bool> {
        self.advance_layer(device,weights,&mut state.chunks,ws,scratch).await
    }
    pub(crate) async fn finish_three_owner512(
        &mut self, owner:usize, device:&mut Device, weights:&Weights,
        state:&PreparedThreeOwner512, ws:&mut Workspace, scratch:&mut Prefill,
    ) -> Result<()> {
        self.finish(state.owner_final_rows[owner],device,weights,&state.chunks,ws,scratch).await
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
        let execution=GroupExecution {
            fused_norms:state.fused_norms, blocked_output:state.blocked_output,
            shard_quant:state.shard_quant, quarter_norm:state.quarter_norm,
            fused_ffn_core:state.fused_ffn_core, quarter_postnorm:state.quarter_postnorm,
            blocked_residual:state.uses_blocked_residual(),
            composed_ffn_epilogue:state.composed_ffn_epilogue,
            projection_owned_ffn:state.projection_owned_ffn,
            preload_ffn_globals:state.preload_ffn_globals,
            weight_stationary_output:state.weight_stationary_output,
            weight_stationary_qkv:state.weight_stationary_qkv,
            grouped_norm_stores:state.grouped_norm_stores,
        };
        self.advance_rows(device,weights,&mut state.states,execution,ws,scratch).await
    }
    async fn advance_rows(
        &mut self, device:&mut Device, weights:&Weights, states:&mut [PreparedWide],
        execution:GroupExecution, ws:&mut Workspace, scratch:&mut Prefill,
    ) -> Result<bool> {
        debug_assert!(matches!(states.len(),2|3|4));
        let index = states[0].next_layer;
        if states.iter().any(|s| s.next_layer != index) {
            return Err("grouped layer cursors diverged".into());
        }
        let layer = weights
            .layers
            .get(index)
            .ok_or("grouped prefill layers already complete")?;
        let mut mark = scratch.profile_mark();
        let blocked_residual = execution.blocked_residual;
        if execution.grouped_norm_stores && !execution.projection_owned_ffn {
            return Err("grouped norm stores require projection-owned FFN".into());
        }
        if execution.projection_owned_ffn && !(execution.composed_ffn_epilogue
            && blocked_residual && execution.fused_ffn_core && execution.preload_ffn_globals
            && self.projection_storage.as_ref().is_some_and(|s|s.layers.len()==crate::LAYERS)) {
            return Err("projection-owned FFN requires prepared storage and the current composed preload route".into());
        }
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
        } else if execution.fused_norms {
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
                if execution.weight_stationary_qkv {
                    // Named candidate factors have the identical blocked Q/K/V
                    // byte order. Each view borrows its existing row owner.
                    launch(crate::ops_qkv_weight_stationary::project_sliding_qkv_four,
                        (device, &a.qkv_input, &a.qkv_scale, &b.qkv_input, &b.qkv_scale,
                         &c.qkv_input, &c.qkv_scale, &d.qkv_input, &d.qkv_scale,
                         &l.q_weight, &l.k_weight, &l.v_weight,
                         &l.q_weight_scale, &l.k_weight_scale, &l.v_weight_scale,
                         unsafe { (oa.q_s.view_mut().reshape(), oa.k_s.view_mut().reshape(), oa.v_s.view_mut().reshape()) },
                         unsafe { (ob.q_s.view_mut().reshape(), ob.k_s.view_mut().reshape(), ob.v_s.view_mut().reshape()) },
                         unsafe { (oc.q_s.view_mut().reshape(), oc.k_s.view_mut().reshape(), oc.v_s.view_mut().reshape()) },
                         unsafe { (od.q_s.view_mut().reshape(), od.k_s.view_mut().reshape(), od.v_s.view_mut().reshape()) })).await?;
                } else {
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
        for row in 0..states.len() {
            let current = &states[row];
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
                if execution.blocked_output && execution.fused_norms {
                    let [oa, ob, oc, od] = &mut self.output_s;
                    if execution.weight_stationary_output {
                        // Same owned H64 buffers and downstream consumer. The
                        // host relabels the proven identical physical byte order.
                        launch(crate::ops_output_weight_stationary::project_sliding_output_four,
                            (device, &a.attn_s, &b.attn_s, &c.attn_s, &d.attn_s,
                             &l.o_weight, &l.o_weight_scale,
                             unsafe { oa.view_mut().reshape() },
                             unsafe { ob.view_mut().reshape() },
                             unsafe { oc.view_mut().reshape() },
                             unsafe { od.view_mut().reshape() })).await?;
                    } else {
                    launch(crate::ops_output_p128_h64_blocked::project_sliding_output_four,
                        (device, &a.attn_s, &b.attn_s, &c.attn_s, &d.attn_s,
                         &l.o_weight, &l.o_weight_scale, oa, ob, oc, od)).await?;
                    }
                } else {
                launch(crate::ops_output_p128_keyrings_h64::project_sliding_output_four,
                    (device, &a.attn_s, &b.attn_s, &c.attn_s, &d.attn_s,
                     &l.o_weight, &l.o_weight_scale, &mut a.projected,
                     &mut b.projected, &mut c.projected, &mut d.projected)).await?;
                }
            }
            Layer::Full(l) => {
                if execution.weight_stationary_output {
                    // Named Ho/Hr/Ht factors preserve canonical [P128,H] bytes.
                    launch(crate::ops_full_output_weight_stationary::project_full_output_four,
                        (device, &a.attn_f, &b.attn_f, &c.attn_f, &d.attn_f,
                         &l.o_weight, &l.o_weight_scale,
                         unsafe { a.projected.view_mut().reshape() },
                         unsafe { b.projected.view_mut().reshape() },
                         unsafe { c.projected.view_mut().reshape() },
                         unsafe { d.projected.view_mut().reshape() })).await?;
                } else {
                launch(crate::ops_output_p128_four_keyrings::project_full_output_four,
                    (device, &a.attn_f, &b.attn_f, &c.attn_f, &d.attn_f,
                     &l.o_weight, &l.o_weight_scale, &mut a.projected,
                     &mut b.projected, &mut c.projected, &mut d.projected)).await?;
                }
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
        if execution.composed_ffn_epilogue && blocked_residual && execution.fused_ffn_core {
            let [a, b, c, d] = &self.rows;
            let norms = (post_attention, pre, post, gate);
            if execution.projection_owned_ffn {
                let storage = self.projection_storage.as_mut().ok_or("joined FFN weights were not prepared")?;
                let joined = storage.layers.get(index).ok_or("joined FFN layer is missing")?;
                if execution.grouped_norm_stores {
                match layer {
                    Layer::Sliding(_) => launch(crate::ops_group_ffn_projection_storegroup::sliding_projection_owned,
                        (device, (&self.output_s[0],&self.output_s[1],&self.output_s[2],&self.output_s[3]),
                         norms,&mut self.residual_blocked,&joined.weight,&joined.scale,&mlp.native_fp8.down,
                         &mlp.native_fp8.down_dequant,&mlp.up_global_scale,&mlp.gate_global_scale,&mlp.down_global_scale,
                         &mut self.q_input,&mut self.input_scale,&mut storage.output,&mut self.q_hidden,
                         &mut self.hidden_scale,&mut self.projected)).await?,
                    Layer::Full(_) => launch(crate::ops_group_ffn_projection_storegroup::full_projection_owned,
                        (device,(&a.projected,&b.projected,&c.projected,&d.projected),
                         norms,&mut self.residual_blocked,&joined.weight,&joined.scale,&mlp.native_fp8.down,
                         &mlp.native_fp8.down_dequant,&mlp.up_global_scale,&mlp.gate_global_scale,&mlp.down_global_scale,
                         &mut self.q_input,&mut self.input_scale,&mut storage.output,&mut self.q_hidden,
                         &mut self.hidden_scale,&mut self.projected)).await?,
                }
                } else {
                match layer {
                    Layer::Sliding(_) => launch(crate::ops_group_ffn_projection::sliding_projection_owned,
                        (device, (&self.output_s[0],&self.output_s[1],&self.output_s[2],&self.output_s[3]),
                         norms,&mut self.residual_blocked,&joined.weight,&joined.scale,&mlp.native_fp8.down,
                         &mlp.native_fp8.down_dequant,&mlp.up_global_scale,&mlp.gate_global_scale,&mlp.down_global_scale,
                         &mut self.q_input,&mut self.input_scale,&mut storage.output,&mut self.q_hidden,
                         &mut self.hidden_scale,&mut self.projected)).await?,
                    Layer::Full(_) => launch(crate::ops_group_ffn_projection::full_projection_owned,
                        (device,(&a.projected,&b.projected,&c.projected,&d.projected),
                         norms,&mut self.residual_blocked,&joined.weight,&joined.scale,&mlp.native_fp8.down,
                         &mlp.native_fp8.down_dequant,&mlp.up_global_scale,&mlp.gate_global_scale,&mlp.down_global_scale,
                         &mut self.q_input,&mut self.input_scale,&mut storage.output,&mut self.q_hidden,
                         &mut self.hidden_scale,&mut self.projected)).await?,
                }
                }
            } else if execution.preload_ffn_globals {
            match layer {
                Layer::Sliding(_) => launch(crate::ops_group_ffn_epilogue::sliding_globals,
                    (device, (&self.output_s[0], &self.output_s[1], &self.output_s[2], &self.output_s[3]),
                     norms, &mut self.residual_blocked,
                     &mlp.native_fp8.up, &mlp.native_fp8.gate, &mlp.native_fp8.down,
                     &mlp.native_fp8.up_dequant, &mlp.native_fp8.gate_dequant, &mlp.native_fp8.down_dequant,
                     &mlp.up_global_scale, &mlp.gate_global_scale, &mlp.down_global_scale,
                     &mut self.q_input, &mut self.input_scale, &mut self.up, &mut self.gate,
                     &mut self.q_hidden, &mut self.hidden_scale, &mut self.projected)).await?,
                Layer::Full(_) => launch(crate::ops_group_ffn_epilogue::full_globals,
                    (device, (&a.projected, &b.projected, &c.projected, &d.projected),
                     norms, &mut self.residual_blocked,
                     &mlp.native_fp8.up, &mlp.native_fp8.gate, &mlp.native_fp8.down,
                     &mlp.native_fp8.up_dequant, &mlp.native_fp8.gate_dequant, &mlp.native_fp8.down_dequant,
                     &mlp.up_global_scale, &mlp.gate_global_scale, &mlp.down_global_scale,
                     &mut self.q_input, &mut self.input_scale, &mut self.up, &mut self.gate,
                     &mut self.q_hidden, &mut self.hidden_scale, &mut self.projected)).await?,
            }
            } else {
            match layer {
                Layer::Sliding(_) => launch(crate::ops_group_ffn_epilogue::sliding,
                    (device, (&self.output_s[0], &self.output_s[1], &self.output_s[2], &self.output_s[3]),
                     norms, &mut self.residual_blocked,
                     &mlp.native_fp8.up, &mlp.native_fp8.gate, &mlp.native_fp8.down,
                     &mlp.native_fp8.up_dequant, &mlp.native_fp8.gate_dequant, &mlp.native_fp8.down_dequant,
                     &mlp.up_global_scale, &mlp.gate_global_scale, &mlp.down_global_scale,
                     &mut self.q_input, &mut self.input_scale, &mut self.up, &mut self.gate,
                     &mut self.q_hidden, &mut self.hidden_scale, &mut self.projected)).await?,
                Layer::Full(_) => launch(crate::ops_group_ffn_epilogue::full,
                    (device, (&a.projected, &b.projected, &c.projected, &d.projected),
                     norms, &mut self.residual_blocked,
                     &mlp.native_fp8.up, &mlp.native_fp8.gate, &mlp.native_fp8.down,
                     &mlp.native_fp8.up_dequant, &mlp.native_fp8.gate_dequant, &mlp.native_fp8.down_dequant,
                     &mlp.up_global_scale, &mlp.gate_global_scale, &mlp.down_global_scale,
                     &mut self.q_input, &mut self.input_scale, &mut self.up, &mut self.gate,
                     &mut self.q_hidden, &mut self.hidden_scale, &mut self.projected)).await?,
            }
            }
            scratch.record_optional_profile("group.ffn.composed_epilogue", &mut mark);
        } else {
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
        } else if execution.fused_norms {
            let [a, b, c, d] = &mut self.rows;
            if execution.blocked_output && matches!(layer, Layer::Sliding(_)) {
                if execution.shard_quant {
                    if execution.quarter_norm {
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
        if execution.fused_ffn_core {
            if execution.preload_ffn_globals {
            launch(crate::ops_native_fp8_p512_core_globals::ffn_core,
                (device, &self.q_input, &self.input_scale,
                 &mlp.native_fp8.up, &mlp.native_fp8.gate, &mlp.native_fp8.down,
                 &mlp.native_fp8.up_dequant, &mlp.native_fp8.gate_dequant, &mlp.native_fp8.down_dequant,
                 &mlp.up_global_scale, &mlp.gate_global_scale, &mlp.down_global_scale,
                 &mut self.up, &mut self.gate, &mut self.q_hidden, &mut self.hidden_scale,
                 &mut self.projected)).await?;
            } else {
            launch(crate::ops_native_fp8_p512_core_fused::ffn_core,
                (device, &self.q_input, &self.input_scale,
                 &mlp.native_fp8.up, &mlp.native_fp8.gate, &mlp.native_fp8.down,
                 &mlp.native_fp8.up_dequant, &mlp.native_fp8.gate_dequant, &mlp.native_fp8.down_dequant,
                 &mlp.up_global_scale, &mlp.gate_global_scale, &mlp.down_global_scale,
                 &mut self.up, &mut self.gate, &mut self.q_hidden, &mut self.hidden_scale,
                 &mut self.projected)).await?;
            }
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
        } else if execution.fused_norms {
            let [a, b, c, d] = &mut self.rows;
            if execution.quarter_postnorm {
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
        }
        if blocked_residual && index + 1 == weights.layers.len() {
            // Restore all canonical quarters once before any owner-specific finish.
            let [a, b, c, d] = &mut self.rows;
            launch(crate::ops_norm_blocked_residual::unpack_four,
                (device, &self.residual_blocked, &mut a.x, &mut b.x, &mut c.x, &mut d.x)).await?;
            scratch.record_optional_profile("group.residual.unpack", &mut mark);
        }
        for current in states.iter_mut() {
            current.next_layer += 1;
        }
        Ok(states[0].next_layer == weights.layers.len())
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

#[cfg(test)]
mod tail256_tests {
    use super::*;
    use crate::host::model::Input;
    fn inputs(count:usize)->Vec<Input<'static>> {
        (0..count).map(|i| Input {token:2+i,soft:None,logits:i+1==count,image:None,audio:None}).collect()
    }
    #[test]
    fn one_owner_halves_are_private_and_do_not_weaken_independent_groups() {
        let a=inputs(256);let blocks=runtime::Blocks{sliding:7,full:vec![255,13,12,11,10]};
        for start in [0,256,896,1920] {
            assert!(GroupPrefill::supports_tail256(&[Sequence{inputs:&a,start_position:start,blocks:&blocks}]));
            let repeated=[Sequence{inputs:&a[..128],start_position:start,blocks:&blocks},
                Sequence{inputs:&a[128..],start_position:start+128,blocks:&blocks}];
            assert!(!GroupPrefill::supports_tail256(&repeated));
            assert!(!GroupPrefill::supports(&repeated));
        }
    }
    #[test]
    fn tail_pair_rejects_aliases_and_does_not_enable_four_owner_constructor() {
        let a=inputs(128);let b=inputs(128);
        let aa=runtime::Blocks{sliding:7,full:vec![255,13,12,11,10]};
        let bb=runtime::Blocks{sliding:9,full:vec![254,23,22,21,20]};
        let pair=[Sequence{inputs:&a,start_position:896,blocks:&aa},Sequence{inputs:&b,start_position:1920,blocks:&bb}];
        assert!(GroupPrefill::supports_tail256(&pair));assert!(!GroupPrefill::supports(&pair));
        for alias in [runtime::Blocks{sliding:7,full:bb.full.clone()},runtime::Blocks{sliding:9,full:aa.full.clone()}] {
            assert!(!GroupPrefill::supports_tail256(&[Sequence{inputs:&a,start_position:896,blocks:&aa},Sequence{inputs:&b,start_position:1920,blocks:&alias}]));
        }
    }
    #[test]
    fn tail_rejects_partial_unaligned_overflow_and_intermediate_logits() {
        let mut a=inputs(256);let blocks=runtime::Blocks{sliding:7,full:vec![255]};
        for (count,start) in [(128,0),(255,0),(256,1),(256,usize::MAX-127)] {
            assert!(!GroupPrefill::supports_tail256(&[Sequence{inputs:&a[..count],start_position:start,blocks:&blocks}]));
        }
        a[127].logits=true;
        assert!(!GroupPrefill::supports_tail256(&[Sequence{inputs:&a,start_position:0,blocks:&blocks}]));
        assert!(!GroupPrefill::supports_tail256(&[]));
    }
}

#[cfg(test)]
mod tail384_tests {
    use super::*;
    use crate::host::model::Input;
    fn inputs(count:usize)->Vec<Input<'static>> {
        (0..count).map(|i|Input{token:i+2,soft:None,logits:i+1==count,image:None,audio:None}).collect()
    }
    #[test]
    fn accepts_only_three_real_quarters_and_keeps_four_owner_guard() {
        let blocks:Vec<_>=(0..4).map(|i|runtime::Blocks{sliding:7+i,full:vec![31-i,63-i,95-i,127-i,159-i]}).collect();
        for counts in [vec![384],vec![256,128],vec![128,256],vec![128,128,128]] {
            let rows:Vec<_>=counts.iter().map(|&n|inputs(n)).collect();
            for start in [0,256,896,1664,1920] {
                let sequences:Vec<_>=rows.iter().zip(&blocks).map(|(inputs,blocks)|Sequence{inputs,start_position:start,blocks}).collect();
                assert!(GroupPrefill::supports_tail384(&sequences));
                assert!(!GroupPrefill::supports(&sequences));
                assert!(!GroupPrefill::supports_tail256(&sequences));
            }
        }
        let row=inputs(128);
        let quad:Vec<_>=blocks.iter().map(|blocks|Sequence{inputs:&row,start_position:0,blocks}).collect();
        assert!(GroupPrefill::supports(&quad));assert!(!GroupPrefill::supports_tail384(&quad));
    }
    #[test]
    fn refuses_aliases_before_deriving_repeated_owner_metadata() {
        let row=inputs(384);let a=runtime::Blocks{sliding:7,full:vec![255,17,23,11,8]};
        let b=runtime::Blocks{sliding:9,full:vec![254,16,22,10,6]};
        for alias in [runtime::Blocks{sliding:a.sliding,full:b.full.clone()},runtime::Blocks{sliding:b.sliding,full:a.full.clone()}] {
            assert!(!GroupPrefill::supports_tail384(&[
                Sequence{inputs:&row[..256],start_position:896,blocks:&a},
                Sequence{inputs:&row[256..],start_position:1920,blocks:&alias}]));
        }
        // Repeated quarters may only be created inside the one-owner constructor.
        let repeated:Vec<_>=row.chunks_exact(128).enumerate().map(|(i,inputs)|Sequence{inputs,start_position:128*i,blocks:&a}).collect();
        assert!(!GroupPrefill::supports_tail384(&repeated));
    }
    #[test]
    fn rejects_short_unaligned_overflow_and_early_logits() {
        let mut row=inputs(384);let blocks=runtime::Blocks{sliding:7,full:vec![255]};
        for (count,start) in [(128,0),(256,0),(383,0),(384,1),(384,usize::MAX-127)] {
            assert!(!GroupPrefill::supports_tail384(&[Sequence{inputs:&row[..count],start_position:start,blocks:&blocks}]));
        }
        row[255].logits=true;
        assert!(!GroupPrefill::supports_tail384(&[Sequence{inputs:&row,start_position:0,blocks:&blocks}]));
        assert!(!GroupPrefill::supports_tail384(&[]));
    }
}

#[cfg(test)]
mod three_owner512_tests {
    use super::*;
    use crate::host::model::Input;
    fn inputs(count:usize)->Vec<Input<'static>> {
        (0..count).map(|i|Input{token:i+2,soft:None,logits:i+1==count,image:None,audio:None}).collect()
    }
    fn blocks()->Vec<runtime::Blocks> {
        (0..4).map(|i|runtime::Blocks{sliding:7+i,full:vec![31-i,63-i,95-i,127-i,159-i]}).collect()
    }
    #[test]
    fn three_real_owner_permutations_keep_original_four_owner_validation() {
        let blocks=blocks();
        for double in 0..3 {
            let rows:Vec<_>=(0..3).map(|i|inputs(if i==double{256}else{128})).collect();
            for start in [0,256,896,1792,1920] {
                let sequences:Vec<_>=rows.iter().zip(&blocks).map(|(inputs,blocks)|Sequence{inputs,start_position:start,blocks}).collect();
                assert!(GroupPrefill::supports_three_owner512(&sequences));
                assert!(!GroupPrefill::supports(&sequences));
                assert!(!GroupPrefill::supports_prompt_pair512(&sequences));
                assert!(!GroupPrefill::supports_tail384(&sequences));
            }
        }
        let row=inputs(128);
        let quad:Vec<_>=blocks.iter().map(|blocks|Sequence{inputs:&row,start_position:0,blocks}).collect();
        assert!(GroupPrefill::supports(&quad));assert!(!GroupPrefill::supports_three_owner512(&quad));
    }
    #[test]
    fn real_owner_aliases_cannot_enter_private_chronological_derivation() {
        let mut blocks=blocks();let rows=[inputs(256),inputs(128),inputs(128)];
        for duplicate in 1..3 {
            let original=blocks[duplicate].clone();blocks[duplicate].sliding=blocks[0].sliding;
            let seq:Vec<_>=rows.iter().zip(&blocks).map(|(inputs,blocks)|Sequence{inputs,start_position:896,blocks}).collect();
            assert!(!GroupPrefill::supports_three_owner512(&seq));blocks[duplicate]=original.clone();
            blocks[duplicate].full[2]=blocks[0].full[2];
            let seq:Vec<_>=rows.iter().zip(&blocks).map(|(inputs,blocks)|Sequence{inputs,start_position:896,blocks}).collect();
            assert!(!GroupPrefill::supports_three_owner512(&seq));blocks[duplicate]=original;
        }
    }
    #[test]
    fn rejects_partial_bad_sum_unaligned_overflow_and_early_logits() {
        let blocks=blocks();
        for counts in [[128,128,128],[256,256,128],[255,128,129],[384,128,0]] {
            let rows:Vec<_>=counts.into_iter().map(inputs).collect();
            let seq:Vec<_>=rows.iter().zip(&blocks).map(|(inputs,blocks)|Sequence{inputs,start_position:0,blocks}).collect();
            assert!(!GroupPrefill::supports_three_owner512(&seq));
        }
        let mut rows=[inputs(256),inputs(128),inputs(128)];
        for start in [1,usize::MAX-127] {
            let seq:Vec<_>=rows.iter().zip(&blocks).map(|(inputs,blocks)|Sequence{inputs,start_position:start,blocks}).collect();
            assert!(!GroupPrefill::supports_three_owner512(&seq));
        }
        rows[0][127].logits=true;
        let seq:Vec<_>=rows.iter().zip(&blocks).map(|(inputs,blocks)|Sequence{inputs,start_position:0,blocks}).collect();
        assert!(!GroupPrefill::supports_three_owner512(&seq));
    }
}
