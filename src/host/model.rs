//! Batched model calculation over caller-owned KV blocks.

use super::{audio::AudioFrames, generate::SoftToken, image::ImagePatches, load, runtime};
use crate::{Chip, axes::{B, W}};
use furiosa_opt_std::prelude::*;

pub use super::prefill::StageTiming;
use super::prefill::wide::group::{GroupPrefill, PreparedGroup, PreparedPrompt512, PreparedPrompt384128, PreparedTail256, PreparedTail384, PreparedThreeOwner512};
use super::prefill::{
    Prefill, PreparedNarrow,
    wide::{PreparedWide, WidePrefill},
};
use runtime::{Blocks, Workspace};
pub type Logits = HbmTensor<bf16, Chip, m![W]>;
pub type Batch16Logits = HbmTensor<bf16, Chip, m![B, W]>;

/// Host-observed phase times for a scheduling quantum. These fields are supplied
/// by the caller, so no timing state survives an aborted model future.
#[derive(Default)]
pub(crate) struct PrefillAdvanceTiming {
    pub layer: Option<std::time::Duration>,
    pub finish: Option<std::time::Duration>,
}

/// Physical output of the current computation. Only input rows requesting logits
/// are valid, just as with `forward_sequences`; blocked rows retain request order.
pub enum ForwardOutput<'a> {
    Rows(&'a [Logits]),
    Batch16(&'a Batch16Logits, usize),
    Blocked64(&'a HbmTensor<bf16, Chip, crate::ops_r64_io_blocked::BlockedLogits>, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputLayout {
    Rows,
    Batch16,
    Blocked64,
}

/// One request's next token. `logits` is false for prompt tokens whose output is discarded.
pub struct Input<'a> {
    pub token: usize,
    pub soft: Option<SoftToken>,
    pub logits: bool,
    pub image: Option<&'a ImagePatches>,
    pub audio: Option<&'a AudioFrames>,
}

/// Consecutive inputs from one request, backed by that request's KV allocation.
/// Only the last input may request logits; earlier rows still compute their full
/// causal history. The caller allocates blocks through the end of the sequence.
pub struct Sequence<'a> {
    pub inputs: &'a [Input<'a>],
    pub start_position: usize,
    pub blocks: &'a Blocks,
}

pub struct Model {
    weights: load::Weights,
    decode: runtime::Decode,
    prefill: Prefill,
    decode64: Option<super::decode_r64::Decode64>,
    wide: WidePrefill,
    group: Option<GroupPrefill>,
    active_group: Option<PreparedGroup>,
    active_prompt384128: Option<PreparedPrompt384128>,
    active_tail256: Option<PreparedTail256>,
    active_tail384: Option<PreparedTail384>,
    active_three_owner512: Option<PreparedThreeOwner512>,
    active_prefill: Option<ActivePrefill>,
    logits: Vec<Logits>,
}

enum ActivePrefill {
    Wide(PreparedWide),
    Narrow(PreparedNarrow),
    Prompt512(PreparedPrompt512),
}
impl ActivePrefill {
    fn blocks(&self) -> &Blocks {
        match self {
            Self::Wide(state) => &state.blocks,
            Self::Narrow(state) => &state.blocks,
            Self::Prompt512(state) => state.blocks(),
        }
    }
}

/// Diagnostic partial-wide dispatch stays explicit; production runs wide only
/// when all128 rows are live. Both paths preserve each request's own history.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dispatch {
    Serial,
    Automatic,
    B16AttentionOutputDiagnostic,
    OriginalB16AttentionOutputDiagnostic,
    WideDiagnostic,
    NativeQkvDiagnostic,
    OriginalAttentionDiagnostic,
    OriginalFullAttentionDiagnostic,
    DirectFullAttentionDiagnostic,
    Bf16QkvDiagnostic,
    Decode64Diagnostic,
    Decode32Diagnostic,
    Decode32OutputDiagnostic,
    OriginalDecodeSlidingDiagnostic,
    OriginalDecodeHeadRowsDiagnostic,
}
impl Dispatch {
    fn use_wide(self, sequence: &Sequence<'_>) -> bool {
        WidePrefill::supports(sequence)
            && match self {
                Self::Serial => false,
                Self::Automatic | Self::B16AttentionOutputDiagnostic | Self::OriginalB16AttentionOutputDiagnostic => sequence.inputs.len() == crate::axes::P128::SIZE,
                Self::WideDiagnostic | Self::NativeQkvDiagnostic | Self::Bf16QkvDiagnostic | Self::OriginalAttentionDiagnostic | Self::OriginalFullAttentionDiagnostic | Self::DirectFullAttentionDiagnostic => true,
                Self::Decode64Diagnostic | Self::Decode32Diagnostic | Self::Decode32OutputDiagnostic | Self::OriginalDecodeSlidingDiagnostic | Self::OriginalDecodeHeadRowsDiagnostic => sequence.inputs.len() == crate::axes::P128::SIZE,
            }
    }
}

impl Model {
    /// Diagnostic storage-control gate. Serving defaults to blocked decode outputs.
    /// This changes only projection scratch layout and its matching head consumer;
    /// request histories, attention, arithmetic and logits selection are unchanged.
    /// Diagnostic control: both choices use identical BF16 partial arithmetic.
    pub fn set_b16_output_hsplit(&mut self, enabled: bool) {
        self.prefill.set_output_hsplit(enabled);
    }

    /// Scoped exact RMS/QKV/head launch-composition diagnostic.
    /// The original four-call prefix remains available as a control.
    pub fn set_b16_prefix_fused(&mut self, enabled: bool) {
        self.prefill.set_prefix_fused(enabled);
    }
    /// Scoped exact post-attention and FFN launch-composition diagnostic.
    pub fn set_b16_epilogue_fused(&mut self, enabled: bool) {
        self.prefill.set_epilogue_fused(enabled);
    }

    /// Diagnostic control for omitting legacy offsets unused by pooled B16.
    pub fn set_b16_compact_decode_metadata(&mut self, enabled: bool) {
        self.prefill.set_compact_decode_metadata(enabled);
    }

    /// Diagnostic B16 FFN dispatch control; scratch, norms and caches stay unchanged.
    pub fn set_b16_staged_ffn(&mut self, enabled: bool) {
        self.prefill.set_staged_ffn(enabled);
    }

    /// Diagnostic arithmetic control for R64 FFN only; both paths use identical norms and caches.
    pub fn set_r64_staged_ffn(&mut self, enabled: bool) {
        if let Some(decode) = &mut self.decode64 {
            decode.set_staged_ffn(enabled);
        }
    }

    pub fn set_decode_blocked_output(&mut self, enabled: bool) {
        self.prefill.set_decode_blocked_output(enabled);
        if let Some(decode) = &mut self.decode64 {
            decode.set_decode_blocked_output(enabled);
        }
    }

    /// Diagnostics only: collect host-observed durations of batched execution stages.
    pub fn begin_stage_profile(&mut self) {
        self.prefill.begin_profile();
        if let Some(decode) = &mut self.decode64 {
            decode.begin_profile();
        }
    }
    pub fn take_stage_profile(&mut self) -> Vec<StageTiming> {
        let mut stages = self.prefill.take_profile();
        if let Some(decode) = &mut self.decode64 {
            stages.extend(decode.take_profile());
        }
        stages
    }

    pub async fn new(
        device: &mut Device,
        weights: load::Weights,
        workspace: &mut Workspace,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut logits = Vec::with_capacity(workspace.requests());
        for _ in 0..workspace.requests() {
            logits.push(
                HostTensor::<bf16, m![W]>::zero()
                    .to_hbm(&mut device.pdma)
                    .await?,
            );
        }
        Ok(Self {
            weights,
            decode: workspace.begin_decode(device).await?,
            prefill: Prefill::new(device).await?,
            decode64: if workspace.is_canonical() {
                Some(super::decode_r64::Decode64::new(device).await?)
            } else {
                None
            },
            wide: WidePrefill::new(device).await?,
            group: None,
            active_group: None,
            active_prompt384128: None,
            active_tail256: None,
            active_tail384: None,
            active_three_owner512: None,
            active_prefill: None,
            logits,
        })
    }

    /// Physical ring currently owning a persistent prompt checkpoint, if any.
    pub fn active_prefill_ring(&self) -> Option<usize> {
        self.active_prefill
            .as_ref()
            .map(|state| state.blocks().sliding)
            .or_else(|| self.active_group.as_ref().map(|state|state.blocks()[0].sliding))
            .or_else(|| self.active_prompt384128.as_ref().map(|state|state.blocks()[0].sliding))
            .or_else(|| self.active_tail256.as_ref().map(|state|state.blocks()[0].sliding))
            .or_else(|| self.active_tail384.as_ref().map(|state|state.blocks()[0].sliding))
            .or_else(|| self.active_three_owner512.as_ref().map(|state|state.blocks()[0].sliding))
    }

    /// Every request allocation held by an active prompt scheduling quantum.
    pub fn active_prefill_rings(&self) -> Vec<usize> {
        self.active_allocations()
            .into_iter()
            .map(|b| b.sliding)
            .collect()
    }
    fn active_allocations(&self) -> Vec<&Blocks> {
        let mut blocks = Vec::new();
        if let Some(active) = &self.active_prefill {
            blocks.push(active.blocks());
        }
        if let Some(active) = &self.active_group {
            blocks.extend(active.blocks());
        }
        if let Some(active) = &self.active_prompt384128 {
            blocks.extend(active.blocks());
        }
        if let Some(active) = &self.active_tail256 { blocks.extend(active.blocks()); }
        if let Some(active) = &self.active_tail384 { blocks.extend(active.blocks()); }
        if let Some(active) = &self.active_three_owner512 { blocks.extend(active.blocks()); }
        blocks
    }
    /// Allocate the optional grouped prompt workspace before serving requests.
    /// Diagnostics may still allocate lazily in `start_prefill_group`.
    pub async fn prepare_group_scratch(
        &mut self,
        device: &mut Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.group.is_none() {
            self.group = Some(GroupPrefill::new(device).await?);
        }
        Ok(())
    }

    /// Scoped exact sliding/full O control. New prepared states default to the
    /// original producer; only a not-yet-started grouped state may opt in.
    /// No model-wide selection survives completion, error or cancellation.
    pub fn set_active_prefill_weight_stationary_output(
        &mut self, enabled: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(state) = self.active_group.as_mut() { return state.weight_stationary_output(enabled); }
        if let Some(state) = self.active_prompt384128.as_mut() { return state.weight_stationary_output(enabled); }
        if let Some(state) = self.active_tail256.as_mut() { return state.weight_stationary_output(enabled); }
        if let Some(state) = self.active_tail384.as_mut() { return state.weight_stationary_output(enabled); }
        if let Some(state) = self.active_three_owner512.as_mut() { return state.weight_stationary_output(enabled); }
        if let Some(ActivePrefill::Prompt512(state)) = self.active_prefill.as_mut() {
            return state.weight_stationary_output(enabled);
        }
        Err("no grouped512-density prompt is awaiting its first layer".into())
    }

    /// Exact grouped FFN launch-composition control, enabled for new prepared
    /// states after all owner-route gates. The selector can change only
    /// before its first layer; no model-wide mode survives completion or drop.
    pub fn set_active_prefill_ffn_epilogue_composed(
        &mut self, enabled: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(state) = self.active_group.as_mut() { return state.composed_ffn_epilogue(enabled); }
        if let Some(state) = self.active_prompt384128.as_mut() { return state.composed_ffn_epilogue(enabled); }
        if let Some(state) = self.active_tail256.as_mut() { return state.composed_ffn_epilogue(enabled); }
        if let Some(state) = self.active_tail384.as_mut() { return state.composed_ffn_epilogue(enabled); }
        if let Some(state) = self.active_three_owner512.as_mut() { return state.composed_ffn_epilogue(enabled); }
        if let Some(ActivePrefill::Prompt512(state)) = self.active_prefill.as_mut() {
            return state.composed_ffn_epilogue(enabled);
        }
        Err("no grouped512-density prompt is awaiting its first layer".into())
    }

    /// Experimental startup-only storage preparation. It adds immutable joined
    /// copies of the existing native UP/GATE tensors; no arithmetic or dispatch
    /// changes until a prepared execution explicitly opts in before layer zero.
    pub async fn prepare_projection_owned_ffn(
        &mut self, device: &mut Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.active_allocations().is_empty() {
            return Err("joined FFN weights must be prepared with no active prompt".into());
        }
        self.prepare_group_scratch(device).await?;
        self.group.as_mut().unwrap().prepare_projection_storage(device, &self.weights).await
    }

    /// Default-off, prepared-state-only diagnostic; existing owner validation,
    /// attention, norms and per-layer completion rules are unchanged.
    pub fn set_active_prefill_projection_owned_ffn(
        &mut self, enabled: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if enabled && !self.group.as_ref().is_some_and(|g|g.has_projection_storage()) {
            return Err("joined FFN weights were not prepared".into());
        }
        if let Some(state) = self.active_group.as_mut() { return state.projection_owned_ffn(enabled); }
        if let Some(state) = self.active_prompt384128.as_mut() { return state.projection_owned_ffn(enabled); }
        if let Some(state) = self.active_tail256.as_mut() { return state.projection_owned_ffn(enabled); }
        if let Some(state) = self.active_tail384.as_mut() { return state.projection_owned_ffn(enabled); }
        if let Some(state) = self.active_three_owner512.as_mut() { return state.projection_owned_ffn(enabled); }
        if let Some(ActivePrefill::Prompt512(state)) = self.active_prefill.as_mut() {
            return state.projection_owned_ffn(enabled);
        }
        Err("no grouped512-density prompt is awaiting its first layer".into())
    }

    /// Scoped exact FFN-global-load control, enabled for new prepared owner
    /// shapes after model/cache gates. Selection is permitted only before layer0;
    /// no persistent Model or Worker mode survives completion/cancellation.
    pub fn set_active_prefill_ffn_global_preload(
        &mut self, enabled:bool,
    ) -> Result<(),Box<dyn std::error::Error>> {
        if let Some(state)=self.active_group.as_mut() {return state.preload_ffn_globals(enabled);}
        if let Some(state)=self.active_prompt384128.as_mut() {return state.preload_ffn_globals(enabled);}
        if let Some(state)=self.active_tail256.as_mut() {return state.preload_ffn_globals(enabled);}
        if let Some(state)=self.active_tail384.as_mut() {return state.preload_ffn_globals(enabled);}
        if let Some(state)=self.active_three_owner512.as_mut() {return state.preload_ffn_globals(enabled);}
        if let Some(ActivePrefill::Prompt512(state))=self.active_prefill.as_mut() {
            return state.preload_ffn_globals(enabled);
        }
        Err("no grouped512-density prompt is awaiting its first layer".into())
    }

    /// Four full P128 chunks share QKV weight reads and blocked P512 FFN.
    /// Keep all four reservations until advance_prefill_group returns four final outputs.
    pub async fn start_prefill_group(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_group_with_media_batch(device, sequences, workspace, super::prefill::wide::MEDIA_BATCH_ENABLED).await
    }

    /// Per-call media preparation choice, applied before any embedding work.
    /// No selector or borrowed media reference survives preparation.
    pub async fn start_prefill_group_with_media_batch(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
        batched_media: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.active_allocations().is_empty() {
            return Err("another prefill is in progress".into());
        }
        if sequences.len() > self.logits.len() {
            return Err("too many grouped request sequences".into());
        }
        if !workspace.is_canonical() || !GroupPrefill::supports(sequences) {
            return Err("grouped prefill needs canonical independent128-token chunks".into());
        }
        for sequence in sequences {
            validate_sequence(sequence, workspace.capacity(), workspace.requests())?;
            validate_physical_pages(sequence.blocks, workspace.physical_full_pages())?;
        }
        self.prepare_group_scratch(device).await?;
        self.active_group = Some(
            self.group
                .as_mut()
                .unwrap()
                .prepare(device, &self.weights, sequences, &mut self.prefill, batched_media)
                .await?,
        );
        Ok(())
    }
    /// Diagnostic control: retain the original normalization/packing launches
    /// for this one independently validated group. The selector belongs to its
    /// prepared state, so no mode persists after completion or cancellation.
    pub async fn start_prefill_group_original_norms(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_group(device, sequences, workspace).await?;
        self.active_group.as_mut().unwrap().original_norms();
        Ok(())
    }
    /// Same prepared group with canonical O stores; all norm fusions remain enabled.
    pub async fn start_prefill_group_original_output(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_group(device, sequences, workspace).await?;
        self.active_group.as_mut().unwrap().original_output();
        Ok(())
    }
    /// Same prepared execution with the previous full-row quantizer; no persistent mode.
    pub async fn start_prefill_group_original_shard_quant(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_group(device, sequences, workspace).await?;
        self.active_group.as_mut().unwrap().original_shard_quant();
        Ok(())
    }
    /// Same prepared execution using the previous independently normalized quarters.
    pub async fn start_prefill_group_original_quarter_norm(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_group(device, sequences, workspace).await?;
        self.active_group.as_mut().unwrap().original_quarter_norm();
        Ok(())
    }
    /// Same prepared group with four FFN core launches; all other kernels are unchanged.
    /// The selector belongs to this execution and is dropped on completion/cancellation.
    pub async fn start_prefill_group_original_ffn_core(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_group(device, sequences, workspace).await?;
        self.active_group.as_mut().unwrap().original_ffn_core();
        Ok(())
    }
    /// Same prepared execution using the previous sequential post-FFN normalization quarters.
    pub async fn start_prefill_group_original_quarter_postnorm(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_group(device, sequences, workspace).await?;
        self.active_group.as_mut().unwrap().original_quarter_postnorm();
        Ok(())
    }
    /// Same prepared execution using the canonical residual storage with every other default kernel unchanged.
    pub async fn start_prefill_group_original_blocked_residual(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_group(device, sequences, workspace).await?;
        self.active_group.as_mut().unwrap().original_blocked_residual();
        Ok(())
    }
    /// A completed layer is the only safe yield point. On an execution error or
    /// an aborted in-flight future, discard this model/workspace; do not retry
    /// a partially updated group. The caller retains all four allocations.
    pub async fn advance_prefill_group(
        &mut self,
        device: &mut Device,
        workspace: &mut Workspace,
    ) -> Result<Option<&[Logits]>, Box<dyn std::error::Error>> {
        self.advance_prefill_group_measured(device, workspace, None)
            .await
    }
    pub(crate) async fn advance_prefill_group_measured(
        &mut self,
        device: &mut Device,
        workspace: &mut Workspace,
        mut timing: Option<&mut PrefillAdvanceTiming>,
    ) -> Result<Option<&[Logits]>, Box<dyn std::error::Error>> {
        if let Some(timing) = timing.as_mut() {
            **timing = PrefillAdvanceTiming::default();
        }
        let state = self
            .active_group
            .as_mut()
            .ok_or("no grouped prefill is active")?;
        let group = self.group.as_mut().ok_or("grouped scratch is missing")?;
        let mark = timing.as_ref().map(|_| std::time::Instant::now());
        let complete = group
            .advance_layer(device, &self.weights, state, workspace, &mut self.prefill)
            .await?;
        if let Some(timing) = timing.as_mut() {
            timing.layer = Some(mark.unwrap().elapsed());
        }
        if !complete {
            return Ok(None);
        }
        let mark = timing.as_ref().map(|_| std::time::Instant::now());
        for row in 0..4 {
            std::mem::swap(&mut self.logits[row], &mut workspace.logits);
            let result = group
                .finish(
                    row,
                    device,
                    &self.weights,
                    state,
                    workspace,
                    &mut self.prefill,
                )
                .await;
            std::mem::swap(&mut self.logits[row], &mut workspace.logits);
            result?;
        }
        self.active_group = None;
        if let Some(timing) = timing.as_mut() {
            timing.finish = Some(mark.unwrap().elapsed());
        }
        Ok(Some(&self.logits[..4]))
    }

    /// Exact256-token checkpoint with two actual P128 execution states.
    /// One request owns both chronological halves; only its final input may request logits.
    pub async fn start_prefill256(
        &mut self, device:&mut Device, sequence:&Sequence<'_>, workspace:&mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if sequence.inputs.len()!=256 { return Err("Prompt256 needs256 inputs".into()); }
        self.start_prefill_tail256_with_padding(device,std::slice::from_ref(sequence),workspace,bf16::from_bits(0)).await
    }
    /// Pair of independent128-token chunks, with owner-ordered outputs.
    pub async fn start_prefill_pair256(
        &mut self, device:&mut Device, sequences:&[Sequence<'_>], workspace:&mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if sequences.len()!=2 { return Err("Pair256 needs two owners".into()); }
        self.start_prefill_tail256_with_padding(device,sequences,workspace,bf16::from_bits(0)).await
    }
    /// Scoped numerical diagnostic: only the two inactive tensor quarters receive
    /// this value. They never own request positions, KV reservations or logits.
    pub async fn start_prefill_tail256_with_padding(
        &mut self, device:&mut Device, sequences:&[Sequence<'_>], workspace:&mut Workspace,
        padding:bf16,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.active_allocations().is_empty() { return Err("another prefill is in progress".into()); }
        if !workspace.is_canonical() || sequences.len()>self.logits.len()
            || !GroupPrefill::supports_tail256(sequences) {
            return Err("Tail256 needs canonical one256 or independent128+128 input ranges".into());
        }
        for sequence in sequences {
            validate_sequence(sequence,workspace.capacity(),workspace.requests())?;
            validate_physical_pages(sequence.blocks,workspace.physical_full_pages())?;
        }
        self.prepare_group_scratch(device).await?;
        self.active_tail256=Some(self.group.as_mut().unwrap().prepare_tail256(
            device,&self.weights,sequences,&mut self.prefill,
            super::prefill::wide::MEDIA_BATCH_ENABLED,padding).await?);
        Ok(())
    }
    /// Complete one layer, then return only the actual one/two owner outputs.
    /// On a failed or cancelled in-flight call, discard Model/Workspace as for
    /// every other resumable prefill; partial cache writes cannot be retried.
    pub async fn advance_prefill_tail256(
        &mut self, device:&mut Device, workspace:&mut Workspace,
    ) -> Result<Option<&[Logits]>, Box<dyn std::error::Error>> {
        self.advance_prefill_tail256_measured(device, workspace, None).await
    }
    pub(crate) async fn advance_prefill_tail256_measured(
        &mut self, device:&mut Device, workspace:&mut Workspace,
        mut timing: Option<&mut PrefillAdvanceTiming>,
    ) -> Result<Option<&[Logits]>, Box<dyn std::error::Error>> {
        if let Some(timing) = timing.as_mut() { **timing = PrefillAdvanceTiming::default(); }
        let state=self.active_tail256.as_mut().ok_or("no Tail256 is active")?;
        let group=self.group.as_mut().ok_or("grouped scratch is missing")?;
        let mark = timing.as_ref().map(|_| std::time::Instant::now());
        let complete = group.advance_tail256(device,&self.weights,state,workspace,&mut self.prefill).await?;
        if let Some(timing) = timing.as_mut() { timing.layer = Some(mark.unwrap().elapsed()); }
        if !complete { return Ok(None); }
        let mark = timing.as_ref().map(|_| std::time::Instant::now());
        let owners=state.owners();
        for owner in 0..owners {
            std::mem::swap(&mut self.logits[owner],&mut workspace.logits);
            let result=group.finish_tail256(owner,device,&self.weights,state,workspace,&mut self.prefill).await;
            std::mem::swap(&mut self.logits[owner],&mut workspace.logits);
            result?;
        }
        self.active_tail256=None;
        if let Some(timing) = timing.as_mut() { timing.finish = Some(mark.unwrap().elapsed()); }
        Ok(Some(&self.logits[..owners]))
    }

    /// Model-only384-token checkpoint over one to three independently owned ranges.
    /// The fourth tensor quarter is padding; only real states attend or commit KV.
    pub async fn start_prefill_tail384(
        &mut self, device:&mut Device, sequences:&[Sequence<'_>], workspace:&mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_tail384_with_padding(device,sequences,workspace,bf16::from_bits(0)).await
    }
    /// Scoped padding-independence diagnostic, with no persistent model selector.
    pub async fn start_prefill_tail384_with_padding(
        &mut self, device:&mut Device, sequences:&[Sequence<'_>], workspace:&mut Workspace,
        padding:bf16,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.active_allocations().is_empty() { return Err("another prefill is in progress".into()); }
        if !workspace.is_canonical() || sequences.len()>self.logits.len()
            || !GroupPrefill::supports_tail384(sequences) {
            return Err("Tail384 needs canonical independent owner ranges totaling384".into());
        }
        for sequence in sequences {
            validate_sequence(sequence,workspace.capacity(),workspace.requests())?;
            validate_physical_pages(sequence.blocks,workspace.physical_full_pages())?;
        }
        self.prepare_group_scratch(device).await?;
        self.active_tail384=Some(self.group.as_mut().unwrap().prepare_tail384(
            device,&self.weights,sequences,&mut self.prefill,
            super::prefill::wide::MEDIA_BATCH_ENABLED,padding).await?);
        Ok(())
    }
    /// One complete layer per call. Failed/cancelled in-flight calls require
    /// discarding Model/Workspace, exactly like the other resumable prompt APIs.
    pub async fn advance_prefill_tail384(
        &mut self, device:&mut Device, workspace:&mut Workspace,
    ) -> Result<Option<&[Logits]>, Box<dyn std::error::Error>> {
        self.advance_prefill_tail384_measured(device,workspace,None).await
    }
    pub(crate) async fn advance_prefill_tail384_measured(
        &mut self, device:&mut Device, workspace:&mut Workspace,
        mut timing:Option<&mut PrefillAdvanceTiming>,
    ) -> Result<Option<&[Logits]>, Box<dyn std::error::Error>> {
        if let Some(timing)=timing.as_mut() {**timing=PrefillAdvanceTiming::default();}
        let state=self.active_tail384.as_mut().ok_or("no Tail384 is active")?;
        let group=self.group.as_mut().ok_or("grouped scratch is missing")?;
        let mark=timing.as_ref().map(|_|std::time::Instant::now());
        let complete=group.advance_tail384(device,&self.weights,state,workspace,&mut self.prefill).await?;
        if let Some(timing)=timing.as_mut() {timing.layer=Some(mark.unwrap().elapsed());}
        if !complete {return Ok(None);}
        let mark=timing.as_ref().map(|_|std::time::Instant::now());
        let owners=state.owners();
        for owner in 0..owners {
            std::mem::swap(&mut self.logits[owner],&mut workspace.logits);
            let result=group.finish_tail384(owner,device,&self.weights,state,workspace,&mut self.prefill).await;
            std::mem::swap(&mut self.logits[owner],&mut workspace.logits);
            result?;
        }
        self.active_tail384=None;
        if let Some(timing)=timing.as_mut() {timing.finish=Some(mark.unwrap().elapsed());}
        Ok(Some(&self.logits[..owners]))
    }

    /// Model-only full512 checkpoint from three independent owners. Exactly one
    /// owner contributes256 consecutive inputs; the other two contribute128 each.
    /// Their four chronological quarters are derived privately after validation.
    pub async fn start_prefill_three_owner512(
        &mut self, device:&mut Device, sequences:&[Sequence<'_>], workspace:&mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.active_allocations().is_empty() { return Err("another prefill is in progress".into()); }
        if !workspace.is_canonical() || self.logits.len()<3
            || !GroupPrefill::supports_three_owner512(sequences) {
            return Err("ThreeOwner512 needs canonical independent256/128/128 ranges in owner order".into());
        }
        for sequence in sequences {
            validate_sequence(sequence,workspace.capacity(),workspace.requests())?;
            validate_physical_pages(sequence.blocks,workspace.physical_full_pages())?;
        }
        self.prepare_group_scratch(device).await?;
        self.active_three_owner512=Some(self.group.as_mut().unwrap().prepare_three_owner512(
            device,&self.weights,sequences,&mut self.prefill,
            super::prefill::wide::MEDIA_BATCH_ENABLED).await?);
        Ok(())
    }
    /// Yield after one whole layer; only the final call returns three outputs.
    /// Failed/cancelled in-flight execution requires discarding Model/Workspace,
    /// as for all resumable prompt APIs; do not retry partial cache updates.
    pub async fn advance_prefill_three_owner512(
        &mut self, device:&mut Device, workspace:&mut Workspace,
    ) -> Result<Option<&[Logits]>, Box<dyn std::error::Error>> {
        self.advance_prefill_three_owner512_measured(device,workspace,None).await
    }
    pub(crate) async fn advance_prefill_three_owner512_measured(
        &mut self, device:&mut Device, workspace:&mut Workspace,
        mut timing:Option<&mut PrefillAdvanceTiming>,
    ) -> Result<Option<&[Logits]>, Box<dyn std::error::Error>> {
        if let Some(timing)=timing.as_mut() { **timing=PrefillAdvanceTiming::default(); }
        let state=self.active_three_owner512.as_mut().ok_or("no ThreeOwner512 is active")?;
        let group=self.group.as_mut().ok_or("grouped scratch is missing")?;
        let mark=timing.as_ref().map(|_|std::time::Instant::now());
        let complete=group.advance_three_owner512(device,&self.weights,state,workspace,&mut self.prefill).await?;
        if let Some(timing)=timing.as_mut() {timing.layer=Some(mark.unwrap().elapsed());}
        if !complete {return Ok(None);}
        let mark=timing.as_ref().map(|_|std::time::Instant::now());
        for owner in 0..3 {
            std::mem::swap(&mut self.logits[owner],&mut workspace.logits);
            let result=group.finish_three_owner512(owner,device,&self.weights,state,workspace,&mut self.prefill).await;
            std::mem::swap(&mut self.logits[owner],&mut workspace.logits);
            result?;
        }
        self.active_three_owner512=None;
        if let Some(timing)=timing.as_mut() {timing.finish=Some(mark.unwrap().elapsed());}
        Ok(Some(&self.logits[..3]))
    }

    /// Diagnostic only: compare every physical K/V value of two canonical
    /// reservations, including untouched ring positions and allocated full pages.
    /// No cache values are modified; transfer time is excluded from model timings.
    pub async fn compare_canonical_kv(
        &self, device:&mut Device, workspace:&Workspace, a:&Blocks, b:&Blocks,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        validate_physical_pages(a,workspace.physical_full_pages())?;
        validate_physical_pages(b,workspace.physical_full_pages())?;
        if !workspace.is_canonical() || a.sliding>=64 || b.sliding>=64 {
            return Err("cache comparison needs canonical reservations".into());
        }
        Ok(super::kv_diagnostic::compare_canonical_kv(device,workspace,a,b).await?/2)
    }

    /// Prepare exactly two independent owners:384 consecutive inputs from A,
    /// then128 from B. The three A quarters are derived here, never supplied by
    /// callers. Retain both request reservations until the final layer completes.
    pub async fn start_prefill384_128(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !GroupPrefill::supports_prompt384128(sequences) {
            return Err("Prompt384128 needs ordered384/128 input ranges".into());
        }
        self.start_prefill_pair512(device, sequences, workspace).await
    }

    /// Pack two owners' chronological quarters without reordering the owners.
    /// Legal input lengths are128+384,256+256, or384+128; no padding is executed.
    pub async fn start_prefill_pair512(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_pair512_with_media_batch(device, sequences, workspace, super::prefill::wide::MEDIA_BATCH_ENABLED).await
    }

    /// Per-call media preparation choice, applied before any embedding work.
    /// No selector or borrowed media reference survives preparation.
    pub async fn start_prefill_pair512_with_media_batch(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
        batched_media: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.active_allocations().is_empty() {
            return Err("another prefill is in progress".into());
        }
        if self.logits.len() < 2 || !workspace.is_canonical()
            || !GroupPrefill::supports_prompt_pair512(sequences) {
            return Err("PromptPair512 needs canonical independent input ranges totaling512".into());
        }
        for sequence in sequences {
            validate_sequence(sequence, workspace.capacity(), workspace.requests())?;
            validate_physical_pages(sequence.blocks, workspace.physical_full_pages())?;
        }
        self.prepare_group_scratch(device).await?;
        self.active_prompt384128 = Some(self.group.as_mut().unwrap()
            .prepare_prompt_pair512(device, &self.weights, sequences, &mut self.prefill, batched_media).await?);
        Ok(())
    }

    pub async fn advance_prefill_pair512(
        &mut self, device: &mut Device, workspace: &mut Workspace,
    ) -> Result<Option<&[Logits]>, Box<dyn std::error::Error>> {
        self.advance_prefill384_128(device, workspace).await
    }

    /// Same two-owner execution with canonical residual storage throughout.
    pub async fn start_prefill384_128_original_blocked_residual(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill384_128(device, sequences, workspace).await?;
        self.active_prompt384128.as_mut().unwrap().original_blocked_residual();
        Ok(())
    }

    /// Advance exactly one complete layer, returning two owner-ordered outputs
    /// only at completion. Only each owner's final input can request logits.
    /// As for other prompt checkpoints, discard Model/Workspace after a failed
    /// or cancelled in-flight device future; do not retry partial cache writes.
    pub async fn advance_prefill384_128(
        &mut self, device: &mut Device, workspace: &mut Workspace,
    ) -> Result<Option<&[Logits]>, Box<dyn std::error::Error>> {
        self.advance_prefill384_128_measured(device, workspace, None).await
    }
    pub(crate) async fn advance_prefill384_128_measured(
        &mut self, device: &mut Device, workspace: &mut Workspace,
        mut timing: Option<&mut PrefillAdvanceTiming>,
    ) -> Result<Option<&[Logits]>, Box<dyn std::error::Error>> {
        if let Some(timing) = timing.as_mut() { **timing = PrefillAdvanceTiming::default(); }
        let state = self.active_prompt384128.as_mut().ok_or("no Prompt384128 is active")?;
        let group = self.group.as_mut().ok_or("grouped scratch is missing")?;
        let mark = timing.as_ref().map(|_| std::time::Instant::now());
        let complete = group.advance_prompt384128(device, &self.weights, state, workspace, &mut self.prefill).await?;
        if let Some(timing) = timing.as_mut() { timing.layer = Some(mark.unwrap().elapsed()); }
        if !complete { return Ok(None); }
        let mark = timing.as_ref().map(|_| std::time::Instant::now());
        for owner in 0..2 {
            std::mem::swap(&mut self.logits[owner], &mut workspace.logits);
            let result = group.finish_prompt384128(owner, device, &self.weights, state, workspace, &mut self.prefill).await;
            std::mem::swap(&mut self.logits[owner], &mut workspace.logits);
            result?;
        }
        self.active_prompt384128 = None;
        if let Some(timing) = timing.as_mut() { timing.finish = Some(mark.unwrap().elapsed()); }
        Ok(Some(&self.logits[..2]))
    }

    /// Diagnostic: process512 consecutive prompt inputs while retaining exactly
    /// one request allocation. Every128-row subchunk keeps its existing causal
    /// attention arithmetic; dense QKV/FFN work is shared within the request.
    /// Complete using `advance_prefill`; only the final input may request logits.
    pub async fn start_prefill512(
        &mut self,
        device: &mut Device,
        sequence: &Sequence<'_>,
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill512_with_media_batch(device, sequence, workspace, super::prefill::wide::MEDIA_BATCH_ENABLED).await
    }

    /// Per-call media preparation choice, applied before any embedding work.
    /// No selector or borrowed media reference survives preparation.
    pub async fn start_prefill512_with_media_batch(
        &mut self,
        device: &mut Device,
        sequence: &Sequence<'_>,
        workspace: &mut Workspace,
        batched_media: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.active_allocations().is_empty() {
            return Err("another prefill is in progress".into());
        }
        validate_sequence(sequence, workspace.capacity(), workspace.requests())?;
        validate_physical_pages(sequence.blocks, workspace.physical_full_pages())?;
        if !workspace.is_canonical() || !GroupPrefill::supports_prompt512(sequence) {
            return Err("Prompt512 needs a canonical512-token sequence aligned to128".into());
        }
        self.prepare_group_scratch(device).await?;
        self.active_prefill = Some(ActivePrefill::Prompt512(
            self.group.as_mut().unwrap().prepare_prompt512(
                device, &self.weights, sequence, &mut self.prefill, batched_media,
            ).await?,
        ));
        Ok(())
    }

    /// Scoped exact heads/attention/commit diagnostic. All other choices persist.
    pub async fn start_prefill512_fused_head_attention(
        &mut self, device: &mut Device, sequence: &Sequence<'_>, workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill512(device, sequence, workspace).await?;
        match self.active_prefill.as_mut().unwrap() {
            ActivePrefill::Prompt512(state) => state.fused_head_attention(true),
            _ => unreachable!("start_prefill512 owns Prompt512 state"),
        }
        Ok(())
    }

    /// Scoped exact fusion across four independent owners; no global selector.
    pub async fn start_prefill_group_fused_head_attention(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_group(device, sequences, workspace).await?;
        self.active_group.as_mut().unwrap().fused_head_attention(true);
        Ok(())
    }

    /// Scoped exact fusion for any supported ordered two-owner512 split.
    pub async fn start_prefill_pair512_fused_head_attention(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_pair512(device, sequences, workspace).await?;
        self.active_prompt384128.as_mut().unwrap().fused_head_attention(true);
        Ok(())
    }

    /// Scoped exact heads/attention/commit diagnostic. All other choices persist.
    pub async fn start_prefill512_original_head_attention(
        &mut self, device: &mut Device, sequence: &Sequence<'_>, workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill512(device, sequence, workspace).await?;
        match self.active_prefill.as_mut().unwrap() {
            ActivePrefill::Prompt512(state) => state.fused_head_attention(false),
            _ => unreachable!("start_prefill512 owns Prompt512 state"),
        }
        Ok(())
    }

    /// Scoped exact fusion across four independent owners; no global selector.
    pub async fn start_prefill_group_original_head_attention(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_group(device, sequences, workspace).await?;
        self.active_group.as_mut().unwrap().fused_head_attention(false);
        Ok(())
    }

    /// Scoped exact fusion for any supported ordered two-owner512 split.
    pub async fn start_prefill_pair512_original_head_attention(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_pair512(device, sequences, workspace).await?;
        self.active_prompt384128.as_mut().unwrap().fused_head_attention(false);
        Ok(())
    }

    /// Original first128 attention control, with old-history work retained.
    pub async fn start_prefill512_original_history(
        &mut self, device: &mut Device, sequence: &Sequence<'_>, workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill512(device, sequence, workspace).await?;
        match self.active_prefill.as_mut().unwrap() {
            ActivePrefill::Prompt512(state) => state.original_empty_history(),
            _ => unreachable!("start_prefill512 owns Prompt512 state"),
        }
        Ok(())
    }

    /// Original first128 attention control across independent owners.
    pub async fn start_prefill_group_original_history(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_group(device, sequences, workspace).await?;
        self.active_group.as_mut().unwrap().original_empty_history();
        Ok(())
    }

    /// Isolated exact causal pruning; only absolute-position-zero quarters use it.
    pub async fn start_prefill512_nohistory(
        &mut self, device: &mut Device, sequence: &Sequence<'_>, workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill512(device, sequence, workspace).await?;
        match self.active_prefill.as_mut().unwrap() {
            ActivePrefill::Prompt512(state) => state.prune_empty_history(),
            _ => unreachable!("start_prefill512 owns Prompt512 state"),
        }
        Ok(())
    }

    /// Isolated exact causal pruning across independent prompt owners.
    pub async fn start_prefill_group_nohistory(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_group(device, sequences, workspace).await?;
        self.active_group.as_mut().unwrap().prune_empty_history();
        Ok(())
    }

    /// Diagnostic control for the same chronological Prompt512 execution,
    /// changing only the exact norm/quantization fusion launch boundary.
    pub async fn start_prefill512_original_norms(
        &mut self,
        device: &mut Device,
        sequence: &Sequence<'_>,
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill512(device, sequence, workspace).await?;
        match self.active_prefill.as_mut().unwrap() {
            ActivePrefill::Prompt512(state) => state.original_norms(),
            _ => unreachable!("start_prefill512 owns Prompt512 state"),
        }
        Ok(())
    }
    /// Same prepared Prompt512 with canonical O stores; no persistent model mode.
    pub async fn start_prefill512_original_output(
        &mut self,
        device: &mut Device,
        sequence: &Sequence<'_>,
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill512(device, sequence, workspace).await?;
        match self.active_prefill.as_mut().unwrap() {
            ActivePrefill::Prompt512(state) => state.original_output(),
            _ => unreachable!("start_prefill512 owns Prompt512 state"),
        }
        Ok(())
    }
    /// Same prepared execution with the previous full-row quantizer; no persistent mode.
    pub async fn start_prefill512_original_shard_quant(
        &mut self,
        device: &mut Device,
        sequence: &Sequence<'_>,
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill512(device, sequence, workspace).await?;
        match self.active_prefill.as_mut().unwrap() {
            ActivePrefill::Prompt512(state) => state.original_shard_quant(),
            _ => unreachable!("start_prefill512 owns Prompt512 state"),
        }
        Ok(())
    }
    /// Same prepared execution using the previous independently normalized quarters.
    pub async fn start_prefill512_original_quarter_norm(
        &mut self,
        device: &mut Device,
        sequence: &Sequence<'_>,
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill512(device, sequence, workspace).await?;
        match self.active_prefill.as_mut().unwrap() {
            ActivePrefill::Prompt512(state) => state.original_quarter_norm(),
            _ => unreachable!("start_prefill512 owns Prompt512 state"),
        }
        Ok(())
    }
    /// Same chronological Prompt512 with four FFN core launches and unchanged norms.
    pub async fn start_prefill512_original_ffn_core(
        &mut self,
        device: &mut Device,
        sequence: &Sequence<'_>,
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill512(device, sequence, workspace).await?;
        match self.active_prefill.as_mut().unwrap() {
            ActivePrefill::Prompt512(state) => state.original_ffn_core(),
            _ => unreachable!("start_prefill512 owns Prompt512 state"),
        }
        Ok(())
    }
    /// Same prepared execution using the previous sequential post-FFN normalization quarters.
    pub async fn start_prefill512_original_quarter_postnorm(
        &mut self,
        device: &mut Device,
        sequence: &Sequence<'_>,
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill512(device, sequence, workspace).await?;
        match self.active_prefill.as_mut().unwrap() {
            ActivePrefill::Prompt512(state) => state.original_quarter_postnorm(),
            _ => unreachable!("start_prefill512 owns Prompt512 state"),
        }
        Ok(())
    }
    /// Same prepared execution using the canonical residual storage with every other default kernel unchanged.
    pub async fn start_prefill512_original_blocked_residual(
        &mut self,
        device: &mut Device,
        sequence: &Sequence<'_>,
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill512(device, sequence, workspace).await?;
        match self.active_prefill.as_mut().unwrap() {
            ActivePrefill::Prompt512(state) => state.original_blocked_residual(),
            _ => unreachable!("start_prefill512 owns Prompt512 state"),
        }
        Ok(())
    }
    /// Prepare one request's embeddings and masks without advancing Generation.
    /// Only one prompt chunk can be in flight; independent request decode may run
    /// between subsequent complete-layer advances. Retain the same Workspace
    /// until completion; its request allocation remains owned by the caller.
    pub async fn start_prefill(
        &mut self,
        device: &mut Device,
        sequence: &Sequence<'_>,
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_prefill_with_media_batch(device, sequence, workspace, super::prefill::wide::MEDIA_BATCH_ENABLED).await
    }

    /// Per-call media preparation choice, applied before any embedding work.
    /// No selector or borrowed media reference survives preparation.
    pub async fn start_prefill_with_media_batch(
        &mut self,
        device: &mut Device,
        sequence: &Sequence<'_>,
        workspace: &mut Workspace,
        batched_media: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.active_allocations().is_empty() {
            return Err("another prefill is in progress".into());
        }
        validate_sequence(sequence, workspace.capacity(), workspace.requests())?;
        validate_physical_pages(sequence.blocks, workspace.physical_full_pages())?;
        if !WidePrefill::supports(sequence) && !Prefill::supports(sequence) {
            return Err("resumable prefill requires a page-local multi-token chunk".into());
        }
        workspace
            .ensure_legacy(device, sequence.blocks, sequence.start_position)
            .await?;
        self.active_prefill = Some(if WidePrefill::supports(sequence) {
            ActivePrefill::Wide(
                self.wide
                    .prepare_with_media_batch(
                        device,
                        &self.weights,
                        sequence,
                        &mut self.prefill,
                        workspace.is_canonical(),
                        batched_media,
                    )
                    .await?,
            )
        } else {
            ActivePrefill::Narrow(
                self.prefill
                    .prepare(device, &self.weights, sequence, workspace.is_canonical())
                    .await?,
            )
        });
        Ok(())
    }

    /// Diagnostic full P128 embedding snapshots: scalar control, batched candidate.
    /// Candidate runs first; neither preparation executes transformer layers or KV writes.
    pub async fn compare_media_prepare(
        &mut self, device: &mut Device, sequence: &Sequence<'_>, workspace: &mut Workspace,
    ) -> Result<(Vec<bf16>, Vec<bf16>), Box<dyn std::error::Error>> {
        if !self.active_allocations().is_empty() { return Err("another prefill is in progress".into()); }
        validate_sequence(sequence, workspace.capacity(), workspace.requests())?;
        validate_physical_pages(sequence.blocks, workspace.physical_full_pages())?;
        if !WidePrefill::supports(sequence) { return Err("media snapshot requires supported P128 preparation".into()); }
        self.wide.prepare_with_media_batch(device, &self.weights, sequence, &mut self.prefill, workspace.is_canonical(), true).await?;
        let candidate = self.wide.embedding_snapshot(device).await?;
        self.wide.prepare_with_media_batch(device, &self.weights, sequence, &mut self.prefill, workspace.is_canonical(), false).await?;
        let control = self.wide.embedding_snapshot(device).await?;
        Ok((control, candidate))
    }

    /// Run one complete layer. A result is returned only after all layers and
    /// the requested final logits complete; the caller may then advance input.
    /// An execution error can leave partial KV writes: discard the model/workspace
    /// rather than retrying that layer. The HTTP worker already takes itself offline.
    pub async fn advance_prefill(
        &mut self,
        device: &mut Device,
        workspace: &mut Workspace,
    ) -> Result<Option<&Logits>, Box<dyn std::error::Error>> {
        self.advance_prefill_measured(device, workspace, None).await
    }
    pub(crate) async fn advance_prefill_measured(
        &mut self,
        device: &mut Device,
        workspace: &mut Workspace,
        mut timing: Option<&mut PrefillAdvanceTiming>,
    ) -> Result<Option<&Logits>, Box<dyn std::error::Error>> {
        if let Some(timing) = timing.as_mut() {
            **timing = PrefillAdvanceTiming::default();
        }
        let state = self
            .active_prefill
            .as_mut()
            .ok_or("no resumable prefill is active")?;
        workspace.blocks.clone_from(state.blocks());
        let mark = timing.as_ref().map(|_| std::time::Instant::now());
        let complete = match state {
            ActivePrefill::Wide(state) => {
                self.wide
                    .advance_layer(device, &self.weights, state, workspace, &mut self.prefill)
                    .await?
            }
            ActivePrefill::Narrow(state) => {
                self.prefill
                    .advance_layer(device, &self.weights, state, workspace)
                    .await?
            }
            ActivePrefill::Prompt512(state) => {
                self.group
                    .as_mut()
                    .expect("Prompt512 owns grouped scratch")
                    .advance_prompt512(device, &self.weights, state, workspace, &mut self.prefill)
                    .await?
            }
        };
        if let Some(timing) = timing.as_mut() {
            timing.layer = Some(mark.unwrap().elapsed());
        }
        if !complete {
            return Ok(None);
        }
        let mark = timing.as_ref().map(|_| std::time::Instant::now());
        std::mem::swap(&mut self.logits[0], &mut workspace.logits);
        let result = match state {
            ActivePrefill::Wide(state) => {
                self.wide
                    .finish(device, &self.weights, state, workspace, &mut self.prefill)
                    .await
            }
            ActivePrefill::Narrow(state) => {
                self.prefill
                    .finish(device, &self.weights, state, workspace)
                    .await
            }
            ActivePrefill::Prompt512(state) => {
                self.group
                    .as_mut()
                    .expect("Prompt512 owns grouped scratch")
                    .finish_prompt512(device, &self.weights, state, workspace, &mut self.prefill)
                    .await
            }
        };
        std::mem::swap(&mut self.logits[0], &mut workspace.logits);
        result?;
        self.active_prefill = None;
        if let Some(timing) = timing.as_mut() {
            timing.finish = Some(mark.unwrap().elapsed());
        }
        Ok(Some(&self.logits[0]))
    }

    /// Append each input to its history and return one logits buffer per input, in input order.
    /// Buffers of inputs with `logits == false` are stale. Independent requests
    /// may share matrix and attention launches while retaining separate histories.
    pub async fn forward(
        &mut self,
        device: &mut Device,
        inputs: &[Input<'_>],
        positions: &[usize],
        blocks: &[&Blocks],
        workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        if inputs.len() != positions.len() || inputs.len() != blocks.len() {
            return Err("incompatible batch dimensions".into());
        }
        let sequences: Vec<_> = inputs
            .iter()
            .zip(positions)
            .zip(blocks)
            .map(|((input, &start_position), &blocks)| Sequence {
                inputs: std::slice::from_ref(input),
                start_position,
                blocks,
            })
            .collect();
        self.forward_sequences(device, &sequences, workspace).await
    }

    /// Compute one consecutive sequence per request, returning its last logits.
    /// Page-local128-token chunks share wide matrix weights; chunks of2..16 use
    /// B16, and independent singleton requests share a B16 decode step. Other
    /// caller-supplied shapes retain token-wise execution.
    pub async fn forward_sequences(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        self.forward_sequences_inner(device, sequences, workspace, Dispatch::Automatic, false)
            .await?;
        Ok(&self.logits[..sequences.len()])
    }

    /// Preserve the native batch output layout for a caller that consumes host logits
    /// or device-selected tokens. The ordinary API still materializes scalar rows.
    pub async fn forward_outputs(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<ForwardOutput<'_>, Box<dyn std::error::Error>> {
        let layout = self.forward_sequences_inner(
            device, sequences, workspace, Dispatch::Automatic, true,
        ).await?;
        Ok(match layout {
            OutputLayout::Blocked64 => ForwardOutput::Blocked64(self.decode64.as_ref().unwrap().blocked_logits(), sequences.len()),
            OutputLayout::Batch16 => ForwardOutput::Batch16(self.prefill.decode_logits(), sequences.len()),
            OutputLayout::Rows => ForwardOutput::Rows(&self.logits[..sequences.len()]),
        })
    }

    /// Per-call exact B16 commit/attention/O composition diagnostic. All other
    /// dispatch and output-layout choices match Automatic, including R32/R64.
    /// No mutable selector survives completion, cancellation, or an error.
    pub async fn forward_outputs_b16_attention_output(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<ForwardOutput<'_>, Box<dyn std::error::Error>> {
        let layout = self.forward_sequences_inner(
            device,sequences,workspace,Dispatch::B16AttentionOutputDiagnostic,true,
        ).await?;
        Ok(match layout {
            OutputLayout::Blocked64=>ForwardOutput::Blocked64(self.decode64.as_ref().unwrap().blocked_logits(),sequences.len()),
            OutputLayout::Batch16=>ForwardOutput::Batch16(self.prefill.decode_logits(),sequences.len()),
            OutputLayout::Rows=>ForwardOutput::Rows(&self.logits[..sequences.len()]),
        })
    }

    /// Explicit original three-launch B16 attention/output control. Other width,
    /// precision and retained-output decisions remain identical to Automatic.
    pub async fn forward_outputs_original_b16_attention_output(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<ForwardOutput<'_>, Box<dyn std::error::Error>> {
        let layout = self.forward_sequences_inner(
            device,sequences,workspace,Dispatch::OriginalB16AttentionOutputDiagnostic,true,
        ).await?;
        Ok(match layout {
            OutputLayout::Blocked64=>ForwardOutput::Blocked64(self.decode64.as_ref().unwrap().blocked_logits(),sequences.len()),
            OutputLayout::Batch16=>ForwardOutput::Batch16(self.prefill.decode_logits(),sequences.len()),
            OutputLayout::Rows=>ForwardOutput::Rows(&self.logits[..sequences.len()]),
        })
    }

    /// Explicit diagnostic control: identical request inputs and original token-wise kernels.
    pub async fn forward_sequences_serial(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        self.forward_sequences_inner(device, sequences, workspace, Dispatch::Serial, false)
            .await?;
        Ok(&self.logits[..sequences.len()])
    }

    /// Explicit P128 correctness/performance candidate, separate from server dispatch.
    pub async fn forward_sequences_wide(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        self.forward_sequences_inner(device, sequences, workspace, Dispatch::WideDiagnostic, false)
            .await?;
        Ok(&self.logits[..sequences.len()])
    }

    /// Numerical diagnostic: only P128 Q/K/V consume materialized FP8 activations.
    /// Default dispatch and the BF16 P128 control remain unchanged.
    pub async fn forward_sequences_native_qkv(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        self.forward_sequences_inner(device, sequences, workspace, Dispatch::NativeQkvDiagnostic, false).await?;
        Ok(&self.logits[..sequences.len()])
    }

    /// Original canonical P128 sliding softmax, with the same native QKV and other kernels.
    /// The selector is owned by each prepared execution and cannot leak across cancellation.
    pub async fn forward_sequences_original_attention(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        self.forward_sequences_inner(device, sequences, workspace, Dispatch::OriginalAttentionDiagnostic, false).await?;
        Ok(&self.logits[..sequences.len()])
    }

    /// Original canonical P128 full-page movement, with identical native QKV,
    /// sliding attention and all other arithmetic. No selector survives the call.
    pub async fn forward_sequences_original_full_attention(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        self.forward_sequences_inner(device, sequences, workspace, Dispatch::OriginalFullAttentionDiagnostic, false).await?;
        Ok(&self.logits[..sequences.len()])
    }

    /// Validated direct full-page kernel with padded statistics, isolating only
    /// the dense state transfer; all other kernels and request history match.
    pub async fn forward_sequences_direct_full_attention(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        self.forward_sequences_inner(device, sequences, workspace, Dispatch::DirectFullAttentionDiagnostic, false).await?;
        Ok(&self.logits[..sequences.len()])
    }

    /// Explicit BF16 P128 QKV reference, independent of the ordinary dispatch default.
    pub async fn forward_sequences_bf16_qkv(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        self.forward_sequences_inner(device, sequences, workspace, Dispatch::Bf16QkvDiagnostic, false).await?;
        Ok(&self.logits[..sequences.len()])
    }

    /// Diagnostic route retaining R64 numerical arithmetic even for a partial batch.
    /// Useful for verifying row regrouping and scheduler ownership independently of
    /// the scalar/B16 numerical differences. Production chooses from live count.
    pub async fn forward_sequences_r64(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        if !workspace.is_canonical() {
            return Err("R64 requires canonical KV storage".into());
        }
        self.forward_sequences_inner(device, sequences, workspace, Dispatch::Decode64Diagnostic, false)
            .await?;
        Ok(&self.logits[..sequences.len()])
    }

    /// Explicit per-call32-capacity hot-stage diagnostic, retaining all R64 arithmetic.
    /// Counts outside17..32 keep the complete existing R64 path; no persistent selector.
    pub async fn forward_outputs_capacity32(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<ForwardOutput<'_>, Box<dyn std::error::Error>> {
        self.forward_outputs_decode_diagnostic(device,sequences,workspace,Dispatch::Decode32Diagnostic).await
    }
    /// Scoped exact32 O diagnostic; every other stage matches capacity32 control.
    /// Counts outside17..32 retain the existing64-row output projection.
    pub async fn forward_outputs_capacity32_output(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<ForwardOutput<'_>, Box<dyn std::error::Error>> {
        self.forward_outputs_decode_diagnostic(device,sequences,workspace,Dispatch::Decode32OutputDiagnostic).await
    }
    /// Matched exact reference with identical native logits ABI and complete64-row work.
    pub async fn forward_outputs_capacity64(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<ForwardOutput<'_>, Box<dyn std::error::Error>> {
        self.forward_outputs_decode_diagnostic(device,sequences,workspace,Dispatch::Decode64Diagnostic).await
    }
    async fn forward_outputs_decode_diagnostic(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace, dispatch:Dispatch,
    ) -> Result<ForwardOutput<'_>, Box<dyn std::error::Error>> {
        if !workspace.is_canonical() || sequences.is_empty() || sequences.iter().any(|s|s.inputs.len()!=1) {
            return Err("capacity diagnostic needs independent canonical singleton rows".into());
        }
        let layout=self.forward_sequences_inner(device,sequences,workspace,dispatch,true).await?;
        Ok(match layout {
            OutputLayout::Blocked64=>ForwardOutput::Blocked64(self.decode64.as_ref().unwrap().blocked_logits(),sequences.len()),
            OutputLayout::Rows=>ForwardOutput::Rows(&self.logits[..sequences.len()]),
            OutputLayout::Batch16=>unreachable!("forced R64 diagnostic cannot select B16"),
        })
    }

    /// Matched R64 control: disable only prefix768; retain existing512/mixed/full
    /// attention, exact request histories, all projection and LM arithmetic.
    pub async fn forward_sequences_r64_original_sliding(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        if !workspace.is_canonical() {
            return Err("R64 requires canonical KV storage".into());
        }
        self.forward_sequences_inner(device, sequences, workspace, Dispatch::OriginalDecodeSlidingDiagnostic, false)
            .await?;
        Ok(&self.logits[..sequences.len()])
    }

    /// Matched control: disable only full-window head-row placement in this call.
    /// Existing prefix512/prefix768/mixed512 routes and all R64 arithmetic remain.
    pub async fn forward_sequences_r64_original_head_rows(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        if !workspace.is_canonical() {
            return Err("R64 requires canonical KV storage".into());
        }
        self.forward_sequences_inner(device, sequences, workspace, Dispatch::OriginalDecodeHeadRowsDiagnostic, false)
            .await?;
        Ok(&self.logits[..sequences.len()])
    }

    /// Compute one actual R64 step, then compare both vocabulary heads on its identical final norm.
    /// Returned raw buffers use BlockedLogits order; KV advances exactly once.
    /// This explicit diagnostic always reruns the original head as its reference.
    pub async fn compare_lm_projections(
        &mut self, device: &mut Device, sequences: &[Sequence<'_>], workspace: &mut Workspace,
    ) -> Result<(Vec<u8>, Vec<u8>, [f64; 2]), Box<dyn std::error::Error>> {
        if sequences.is_empty() || sequences.iter().any(|s| s.inputs.len() != 1 || !s.inputs[0].logits) {
            return Err("LM comparison requires single-token rows with requested logits".into());
        }
        let blocked = self.forward_sequences_inner(device, sequences, workspace, Dispatch::Decode64Diagnostic, true).await?;
        if blocked != OutputLayout::Blocked64 { return Err("LM comparison requires a canonical R64 output".into()); }
        let decode = self.decode64.as_mut().ok_or("R64 scratch missing")?;
        let mut raw = [Vec::new(), Vec::new()];
        let mut milliseconds = [0.0; 2];
        for mode in 0..2 {
            let start = std::time::Instant::now();
            let output = decode.reproject_lm(device, &self.weights, mode == 1).await?;
            milliseconds[mode] = start.elapsed().as_secs_f64() * 1000.0;
            raw[mode] = output.to_host::<crate::ops_r64_io_blocked::BlockedLogits>(&mut device.pdma)
                .await?.into_inner().into_buf();
        }
        let [control, candidate] = raw;
        Ok((control, candidate, milliseconds))
    }

    async fn forward_sequences_inner(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
        dispatch: Dispatch,
        prefer_native: bool,
    ) -> Result<OutputLayout, Box<dyn std::error::Error>> {
        if sequences.len() > self.logits.len() {
            return Err("too many request sequences".into());
        }
        for sequence in sequences {
            validate_sequence(sequence, workspace.capacity(), workspace.requests())?;
            validate_physical_pages(sequence.blocks, workspace.physical_full_pages())?;
        }
        let active = self.active_allocations();
        if !active.is_empty() {
            for sequence in sequences {
                if active.iter().any(|blocks| {
                    blocks.sliding == sequence.blocks.sliding
                        || blocks
                            .full
                            .iter()
                            .any(|page| sequence.blocks.full.contains(page))
                }) {
                    return Err("request aliases an active prefill KV allocation".into());
                }
                if dispatch.use_wide(sequence)
                    || (dispatch != Dispatch::Serial && Prefill::supports(sequence))
                {
                    return Err("prompt scratch is owned by the active prefill".into());
                }
            }
        }
        let singletons: Vec<usize> = sequences
            .iter()
            .enumerate()
            .filter_map(|(row, s)| (s.inputs.len() == 1).then_some(row))
            .collect();
        let independent = super::request_pool::independent_sequences(sequences);
        let use_r64 = workspace.is_canonical()
            && independent
            && (1..=crate::axes::R64::SIZE).contains(&singletons.len())
            && (matches!(dispatch, Dispatch::Decode64Diagnostic | Dispatch::Decode32Diagnostic | Dispatch::Decode32OutputDiagnostic | Dispatch::OriginalDecodeSlidingDiagnostic | Dispatch::OriginalDecodeHeadRowsDiagnostic)
                || (dispatch != Dispatch::Serial && singletons.len() > crate::axes::B::SIZE));
        let batched_singletons = use_r64
            || (dispatch != Dispatch::Serial
                && independent
                && (2..=crate::axes::B::SIZE).contains(&singletons.len()));
        // Mixed prompt/decode calls retain the existing scalar-row contract.
        let output_layout = if prefer_native && batched_singletons
            && singletons.len() == sequences.len()
            && sequences.iter().any(|s| s.inputs[0].logits)
        {
            if use_r64 { OutputLayout::Blocked64 } else { OutputLayout::Batch16 }
        } else {
            OutputLayout::Rows
        };
        if batched_singletons {
            let rows: Vec<_> = singletons
                .iter()
                .map(|&row| Sequence {
                    inputs: sequences[row].inputs,
                    start_position: sequences[row].start_position,
                    blocks: sequences[row].blocks,
                })
                .collect();
            if use_r64 {
                self.decode64
                    .as_mut()
                    .ok_or("R64 scratch was not allocated for this model")?
                    .run(
                        device,
                        &self.weights,
                        &rows,
                        workspace,
                        &mut self.logits,
                        &singletons,
                        output_layout == OutputLayout::Rows,
                        dispatch != Dispatch::OriginalDecodeSlidingDiagnostic,
                        dispatch != Dispatch::OriginalDecodeHeadRowsDiagnostic,
                        matches!(dispatch, Dispatch::Automatic | Dispatch::B16AttentionOutputDiagnostic | Dispatch::OriginalB16AttentionOutputDiagnostic | Dispatch::Decode32Diagnostic | Dispatch::Decode32OutputDiagnostic),
                        matches!(dispatch, Dispatch::Automatic | Dispatch::B16AttentionOutputDiagnostic | Dispatch::OriginalB16AttentionOutputDiagnostic | Dispatch::Decode32OutputDiagnostic),
                    )
                    .await?;
            } else {
                self.prefill
                    .decode_batch(
                        device,
                        &self.weights,
                        &rows,
                        workspace,
                        &mut self.logits,
                        &singletons,
                        output_layout == OutputLayout::Rows,
                        matches!(dispatch, Dispatch::Automatic | Dispatch::B16AttentionOutputDiagnostic) && singletons.len() == sequences.len(),
                    )
                    .await?;
            }
        }
        for (row, sequence) in sequences.iter().enumerate() {
            if batched_singletons && sequence.inputs.len() == 1 {
                continue;
            }
            workspace
                .ensure_legacy(device, sequence.blocks, sequence.start_position)
                .await?;
            workspace.blocks.full.clone_from(&sequence.blocks.full);
            workspace.blocks.sliding = sequence.blocks.sliding;
            std::mem::swap(&mut self.logits[row], &mut workspace.logits);
            let result = async {
                if dispatch == Dispatch::DirectFullAttentionDiagnostic && dispatch.use_wide(sequence) {
                    self.wide.run_direct_full_attention(device, &self.weights, sequence, workspace, &mut self.prefill).await?;
                } else if dispatch == Dispatch::OriginalFullAttentionDiagnostic && dispatch.use_wide(sequence) {
                    self.wide.run_original_full_attention(device, &self.weights, sequence, workspace, &mut self.prefill).await?;
                } else if dispatch == Dispatch::OriginalAttentionDiagnostic && dispatch.use_wide(sequence) {
                    self.wide.run_original_attention(device, &self.weights, sequence, workspace, &mut self.prefill).await?;
                } else if dispatch == Dispatch::NativeQkvDiagnostic && dispatch.use_wide(sequence) {
                    self.wide.run_native_qkv(device, &self.weights, sequence, workspace, &mut self.prefill).await?;
                } else if dispatch == Dispatch::Bf16QkvDiagnostic && dispatch.use_wide(sequence) {
                    self.wide.run_bf16_qkv(device, &self.weights, sequence, workspace, &mut self.prefill).await?;
                } else if dispatch.use_wide(sequence) {
                    self.wide
                        .run(
                            device,
                            &self.weights,
                            sequence,
                            workspace,
                            &mut self.prefill,
                        )
                        .await?;
                } else if dispatch != Dispatch::Serial && Prefill::supports(sequence) {
                    self.prefill
                        .run(device, &self.weights, sequence, workspace)
                        .await?;
                } else {
                    for (offset, input) in sequence.inputs.iter().enumerate() {
                        self.step(device, input, sequence.start_position + offset, workspace)
                            .await?;
                    }
                }
                Ok::<_, Box<dyn std::error::Error>>(())
            }
            .await;
            std::mem::swap(&mut self.logits[row], &mut workspace.logits);
            result?;
        }
        Ok(output_layout)
    }

    async fn step(
        &mut self,
        device: &mut Device,
        input: &Input<'_>,
        position: usize,
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let embedding = match input.soft {
            Some(SoftToken::Image(index)) => {
                let image = input.image.ok_or("image is missing")?;
                runtime::encode_vision_patch(
                    device,
                    &self.weights,
                    &image.pixels[index],
                    image.positions[index],
                )
                .await?
            }
            Some(SoftToken::Audio(index)) => {
                let audio = input.audio.ok_or("audio is missing")?;
                runtime::encode_audio_frame(device, &self.weights, &audio.frames[index]).await?
            }
            None => {
                self.decode
                    .run(
                        device,
                        workspace,
                        input.token,
                        &self.weights,
                        position,
                        input.logits,
                    )
                    .await?;
                return Ok(());
            }
        };
        self.decode
            .run_with_embedding(
                device,
                workspace,
                embedding,
                &self.weights,
                position,
                input.logits,
            )
            .await?;
        Ok(())
    }
}

fn validate_physical_pages(blocks: &Blocks, pages: usize) -> Result<(), &'static str> {
    let mut seen = std::collections::HashSet::new();
    if blocks
        .full
        .iter()
        .any(|&page| page >= pages || !seen.insert(page))
    {
        return Err("full-attention pages must be valid and distinct");
    }
    Ok(())
}

fn validate_sequence(
    sequence: &Sequence<'_>,
    capacity: usize,
    requests: usize,
) -> Result<(), &'static str> {
    let end = sequence
        .start_position
        .checked_add(sequence.inputs.len())
        .ok_or("sequence position overflow")?;
    if sequence.inputs.is_empty()
        || end > capacity
        || sequence.blocks.full.len() < Workspace::blocks_for(end)
        || sequence.blocks.sliding >= requests
    {
        return Err("invalid sequence positions or blocks");
    }
    if sequence.inputs[..sequence.inputs.len() - 1]
        .iter()
        .any(|input| input.logits)
    {
        return Err("only the last sequence input may request logits");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(logits: bool) -> Input<'static> {
        Input {
            token: 7,
            soft: None,
            logits,
            image: None,
            audio: None,
        }
    }

    #[test]
    fn sequence_crossing_full_page_requires_next_physical_block() {
        let inputs = [input(false), input(true)];
        let mut blocks = Blocks {
            full: vec![5],
            sliding: 0,
        };
        assert!(
            validate_sequence(
                &Sequence {
                    inputs: &inputs,
                    start_position: 511,
                    blocks: &blocks
                },
                8192,
                4
            )
            .is_err()
        );
        blocks.full.push(9);
        assert!(
            validate_sequence(
                &Sequence {
                    inputs: &inputs,
                    start_position: 511,
                    blocks: &blocks
                },
                8192,
                4
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_discarding_requested_logits_and_out_of_bounds_sequences() {
        let blocks = Blocks {
            full: vec![0, 1],
            sliding: 0,
        };
        let wrong = [input(true), input(true)];
        assert!(
            validate_sequence(
                &Sequence {
                    inputs: &wrong,
                    start_position: 0,
                    blocks: &blocks
                },
                1024,
                4
            )
            .is_err()
        );
        let inputs = [input(false), input(true)];
        assert!(
            validate_sequence(
                &Sequence {
                    inputs: &inputs,
                    start_position: 1023,
                    blocks: &blocks
                },
                1024,
                4
            )
            .is_err()
        );
        assert!(
            validate_sequence(
                &Sequence {
                    inputs: &inputs,
                    start_position: usize::MAX,
                    blocks: &blocks
                },
                1024,
                4
            )
            .is_err()
        );
        assert!(
            validate_sequence(
                &Sequence {
                    inputs: &[],
                    start_position: 0,
                    blocks: &blocks
                },
                1024,
                4
            )
            .is_err()
        );
    }

    #[test]
    fn accepts_final_partial_chunk_and_rejects_another_requests_ring() {
        let inputs = [input(false), input(false), input(true)];
        let mut blocks = Blocks {
            full: vec![7, 3],
            sliding: 3,
        };
        assert!(
            validate_sequence(
                &Sequence {
                    inputs: &inputs,
                    start_position: 1021,
                    blocks: &blocks
                },
                1024,
                4
            )
            .is_ok()
        );
        blocks.sliding = 4;
        assert!(
            validate_sequence(
                &Sequence {
                    inputs: &inputs,
                    start_position: 1021,
                    blocks: &blocks
                },
                1024,
                4
            )
            .is_err()
        );
    }
}
