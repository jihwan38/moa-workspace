//! Batched model calculation over caller-owned KV blocks.

use super::{audio::AudioFrames, generate::SoftToken, image::ImagePatches, load, runtime};
use crate::{Chip, axes::W};
use furiosa_opt_std::prelude::*;

pub use super::prefill::StageTiming;
use super::prefill::wide::pair::{PairPrefill, PreparedPair};
use super::prefill::{
    Prefill, PreparedNarrow,
    wide::{PreparedWide, WidePrefill},
};
use runtime::{Blocks, Workspace};
pub type Logits = HbmTensor<bf16, Chip, m![W]>;

/// Physical output of the current computation. Only input rows requesting logits
/// are valid, just as with `forward_sequences`; blocked rows retain request order.
pub enum ForwardOutput<'a> {
    Rows(&'a [Logits]),
    Blocked64(&'a HbmTensor<bf16, Chip, crate::ops_r64_io_blocked::BlockedLogits>, usize),
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
    pair: Option<PairPrefill>,
    active_pair: Option<PreparedPair>,
    active_prefill: Option<ActivePrefill>,
    logits: Vec<Logits>,
}

enum ActivePrefill {
    Wide(PreparedWide),
    Narrow(PreparedNarrow),
}
impl ActivePrefill {
    fn blocks(&self) -> &Blocks {
        match self {
            Self::Wide(state) => &state.blocks,
            Self::Narrow(state) => &state.blocks,
        }
    }
}

/// Diagnostic partial-wide dispatch stays explicit; production runs wide only
/// when all128 rows are live. Both paths preserve each request's own history.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dispatch {
    Serial,
    Automatic,
    WideDiagnostic,
    Decode64Diagnostic,
}
impl Dispatch {
    fn use_wide(self, sequence: &Sequence<'_>) -> bool {
        WidePrefill::supports(sequence)
            && match self {
                Self::Serial => false,
                Self::Automatic => sequence.inputs.len() == crate::axes::P128::SIZE,
                Self::WideDiagnostic => true,
                Self::Decode64Diagnostic => sequence.inputs.len() == crate::axes::P128::SIZE,
            }
    }
}

impl Model {
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
            pair: None,
            active_pair: None,
            active_prefill: None,
            logits,
        })
    }

    /// Physical ring currently owning a persistent prompt checkpoint, if any.
    pub fn active_prefill_ring(&self) -> Option<usize> {
        self.active_prefill
            .as_ref()
            .map(|state| state.blocks().sliding)
            .or_else(|| {
                self.active_pair
                    .as_ref()
                    .map(|state| state.blocks()[0].sliding)
            })
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
        if let Some(active) = &self.active_pair {
            blocks.extend(active.blocks());
        }
        blocks
    }
    /// Diagnostic candidate: two distinct full P128 chunks share a P256 FFN pass.
    /// Keep both reservations until advance_prefill_pair returns two final outputs.
    pub async fn start_prefill_pair(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.active_allocations().is_empty() {
            return Err("another prefill is in progress".into());
        }
        if !workspace.is_canonical() || !PairPrefill::supports(sequences) {
            return Err("paired prefill needs canonical independent128-token chunks".into());
        }
        for sequence in sequences {
            validate_sequence(sequence, workspace.capacity(), workspace.requests())?;
            validate_physical_pages(sequence.blocks, workspace.physical_full_pages())?;
        }
        if self.pair.is_none() {
            self.pair = Some(PairPrefill::new(device).await?);
        }
        self.active_pair = Some(
            self.pair
                .as_mut()
                .unwrap()
                .prepare(device, &self.weights, sequences, &mut self.prefill)
                .await?,
        );
        Ok(())
    }
    pub async fn advance_prefill_pair(
        &mut self,
        device: &mut Device,
        workspace: &mut Workspace,
    ) -> Result<Option<&[Logits]>, Box<dyn std::error::Error>> {
        let state = self
            .active_pair
            .as_mut()
            .ok_or("no paired prefill is active")?;
        let pair = self.pair.as_mut().ok_or("paired scratch is missing")?;
        if !pair
            .advance_layer(device, &self.weights, state, workspace, &mut self.prefill)
            .await?
        {
            return Ok(None);
        }
        for row in 0..2 {
            std::mem::swap(&mut self.logits[row], &mut workspace.logits);
            let result = pair
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
        self.active_pair = None;
        Ok(Some(&self.logits[..2]))
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
                    .prepare(
                        device,
                        &self.weights,
                        sequence,
                        &mut self.prefill,
                        workspace.is_canonical(),
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

    /// Run one complete layer. A result is returned only after all layers and
    /// the requested final logits complete; the caller may then advance input.
    /// An execution error can leave partial KV writes: discard the model/workspace
    /// rather than retrying that layer. The HTTP worker already takes itself offline.
    pub async fn advance_prefill(
        &mut self,
        device: &mut Device,
        workspace: &mut Workspace,
    ) -> Result<Option<&Logits>, Box<dyn std::error::Error>> {
        let state = self
            .active_prefill
            .as_mut()
            .ok_or("no resumable prefill is active")?;
        workspace.blocks.clone_from(state.blocks());
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
        };
        if !complete {
            return Ok(None);
        }
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
        };
        std::mem::swap(&mut self.logits[0], &mut workspace.logits);
        result?;
        self.active_prefill = None;
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

    /// Preserve the native R64 output layout for a caller that consumes host logits
    /// or device-selected tokens. The ordinary API still materializes scalar rows.
    pub async fn forward_outputs(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<ForwardOutput<'_>, Box<dyn std::error::Error>> {
        let blocked = self.forward_sequences_inner(
            device, sequences, workspace, Dispatch::Automatic, true,
        ).await?;
        Ok(if blocked {
            ForwardOutput::Blocked64(self.decode64.as_ref().unwrap().blocked_logits(), sequences.len())
        } else {
            ForwardOutput::Rows(&self.logits[..sequences.len()])
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

    async fn forward_sequences_inner(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
        dispatch: Dispatch,
        prefer_blocked: bool,
    ) -> Result<bool, Box<dyn std::error::Error>> {
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
            && (dispatch == Dispatch::Decode64Diagnostic
                || (dispatch != Dispatch::Serial && singletons.len() > crate::axes::B::SIZE));
        let batched_singletons = use_r64
            || (dispatch != Dispatch::Serial
                && independent
                && (2..=crate::axes::B::SIZE).contains(&singletons.len()));
        // Mixed prompt/decode calls retain the existing scalar-row contract.
        let blocked = prefer_blocked && use_r64 && singletons.len() == sequences.len()
            && sequences.iter().any(|s| s.inputs[0].logits);
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
                        !blocked,
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
                if dispatch.use_wide(sequence) {
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
        Ok(blocked)
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
