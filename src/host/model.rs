//! Batched model calculation over caller-owned KV blocks.

use super::{audio::AudioFrames, generate::SoftToken, image::ImagePatches, load, runtime};
use crate::{Chip, axes::W};
use furiosa_opt_std::prelude::*;

use super::prefill::Prefill;
use runtime::{Blocks, Workspace};
pub type Logits = HbmTensor<bf16, Chip, m![W]>;

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
    logits: Vec<Logits>,
}

impl Model {
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
            logits,
        })
    }

    /// Append each input to its history and return one logits buffer per input, in input order.
    /// Buffers of inputs with `logits == false` are stale. The reference runs rows serially.
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
    /// Multi-token page-local chunks use layer-wise B16 execution. Singleton
    /// or externally supplied larger sequences retain token-wise execution.
    pub async fn forward_sequences(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        self.forward_sequences_inner(device, sequences, workspace, true)
            .await
    }

    /// Explicit diagnostic control: identical request inputs and original token-wise kernels.
    pub async fn forward_sequences_serial(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        self.forward_sequences_inner(device, sequences, workspace, false)
            .await
    }

    async fn forward_sequences_inner(
        &mut self,
        device: &mut Device,
        sequences: &[Sequence<'_>],
        workspace: &mut Workspace,
        use_prefill: bool,
    ) -> Result<&[Logits], Box<dyn std::error::Error>> {
        if sequences.len() > self.logits.len() {
            return Err("too many request sequences".into());
        }
        for sequence in sequences {
            validate_sequence(sequence, workspace.capacity(), workspace.requests())?;
        }
        let singletons: Vec<usize> = sequences
            .iter()
            .enumerate()
            .filter_map(|(row, s)| (s.inputs.len() == 1).then_some(row))
            .collect();
        let batched_singletons =
            use_prefill && (2..=crate::axes::B::SIZE).contains(&singletons.len());
        if batched_singletons {
            let rows: Vec<_> = singletons
                .iter()
                .map(|&row| Sequence {
                    inputs: sequences[row].inputs,
                    start_position: sequences[row].start_position,
                    blocks: sequences[row].blocks,
                })
                .collect();
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
        for (row, sequence) in sequences.iter().enumerate() {
            if batched_singletons && sequence.inputs.len() == 1 {
                continue;
            }
            workspace.blocks.full.clone_from(&sequence.blocks.full);
            workspace.blocks.sliding = sequence.blocks.sliding;
            std::mem::swap(&mut self.logits[row], &mut workspace.logits);
            let result = async {
                if use_prefill && Prefill::supports(sequence) {
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
        Ok(&self.logits[..sequences.len()])
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
