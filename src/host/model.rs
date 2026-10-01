//! Batched model calculation over caller-owned KV blocks.

use super::{audio::AudioFrames, generate::SoftToken, image::ImagePatches, load, runtime};
use crate::{Chip, axes::W};
use furiosa_opt_std::prelude::*;

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

pub struct Model {
    weights: load::Weights,
    decode: runtime::Decode,
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
            logits.push(HostTensor::<bf16, m![W]>::zero().to_hbm(&mut device.pdma).await?);
        }
        Ok(Self {
            weights,
            decode: workspace.begin_decode(device).await?,
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
        if inputs.len() > self.logits.len() || inputs.len() != positions.len() || inputs.len() != blocks.len() {
            return Err("incompatible batch dimensions".into());
        }
        for (row, ((input, &position), blocks)) in inputs.iter().zip(positions).zip(blocks).enumerate() {
            if position >= workspace.capacity()
                || blocks.full.len() * crate::axes::Tf::SIZE <= position
                || blocks.sliding >= workspace.requests()
            {
                return Err("invalid position or blocks".into());
            }
            workspace.blocks.full.clone_from(&blocks.full);
            workspace.blocks.sliding = blocks.sliding;
            std::mem::swap(&mut self.logits[row], &mut workspace.logits);
            let result = self.step(device, input, position, workspace).await;
            std::mem::swap(&mut self.logits[row], &mut workspace.logits);
            result?;
        }
        Ok(&self.logits[..inputs.len()])
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
                runtime::encode_vision_patch(device, &self.weights, &image.pixels[index], image.positions[index])
                    .await?
            }
            Some(SoftToken::Audio(index)) => {
                let audio = input.audio.ok_or("audio is missing")?;
                runtime::encode_audio_frame(device, &self.weights, &audio.frames[index]).await?
            }
            None => {
                self.decode
                    .run(device, workspace, input.token, &self.weights, position, input.logits)
                    .await?;
                return Ok(());
            }
        };
        self.decode
            .run_with_embedding(device, workspace, embedding, &self.weights, position, input.logits)
            .await?;
        Ok(())
    }
}
