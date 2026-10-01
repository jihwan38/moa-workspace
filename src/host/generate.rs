use crate::Chip;
use crate::axes::W;
use crate::host::audio::AudioFrames;
use crate::host::image::ImagePatches;
use crate::host::model::{Input, Model};
use crate::host::runtime::{Blocks, Workspace};
use crate::host::sampling::{SamplingConfig, sample};
use crate::host::tokenizer::Tokenizer;
use furiosa_opt_std::prelude::*;
use llm_tokenizer::DecodeStream;
use std::collections::HashSet;

pub const EOS_TOKEN: usize = 1;
pub const END_OF_TURN_TOKEN: usize = 106;
pub const TOOL_RESPONSE_TOKEN: usize = 50;

pub const EOS_TOKENS: [usize; 3] = [EOS_TOKEN, END_OF_TURN_TOKEN, TOOL_RESPONSE_TOKEN];

pub const CHANNEL_OPEN_TOKEN: usize = 100;
pub const CHANNEL_CLOSE_TOKEN: usize = 101;

const CHANNEL_OPEN_TEXT: &str = "<|channel>";
const CHANNEL_CLOSE_TEXT: &str = "<channel|>";

pub const DEFAULT_MAX_NEW_TOKENS: usize = 300;
pub const BOI_TOKEN: usize = 255999;
pub const IMAGE_TOKEN: usize = 258880;
pub const EOI_TOKEN: usize = 258882;
pub const BOA_TOKEN: usize = 256000;
pub const AUDIO_TOKEN: usize = 258881;
pub const EOA_TOKEN: usize = 258883;

const SUPPRESSED_TOKENS: [usize; 2] = [EOI_TOKEN, EOA_TOKEN];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Channel {
    Answer,
    Name,
    Reasoning,
}

#[derive(Clone, Copy, Debug)]
pub enum Delta<'a> {
    Content(&'a str),
    Reasoning(&'a str),
}

#[derive(Clone, Copy)]
pub enum SoftToken {
    Image(usize),
    Audio(usize),
}

pub fn splice_multimodal_tokens(
    ids: Vec<usize>,
    image: Option<&ImagePatches>,
    audio: Option<&AudioFrames>,
) -> Result<(Vec<usize>, Vec<Option<SoftToken>>), String> {
    let image_markers = ids.iter().filter(|&&id| id == IMAGE_TOKEN).count();
    let audio_markers = ids.iter().filter(|&&id| id == AUDIO_TOKEN).count();
    if image.is_some() && image_markers != 1 {
        return Err(format!("expected one image placeholder, found {image_markers}"));
    }
    if audio.is_some() && audio_markers != 1 {
        return Err(format!("expected one audio placeholder, found {audio_markers}"));
    }

    let image_len = image.map_or(0, |value| value.pixels.len());
    let audio_len = audio.map_or(0, AudioFrames::len);
    let mut out_ids = Vec::with_capacity(ids.len() + image_len + audio_len + 4);
    let mut soft_at = Vec::with_capacity(out_ids.capacity());
    for id in ids {
        match id {
            IMAGE_TOKEN if image.is_some() => {
                out_ids.push(BOI_TOKEN);
                soft_at.push(None);
                for index in 0..image_len {
                    out_ids.push(IMAGE_TOKEN);
                    soft_at.push(Some(SoftToken::Image(index)));
                }
                out_ids.push(EOI_TOKEN);
                soft_at.push(None);
            }
            AUDIO_TOKEN if audio.is_some() => {
                out_ids.push(BOA_TOKEN);
                soft_at.push(None);
                for index in 0..audio_len {
                    out_ids.push(AUDIO_TOKEN);
                    soft_at.push(Some(SoftToken::Audio(index)));
                }
                out_ids.push(EOA_TOKEN);
                soft_at.push(None);
            }
            _ => {
                out_ids.push(id);
                soft_at.push(None);
            }
        }
    }
    Ok((out_ids, soft_at))
}

fn pending_stop_overlap(text: &str, stop_strings: &[&str]) -> usize {
    let mut hold = 0usize;
    for &stop in stop_strings {
        let max_len = (stop.len() - 1).min(text.len());
        for len in (1..=max_len).rev() {
            if stop.is_char_boundary(len) && text.ends_with(&stop[..len]) {
                hold = hold.max(len);
                break;
            }
        }
    }
    hold
}

fn step_over_marker(
    stream: &mut DecodeStream,
    id: usize,
    marker: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let Some(delta) = stream.step(id as u32)? else {
        return Ok(None);
    };
    let held = delta.strip_suffix(marker).unwrap_or(&delta);
    Ok((!held.is_empty()).then(|| held.to_owned()))
}

pub async fn read_logits(
    device: &mut Device,
    logits: &HbmTensor<bf16, Chip, m![W]>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let logits: HostTensor<bf16, m![W]> = logits.to_host(&mut device.pdma).await?;
    Ok(logits.into_vec().into_iter().map(bf16::to_f32).collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

pub struct GenerationRequest {
    pub prompt_ids: Vec<usize>,
    pub soft_at: Vec<Option<SoftToken>>,
    pub image: Option<ImagePatches>,
    pub audio: Option<AudioFrames>,
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_strings: Vec<String>,
    pub separate_reasoning: bool,
    /// Never stop at an end-of-sequence token: generate exactly `max_new_tokens` tokens.
    /// Set only by performance tests, to give every request the same output length.
    pub ignore_eos: bool,
}

pub struct GenerationOutput {
    pub text: String,
    pub reasoning: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
}

/// One conversation's progress, independent of the order in which requests run.
pub struct Generation {
    request: GenerationRequest,
    phase: Phase,
    stream: DecodeStream,
    channel: Channel,
    output: GenerationOutput,
    emitted: usize,
}

#[derive(Clone, Copy)]
enum Phase {
    Prompt(usize),
    Decode(usize),
    Complete,
}

impl Generation {
    pub fn new(
        tokenizer: &Tokenizer,
        request: GenerationRequest,
        capacity: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if request.prompt_ids.is_empty() {
            return Err("prompt produced no tokens".into());
        }
        if request.prompt_ids.len().saturating_add(request.max_new_tokens) > capacity {
            return Err(format!("prompt plus generation exceeds {capacity} positions").into());
        }
        if request.prompt_ids.len() != request.soft_at.len() {
            return Err(format!(
                "prompt_ids ({}) and soft_at ({}) must be parallel",
                request.prompt_ids.len(),
                request.soft_at.len()
            )
            .into());
        }
        Ok(Self {
            stream: tokenizer.decode_stream(&request.prompt_ids.iter().map(|&id| id as u32).collect::<Vec<_>>()),
            output: GenerationOutput {
                text: String::new(),
                reasoning: String::new(),
                prompt_tokens: request.prompt_ids.len(),
                completion_tokens: 0,
                finish_reason: FinishReason::Length,
            },
            request,
            phase: Phase::Prompt(0),
            channel: Channel::Answer,
            emitted: 0,
        })
    }

    /// The next token to compute and the position it is written at.
    pub fn input(&self) -> (Input<'_>, usize) {
        let (token, soft, logits, position) = match self.phase {
            Phase::Prompt(pos) => (
                self.request.prompt_ids[pos],
                self.request.soft_at[pos],
                pos + 1 == self.request.prompt_ids.len(),
                pos,
            ),
            Phase::Decode(token) => (
                token,
                None,
                true,
                self.request.prompt_ids.len() + self.output.completion_tokens - 1,
            ),
            Phase::Complete => unreachable!("completed requests have no input"),
        };
        (
            Input {
                token,
                soft,
                logits,
                image: self.request.image.as_ref(),
                audio: self.request.audio.as_ref(),
            },
            position,
        )
    }

    /// Whether the pending input produces logits that `advance` needs.
    pub fn wants_logits(&self) -> bool {
        self.request.max_new_tokens > 0 && self.input().0.logits
    }

    /// Consume the logits of the pending input (empty if `wants_logits` was false).
    pub fn advance(
        &mut self,
        logits: &[f32],
        mut on_delta: impl FnMut(Delta) -> bool,
    ) -> Result<Option<GenerationOutput>, Box<dyn std::error::Error>> {
        let phase = std::mem::replace(&mut self.phase, Phase::Complete);
        if matches!(phase, Phase::Complete) {
            return Err("generation is already complete".into());
        }
        if self.request.max_new_tokens == 0 {
            return Ok(Some(self.finish(FinishReason::Length, &mut on_delta)));
        }
        if let Phase::Prompt(pos) = phase
            && pos + 1 < self.request.prompt_ids.len()
        {
            self.phase = Phase::Prompt(pos + 1);
            return Ok(None);
        }
        let mut banned: HashSet<usize> = SUPPRESSED_TOKENS.into_iter().collect();
        if self.request.ignore_eos {
            // Channel tokens are banned too: text in the reasoning channel is not part of the
            // answer, so it would shorten the visible output.
            banned.extend(EOS_TOKENS);
            banned.extend([CHANNEL_OPEN_TOKEN, CHANNEL_CLOSE_TOKEN]);
        }
        let next = sample(logits, &self.request.sampling, &banned, &mut rand::rng());
        if EOS_TOKENS.contains(&next) {
            return Ok(Some(self.finish(FinishReason::Stop, &mut on_delta)));
        }
        if let Some(reason) = self.emit(next, &mut on_delta)? {
            return Ok(Some(self.finish(reason, &mut on_delta)));
        }
        self.output.completion_tokens += 1;
        if self.output.completion_tokens == self.request.max_new_tokens {
            return Ok(Some(self.finish(FinishReason::Length, &mut on_delta)));
        }
        // The next turn computes the token just emitted, so every turn has one forward.
        self.phase = Phase::Decode(next);
        Ok(None)
    }

    fn emit(
        &mut self,
        token: usize,
        on_delta: &mut impl FnMut(Delta) -> bool,
    ) -> Result<Option<FinishReason>, Box<dyn std::error::Error>> {
        let (delta, switch_to) = match token {
            CHANNEL_OPEN_TOKEN if self.request.separate_reasoning => (
                step_over_marker(&mut self.stream, token, CHANNEL_OPEN_TEXT)?,
                Some(Channel::Name),
            ),
            CHANNEL_CLOSE_TOKEN if self.request.separate_reasoning => (
                step_over_marker(&mut self.stream, token, CHANNEL_CLOSE_TEXT)?,
                Some(Channel::Answer),
            ),
            _ => (self.stream.step(token as u32)?, None),
        };
        if let Some(delta) = delta {
            let keep = match self.channel {
                Channel::Name => {
                    if let Some((_, rest)) = delta.split_once('\n') {
                        self.channel = Channel::Reasoning;
                        self.output.reasoning.push_str(rest);
                        rest.is_empty() || on_delta(Delta::Reasoning(rest))
                    } else {
                        true
                    }
                }
                Channel::Reasoning => {
                    self.output.reasoning.push_str(&delta);
                    on_delta(Delta::Reasoning(&delta))
                }
                Channel::Answer => {
                    self.output.text.push_str(&delta);
                    let stops: Vec<_> = self
                        .request
                        .stop_strings
                        .iter()
                        .map(String::as_str)
                        .filter(|text| !text.is_empty())
                        .collect();
                    if let Some(cut) = stops.iter().filter_map(|s| self.output.text.find(s)).min() {
                        if cut > self.emitted {
                            let _ = on_delta(Delta::Content(&self.output.text[self.emitted..cut]));
                        }
                        self.output.text.truncate(cut);
                        self.emitted = cut;
                        return Ok(Some(FinishReason::Stop));
                    }
                    let safe = self.output.text.len() - pending_stop_overlap(&self.output.text, &stops);
                    if safe > self.emitted {
                        let keep = on_delta(Delta::Content(&self.output.text[self.emitted..safe]));
                        self.emitted = safe;
                        keep
                    } else {
                        true
                    }
                }
            };
            if !keep {
                return Ok(Some(FinishReason::Length));
            }
        }
        if let Some(channel) = switch_to {
            self.channel = channel;
        }
        Ok(None)
    }

    fn finish(&mut self, reason: FinishReason, on_delta: &mut impl FnMut(Delta) -> bool) -> GenerationOutput {
        if self.emitted < self.output.text.len() {
            let _ = on_delta(Delta::Content(&self.output.text[self.emitted..]));
        }
        self.phase = Phase::Complete;
        GenerationOutput {
            text: std::mem::take(&mut self.output.text),
            reasoning: std::mem::take(&mut self.output.reasoning),
            prompt_tokens: self.output.prompt_tokens,
            completion_tokens: self.output.completion_tokens,
            finish_reason: reason,
        }
    }
}

/// Run one request alone, using the first `blocks_for(capacity)` pool blocks.
pub async fn generate(
    device: &mut Device,
    model: &mut Model,
    tokenizer: &Tokenizer,
    workspace: &mut Workspace,
    request: GenerationRequest,
    mut on_delta: impl FnMut(Delta) -> bool,
) -> Result<GenerationOutput, Box<dyn std::error::Error>> {
    let empty = request.max_new_tokens == 0;
    let mut generation = Generation::new(tokenizer, request, workspace.capacity())?;
    if empty {
        return Ok(generation.advance(&[], &mut on_delta)?.expect("empty request finishes"));
    }
    let blocks = Blocks {
        full: (0..Workspace::blocks_for(workspace.capacity())).collect(),
        sliding: 0,
    };
    loop {
        let wants = generation.wants_logits();
        let (input, position) = generation.input();
        let outputs = model.forward(device, &[input], &[position], &[&blocks], workspace).await?;
        let logits = if wants {
            read_logits(device, &outputs[0]).await?
        } else {
            Vec::new()
        };
        if let Some(output) = generation.advance(&logits, &mut on_delta)? {
            return Ok(output);
        }
    }
}
