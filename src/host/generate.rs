use crate::Chip;
use crate::axes::{B, P128, Tf, W};
use crate::host::audio::AudioFrames;
use crate::host::image::ImagePatches;
use crate::host::model::{Input, Model, Sequence};
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
/// Prompt positions dispatched together. Generated tokens remain autoregressive.
pub const PREFILL_CHUNK: usize = P128::SIZE;
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
        return Err(format!(
            "expected one image placeholder, found {image_markers}"
        ));
    }
    if audio.is_some() && audio_markers != 1 {
        return Err(format!(
            "expected one audio placeholder, found {audio_markers}"
        ));
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
    banned_tokens: HashSet<usize>,
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
        if request
            .prompt_ids
            .len()
            .saturating_add(request.max_new_tokens)
            > capacity
        {
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
        let mut banned_tokens: HashSet<usize> = SUPPRESSED_TOKENS.into_iter().collect();
        if request.ignore_eos {
            // Suppressing channel markers keeps all requested generated tokens visible.
            banned_tokens.extend(EOS_TOKENS);
            banned_tokens.extend([CHANNEL_OPEN_TOKEN, CHANNEL_CLOSE_TOKEN]);
        }
        Ok(Self {
            banned_tokens,
            stream: tokenizer.decode_stream(
                &request
                    .prompt_ids
                    .iter()
                    .map(|&id| id as u32)
                    .collect::<Vec<_>>(),
            ),
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

    /// Number of consecutive inputs available without sampling another token.
    pub fn input_count(&self, limit: usize) -> usize {
        assert!(limit > 0, "an input chunk must have positive capacity");
        match self.phase {
            Phase::Prompt(position) => {
                prompt_chunk_count(position, self.request.prompt_ids.len() - position, limit)
            }
            Phase::Decode(_) => 1,
            Phase::Complete => unreachable!("completed requests have no input"),
        }
    }

    /// Borrow the next prompt chunk, or one generated token. Media descriptors
    /// stay attached to their original positions and are not shared across requests.
    pub fn inputs(&self, limit: usize) -> (Vec<Input<'_>>, usize) {
        let count = self.input_count(limit);
        if let Phase::Prompt(position) = self.phase {
            let inputs = (position..position + count)
                .map(|index| Input {
                    token: self.request.prompt_ids[index],
                    soft: self.request.soft_at[index],
                    logits: index + 1 == self.request.prompt_ids.len(),
                    image: self.request.image.as_ref(),
                    audio: self.request.audio.as_ref(),
                })
                .collect();
            (inputs, position)
        } else {
            let (input, position) = self.input();
            (vec![input], position)
        }
    }

    /// The unmodified per-request sampling policy for an external selector.
    pub fn sampling_config(&self) -> &SamplingConfig {
        &self.request.sampling
    }

    /// Immutable request bans, shared by CPU sampling and external greedy selection.
    pub fn banned_tokens(&self) -> &HashSet<usize> {
        &self.banned_tokens
    }

    fn phase_after_inputs(&self, count: usize) -> Result<Phase, Box<dyn std::error::Error>> {
        if count == 0 {
            return Err("cannot advance an empty input chunk".into());
        }
        match self.phase {
            Phase::Prompt(position) => {
                if count > self.request.prompt_ids.len() - position {
                    return Err("input chunk exceeds the remaining prompt".into());
                }
                Ok(Phase::Prompt(position + count - 1))
            }
            Phase::Decode(_) if count != 1 => Err("decode needs exactly one sampled input".into()),
            Phase::Complete => Err("generation is already complete".into()),
            phase => Ok(phase),
        }
    }

    /// Consume an actually computed chunk. Intermediate prompt rows do not sample
    /// or emit tokens; the final prompt row and every decode row use `advance`.
    pub fn advance_inputs(
        &mut self,
        count: usize,
        logits: &[f32],
        on_delta: impl FnMut(Delta) -> bool,
    ) -> Result<Option<GenerationOutput>, Box<dyn std::error::Error>> {
        self.phase = self.phase_after_inputs(count)?;
        self.advance(logits, on_delta)
    }

    /// Consume the exact greedy result of a computed chunk without downloading its logits.
    /// The caller must apply `banned_tokens()` and the same tie order as CPU sampling.
    /// This accepts only a final prompt chunk or one decode input. Invalid selections
    /// leave all generation state unchanged; stochastic requests keep `advance_inputs`.
    pub fn advance_inputs_greedy(
        &mut self,
        count: usize,
        token: usize,
        on_delta: impl FnMut(Delta) -> bool,
    ) -> Result<Option<GenerationOutput>, Box<dyn std::error::Error>> {
        let phase = self.phase_after_inputs(count)?;
        if !self.request.sampling.is_greedy() {
            return Err("external greedy selection requires greedy sampling".into());
        }
        if self.request.max_new_tokens == 0
            || matches!(phase, Phase::Prompt(pos) if pos + 1 < self.request.prompt_ids.len())
        {
            return Err("this input does not consume a selected token".into());
        }
        // `sample` deliberately falls back to unmasked argmax when every valid
        // vocabulary ID is banned. Current request bans cannot cover W, but keep
        // that contract if the request policy later gains arbitrary bans.
        let banned_without_fallback = self.banned_tokens.contains(&token)
            && !(self.banned_tokens.len() >= W::SIZE
                && (0..W::SIZE).all(|id| self.banned_tokens.contains(&id)));
        if token >= W::SIZE || banned_without_fallback {
            return Err("selected token is outside the vocabulary or banned".into());
        }
        self.phase = phase;
        self.advance_selected(|_, _| token, on_delta)
    }

    /// Consume the logits of the pending input (empty if `wants_logits` was false).
    pub fn advance(
        &mut self,
        logits: &[f32],
        on_delta: impl FnMut(Delta) -> bool,
    ) -> Result<Option<GenerationOutput>, Box<dyn std::error::Error>> {
        self.advance_selected(
            |config, banned| sample(logits, config, banned, &mut rand::rng()),
            on_delta,
        )
    }

    fn advance_selected(
        &mut self,
        select: impl FnOnce(&SamplingConfig, &HashSet<usize>) -> usize,
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
        let next = select(&self.request.sampling, &self.banned_tokens);
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
                    let safe =
                        self.output.text.len() - pending_stop_overlap(&self.output.text, &stops);
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

    fn finish(
        &mut self,
        reason: FinishReason,
        on_delta: &mut impl FnMut(Delta) -> bool,
    ) -> GenerationOutput {
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
        return Ok(generation
            .advance(&[], &mut on_delta)?
            .expect("empty request finishes"));
    }
    let blocks = Blocks {
        full: (0..Workspace::blocks_for(workspace.capacity())).collect(),
        sliding: 0,
    };
    loop {
        let (inputs, start_position) = generation.inputs(PREFILL_CHUNK);
        let count = inputs.len();
        let wants = inputs.last().expect("nonempty input chunk").logits;
        let sequence = Sequence {
            inputs: &inputs,
            start_position,
            blocks: &blocks,
        };
        let outputs = model
            .forward_sequences(device, &[sequence], workspace)
            .await?;
        let logits = if wants {
            read_logits(device, &outputs[0]).await?
        } else {
            Vec::new()
        };
        if let Some(output) = generation.advance_inputs(count, &logits, &mut on_delta)? {
            return Ok(output);
        }
    }
}

/// Match the two implemented prompt widths while staying in the current full-KV page.
/// Short tails use B16 instead of paying for all128 matrix rows.
fn prompt_chunk_count(position: usize, remaining: usize, limit: usize) -> usize {
    let available = remaining.min(limit).min(Tf::SIZE - position % Tf::SIZE);
    if available >= P128::SIZE {
        P128::SIZE
    } else {
        available.min(B::SIZE)
    }
}

#[cfg(test)]
mod chunk_tests {
    use super::*;

    #[test]
    fn prompt_chunks_cover_page_and_ring_boundaries_without_skipping_rows() {
        for (start, len) in [
            (0, 127),
            (0, 128),
            (0, 513),
            (0, 1025),
            (500, 140),
            (1020, 140),
        ] {
            let mut position = start;
            let mut remaining = len;
            while remaining > 0 {
                let count = prompt_chunk_count(position, remaining, PREFILL_CHUNK);
                assert!(count == P128::SIZE || (1..=B::SIZE).contains(&count));
                assert!(count <= remaining);
                assert_eq!(position / Tf::SIZE, (position + count - 1) / Tf::SIZE);
                position += count;
                remaining -= count;
            }
            assert_eq!(position, start + len);
        }
        assert_eq!(prompt_chunk_count(0, 127, PREFILL_CHUNK), 16);
        assert_eq!(prompt_chunk_count(0, 128, PREFILL_CHUNK), 128);
        assert_eq!(prompt_chunk_count(500, 140, PREFILL_CHUNK), 12);
        assert_eq!(prompt_chunk_count(0, 512, 8), 8);
    }
}

#[cfg(test)]
mod selected_token_tests {
    use super::*;
    use std::sync::OnceLock;

    fn tokenizer() -> &'static Tokenizer {
        static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
        TOKENIZER.get_or_init(|| {
            let mut names: Vec<_> = (0..107).map(|i| format!("token{i}")).collect();
            for (id, text) in [
                (0, "[UNK]"),
                (1, "<eos>"),
                (2, "prompt"),
                (3, "a"),
                (4, "b"),
                (5, "S"),
                (6, "TO"),
                (7, "P"),
                (8, "analysis\n"),
                (9, "why"),
                (10, "answer"),
                (50, "<tool>"),
                (100, CHANNEL_OPEN_TEXT),
                (101, CHANNEL_CLOSE_TEXT),
                (106, "<end>"),
            ] {
                names[id] = text.to_owned();
            }
            let vocab: serde_json::Map<_, _> = names
                .into_iter()
                .enumerate()
                .map(|(id, name)| (name, serde_json::json!(id)))
                .collect();
            let definition = serde_json::json!({"version":"1.0", "added_tokens":[],
                "model":{"type":"WordLevel", "unk_token":"[UNK]", "vocab":vocab},
                "decoder":{"type":"Fuse"}});
            let path = std::env::temp_dir().join(format!(
                "moa-generation-test-{}-{}.json",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::write(&path, serde_json::to_vec(&definition).unwrap()).unwrap();
            let tokenizer = Tokenizer::new(&path).unwrap();
            std::fs::remove_file(path).unwrap();
            tokenizer
        })
    }
    fn request(max: usize, stops: &[&str], reasoning: bool) -> GenerationRequest {
        GenerationRequest {
            prompt_ids: vec![2; 3],
            soft_at: vec![None; 3],
            image: None,
            audio: None,
            max_new_tokens: max,
            sampling: SamplingConfig::DEFAULT,
            stop_strings: stops.iter().map(|s| s.to_string()).collect(),
            separate_reasoning: reasoning,
            ignore_eos: false,
        }
    }
    fn run(
        external: bool,
        max: usize,
        stops: &[&str],
        reasoning: bool,
        tokens: &[usize],
        cancel_after: usize,
    ) -> (GenerationOutput, Vec<(bool, String)>) {
        let mut g = Generation::new(tokenizer(), request(max, stops, reasoning), 256).unwrap();
        let mut events = Vec::new();
        for (step, &token) in tokens.iter().enumerate() {
            let mut callback = |delta: Delta<'_>| {
                events.push(match delta {
                    Delta::Content(s) => (false, s.to_owned()),
                    Delta::Reasoning(s) => (true, s.to_owned()),
                });
                events.len() < cancel_after
            };
            let count = if step == 0 { 3 } else { 1 };
            let out = if external {
                g.advance_inputs_greedy(count, token, &mut callback)
            } else {
                let mut logits = vec![-1.0; 107];
                logits[token] = 1.0;
                g.advance_inputs(count, &logits, &mut callback)
            }
            .unwrap();
            if let Some(out) = out {
                return (out, events);
            }
            assert_eq!(g.input().0.token, token);
            assert_eq!(g.input().1, 3 + g.output.completion_tokens - 1);
        }
        panic!("test token sequence did not finish generation");
    }
    fn same(
        a: &(GenerationOutput, Vec<(bool, String)>),
        b: &(GenerationOutput, Vec<(bool, String)>),
    ) {
        assert_eq!(
            (
                &a.0.text,
                &a.0.reasoning,
                a.0.prompt_tokens,
                a.0.completion_tokens,
                a.0.finish_reason,
                &a.1
            ),
            (
                &b.0.text,
                &b.0.reasoning,
                b.0.prompt_tokens,
                b.0.completion_tokens,
                b.0.finish_reason,
                &b.1
            )
        );
    }
    #[test]
    fn selected_tokens_preserve_stop_overlap_eos_length_and_cancellation() {
        for (max, stops, tokens, cancel) in [
            (32, vec!["STOP"], vec![3, 5, 6, 7], usize::MAX),
            (32, vec![], vec![3, 4, EOS_TOKEN], usize::MAX),
            (2, vec![], vec![3, 4], usize::MAX),
            (32, vec![], vec![3], 1),
        ] {
            let cpu = run(false, max, &stops, false, &tokens, cancel);
            let selected = run(true, max, &stops, false, &tokens, cancel);
            same(&cpu, &selected);
            assert_eq!(selected.0.prompt_tokens, 3);
            if !stops.is_empty() {
                assert_eq!(selected.0.text, "a");
                assert_eq!(selected.0.completion_tokens, 3);
                assert_eq!(selected.0.finish_reason, FinishReason::Stop);
                assert_eq!(selected.1, vec![(false, "a".into())]);
            }
        }
    }
    #[test]
    fn selected_tokens_preserve_reasoning_channel_and_marker_suppression() {
        let tokens = [
            CHANNEL_OPEN_TOKEN,
            8,
            9,
            CHANNEL_CLOSE_TOKEN,
            10,
            END_OF_TURN_TOKEN,
        ];
        let cpu = run(false, 32, &[], true, &tokens, usize::MAX);
        let selected = run(true, 32, &[], true, &tokens, usize::MAX);
        same(&cpu, &selected);
        assert_eq!(selected.0.reasoning, "why");
        assert_eq!(selected.0.text, "answer");
        assert_eq!(
            selected.1,
            vec![(true, "why".into()), (false, "answer".into())]
        );
    }
    #[test]
    fn rejected_external_selection_keeps_phase_and_sampling_fallback() {
        let mut req = request(4, &[], false);
        req.ignore_eos = true;
        let mut g = Generation::new(tokenizer(), req, 256).unwrap();
        assert!(g.sampling_config().is_greedy());
        assert_eq!(g.banned_tokens().len(), 7);
        for (count, token) in [
            (0, 3),
            (4, 3),
            (1, 3),
            (3, W::SIZE),
            (3, EOI_TOKEN),
            (3, EOA_TOKEN),
            (3, EOS_TOKEN),
            (3, CHANNEL_OPEN_TOKEN),
        ] {
            assert!(g.advance_inputs_greedy(count, token, |_| true).is_err());
            assert_eq!(g.input().1, 0);
            assert_eq!(g.output.completion_tokens, 0);
        }
        assert!(g.advance_inputs(2, &[], |_| true).unwrap().is_none());
        assert_eq!(g.input().1, 2);
        assert!(g.advance_inputs_greedy(1, 3, |_| true).unwrap().is_none());
        assert_eq!(g.input().1, 3);
        assert_eq!(g.output.completion_tokens, 1);
        assert!(g.advance_inputs_greedy(2, 4, |_| true).is_err());
        assert_eq!(g.input().0.token, 3);

        let mut req = request(1, &[], false);
        req.sampling.temperature = 1.0;
        req.sampling.top_k = 1;
        let mut g = Generation::new(tokenizer(), req, 256).unwrap();
        assert!(!g.sampling_config().is_greedy());
        assert!(g.advance_inputs_greedy(3, 3, |_| true).is_err());
        assert_eq!(g.input().1, 0);
        let out = g
            .advance_inputs(3, &[-1.0, -1.0, -1.0, 2.0], |_| true)
            .unwrap()
            .unwrap();
        assert_eq!(out.text, "a");
        assert_eq!(out.finish_reason, FinishReason::Length);
    }
    #[test]
    fn all_banned_fallback_matches_existing_sampler() {
        let mut cpu = Generation::new(tokenizer(), request(1, &[], false), 256).unwrap();
        let mut selected = Generation::new(tokenizer(), request(1, &[], false), 256).unwrap();
        cpu.banned_tokens = (0..W::SIZE).collect();
        selected.banned_tokens = cpu.banned_tokens.clone();
        let mut logits = vec![-1.0; W::SIZE];
        logits[3] = 1.0;
        let a = cpu.advance_inputs(3, &logits, |_| true).unwrap().unwrap();
        let b = selected
            .advance_inputs_greedy(3, 3, |_| true)
            .unwrap()
            .unwrap();
        assert_eq!(
            (a.text, a.completion_tokens, a.finish_reason),
            (b.text, b.completion_tokens, b.finish_reason)
        );
    }

    #[test]
    fn zero_completion_and_intermediate_prompt_never_consume_a_selection() {
        let mut g = Generation::new(tokenizer(), request(0, &[], false), 256).unwrap();
        assert!(g.advance_inputs_greedy(3, 3, |_| true).is_err());
        assert_eq!(g.input().1, 0);
        let out = g.advance(&[], |_| true).unwrap().unwrap();
        assert_eq!(out.completion_tokens, 0);
        assert!(out.text.is_empty());
        assert!(g.advance_inputs_greedy(1, 3, |_| true).is_err());
        let mut config = SamplingConfig::DEFAULT;
        for (temperature, greedy) in [
            (0.0, true),
            (-0.0, true),
            (-1.0, true),
            (1.0, false),
            (f32::NAN, false),
        ] {
            config.temperature = temperature;
            assert_eq!(config.is_greedy(), greedy);
        }
    }
}
