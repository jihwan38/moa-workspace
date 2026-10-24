//! Bounded request scheduling and KV ownership for the fixed HTTP frontend.

use std::sync::mpsc;

use crate::api::worker::{Job, StreamEvent};
use crate::host::generate::{Delta, Generation, PREFILL_CHUNK, read_logits};
use crate::host::load;
use crate::host::model::{Model, Sequence};
use crate::host::runtime::{Blocks, Workspace};
use crate::host::tokenizer::Tokenizer;
use furiosa_opt_std::prelude::*;

/// Requests computed together; more wait in the queue.
pub const CAPACITY: usize = crate::axes::B::SIZE;
/// Prompt plus generated positions allowed per request.
pub const CONTEXT: usize = 8192;

pub struct Worker {
    runtime: tokio::runtime::Runtime,
    device: Device,
    model: Model,
    workspace: Workspace,
    tokenizer: Tokenizer,
}

struct Session {
    generation: Generation,
    blocks: Blocks,
    events: mpsc::Sender<StreamEvent>,
    stream: bool,
}

impl Worker {
    pub fn new(tokenizer: Tokenizer) -> Result<Self, Box<dyn std::error::Error>> {
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let mut device = Device::new(crate::ops::embed_token.topology())?;
        let (model, workspace) = runtime.block_on(async {
            let weights = load::load_weights(&mut device).await?;
            let mut workspace = Workspace::new(&mut device, &weights, CONTEXT, CAPACITY).await?;
            let model = Model::new(&mut device, weights, &mut workspace).await?;
            Ok::<_, Box<dyn std::error::Error>>((model, workspace))
        })?;
        Ok(Self {
            runtime,
            device,
            model,
            workspace,
            tokenizer,
        })
    }

    pub fn run(mut self, jobs: mpsc::Receiver<Job>) {
        let mut sessions = Vec::<Session>::new();
        let mut rings: Vec<_> = (0..CAPACITY).rev().collect();
        // Every admitted request can reach its limit without reclaiming another request's KV.
        let mut free: Vec<_> = (0..CAPACITY * Workspace::blocks_for(CONTEXT))
            .rev()
            .collect();
        let mut prefer_prefill = false;
        let mut next_prefill_ring = 0;
        loop {
            let incoming: Vec<_> = if sessions.is_empty() {
                let Ok(job) = jobs.recv() else { break };
                std::iter::once(job)
                    .chain(jobs.try_iter())
                    .take(CAPACITY)
                    .collect()
            } else {
                jobs.try_iter().take(CAPACITY - sessions.len()).collect()
            };
            for job in incoming {
                let empty = job.request.max_new_tokens == 0;
                match Generation::new(&self.tokenizer, job.request, CONTEXT) {
                    Ok(mut generation) if empty => {
                        let output = generation.advance(&[], |_| true).unwrap().unwrap();
                        let _ = job.events.send(StreamEvent::Done(output));
                    }
                    Ok(generation) => sessions.push(Session {
                        generation,
                        blocks: Blocks {
                            full: Vec::new(),
                            sliding: rings.pop().unwrap(),
                        },
                        events: job.events,
                        stream: job.stream,
                    }),
                    Err(error) => {
                        let _ = job.events.send(StreamEvent::Error(error.to_string()));
                    }
                }
            }
            if sessions.is_empty() {
                continue;
            }
            let available: Vec<_> = sessions
                .iter()
                .map(|s| (s.blocks.sliding, s.generation.input_count(PREFILL_CHUNK)))
                .collect();
            let selected = select_work(&available, &mut prefer_prefill, &mut next_prefill_ring);
            for &index in &selected {
                let session = &mut sessions[index];
                let (_, position) = session.generation.input();
                let count = session.generation.input_count(PREFILL_CHUNK);
                while session.blocks.full.len() < Workspace::blocks_for(position + count) {
                    session
                        .blocks
                        .full
                        .push(free.pop().expect("admission bounds KV usage"));
                }
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.runtime.block_on(async {
                    let chunks: Vec<_> = selected
                        .iter()
                        .map(|&index| sessions[index].generation.inputs(PREFILL_CHUNK))
                        .collect();
                    let sequences: Vec<_> = chunks
                        .iter()
                        .zip(&selected)
                        .map(|((inputs, start_position), &index)| Sequence {
                            inputs,
                            start_position: *start_position,
                            blocks: &sessions[index].blocks,
                        })
                        .collect();
                    let outputs = self
                        .model
                        .forward_sequences(&mut self.device, &sequences, &mut self.workspace)
                        .await?;
                    let mut logits = Vec::with_capacity(outputs.len());
                    for (output, (inputs, _)) in outputs.iter().zip(&chunks) {
                        logits.push(if inputs.last().expect("nonempty input chunk").logits {
                            read_logits(&mut self.device, output).await?
                        } else {
                            Vec::new()
                        });
                    }
                    Ok::<_, Box<dyn std::error::Error>>(logits)
                })
            }));
            let logits = match result {
                Ok(Ok(logits)) => logits,
                failed => {
                    let message = match failed {
                        Ok(Err(error)) => error.to_string(),
                        _ => "generation panicked; worker is offline".to_owned(),
                    };
                    for session in &sessions {
                        let _ = session.events.send(StreamEvent::Error(message.clone()));
                    }
                    return;
                }
            };
            let mut session_index = 0;
            sessions.retain_mut(|session| {
                let index = session_index;
                session_index += 1;
                let Some(row) = selected.iter().position(|&selected| selected == index) else {
                    return true;
                };
                let count = session.generation.input_count(PREFILL_CHUNK);
                let result = session
                    .generation
                    .advance_inputs(count, &logits[row], |delta| {
                        !session.stream
                            || session
                                .events
                                .send(match delta {
                                    Delta::Content(text) => StreamEvent::Delta(text.to_owned()),
                                    Delta::Reasoning(text) => {
                                        StreamEvent::ReasoningDelta(text.to_owned())
                                    }
                                })
                                .is_ok()
                    });
                let event = match result {
                    Ok(None) => return true,
                    Ok(Some(output)) => StreamEvent::Done(output),
                    Err(error) => StreamEvent::Error(error.to_string()),
                };
                let _ = session.events.send(event);
                free.append(&mut session.blocks.full);
                rings.push(session.blocks.sliding);
                false
            });
        }
    }
}

/// Emit a ready decode batch before running another prompt chunk. A single
/// prompt chunk alternates with decode batches so continued arrivals cannot
/// serialize all active decode requests behind a whole round of prefill.
fn select_work(
    available: &[(usize, usize)],
    prefer_prefill: &mut bool,
    next_ring: &mut usize,
) -> Vec<usize> {
    let singletons: Vec<_> = available
        .iter()
        .enumerate()
        .filter_map(|(index, (_, count))| (*count == 1).then_some(index))
        .collect();
    let prefill = available
        .iter()
        .enumerate()
        .filter(|(_, (_, count))| *count > 1)
        .min_by_key(|(_, (ring, _))| (*ring + CAPACITY - *next_ring) % CAPACITY)
        .map(|(index, _)| index);
    if !singletons.is_empty() && (!*prefer_prefill || prefill.is_none()) {
        *prefer_prefill = true;
        singletons
    } else if let Some(index) = prefill {
        *next_ring = (available[index].0 + 1) % CAPACITY;
        *prefer_prefill = false;
        vec![index]
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ready_decode_requests_stay_batched_while_prompt_progresses() {
        let work = [(2, 1), (7, 16), (12, 1)];
        let mut prefer = false;
        let mut ring = 0;
        assert_eq!(select_work(&work, &mut prefer, &mut ring), vec![0, 2]);
        assert_eq!(select_work(&work, &mut prefer, &mut ring), vec![1]);
        assert_eq!(select_work(&work, &mut prefer, &mut ring), vec![0, 2]);
    }
    #[test]
    fn prompt_round_robin_uses_stable_ring_ids_across_session_removal() {
        let mut prefer = false;
        let mut ring = 0;
        assert_eq!(
            select_work(&[(8, 16), (2, 16), (15, 3)], &mut prefer, &mut ring),
            vec![1]
        );
        assert_eq!(
            select_work(&[(8, 16), (15, 3)], &mut prefer, &mut ring),
            vec![0]
        );
        assert_eq!(
            select_work(&[(8, 16), (15, 3)], &mut prefer, &mut ring),
            vec![1]
        );
        assert_eq!(
            select_work(&[(8, 16), (1, 16)], &mut prefer, &mut ring),
            vec![1]
        );
    }
}
