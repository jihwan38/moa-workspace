//! Bounded request scheduling and KV ownership for the fixed HTTP frontend.

use std::sync::mpsc;

use crate::api::worker::{Job, StreamEvent};
use crate::host::generate::{Delta, Generation, read_logits};
use crate::host::load;
use crate::host::model::Model;
use crate::host::runtime::{Blocks, Workspace};
use crate::host::tokenizer::Tokenizer;
use furiosa_opt_std::prelude::*;

/// Requests computed together; more wait in the queue.
pub const CAPACITY: usize = 4;
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
        let mut free: Vec<_> = (0..CAPACITY * Workspace::blocks_for(CONTEXT)).rev().collect();
        loop {
            let incoming: Vec<_> = if sessions.is_empty() {
                let Ok(job) = jobs.recv() else { break };
                std::iter::once(job).chain(jobs.try_iter()).take(CAPACITY).collect()
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
            for session in &mut sessions {
                let (_, position) = session.generation.input();
                while session.blocks.full.len() < Workspace::blocks_for(position + 1) {
                    session
                        .blocks
                        .full
                        .push(free.pop().expect("admission bounds KV usage"));
                }
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.runtime.block_on(async {
                    let (inputs, positions): (Vec<_>, Vec<_>) =
                        sessions.iter().map(|session| session.generation.input()).unzip();
                    let blocks: Vec<_> = sessions.iter().map(|session| &session.blocks).collect();
                    let outputs = self
                        .model
                        .forward(&mut self.device, &inputs, &positions, &blocks, &mut self.workspace)
                        .await?;
                    let mut logits = Vec::with_capacity(outputs.len());
                    for (output, session) in outputs.iter().zip(&sessions) {
                        logits.push(if session.generation.wants_logits() {
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
            let mut row = 0;
            sessions.retain_mut(|session| {
                let result = session.generation.advance(&logits[row], |delta| {
                    !session.stream
                        || session
                            .events
                            .send(match delta {
                                Delta::Content(text) => StreamEvent::Delta(text.to_owned()),
                                Delta::Reasoning(text) => StreamEvent::ReasoningDelta(text.to_owned()),
                            })
                            .is_ok()
                });
                row += 1;
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
