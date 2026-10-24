//! Bounded request scheduling and KV ownership for the fixed HTTP frontend.

use super::admission::{PageReservations, Reservation};
use std::{collections::VecDeque, sync::mpsc};

use crate::api::worker::{Job, StreamEvent};
use crate::host::generate::{Delta, Generation, PREFILL_CHUNK, read_logits};
use crate::host::load;
use crate::host::model::{ForwardOutput, Model, Sequence};
use crate::host::runtime::{Blocks, Workspace};
use crate::host::tokenizer::Tokenizer;
use furiosa_opt_std::prelude::*;
use std::time::{Duration, Instant};

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
    greedy: super::greedy::GreedySelector,
}

enum SampleOutput { Logits(Vec<f32>), Greedy(usize) }

struct Session {
    generation: Generation,
    blocks: Blocks,
    reservation: Option<Reservation>,
    events: mpsc::Sender<StreamEvent>,
    stream: bool,
}

struct Pending {
    generation: Generation,
    positions: usize,
    events: mpsc::Sender<StreamEvent>,
    stream: bool,
}

/// Spend the interval between decode rounds on complete prompt layers. A single
/// layer always remains eligible after a decode so slow rounds cannot starve it.
/// Measurements include host logit reads, sampling and response event emission.
struct DecodeCadence {
    last: Option<(Instant, usize)>,
    costs: [Duration; crate::axes::R64::SIZE + 1],
    prompt_cost: Duration,
}
impl DecodeCadence {
    fn new() -> Self {
        Self { last: None, costs: [Duration::ZERO; crate::axes::R64::SIZE + 1], prompt_cost: Duration::from_millis(8) }
    }
    fn prompt_fits(&self, now: Instant, ready: usize) -> bool {
        if ready == 0 { return true; }
        let Some((last, width)) = self.last else { return false; };
        // An unseen width must run once before using its measured cost.
        width == ready && self.costs[ready] != Duration::ZERO
            && now.saturating_duration_since(last) + self.prompt_cost + self.costs[ready]
                <= Duration::from_millis(90)
    }
    fn observe_prompt(&mut self, elapsed: Duration) {
        self.prompt_cost = elapsed.max(self.prompt_cost.mul_f64(0.9));
    }
    fn observe_decode(&mut self, finished: Instant, width: usize, elapsed: Duration) {
        self.costs[width] = elapsed.max(self.costs[width].mul_f64(0.9));
        self.last = Some((finished, width));
    }
}

impl Worker {
    pub fn new(tokenizer: Tokenizer) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_storage(tokenizer, true)
    }
    /// Canonical KV pool and R64 request capacity, also used by the HTTP worker.
    pub fn new_canonical64(tokenizer: Tokenizer) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_storage(tokenizer, true)
    }
    fn new_with_storage(
        tokenizer: Tokenizer,
        canonical: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let mut device = Device::new(crate::ops::embed_token.topology())?;
        let (model, workspace, greedy) = runtime.block_on(async {
            let started = std::time::Instant::now();
            let weights = load::load_weights(&mut device).await?;
            eprintln!("startup.weights_ms={:.3}", started.elapsed().as_secs_f64()*1000.0);
            let started = std::time::Instant::now();
            let mut workspace = if canonical {
                Workspace::new_canonical64(&mut device, &weights, CONTEXT, crate::axes::R64::SIZE)
                    .await?
            } else {
                Workspace::new(&mut device, &weights, CONTEXT, CAPACITY).await?
            };
            eprintln!("startup.workspace_ms={:.3}", started.elapsed().as_secs_f64()*1000.0);
            let started = std::time::Instant::now();
            let model = Model::new(&mut device, weights, &mut workspace).await?;
            eprintln!("startup.model_scratch_ms={:.3}", started.elapsed().as_secs_f64()*1000.0);
            let greedy = super::greedy::GreedySelector::new(&mut device).await?;
            Ok::<_, Box<dyn std::error::Error>>((model, workspace, greedy))
        })?;
        Ok(Self {
            runtime,
            device,
            model,
            workspace,
            tokenizer,
            greedy,
        })
    }

    pub fn run(mut self, jobs: mpsc::Receiver<Job>) {
        let canonical = self.workspace.is_canonical();
        let capacity = self.workspace.requests();
        let mut reservations = canonical.then(|| {
            let allocator =
                PageReservations::new(capacity, self.workspace.physical_full_pages(), CONTEXT)
                    .expect("canonical page budget can finish any admitted request");
            debug_assert_eq!(allocator.available_pages(), allocator.total_pages());
            allocator
        });
        let mut pending = VecDeque::<Pending>::new();
        let mut sessions = Vec::<Session>::new();
        let mut rings: Vec<_> = (0..capacity).rev().collect();
        // Every admitted request can reach its limit without reclaiming another request's KV.
        let mut free: Vec<_> = (0..capacity * Workspace::blocks_for(CONTEXT))
            .rev()
            .collect();
        let mut prefer_prefill = false;
        let mut cadence = DecodeCadence::new();
        let mut next_prefill_ring = 0;
        loop {
            let incoming: Vec<_> = if sessions.is_empty() && pending.is_empty() {
                let Ok(job) = jobs.recv() else { break };
                std::iter::once(job)
                    .chain(jobs.try_iter())
                    .take(capacity)
                    .collect()
            } else {
                jobs.try_iter()
                    .take(capacity - sessions.len() - pending.len())
                    .collect()
            };
            for job in incoming {
                let empty = job.request.max_new_tokens == 0;
                let Some(positions) = job
                    .request
                    .prompt_ids
                    .len()
                    .checked_add(job.request.max_new_tokens)
                else {
                    let _ = job
                        .events
                        .send(StreamEvent::Error("request position count overflow".into()));
                    continue;
                };
                match Generation::new(&self.tokenizer, job.request, CONTEXT) {
                    Ok(mut generation) if empty => {
                        let output = generation.advance(&[], |_| true).unwrap().unwrap();
                        let _ = job.events.send(StreamEvent::Done(output));
                    }
                    Ok(generation) if canonical => pending.push_back(Pending {
                        generation,
                        positions,
                        events: job.events,
                        stream: job.stream,
                    }),
                    Ok(generation) => sessions.push(Session {
                        generation,
                        reservation: None,
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
            if let Some(allocator) = &mut reservations {
                // FIFO admission: a request holds every page needed to finish before
                // its first token executes; smaller arrivals never bypass the head.
                while let Some(front) = pending.front() {
                    let Some(reservation) = allocator
                        .try_reserve(front.positions)
                        .expect("Generation validated the complete position budget")
                    else {
                        break;
                    };
                    let request = pending.pop_front().unwrap();
                    sessions.push(Session {
                        generation: request.generation,
                        blocks: reservation.blocks.clone(),
                        reservation: Some(reservation),
                        events: request.events,
                        stream: request.stream,
                    });
                }
            }
            if sessions.is_empty() {
                continue;
            }
            let available: Vec<_> = sessions
                .iter()
                .map(|s| (s.blocks.sliding, s.generation.input_count(PREFILL_CHUNK)))
                .collect();
            let active_ring = self.model.active_prefill_ring();
            let ready = available.iter().filter(|(ring, count)| *count == 1 && Some(*ring) != active_ring).count();
            // Keep all ready decoders together, while allowing several short
            // prompt layers inside the time left before their next round.
            prefer_prefill |= cadence.prompt_fits(Instant::now(), ready);
            let selected = select_work_in(
                capacity,
                &available,
                &mut prefer_prefill,
                &mut next_prefill_ring,
                active_ring,
            );
            let decode_width = if selected.iter().all(|&i| available[i].1 == 1 && Some(available[i].0) != active_ring) { selected.len() } else { 0 };
            let work_started = Instant::now();
            for &index in &selected {
                let session = &mut sessions[index];
                let (_, position) = session.generation.input();
                let count = session.generation.input_count(PREFILL_CHUNK);
                if canonical {
                    assert!(
                        session.blocks.full.len() >= Workspace::blocks_for(position + count),
                        "reserved completion budget covers every scheduled input"
                    );
                }
                while !canonical
                    && session.blocks.full.len() < Workspace::blocks_for(position + count)
                {
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
                    if sequences.len() == 1
                        && (self.model.active_prefill_ring() == Some(sequences[0].blocks.sliding)
                            || (self.model.active_prefill_ring().is_none()
                                && sequences[0].inputs.len() > 1))
                    {
                        if self.model.active_prefill_ring().is_none() {
                            self.model
                                .start_prefill(&mut self.device, &sequences[0], &mut self.workspace)
                                .await?;
                        }
                        let Some(output) = self
                            .model
                            .advance_prefill(&mut self.device, &mut self.workspace)
                            .await?
                        else {
                            return Ok(None);
                        };
                        let logits = if sequences[0].inputs.last().unwrap().logits {
                            read_logits(&mut self.device, output).await?
                        } else {
                            Vec::new()
                        };
                        return Ok(Some(vec![SampleOutput::Logits(logits)]));
                    }
                    let outputs = self
                        .model
                        .forward_outputs(&mut self.device, &sequences, &mut self.workspace)
                        .await?;
                    let logits = match outputs {
                        ForwardOutput::Blocked64(output, live) => {
                            let all_greedy = selected.iter().zip(&chunks).all(|(&index,(inputs,_))|
                                !inputs.last().unwrap().logits || sessions[index].generation.sampling_config().is_greedy());
                            if all_greedy {
                                let bans: Vec<_> = selected.iter().zip(&chunks).map(|(&index,(inputs,_))|
                                    inputs.last().unwrap().logits.then(||sessions[index].generation.banned_tokens())).collect();
                                let tokens = self.greedy.select(&mut self.device, output, &bans).await?;
                                let mut result = Vec::with_capacity(live);
                                for row in 0..live {
                                    result.push(if !chunks[row].0.last().unwrap().logits {
                                        SampleOutput::Logits(Vec::new())
                                    } else if let Some(token) = tokens[row] {
                                        SampleOutput::Greedy(token)
                                    } else {
                                        SampleOutput::Logits(self.greedy.read_fallback(&mut self.device, output, row).await?)
                                    });
                                }
                                result
                            } else {
                                let host = output.to_host::<crate::ops_r64_io_blocked::BlockedLogits>(&mut self.device.pdma).await?;
                                let bytes = host.into_inner().into_buf();
                                let workers = std::thread::available_parallelism().map_or(1, usize::from);
                                crate::host_logit_bytes::decode_blocked_r64(&bytes, live, workers)
                                    .into_iter().map(SampleOutput::Logits).collect()
                            }
                        }
                        ForwardOutput::Rows(outputs) => {
                            let mut logits = Vec::with_capacity(outputs.len());
                            for (output, (inputs, _)) in outputs.iter().zip(&chunks) {
                                logits.push(SampleOutput::Logits(if inputs.last().expect("nonempty input chunk").logits {
                                    read_logits(&mut self.device, output).await?
                                } else {
                                    Vec::new()
                                }));
                            }
                            logits
                        }
                    };
                    Ok::<_, Box<dyn std::error::Error>>(Some(logits))
                })
            }));
            let logits = match result {
                Ok(Ok(Some(logits))) => logits,
                Ok(Ok(None)) => {
                    cadence.observe_prompt(work_started.elapsed());
                    continue;
                }
                failed => {
                    let message = match failed {
                        Ok(Err(error)) => error.to_string(),
                        _ => "generation panicked; worker is offline".to_owned(),
                    };
                    for session in &mut sessions {
                        let _ = session.events.send(StreamEvent::Error(message.clone()));
                        if let Some(reservation) = session.reservation.take() {
                            reservations
                                .as_mut()
                                .unwrap()
                                .release(reservation)
                                .expect("live session owns reservation");
                        }
                    }
                    for queued in &pending {
                        let _ = queued.events.send(StreamEvent::Error(message.clone()));
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
                let emit = |delta| {
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
                    };
                let result = match &logits[row] {
                    SampleOutput::Logits(logits) => session.generation.advance_inputs(count, logits, emit),
                    SampleOutput::Greedy(token) => session.generation.advance_inputs_greedy(count, *token, emit),
                };
                let event = match result {
                    Ok(None) => return true,
                    Ok(Some(output)) => StreamEvent::Done(output),
                    Err(error) => StreamEvent::Error(error.to_string()),
                };
                let _ = session.events.send(event);
                if let Some(reservation) = session.reservation.take() {
                    reservations
                        .as_mut()
                        .unwrap()
                        .release(reservation)
                        .expect("completed session owns reservation");
                } else {
                    free.append(&mut session.blocks.full);
                    rings.push(session.blocks.sliding);
                }
                false
            });
            let finished = Instant::now();
            let elapsed = finished.duration_since(work_started);
            if decode_width > 0 {
                cadence.observe_decode(finished, decode_width, elapsed);
            } else {
                cadence.observe_prompt(elapsed);
            }
        }
    }
}

/// Start prompt chunks in stable ring order. The caller may extend prompt service
/// while the measured decode deadline permits it. Otherwise alternate complete
/// prompt layers and ready decode batches so neither kind of work can starve.
fn select_work_in(
    capacity: usize,
    available: &[(usize, usize)],
    prefer_prefill: &mut bool,
    next_ring: &mut usize,
    active_prefill_ring: Option<usize>,
) -> Vec<usize> {
    if let Some(ring) = active_prefill_ring {
        let pending = available
            .iter()
            .position(|(candidate, _)| *candidate == ring)
            .expect("active prefill retains its session and KV allocation");
        let ready: Vec<_> = available
            .iter()
            .enumerate()
            .filter_map(|(index, (_, count))| (index != pending && *count == 1).then_some(index))
            .collect();
        if !ready.is_empty() && !*prefer_prefill {
            *prefer_prefill = true;
            return ready;
        }
        *prefer_prefill = false;
        return vec![pending];
    }
    let singletons: Vec<_> = available
        .iter()
        .enumerate()
        .filter_map(|(index, (_, count))| (*count == 1).then_some(index))
        .collect();
    let prefill = available
        .iter()
        .enumerate()
        .filter(|(_, (_, count))| *count > 1)
        .min_by_key(|(_, (ring, _))| (*ring + capacity - *next_ring) % capacity)
        .map(|(index, _)| index);
    if !singletons.is_empty() && (!*prefer_prefill || prefill.is_none()) {
        *prefer_prefill = true;
        singletons
    } else if let Some(index) = prefill {
        *next_ring = (available[index].0 + 1) % capacity;
        *prefer_prefill = false;
        vec![index]
    } else {
        Vec::new()
    }
}

#[cfg(test)]
fn select_work(
    available: &[(usize, usize)],
    prefer_prefill: &mut bool,
    next_ring: &mut usize,
    active_prefill_ring: Option<usize>,
) -> Vec<usize> {
    select_work_in(
        CAPACITY,
        available,
        prefer_prefill,
        next_ring,
        active_prefill_ring,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn measured_decode_budget_fits_multiple_prompt_layers_and_reserves_next_round() {
        let start = Instant::now();
        let mut cadence = DecodeCadence::new();
        assert!(!cadence.prompt_fits(start, 4));
        assert!(cadence.prompt_fits(start, 0));
        cadence.prompt_cost = Duration::from_millis(3);
        cadence.observe_decode(start, 4, Duration::from_millis(30));
        assert!(cadence.prompt_fits(start + Duration::from_millis(55), 4));
        assert!(!cadence.prompt_fits(start + Duration::from_millis(58), 4));
        assert!(!cadence.prompt_fits(start, 5), "changed width needs its own observation");
        cadence.observe_decode(start, 64, Duration::from_millis(110));
        assert!(!cadence.prompt_fits(start, 64), "slow decode gets no extra prompt layers");
    }
    #[test]
    fn a_slow_prompt_quantum_reduces_the_remaining_decode_budget() {
        let start = Instant::now();
        let mut cadence = DecodeCadence::new();
        cadence.observe_decode(start, 2, Duration::from_millis(30));
        assert!(cadence.prompt_fits(start + Duration::from_millis(40), 2));
        cadence.observe_prompt(Duration::from_millis(25));
        assert!(!cadence.prompt_fits(start + Duration::from_millis(40), 2));
    }
    #[test]
    fn canonical64_scheduler_visits_high_slots_and_keeps_all_ready_decoders() {
        let mut prefer = true;
        let mut next = 62;
        let work = [(31, 128), (63, 128), (7, 1)];
        assert_eq!(
            select_work_in(64, &work, &mut prefer, &mut next, None),
            vec![1]
        );
        assert_eq!(next, 0);
        prefer = true;
        assert_eq!(
            select_work_in(64, &work, &mut prefer, &mut next, None),
            vec![0]
        );
        let all: Vec<_> = (0..64).map(|ring| (ring, 1)).collect();
        assert_eq!(
            select_work_in(64, &all, &mut prefer, &mut next, None),
            (0..64).collect::<Vec<_>>()
        );
    }
    #[test]
    fn active_prompt_chunk_yields_to_decode_and_keeps_other_prompts_pending() {
        let work = [(2, 1), (7, 128), (12, 1), (5, 128)];
        let mut prefer = false;
        let mut ring = 8;
        assert_eq!(
            select_work(&work, &mut prefer, &mut ring, Some(7)),
            vec![0, 2]
        );
        assert_eq!(select_work(&work, &mut prefer, &mut ring, Some(7)), vec![1]);
        assert_eq!(
            select_work(&work, &mut prefer, &mut ring, Some(7)),
            vec![0, 2]
        );
        assert_eq!(
            select_work(&[(7, 128), (5, 128)], &mut prefer, &mut ring, Some(7)),
            vec![0]
        );
        assert_eq!(
            ring, 8,
            "layer resumes must not advance chunk-level round robin"
        );
    }
    #[test]
    fn ready_decode_requests_stay_batched_while_prompt_progresses() {
        let work = [(2, 1), (7, 16), (12, 1)];
        let mut prefer = false;
        let mut ring = 0;
        assert_eq!(select_work(&work, &mut prefer, &mut ring, None), vec![0, 2]);
        assert_eq!(select_work(&work, &mut prefer, &mut ring, None), vec![1]);
        assert_eq!(select_work(&work, &mut prefer, &mut ring, None), vec![0, 2]);
    }
    #[test]
    fn prompt_round_robin_uses_stable_ring_ids_across_session_removal() {
        let mut prefer = false;
        let mut ring = 0;
        assert_eq!(
            select_work(&[(8, 16), (2, 16), (15, 3)], &mut prefer, &mut ring, None),
            vec![1]
        );
        assert_eq!(
            select_work(&[(8, 16), (15, 3)], &mut prefer, &mut ring, None),
            vec![0]
        );
        assert_eq!(
            select_work(&[(8, 16), (15, 3)], &mut prefer, &mut ring, None),
            vec![1]
        );
        assert_eq!(
            select_work(&[(8, 16), (1, 16)], &mut prefer, &mut ring, None),
            vec![1]
        );
    }
}
