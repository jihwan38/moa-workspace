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
/// Grouped execution uses the same request-owned state as the validated model path.
const GROUP_PREFILL_ENABLED: bool = true;
/// Keep disabled until the chronological model and HTTP gates pass.
const PROMPT512_ENABLED: bool = true;
/// Diagnostic candidates flip these constants globally, never from request data.
/// Public TPOT averages request latency; the private evaluator's exact formula
/// is unverified. This candidate needs measured output-gap and HTTP gates.
const CREDIT_SCHEDULER_ENABLED: bool = false;
const SCHEDULER_TRACE_ENABLED: bool = false;
const DECODE_CREDIT_TARGET: Duration = Duration::from_millis(90);
/// A soft per-delta bound complements the earned average budget. Slow model
/// work and the existing minimum-progress quantum can exceed a prediction.
const DECODE_GAP_SOFTCAP: Duration = Duration::from_millis(150);

pub struct Worker {
    runtime: tokio::runtime::Runtime,
    device: Device,
    model: Model,
    workspace: Workspace,
    tokenizer: Tokenizer,
    greedy: super::greedy::GreedySelector,
}

enum SampleOutput {
    Logits(Vec<f32>),
    Greedy(usize),
}

struct Session {
    generation: Generation,
    blocks: Blocks,
    reservation: Option<Reservation>,
    events: mpsc::Sender<StreamEvent>,
    stream: bool,
    timing: SessionTiming,
}

struct Pending {
    generation: Generation,
    positions: usize,
    events: mpsc::Sender<StreamEvent>,
    stream: bool,
    timing: SessionTiming,
}

#[derive(Default)]
struct TokenProgress {
    first: Option<Instant>,
    last: Option<Instant>,
    tokens: usize,
    max_gap: Duration,
}
impl TokenProgress {
    fn emitted(&mut self, now: Instant) {
        self.first.get_or_insert(now);
        if let Some(last) = self.last {
            self.max_gap = self.max_gap.max(now.saturating_duration_since(last));
        }
        self.last = Some(now);
    }
    fn deadline(&self, target: Duration, gap_cap: Duration) -> Option<Instant> {
        if self.tokens == 0 {
            return None;
        }
        let count = u32::try_from(self.tokens).ok()?;
        let earned = self.first?.checked_add(target.checked_mul(count)?)?;
        let per_gap = self.last?.checked_add(gap_cap)?;
        Some(earned.min(per_gap))
    }
}

struct SessionTiming {
    id: u64,
    prompt_tokens: usize,
    received: Instant,
    admitted: Instant,
    progress: TokenProgress,
    own: Duration,
    decode_wait: Duration,
    prompt_wait: Duration,
    decode_rounds: usize,
    credited: usize,
}
impl SessionTiming {
    fn new(id: u64, prompt_tokens: usize, now: Instant) -> Self {
        Self {
            id,
            prompt_tokens,
            received: now,
            admitted: now,
            progress: TokenProgress::default(),
            own: Duration::ZERO,
            decode_wait: Duration::ZERO,
            prompt_wait: Duration::ZERO,
            decode_rounds: 0,
            credited: 0,
        }
    }
    fn report(&self, finished: Instant) {
        let ms = |duration: Duration| duration.as_secs_f64() * 1000.0;
        let first = self
            .progress
            .first
            .map(|t| t.saturating_duration_since(self.admitted));
        let accounted = self.own + self.decode_wait + self.prompt_wait;
        let average = self
            .progress
            .first
            .filter(|_| self.progress.tokens > 1)
            .map(|t| ms(finished.saturating_duration_since(t)) / (self.progress.tokens - 1) as f64)
            .unwrap_or(0.0);
        eprintln!(
            "SCHED_REQ id={} p={} q={:.1} first={:.1} own={:.1} dw={:.1}/{} pw={:.1} host={:.1} n={} avg={:.2} maxgap={:.1} credited={}",
            self.id,
            self.prompt_tokens,
            ms(self.admitted.saturating_duration_since(self.received)),
            first.map(ms).unwrap_or(-1.0),
            ms(self.own),
            ms(self.decode_wait),
            self.decode_rounds,
            ms(self.prompt_wait),
            first
                .map(|f| ms(f.saturating_sub(accounted)))
                .unwrap_or(0.0),
            self.progress.tokens,
            average,
            ms(self.progress.max_gap),
            self.credited
        );
    }
}

#[derive(Clone, Copy, Default)]
struct WorkCost {
    calls: usize,
    total: Duration,
    max: Duration,
}
impl WorkCost {
    fn add(&mut self, elapsed: Duration) {
        self.calls += 1;
        self.total += elapsed;
        self.max = self.max.max(elapsed);
    }
    fn compact(&self) -> String {
        format!(
            "{}:{:.2}/{:.2}",
            self.calls,
            self.total.as_secs_f64() * 1000.0 / self.calls.max(1) as f64,
            self.max.as_secs_f64() * 1000.0
        )
    }
}
struct SchedulerTrace {
    decode: [WorkCost; crate::axes::R64::SIZE + 1],
    prompt: [WorkCost; 2],
    // Actual owners per dispatch, separate from the cost class: Prompt512 has
    // one owner but shares the four-chunk cost class with four independent rows.
    prompt_dispatches: [usize; 5],
    credit_opportunities: usize,
    unknown_credit: usize,
    credited: WorkCost,
}
impl SchedulerTrace {
    fn new() -> Self {
        Self {
            decode: [WorkCost::default(); crate::axes::R64::SIZE + 1],
            prompt: [WorkCost::default(); 2],
            prompt_dispatches: [0; 5],
            credit_opportunities: 0,
            unknown_credit: 0,
            credited: WorkCost::default(),
        }
    }
    fn work(
        &mut self,
        width: usize,
        scheduled: &[ScheduledInput],
        elapsed: Duration,
        credited: bool,
    ) {
        if width > 0 {
            self.decode[width].add(elapsed);
        } else {
            self.prompt[usize::from(prompt_cost_width(scheduled) == 4)].add(elapsed);
            self.prompt_dispatches[scheduled.len()] += 1;
        }
        if credited {
            self.credited.add(elapsed);
        }
    }
    fn report(&self) {
        let widths: Vec<_> = self
            .decode
            .iter()
            .enumerate()
            .filter(|(_, s)| s.calls > 0)
            .map(|(width, s)| format!("d{width}={}", s.compact()))
            .collect();
        let prompt_widths: Vec<_> = self
            .prompt_dispatches
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(width, count)| format!("p{width}={count}"))
            .collect();
        eprintln!(
            "SCHED_TOTAL credit={} cap_ms={} {} {} pc1={} pc4={} credit_fit={} unknown={} used={}",
            CREDIT_SCHEDULER_ENABLED,
            DECODE_GAP_SOFTCAP.as_millis(),
            widths.join(" "),
            prompt_widths.join(" "),
            self.prompt[0].compact(),
            self.prompt[1].compact(),
            self.credit_opportunities,
            self.unknown_credit,
            self.credited.compact()
        );
    }
}

#[derive(Clone, Copy)]
struct ReadyInput {
    ring: usize,
    position: usize,
    count: usize,
    prompt512: bool,
}

/// Stable request identity and the exact input range whose model state is in
/// flight. Session-vector indices may change while a different decoder finishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScheduledInput {
    ring: usize,
    position: usize,
    count: usize,
}

impl ReadyInput {
    fn groupable(self) -> bool {
        self.count == crate::axes::P128::SIZE
            && self.position % crate::axes::Tf::SIZE + self.count <= crate::axes::Tf::SIZE
    }
}

/// Spend the interval between decode rounds on complete prompt layers. A single
/// layer always remains eligible after a decode so slow rounds cannot starve it.
/// Measurements include host logit reads, sampling and response event emission.
struct DecodeCadence {
    last: Option<(Instant, usize)>,
    costs: [Duration; crate::axes::R64::SIZE + 1],
    // A single layer must not erase the measured cost of a four-request layer.
    prompt_cost: [Duration; 2],
}
impl DecodeCadence {
    fn new() -> Self {
        Self {
            last: None,
            costs: [Duration::ZERO; crate::axes::R64::SIZE + 1],
            prompt_cost: [Duration::from_millis(8), Duration::from_millis(16)],
        }
    }
    fn prompt_fits(&self, now: Instant, ready: usize, prompt_width: usize) -> bool {
        if ready == 0 {
            return true;
        }
        let Some((last, width)) = self.last else {
            return false;
        };
        // An unseen width must run once before using its measured cost.
        width == ready
            && self.costs[ready] != Duration::ZERO
            && now.saturating_duration_since(last)
                + self.prompt_cost[usize::from(prompt_width == 4)]
                + self.costs[ready]
                <= DECODE_CREDIT_TARGET
    }
    /// None retains the conventional policy until both width cost and actual
    /// stream progress are known. Credit comes only from already emitted usage.
    fn earned_prompt_fits(
        &self,
        now: Instant,
        ready: &[&TokenProgress],
        prompt_width: usize,
        gap_cap: Duration,
    ) -> Option<bool> {
        if ready.is_empty() {
            return Some(true);
        }
        let decode = self.costs[ready.len()];
        if decode == Duration::ZERO {
            return None;
        }
        let mut deadline = None::<Instant>;
        for stream in ready {
            let next = stream.deadline(DECODE_CREDIT_TARGET, gap_cap)?;
            deadline = Some(deadline.map_or(next, |current| current.min(next)));
        }
        let finish = now + self.prompt_cost[usize::from(prompt_width == 4)] + decode;
        Some(finish <= deadline.unwrap())
    }
    fn observe_prompt(&mut self, elapsed: Duration, width: usize) {
        let cost = &mut self.prompt_cost[usize::from(width == 4)];
        *cost = elapsed.max(cost.mul_f64(0.9));
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
            eprintln!(
                "startup.weights_ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            );
            let started = std::time::Instant::now();
            let mut workspace = if canonical {
                Workspace::new_canonical64(&mut device, &weights, CONTEXT, crate::axes::R64::SIZE)
                    .await?
            } else {
                Workspace::new(&mut device, &weights, CONTEXT, CAPACITY).await?
            };
            eprintln!(
                "startup.workspace_ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            );
            let started = std::time::Instant::now();
            let mut model = Model::new(&mut device, weights, &mut workspace).await?;
            if (GROUP_PREFILL_ENABLED || PROMPT512_ENABLED) && canonical {
                model.prepare_group_scratch(&mut device).await?;
            }
            eprintln!(
                "startup.model_scratch_ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            );
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
        let mut active_work = Vec::<ScheduledInput>::new();
        let mut request_id = 0u64;
        let mut trace = SchedulerTrace::new();
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
                request_id += 1;
                let timing =
                    SessionTiming::new(request_id, job.request.prompt_ids.len(), Instant::now());
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
                        timing,
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
                        timing,
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
                    let mut request = pending.pop_front().unwrap();
                    request.timing.admitted = Instant::now();
                    sessions.push(Session {
                        generation: request.generation,
                        blocks: reservation.blocks.clone(),
                        reservation: Some(reservation),
                        events: request.events,
                        stream: request.stream,
                        timing: request.timing,
                    });
                }
            }
            if sessions.is_empty() {
                continue;
            }
            let available: Vec<_> = sessions
                .iter()
                .map(|s| ReadyInput {
                    ring: s.blocks.sliding,
                    position: s.generation.input().1,
                    count: s.generation.input_count(PREFILL_CHUNK),
                    prompt512: s.generation.input_count(crate::axes::P512::SIZE)
                        == crate::axes::P512::SIZE,
                })
                .collect();
            let active_rings = self.model.active_prefill_rings();
            debug_assert_eq!(
                active_rings,
                active_work
                    .iter()
                    .map(|input| input.ring)
                    .collect::<Vec<_>>()
            );
            let group_enabled = GROUP_PREFILL_ENABLED && canonical;
            let prompt512_enabled = PROMPT512_ENABLED && canonical;
            let ready = available
                .iter()
                .filter(|input| input.count == 1 && !active_rings.contains(&input.ring))
                .count();
            let pending_selection = pending_prompt(
                capacity,
                &available,
                next_prefill_ring,
                &active_rings,
                group_enabled,
            );
            let pending_inputs = schedule_inputs(
                &available,
                &pending_selection,
                &active_work,
                prompt512_enabled,
            );
            let prompt_width = prompt_cost_width(&pending_inputs);
            // Keep all ready decoders together, while allowing several short
            // prompt layers inside the time left before their next round.
            let now = Instant::now();
            let conventional_fits = cadence.prompt_fits(now, ready, prompt_width);
            let ready_progress: Vec<_> = sessions
                .iter()
                .zip(&available)
                .filter(|(_, input)| input.count == 1 && !active_rings.contains(&input.ring))
                .map(|(session, _)| &session.timing.progress)
                .collect();
            let earned_fits =
                cadence.earned_prompt_fits(now, &ready_progress, prompt_width, DECODE_GAP_SOFTCAP);
            let credit_opportunity = !prefer_prefill
                && !conventional_fits
                && !pending_selection.is_empty()
                && earned_fits == Some(true);
            if SCHEDULER_TRACE_ENABLED {
                trace.credit_opportunities += usize::from(credit_opportunity);
                trace.unknown_credit += usize::from(ready > 0 && earned_fits.is_none());
            }
            prefer_prefill |= if CREDIT_SCHEDULER_ENABLED {
                earned_fits.unwrap_or(conventional_fits)
            } else {
                conventional_fits
            };
            let selected = select_work_in(
                capacity,
                &available,
                &mut prefer_prefill,
                &mut next_prefill_ring,
                &active_rings,
                group_enabled,
            );
            let scheduled = schedule_inputs(&available, &selected, &active_work, prompt512_enabled);
            let decode_width = if selected
                .iter()
                .all(|&i| available[i].count == 1 && !active_rings.contains(&available[i].ring))
            {
                selected.len()
            } else {
                0
            };
            let grouped = selected.len() == 4 && decode_width == 0;
            let cost_width = prompt_cost_width(&scheduled);
            let credited = CREDIT_SCHEDULER_ENABLED && credit_opportunity && decode_width == 0;
            if decode_width == 0 && active_work.is_empty() {
                active_work.clone_from(&scheduled);
            }
            let work_started = Instant::now();
            for (&index, input) in selected.iter().zip(&scheduled) {
                let session = &mut sessions[index];
                let (_, position) = session.generation.input();
                let count = input.count;
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
                        .zip(&scheduled)
                        .map(|(&index, input)| sessions[index].generation.inputs(input.count))
                        .collect();
                    if chunks.iter().zip(&scheduled).any(|((rows, start), input)| {
                        rows.len() != input.count || *start != input.position
                    }) {
                        return Err("scheduled input range changed before model execution".into());
                    }
                    let sequences: Vec<_> = chunks
                        .iter()
                        .zip(&selected)
                        .map(|((inputs, start_position), &index)| Sequence {
                            inputs,
                            start_position: *start_position,
                            blocks: &sessions[index].blocks,
                        })
                        .collect();
                    if grouped {
                        if active_rings.is_empty() {
                            self.model
                                .start_prefill_group(
                                    &mut self.device,
                                    &sequences,
                                    &mut self.workspace,
                                )
                                .await?;
                        }
                        let Some(outputs) = self
                            .model
                            .advance_prefill_group(&mut self.device, &mut self.workspace)
                            .await?
                        else {
                            return Ok(None);
                        };
                        if outputs.len() != sequences.len() {
                            return Err("grouped output count does not match its requests".into());
                        }
                        let mut logits = Vec::with_capacity(outputs.len());
                        for (output, sequence) in outputs.iter().zip(&sequences) {
                            logits.push(SampleOutput::Logits(
                                if sequence.inputs.last().unwrap().logits {
                                    read_logits(&mut self.device, output).await?
                                } else {
                                    Vec::new()
                                },
                            ));
                        }
                        return Ok(Some(logits));
                    }
                    if sequences.len() == 1
                        && (active_rings.as_slice() == [sequences[0].blocks.sliding]
                            || (active_rings.is_empty() && sequences[0].inputs.len() > 1))
                    {
                        if active_rings.is_empty() {
                            if scheduled[0].count == crate::axes::P512::SIZE {
                                self.model
                                    .start_prefill512(
                                        &mut self.device,
                                        &sequences[0],
                                        &mut self.workspace,
                                    )
                                    .await?;
                            } else {
                                self.model
                                    .start_prefill(
                                        &mut self.device,
                                        &sequences[0],
                                        &mut self.workspace,
                                    )
                                    .await?;
                            }
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
                            if live != selected.len() {
                                return Err(
                                    "blocked output count does not match its requests".into()
                                );
                            }
                            let all_greedy =
                                selected.iter().zip(&chunks).all(|(&index, (inputs, _))| {
                                    !inputs.last().unwrap().logits
                                        || sessions[index].generation.sampling_config().is_greedy()
                                });
                            if all_greedy {
                                let bans: Vec<_> = selected
                                    .iter()
                                    .zip(&chunks)
                                    .map(|(&index, (inputs, _))| {
                                        inputs
                                            .last()
                                            .unwrap()
                                            .logits
                                            .then(|| sessions[index].generation.banned_tokens())
                                    })
                                    .collect();
                                let tokens =
                                    self.greedy.select(&mut self.device, output, &bans).await?;
                                let mut result = Vec::with_capacity(live);
                                for row in 0..live {
                                    result.push(if !chunks[row].0.last().unwrap().logits {
                                        SampleOutput::Logits(Vec::new())
                                    } else if let Some(token) = tokens[row] {
                                        SampleOutput::Greedy(token)
                                    } else {
                                        SampleOutput::Logits(
                                            self.greedy
                                                .read_fallback(&mut self.device, output, row)
                                                .await?,
                                        )
                                    });
                                }
                                result
                            } else {
                                let host = output
                                    .to_host::<crate::ops_r64_io_blocked::BlockedLogits>(
                                        &mut self.device.pdma,
                                    )
                                    .await?;
                                let bytes = host.into_inner().into_buf();
                                let workers =
                                    std::thread::available_parallelism().map_or(1, usize::from);
                                crate::host_logit_bytes::decode_blocked_r64(&bytes, live, workers)
                                    .into_iter()
                                    .map(SampleOutput::Logits)
                                    .collect()
                            }
                        }
                        ForwardOutput::Rows(outputs) => {
                            if outputs.len() != selected.len() {
                                return Err("model output count does not match its requests".into());
                            }
                            let mut logits = Vec::with_capacity(outputs.len());
                            for (output, (inputs, _)) in outputs.iter().zip(&chunks) {
                                logits.push(SampleOutput::Logits(
                                    if inputs.last().expect("nonempty input chunk").logits {
                                        read_logits(&mut self.device, output).await?
                                    } else {
                                        Vec::new()
                                    },
                                ));
                            }
                            logits
                        }
                    };
                    Ok::<_, Box<dyn std::error::Error>>(Some(logits))
                })
            }));
            if SCHEDULER_TRACE_ENABLED {
                account_prefill_wait(
                    &mut sessions,
                    &scheduled,
                    decode_width,
                    work_started.elapsed(),
                    credited,
                );
            }
            let completed = match result {
                Ok(Ok(outputs)) => complete_work(
                    &mut sessions,
                    &scheduled,
                    outputs.as_deref(),
                    &mut reservations,
                    &mut free,
                    &mut rings,
                )
                .map_err(str::to_owned),
                Ok(Err(error)) => Err(error.to_string()),
                Err(_) => Err("generation panicked; worker is offline".to_owned()),
            };
            match completed {
                Ok(false) => {
                    let elapsed = work_started.elapsed();
                    cadence.observe_prompt(elapsed, cost_width);
                    if SCHEDULER_TRACE_ENABLED {
                        trace.work(0, &scheduled, elapsed, credited);
                    }
                    continue;
                }
                Ok(true) => {
                    if decode_width == 0 {
                        active_work.clear();
                    }
                }
                Err(message) => {
                    // Any partially executed prompt layer makes the entire model
                    // unusable. Release every live reservation only as we go offline.
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
                    if SCHEDULER_TRACE_ENABLED {
                        trace.report();
                    }
                    return;
                }
            }
            let finished = Instant::now();
            let elapsed = finished.duration_since(work_started);
            if SCHEDULER_TRACE_ENABLED {
                trace.work(decode_width, &scheduled, elapsed, credited);
                if sessions.is_empty() && pending.is_empty() {
                    trace.report();
                }
            }
            if decode_width > 0 {
                cadence.observe_decode(finished, decode_width, elapsed);
            } else {
                cadence.observe_prompt(elapsed, cost_width);
            }
        }
    }
}

fn account_prefill_wait(
    sessions: &mut [Session],
    scheduled: &[ScheduledInput],
    decode_width: usize,
    elapsed: Duration,
    credited: bool,
) {
    for session in sessions {
        if session.timing.progress.first.is_some() {
            continue;
        }
        if scheduled
            .iter()
            .any(|input| input.ring == session.blocks.sliding)
        {
            session.timing.own += elapsed;
            session.timing.credited += usize::from(credited);
        } else if decode_width > 0 {
            session.timing.decode_wait += elapsed;
            session.timing.decode_rounds += 1;
        } else {
            session.timing.prompt_wait += elapsed;
        }
    }
}

/// Completion is atomic with respect to output shape. A layer yield retains
/// every session and reservation, even if its stream receiver has disconnected.
fn complete_work(
    sessions: &mut Vec<Session>,
    scheduled: &[ScheduledInput],
    outputs: Option<&[SampleOutput]>,
    reservations: &mut Option<PageReservations>,
    free: &mut Vec<usize>,
    rings: &mut Vec<usize>,
) -> Result<bool, &'static str> {
    let Some(outputs) = outputs else {
        return Ok(false);
    };
    if outputs.len() != scheduled.len() || scheduled.is_empty() {
        return Err("model output count does not match its requests");
    }
    // Validate every range before advancing any row. In particular, a512-token
    // completion must not silently shrink back to the default128-token limit.
    for (row, input) in scheduled.iter().enumerate() {
        let Some(session) = sessions.iter().find(|s| s.blocks.sliding == input.ring) else {
            return Err("scheduled request is no longer live");
        };
        if input.count == 0
            || scheduled[..row]
                .iter()
                .any(|other| other.ring == input.ring)
            || session.generation.input().1 != input.position
            || session.generation.input_count(input.count) != input.count
        {
            return Err("scheduled input range changed before completion");
        }
    }
    sessions.retain_mut(|session| {
        let Some(row) = scheduled
            .iter()
            .position(|input| input.ring == session.blocks.sliding)
        else {
            return true;
        };
        let count = scheduled[row].count;
        let emit = |delta: Delta<'_>| {
            let text = match delta {
                Delta::Content(text) | Delta::Reasoning(text) => text,
            };
            if !text.is_empty() {
                session.timing.progress.emitted(Instant::now());
            }
            !session.stream
                || session
                    .events
                    .send(match delta {
                        Delta::Content(text) => StreamEvent::Delta(text.to_owned()),
                        Delta::Reasoning(text) => StreamEvent::ReasoningDelta(text.to_owned()),
                    })
                    .is_ok()
        };
        let result = match &outputs[row] {
            SampleOutput::Logits(logits) => session.generation.advance_inputs(count, logits, emit),
            SampleOutput::Greedy(token) => session
                .generation
                .advance_inputs_greedy(count, *token, emit),
        };
        session.timing.progress.tokens = session.generation.completion_tokens();
        let event = match result {
            Ok(None) => return true,
            Ok(Some(output)) => StreamEvent::Done(output),
            Err(error) => StreamEvent::Error(error.to_string()),
        };
        let _ = session.events.send(event);
        if SCHEDULER_TRACE_ENABLED {
            session.timing.report(Instant::now());
        }
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
    Ok(true)
}

/// Four independent requests take priority because `pending_prompt` selects
/// them first. Only its single-request fallback can expand to512 chronological
/// rows. Active ranges are copied exactly until that model checkpoint completes.
fn schedule_inputs(
    available: &[ReadyInput],
    selected: &[usize],
    active: &[ScheduledInput],
    prompt512_enabled: bool,
) -> Vec<ScheduledInput> {
    selected
        .iter()
        .map(|&index| {
            let ready = available[index];
            if let Some(input) = active.iter().find(|input| input.ring == ready.ring) {
                return *input;
            }
            ScheduledInput {
                ring: ready.ring,
                position: ready.position,
                count: if prompt512_enabled
                    && active.is_empty()
                    && selected.len() == 1
                    && ready.prompt512
                {
                    crate::axes::P512::SIZE
                } else {
                    ready.count
                },
            }
        })
        .collect()
}

/// Prompt512 owns one request but runs the same four-P128 dense work as a quad.
fn prompt_cost_width(inputs: &[ScheduledInput]) -> usize {
    if inputs.iter().map(|input| input.count).sum::<usize>() >= crate::axes::P512::SIZE {
        4
    } else {
        inputs.len()
    }
}

/// Resolve active rows in Model order after unrelated sessions may have left.
/// For a new group, the oldest four prompt requests must all be eligible. A
/// short tail keeps its turn instead of being repeatedly bypassed by full chunks.
fn pending_prompt(
    capacity: usize,
    available: &[ReadyInput],
    next_ring: usize,
    active: &[usize],
    group_enabled: bool,
) -> Vec<usize> {
    if !active.is_empty() {
        return active
            .iter()
            .map(|ring| {
                available
                    .iter()
                    .position(|input| input.ring == *ring)
                    .expect("active prefill retains its session and KV allocation")
            })
            .collect();
    }
    let mut pending: Vec<_> = available
        .iter()
        .enumerate()
        .filter_map(|(index, input)| (input.count > 1).then_some(index))
        .collect();
    pending
        .sort_unstable_by_key(|&index| (available[index].ring + capacity - next_ring) % capacity);
    let group = group_enabled
        && pending.len() >= 4
        && pending[..4]
            .iter()
            .all(|&index| available[index].groupable());
    pending.truncate(if group { 4 } else { 1 });
    pending
}

/// Start prompt chunks in stable ring order. The caller may extend prompt service
/// while the measured decode deadline permits it. Otherwise alternate complete
/// prompt layers and ready decode batches so neither kind of work can starve.
fn select_work_in(
    capacity: usize,
    available: &[ReadyInput],
    prefer_prefill: &mut bool,
    next_ring: &mut usize,
    active: &[usize],
    group_enabled: bool,
) -> Vec<usize> {
    let pending = pending_prompt(capacity, available, *next_ring, active, group_enabled);
    let ready: Vec<_> = available
        .iter()
        .enumerate()
        .filter_map(|(index, input)| {
            (input.count == 1 && !active.contains(&input.ring)).then_some(index)
        })
        .collect();
    if !ready.is_empty() && (!*prefer_prefill || pending.is_empty()) {
        *prefer_prefill = true;
        return ready;
    }
    if let Some(&last) = pending.last() {
        if active.is_empty() {
            *next_ring = (available[last].ring + 1) % capacity;
        }
        *prefer_prefill = false;
    }
    pending
}

#[cfg(test)]
fn select_work(
    available: &[(usize, usize)],
    prefer_prefill: &mut bool,
    next_ring: &mut usize,
    active_prefill_ring: Option<usize>,
) -> Vec<usize> {
    let work: Vec<_> = available
        .iter()
        .map(|&(ring, count)| ReadyInput {
            ring,
            position: 0,
            count,
            prompt512: false,
        })
        .collect();
    select_work_in(
        CAPACITY,
        &work,
        prefer_prefill,
        next_ring,
        active_prefill_ring.as_slice(),
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ready_inputs(inputs: &[(usize, usize)]) -> Vec<ReadyInput> {
        inputs
            .iter()
            .map(|&(ring, count)| ReadyInput {
                ring,
                position: 0,
                count,
                prompt512: false,
            })
            .collect()
    }
    #[test]
    fn measured_decode_budget_fits_multiple_prompt_layers_and_reserves_next_round() {
        let start = Instant::now();
        let mut cadence = DecodeCadence::new();
        assert!(!cadence.prompt_fits(start, 4, 1));
        assert!(cadence.prompt_fits(start, 0, 1));
        cadence.prompt_cost[0] = Duration::from_millis(3);
        cadence.observe_decode(start, 4, Duration::from_millis(30));
        assert!(cadence.prompt_fits(start + Duration::from_millis(55), 4, 1));
        assert!(!cadence.prompt_fits(start + Duration::from_millis(58), 4, 1));
        assert!(
            !cadence.prompt_fits(start, 5, 1),
            "changed width needs its own observation"
        );
        cadence.observe_decode(start, 64, Duration::from_millis(110));
        assert!(
            !cadence.prompt_fits(start, 64, 1),
            "slow decode gets no extra prompt layers"
        );
    }
    #[test]
    fn a_slow_prompt_quantum_reduces_the_remaining_decode_budget() {
        let start = Instant::now();
        let mut cadence = DecodeCadence::new();
        cadence.observe_decode(start, 2, Duration::from_millis(30));
        assert!(cadence.prompt_fits(start + Duration::from_millis(40), 2, 1));
        cadence.observe_prompt(Duration::from_millis(25), 1);
        assert!(!cadence.prompt_fits(start + Duration::from_millis(40), 2, 1));
    }
    #[test]
    fn canonical64_scheduler_visits_high_slots_and_keeps_all_ready_decoders() {
        let mut prefer = true;
        let mut next = 62;
        let work = ready_inputs(&[(31, 128), (63, 128), (7, 1)]);
        assert_eq!(
            select_work_in(64, &work, &mut prefer, &mut next, &[], false),
            vec![1]
        );
        assert_eq!(next, 0);
        prefer = true;
        assert_eq!(
            select_work_in(64, &work, &mut prefer, &mut next, &[], false),
            vec![0]
        );
        let all: Vec<_> = (0..64)
            .map(|ring| ReadyInput {
                ring,
                position: 0,
                count: 1,
                prompt512: false,
            })
            .collect();
        assert_eq!(
            select_work_in(64, &all, &mut prefer, &mut next, &[], false),
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
    #[test]
    fn grouped_prompt_cost_survives_interleaved_single_prompt_measurements() {
        let start = Instant::now();
        let mut cadence = DecodeCadence::new();
        cadence.observe_decode(start, 4, Duration::from_millis(30));
        cadence.observe_prompt(Duration::from_millis(16), 4);
        for _ in 0..10 {
            cadence.observe_prompt(Duration::from_millis(2), 1);
        }
        assert!(cadence.prompt_fits(start + Duration::from_millis(50), 4, 1));
        assert!(!cadence.prompt_fits(start + Duration::from_millis(50), 4, 4));
    }

    #[test]
    fn group_selection_keeps_cyclic_order_and_short_or_cross_page_tails_get_their_turn() {
        let mut work = ready_inputs(&[(31, 128), (63, 128), (7, 128), (2, 128), (9, 1)]);
        let mut prefer = true;
        let mut next = 62;
        assert_eq!(
            select_work_in(64, &work, &mut prefer, &mut next, &[], true),
            vec![1, 3, 2, 0]
        );
        assert_eq!(next, 32);
        // A pending short tail in the next four must not be skipped to fill a group.
        work.insert(
            3,
            ReadyInput {
                ring: 1,
                position: 509,
                count: 3,
                prompt512: false,
            },
        );
        assert_eq!(pending_prompt(64, &work, 62, &[], true), vec![1]);
        assert_eq!(pending_prompt(64, &work, 0, &[], true), vec![3]);
        work.remove(3);
        work[0].position = 449; // Synthetic malformed chunk: the model must not group across a page.
        assert_eq!(pending_prompt(64, &work, 62, &[], true), vec![1]);
        work[0].position = 384;
        assert_eq!(pending_prompt(64, &work, 62, &[], true), vec![1, 3, 2, 0]);
        assert_eq!(pending_prompt(64, &work, 62, &[], false), vec![1]);
        assert_eq!(pending_prompt(64, &work[..3], 62, &[], true), vec![1]);
    }

    fn tokenizer() -> &'static Tokenizer {
        static TOKENIZER: std::sync::OnceLock<Tokenizer> = std::sync::OnceLock::new();
        TOKENIZER.get_or_init(|| {
            let mut names: Vec<_> = (0..107).map(|i| format!("token{i}")).collect();
            for (id, text) in [
                (0, "[UNK]"),
                (1, "<eos>"),
                (2, "prompt"),
                (3, "a"),
                (4, "b"),
                (5, "c"),
                (6, "d"),
                (50, "<tool>"),
                (100, "<|channel>"),
                (101, "<channel|>"),
                (106, "<end>"),
            ] {
                names[id] = text.into();
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
                "moa-worker-test-{}-{}.json",
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

    fn session(
        pool: &mut PageReservations,
        prompt: usize,
        max: usize,
    ) -> (Session, mpsc::Receiver<StreamEvent>) {
        let request = crate::host::generate::GenerationRequest {
            prompt_ids: vec![2; prompt],
            soft_at: vec![None; prompt],
            image: None,
            audio: None,
            max_new_tokens: max,
            sampling: crate::host::sampling::SamplingConfig::DEFAULT,
            stop_strings: vec![],
            separate_reasoning: false,
            ignore_eos: false,
        };
        let reservation = pool.try_reserve(prompt + max).unwrap().unwrap();
        let (events, receiver) = mpsc::channel();
        (
            Session {
                generation: Generation::new(tokenizer(), request, CONTEXT).unwrap(),
                blocks: reservation.blocks.clone(),
                reservation: Some(reservation),
                events,
                stream: true,
                timing: SessionTiming::new(0, prompt, Instant::now()),
            },
            receiver,
        )
    }

    fn ready_sessions(sessions: &[Session]) -> Vec<ReadyInput> {
        sessions
            .iter()
            .map(|s| ReadyInput {
                ring: s.blocks.sliding,
                position: s.generation.input().1,
                count: s.generation.input_count(PREFILL_CHUNK),
                prompt512: s.generation.input_count(crate::axes::P512::SIZE)
                    == crate::axes::P512::SIZE,
            })
            .collect()
    }

    fn complete_selected(
        sessions: &mut Vec<Session>,
        selected: &[usize],
        outputs: Option<&[SampleOutput]>,
        pool: &mut Option<PageReservations>,
        free: &mut Vec<usize>,
        rings: &mut Vec<usize>,
    ) -> Result<bool, &'static str> {
        let work = ready_sessions(sessions);
        let scheduled = schedule_inputs(&work, selected, &[], false);
        complete_work(sessions, &scheduled, outputs, pool, free, rings)
    }

    #[test]
    fn incomplete_outputs_cannot_advance_or_release_any_group_member() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (mut sessions, _receivers): (Vec<_>, Vec<_>) = (0..4)
            .map(|_| session(pool.as_mut().unwrap(), 128, 4))
            .unzip();
        let selected = [3, 1, 0, 2];
        let mut free = vec![];
        let mut rings = vec![];
        let partial = vec![
            SampleOutput::Greedy(3),
            SampleOutput::Greedy(4),
            SampleOutput::Greedy(5),
        ];
        assert!(
            complete_selected(
                &mut sessions,
                &selected,
                Some(&partial),
                &mut pool,
                &mut free,
                &mut rings
            )
            .is_err()
        );
        assert_eq!(sessions.len(), 4);
        assert!(
            sessions
                .iter()
                .all(|s| s.generation.input().1 == 0 && s.reservation.is_some())
        );
        assert_eq!(pool.as_ref().unwrap().available_pages(), 252);
        assert!(free.is_empty() && rings.is_empty());
    }

    #[test]
    fn cancelled_stream_retains_group_reservation_until_complete_outputs_arrive() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (mut sessions, receivers): (Vec<_>, Vec<_>) = (0..4)
            .map(|_| session(pool.as_mut().unwrap(), 128, 4))
            .unzip();
        let mut receivers: Vec<_> = receivers.into_iter().map(Some).collect();
        drop(receivers[2].take());
        let selected = [3, 1, 0, 2];
        let mut free = vec![];
        let mut rings = vec![];
        assert!(
            !complete_selected(
                &mut sessions,
                &selected,
                None,
                &mut pool,
                &mut free,
                &mut rings
            )
            .unwrap()
        );
        assert_eq!(sessions.len(), 4);
        assert_eq!(pool.as_ref().unwrap().available_pages(), 252);
        assert!(
            sessions
                .iter()
                .all(|s| s.generation.input().1 == 0 && s.reservation.is_some())
        );
        let outputs = [
            SampleOutput::Greedy(3),
            SampleOutput::Greedy(4),
            SampleOutput::Greedy(5),
            SampleOutput::Greedy(6),
        ];
        assert!(
            complete_selected(
                &mut sessions,
                &selected,
                Some(&outputs),
                &mut pool,
                &mut free,
                &mut rings
            )
            .unwrap()
        );
        assert_eq!(
            sessions
                .iter()
                .map(|s| (s.blocks.sliding, s.generation.input().0.token))
                .collect::<Vec<_>>(),
            vec![(0, 5), (1, 4), (3, 3)]
        );
        assert!(sessions.iter().all(|s| s.generation.input().1 == 128));
        assert_eq!(pool.as_ref().unwrap().available_pages(), 253);
        assert!(free.is_empty() && rings.is_empty());
    }

    #[test]
    fn active_group_keeps_row_identity_after_an_interleaved_decoder_finishes() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (mut sessions, _receivers): (Vec<_>, Vec<_>) = [128, 1, 256, 128, 256]
            .into_iter()
            .map(|prompt| session(pool.as_mut().unwrap(), prompt, 3))
            .unzip();
        // The group is deliberately stored in a different order from the session vector.
        let active = [4, 0, 3, 2];
        let mut prefer = false;
        let mut next = 5;
        let mut free = vec![];
        let mut rings = vec![];
        let ready = ready_sessions(&sessions);
        let decode = select_work_in(64, &ready, &mut prefer, &mut next, &active, true);
        assert_eq!(decode, vec![1]);
        complete_selected(
            &mut sessions,
            &decode,
            Some(&[SampleOutput::Greedy(1)]),
            &mut pool,
            &mut free,
            &mut rings,
        )
        .unwrap();
        assert_eq!(pool.as_ref().unwrap().available_pages(), 252);
        let ready = ready_sessions(&sessions);
        let resumed = select_work_in(64, &ready, &mut prefer, &mut next, &active, true);
        assert_eq!(resumed, vec![3, 0, 2, 1]);
        assert_eq!(
            next, 5,
            "layer resumes cannot rotate the chunk-level cursor"
        );
        assert!(
            !complete_selected(
                &mut sessions,
                &resumed,
                None,
                &mut pool,
                &mut free,
                &mut rings
            )
            .unwrap()
        );
        let outputs = [
            SampleOutput::Logits(vec![]),
            SampleOutput::Greedy(3),
            SampleOutput::Greedy(4),
            SampleOutput::Logits(vec![]),
        ];
        complete_selected(
            &mut sessions,
            &resumed,
            Some(&outputs),
            &mut pool,
            &mut free,
            &mut rings,
        )
        .unwrap();
        for session in &sessions {
            assert_eq!(session.generation.input().1, 128);
            let token = session.generation.input().0.token;
            assert_eq!(
                token,
                match session.blocks.sliding {
                    0 => 3,
                    3 => 4,
                    _ => 2,
                }
            );
        }
        // Completed prompt rows decode together; intermediate chunks keep their turn.
        let work = ready_sessions(&sessions);
        assert_eq!(
            select_work_in(64, &work, &mut prefer, &mut next, &[], true),
            vec![0, 2]
        );
        assert_eq!(pool.as_ref().unwrap().available_pages(), 252);
    }
    #[test]
    fn four_independent_prompts_precede_single_owner512_and_share_its_cost_class() {
        let mut work = ready_inputs(&[(7, 128), (2, 128), (31, 128), (63, 128), (9, 1)]);
        for input in &mut work[..4] {
            input.prompt512 = true;
        }
        let quad = pending_prompt(64, &work, 0, &[], true);
        let scheduled = schedule_inputs(&work, &quad, &[], true);
        assert_eq!(
            scheduled
                .iter()
                .map(|input| input.count)
                .collect::<Vec<_>>(),
            vec![128; 4]
        );
        assert_eq!(prompt_cost_width(&scheduled), 4);
        let pending = pending_prompt(64, &work[..3], 0, &[], true);
        let single = schedule_inputs(&work[..3], &pending, &[], true);
        assert_eq!(
            single,
            vec![ScheduledInput {
                ring: 2,
                position: 0,
                count: 512
            }]
        );
        assert_eq!(prompt_cost_width(&single), 4);
        assert_eq!(schedule_inputs(&work, &pending, &[], false)[0].count, 128);
        // The oldest short prompt keeps its turn, even when newer prompts can supply512.
        work[1].prompt512 = false;
        work[1].count = 3;
        let pending = pending_prompt(64, &work, 0, &[], true);
        assert_eq!(schedule_inputs(&work, &pending, &[], true)[0].count, 3);
    }

    fn advance_prompt_to(session: &mut Session, position: usize) {
        while session.generation.input().1 < position {
            let count = session.generation.input_count(PREFILL_CHUNK);
            assert!(session.generation.input().1 + count <= position);
            assert!(
                session
                    .generation
                    .advance_inputs(count, &[], |_| true)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn prompt512_range_survives_decoder_removal_yields_and_a_cross_page_completion() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (decoder, decoder_rx) = session(pool.as_mut().unwrap(), 1, 3);
        let (mut prompt, prompt_rx) = session(pool.as_mut().unwrap(), 1409, 4);
        advance_prompt_to(&mut prompt, 896);
        assert_eq!(
            prompt.blocks.full.len(),
            3,
            "all future pages were reserved at admission"
        );
        let prompt_ring = prompt.blocks.sliding;
        let mut sessions = vec![decoder, prompt];
        let _receivers = [decoder_rx, prompt_rx];
        let mut free = vec![];
        let mut rings = vec![];
        let work = ready_sessions(&sessions);
        let active = schedule_inputs(&work, &[1], &[], true);
        assert_eq!(
            active,
            vec![ScheduledInput {
                ring: prompt_ring,
                position: 896,
                count: 512
            }]
        );
        assert_eq!(
            Workspace::blocks_for(active[0].position + active[0].count),
            3
        );
        let (inputs, start) = sessions[1].generation.inputs(active[0].count);
        assert_eq!((start, inputs.len()), (896, 512));
        assert!(inputs.iter().all(|input| !input.logits));
        let mut prefer = false;
        let mut next = 2;
        let selected = select_work_in(64, &work, &mut prefer, &mut next, &[prompt_ring], true);
        assert_eq!(selected, vec![0]);
        complete_selected(
            &mut sessions,
            &selected,
            Some(&[SampleOutput::Greedy(1)]),
            &mut pool,
            &mut free,
            &mut rings,
        )
        .unwrap();
        let work = ready_sessions(&sessions);
        let selected = select_work_in(64, &work, &mut prefer, &mut next, &[prompt_ring], true);
        assert_eq!(selected, vec![0]);
        let resumed = schedule_inputs(&work, &selected, &active, false);
        assert_eq!(
            resumed, active,
            "a resume cannot recompute the default128-token count"
        );
        assert!(
            !complete_work(
                &mut sessions,
                &resumed,
                None,
                &mut pool,
                &mut free,
                &mut rings
            )
            .unwrap()
        );
        assert_eq!(sessions[0].generation.input().1, 896);
        assert_eq!(pool.as_ref().unwrap().available_pages(), 253);
        let mut stale = resumed.clone();
        stale[0].position += 128;
        assert!(
            complete_work(
                &mut sessions,
                &stale,
                Some(&[SampleOutput::Logits(vec![])]),
                &mut pool,
                &mut free,
                &mut rings
            )
            .is_err()
        );
        assert_eq!(sessions[0].generation.input().1, 896);
        complete_work(
            &mut sessions,
            &resumed,
            Some(&[SampleOutput::Logits(vec![])]),
            &mut pool,
            &mut free,
            &mut rings,
        )
        .unwrap();
        assert_eq!(sessions[0].generation.input().1, 1408);
        assert_eq!(sessions[0].generation.input_count(PREFILL_CHUNK), 1);
        assert_eq!(pool.as_ref().unwrap().available_pages(), 253);
        assert_eq!(next, 2);
    }

    #[test]
    fn cancelled_final_prompt512_holds_one_reservation_until_its_output_completes() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (mut prompt, receiver) = session(pool.as_mut().unwrap(), 896, 4);
        advance_prompt_to(&mut prompt, 384);
        let mut sessions = vec![prompt];
        let work = ready_sessions(&sessions);
        let scheduled = schedule_inputs(&work, &[0], &[], true);
        assert_eq!(scheduled[0].count, 512);
        let (inputs, start) = sessions[0].generation.inputs(scheduled[0].count);
        assert_eq!((start, inputs.len()), (384, 512));
        assert!(inputs[..511].iter().all(|input| !input.logits));
        assert!(inputs[511].logits);
        drop(receiver);
        let mut free = vec![];
        let mut rings = vec![];
        assert!(
            !complete_work(
                &mut sessions,
                &scheduled,
                None,
                &mut pool,
                &mut free,
                &mut rings
            )
            .unwrap()
        );
        assert_eq!(sessions[0].generation.input().1, 384);
        assert_eq!(pool.as_ref().unwrap().available_pages(), 254);
        assert!(
            complete_work(
                &mut sessions,
                &scheduled,
                Some(&[]),
                &mut pool,
                &mut free,
                &mut rings
            )
            .is_err()
        );
        assert_eq!(sessions[0].generation.input().1, 384);
        complete_work(
            &mut sessions,
            &scheduled,
            Some(&[SampleOutput::Greedy(3)]),
            &mut pool,
            &mut free,
            &mut rings,
        )
        .unwrap();
        assert!(sessions.is_empty());
        assert_eq!(pool.as_ref().unwrap().available_pages(), 256);
        assert!(free.is_empty() && rings.is_empty());
    }
    #[test]
    fn earned_credit_uses_only_completed_tokens_and_the_earliest_stream_deadline() {
        let start = Instant::now();
        let mut cadence = DecodeCadence::new();
        cadence.prompt_cost[0] = Duration::from_millis(20);
        cadence.observe_decode(
            start + Duration::from_millis(30),
            1,
            Duration::from_millis(30),
        );
        let fast = TokenProgress {
            first: Some(start),
            last: Some(start + Duration::from_millis(30)),
            tokens: 2,
            max_gap: Duration::from_millis(30),
        };
        let now = start + Duration::from_millis(100);
        assert!(!cadence.prompt_fits(now, 1, 1));
        assert_eq!(
            cadence.earned_prompt_fits(now, &[&fast], 1, DECODE_GAP_SOFTCAP),
            Some(true)
        );
        let unearned = TokenProgress { tokens: 1, ..fast };
        assert_eq!(
            cadence.earned_prompt_fits(now, &[&unearned], 1, DECODE_GAP_SOFTCAP),
            Some(false)
        );
        cadence.costs[2] = Duration::from_millis(30);
        let newly_ready = TokenProgress {
            first: Some(start + Duration::from_millis(50)),
            last: Some(start + Duration::from_millis(50)),
            tokens: 1,
            max_gap: Duration::ZERO,
        };
        assert_eq!(
            cadence.earned_prompt_fits(now, &[&fast, &newly_ready], 1, DECODE_GAP_SOFTCAP),
            Some(false)
        );
    }

    #[test]
    fn soft_gap_cap_limits_banked_credit_and_unknown_progress_keeps_the_control_policy() {
        let start = Instant::now();
        let mut cadence = DecodeCadence::new();
        let rich = TokenProgress {
            first: Some(start),
            last: Some(start + Duration::from_millis(90)),
            tokens: 100,
            max_gap: Duration::ZERO,
        };
        let now = start + Duration::from_millis(100);
        assert_eq!(
            cadence.earned_prompt_fits(now, &[&rich], 1, DECODE_GAP_SOFTCAP),
            None
        );
        cadence.costs[1] = Duration::from_millis(30);
        cadence.prompt_cost[0] = Duration::from_millis(110);
        assert_eq!(
            cadence.earned_prompt_fits(now, &[&rich], 1, DECODE_GAP_SOFTCAP),
            Some(true)
        );
        cadence.prompt_cost[0] = Duration::from_millis(111);
        assert_eq!(
            cadence.earned_prompt_fits(now, &[&rich], 1, DECODE_GAP_SOFTCAP),
            Some(false)
        );
        assert_eq!(
            cadence.earned_prompt_fits(now, &[&TokenProgress::default()], 1, DECODE_GAP_SOFTCAP),
            None
        );
        // The established minimum-progress alternation remains available even
        // when an overloaded batch cannot satisfy a predicted soft deadline.
        let work = ready_inputs(&[(0, 1), (1, 128)]);
        let mut prefer = true;
        let mut next = 0;
        assert_eq!(
            select_work_in(64, &work, &mut prefer, &mut next, &[], true),
            vec![1]
        );
        assert_eq!(
            select_work_in(64, &work, &mut prefer, &mut next, &[], true),
            vec![0]
        );
    }

    #[test]
    fn progress_tracks_real_generation_usage_and_first_nonempty_delta() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (session, receiver) = session(pool.as_mut().unwrap(), 256, 4);
        let mut sessions = vec![session];
        let mut free = vec![];
        let mut rings = vec![];
        complete_selected(
            &mut sessions,
            &[0],
            Some(&[SampleOutput::Logits(vec![])]),
            &mut pool,
            &mut free,
            &mut rings,
        )
        .unwrap();
        assert!(sessions[0].timing.progress.first.is_none());
        assert_eq!(sessions[0].timing.progress.tokens, 0);
        complete_selected(
            &mut sessions,
            &[0],
            Some(&[SampleOutput::Greedy(3)]),
            &mut pool,
            &mut free,
            &mut rings,
        )
        .unwrap();
        let first = sessions[0].timing.progress.first.unwrap();
        assert_eq!(sessions[0].timing.progress.tokens, 1);
        assert!(matches!(receiver.try_recv(), Ok(StreamEvent::Delta(_))));
        complete_selected(
            &mut sessions,
            &[0],
            Some(&[SampleOutput::Greedy(4)]),
            &mut pool,
            &mut free,
            &mut rings,
        )
        .unwrap();
        assert_eq!(sessions[0].timing.progress.first, Some(first));
        assert_eq!(sessions[0].timing.progress.tokens, 2);
        assert!(sessions[0].timing.progress.last.unwrap() >= first);
        assert_eq!(sessions[0].generation.completion_tokens(), 2);
    }

    #[test]
    fn compact_trace_separates_own_compute_decoder_wait_and_other_prompt_wait() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (mut sessions, _receivers): (Vec<_>, Vec<_>) = (0..3)
            .map(|_| session(pool.as_mut().unwrap(), 256, 4))
            .unzip();
        let work = ready_sessions(&sessions);
        let own = schedule_inputs(&work, &[0], &[], false);
        let decoder = schedule_inputs(&work, &[1], &[], false);
        let other = schedule_inputs(&work, &[2], &[], false);
        account_prefill_wait(&mut sessions, &own, 0, Duration::from_millis(2), true);
        account_prefill_wait(&mut sessions, &decoder, 1, Duration::from_millis(20), false);
        account_prefill_wait(&mut sessions, &other, 0, Duration::from_millis(7), false);
        let stats = &sessions[0].timing;
        assert_eq!(
            (stats.own, stats.decode_wait, stats.prompt_wait),
            (
                Duration::from_millis(2),
                Duration::from_millis(20),
                Duration::from_millis(7)
            )
        );
        assert_eq!((stats.decode_rounds, stats.credited), (1, 1));
        sessions[0].timing.progress.emitted(Instant::now());
        account_prefill_wait(&mut sessions, &other, 0, Duration::from_millis(100), false);
        assert_eq!(sessions[0].timing.prompt_wait, Duration::from_millis(7));
    }

    #[test]
    fn dispatch_trace_distinguishes_r64_width_and_prompt_owner_width_from_cost_class() {
        let mut trace = SchedulerTrace::new();
        let decode: Vec<_> = (0..17)
            .map(|ring| ScheduledInput {
                ring,
                position: 128,
                count: 1,
            })
            .collect();
        trace.work(17, &decode, Duration::from_millis(70), false);
        let single = [ScheduledInput {
            ring: 31,
            position: 0,
            count: 512,
        }];
        trace.work(0, &single, Duration::from_millis(8), true);
        let quad: Vec<_> = (0..4)
            .map(|ring| ScheduledInput {
                ring,
                position: 0,
                count: 128,
            })
            .collect();
        trace.work(0, &quad, Duration::from_millis(9), false);
        assert_eq!(trace.decode[17].calls, 1);
        assert_eq!(trace.decode[16].calls, 0);
        assert_eq!(trace.prompt_dispatches[1], 1);
        assert_eq!(trace.prompt_dispatches[4], 1);
        assert_eq!(trace.prompt[0].calls, 0);
        assert_eq!(trace.prompt[1].calls, 2);
        assert_eq!(trace.credited.calls, 1);
    }
}
