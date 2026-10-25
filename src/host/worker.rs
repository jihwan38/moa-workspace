//! Bounded request scheduling and KV ownership for the fixed HTTP frontend.

use super::admission::{PageReservations, Reservation};
use std::{collections::VecDeque, sync::mpsc};

use crate::api::worker::{Job, StreamEvent};
use crate::host::generate::{Delta, Generation, PREFILL_CHUNK, read_logits};
use crate::host::load;
use crate::host::model::{ForwardOutput, Model, PrefillAdvanceTiming, Sequence};
use crate::host::runtime::{Blocks, Workspace};
use crate::host::tokenizer::Tokenizer;
use furiosa_opt_std::prelude::*;
use std::time::{Duration, Instant};

#[path = "startup_cohort.rs"]
mod startup_cohort;
use startup_cohort::{StartupCohort, StartupDelivery};
#[path = "chunk_trace.rs"]
mod chunk_trace;
#[path = "idle_batch.rs"]
mod idle_batch;

/// Requests computed together; more wait in the queue.
pub const CAPACITY: usize = crate::axes::B::SIZE;
/// Prompt plus generated positions allowed per request.
pub const CONTEXT: usize = 8192;
/// Grouped execution uses the same request-owned state as the validated model path.
const GROUP_PREFILL_ENABLED: bool = true;
/// Keep disabled until the chronological model and HTTP gates pass.
const PROMPT512_ENABLED: bool = true;
/// Pack four chronological P128 quarters from the two oldest owners after quad priority.
const PROMPT_PAIR512_ENABLED: bool = true;
/// Two real P128 quarters share the P512 dense path; inactive quarters own no KV.
const TAIL256_ENABLED: bool = true;
/// Three actual owners can fill512 before any tensor-padded tail is selected.
const THREE_OWNER512_ENABLED: bool = true;
/// Three real quarters share P512 dense work, with no inactive KV owner.
const TAIL384_ENABLED: bool = true;
/// Diagnostic candidates flip these constants globally, never from request data.
/// Public TPOT averages request latency; the private evaluator's exact formula
/// is unverified. This candidate needs measured output-gap and HTTP gates.
const CREDIT_SCHEDULER_ENABLED: bool = true;
const SCHEDULER_TRACE_ENABLED: bool = true;
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
    delivery: StartupDelivery,
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
    deadline_overruns: usize,
    max_deadline_overrun: Duration,
}
impl TokenProgress {
    fn emitted(&mut self, now: Instant) {
        if let Some(deadline) = self.deadline(DECODE_CREDIT_TARGET, DECODE_GAP_SOFTCAP) {
            if now > deadline {
                self.deadline_overruns += 1;
                self.max_deadline_overrun =
                    self.max_deadline_overrun.max(now.duration_since(deadline));
            }
        }
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
    phases: [Duration; 3],
    cohort_hold: Duration,
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
            phases: [Duration::ZERO; 3],
            cohort_hold: Duration::ZERO,
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
            "SCHED_REQ id={} p={} q={:.1} first={:.1} own={:.1} dw={:.1}/{} pw={:.1} host={:.1} n={} avg={:.2} maxgap={:.1} over={}/{:.1} credited={} ph={:.1}/{:.1}/{:.1} hold={:.1}",
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
            self.progress.deadline_overruns,
            ms(self.progress.max_deadline_overrun),
            self.credited,
            ms(self.phases[0]),
            ms(self.phases[1]),
            ms(self.phases[2]),
            ms(self.cohort_hold)
        );
    }
}

#[derive(Default)]
struct PromptPhases {
    preparation: Option<Duration>,
    advance: PrefillAdvanceTiming,
    read: Duration,
}
impl PromptPhases {
    // Preparation can encode media and finalization can project/read logits.
    // Neither repeats on every layer; only the measured layer trains its cost.
    fn recurring_cost(&self, fallback: Duration) -> Duration {
        self.advance.layer.unwrap_or(fallback)
    }
    fn times(&self) -> [Option<Duration>; 3] {
        [
            self.preparation,
            self.advance.layer,
            self.advance.finish.map(|t| t + self.read),
        ]
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
    phases: [[WorkCost; 2]; 3],
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
            phases: [[WorkCost::default(); 2]; 3],
            credit_opportunities: 0,
            unknown_credit: 0,
            credited: WorkCost::default(),
        }
    }
    fn prompt_phases(&mut self, width: usize, phases: &PromptPhases) {
        for (stats, elapsed) in self.phases.iter_mut().zip(phases.times()) {
            if let Some(elapsed) = elapsed {
                stats[usize::from(width == 4)].add(elapsed);
            }
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
            "SCHED_TOTAL credit={} cap_ms={} {} {} pc1={} pc4={} credit_fit={} unknown={} used={} prep1={} prep4={} layer1={} layer4={} final1={} final4={}",
            CREDIT_SCHEDULER_ENABLED,
            DECODE_GAP_SOFTCAP.as_millis(),
            widths.join(" "),
            prompt_widths.join(" "),
            self.prompt[0].compact(),
            self.prompt[1].compact(),
            self.credit_opportunities,
            self.unknown_credit,
            self.credited.compact(),
            self.phases[0][0].compact(),
            self.phases[0][1].compact(),
            self.phases[1][0].compact(),
            self.phases[1][1].compact(),
            self.phases[2][0].compact(),
            self.phases[2][1].compact()
        );
    }
}

#[derive(Clone, Copy)]
struct ReadyInput {
    ring: usize,
    position: usize,
    count: usize,
    prompt512: bool,
    prompt384: bool,
    prompt256: bool,
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
    fn aligned_quarter(self) -> bool {
        self.groupable() && self.position % crate::axes::P128::SIZE == 0
    }
}

/// Spend the interval between decode rounds on complete prompt layers. A single
/// layer always remains eligible after a decode so slow rounds cannot starve it.
/// Decode measurements include host reads, sampling and response event emission.
/// Prompt estimates use recurring layer time; preparation and finalization are
/// atomic and separately traced, so their stalls cannot inflate later layers.
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

/// Validate the real server's admission and owning allocations before any job.
/// Public Buffer metadata is read only; no cache data is transferred or changed.
fn verify_canonical_storage_identity(workspace: &Workspace) -> Result<(), Box<dyn std::error::Error>> {
    let physical = crate::ops_pool_views::Rc64::SIZE;
    let query_rows = crate::axes::R64::SIZE;
    let pages = crate::ops_request_attention::Fc::SIZE;
    if !workspace.is_canonical() || physical != 32 || query_rows != 64
        || workspace.requests() != physical || workspace.physical_full_pages() != pages
        || pages != 256 || workspace.capacity() != CONTEXT
    {
        return Err("physical32 server storage/admission identity mismatch".into());
    }
    #[cfg(backend = "npu")]
    {
        use furiosa_opt_std::runtime::{Buffers, DeviceSend};
        use super::runtime::KvCache;
        let sliding_bytes = physical * crate::axes::Ts::SIZE * crate::axes::Ns::SIZE
            * crate::axes::Ds::SIZE * 2;
        let full_bytes = pages * crate::axes::Tf::SIZE * crate::axes::Df::SIZE * 2;
        let mut sliding_layers = 0usize;
        let mut full_layers = 0usize;
        let mut allocated_bytes = 0usize;
        for cache in &workspace.kv_cache {
            let mut buffers = Buffers::new();
            let expected = match cache {
                KvCache::SlidingCanonical { k, v } => {
                    DeviceSend::bind(k, &mut buffers)?;
                    DeviceSend::bind(v, &mut buffers)?;
                    sliding_layers += 1;
                    sliding_bytes
                }
                KvCache::FullCanonical { k, v } => {
                    DeviceSend::bind(k, &mut buffers)?;
                    DeviceSend::bind(v, &mut buffers)?;
                    full_layers += 1;
                    full_bytes
                }
                _ => return Err("physical32 server needs canonical owning KV pools".into()),
            };
            if buffers.len() != 2 || buffers.iter().any(|buffer| buffer.size() != expected) {
                return Err("physical32 server allocated KV extent mismatch".into());
            }
            allocated_bytes += buffers.iter().map(|buffer| buffer.size()).sum::<usize>();
        }
        if sliding_layers + full_layers != crate::LAYERS {
            return Err("physical32 server KV layer count mismatch".into());
        }
        eprintln!("startup.storage_identity physical_rings={physical} admission={} query_rows={query_rows} full_pages={pages} context={} sliding_pool_bytes={sliding_bytes} full_pool_bytes={full_bytes} sliding_layers={sliding_layers} full_layers={full_layers} allocated_kv_bytes={allocated_bytes} verified_owning_buffers=true",
            workspace.requests(), workspace.capacity());
    }
    Ok(())
}

impl Worker {
    pub fn new(tokenizer: Tokenizer) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_storage(tokenizer, true)
    }
    /// Canonical KV pool with32 resident requests; R64 compute capacity is unchanged.
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
                Workspace::new_canonical64(&mut device, &weights, CONTEXT, crate::ops_pool_views::Rc64::SIZE)
                    .await?
            } else {
                Workspace::new(&mut device, &weights, CONTEXT, CAPACITY).await?
            };
            if canonical {
                verify_canonical_storage_identity(&workspace)?;
            }
            eprintln!(
                "startup.workspace_ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            );
            let started = std::time::Instant::now();
            let mut model = Model::new(&mut device, weights, &mut workspace).await?;
            if (GROUP_PREFILL_ENABLED || PROMPT512_ENABLED) && canonical {
                model.prepare_group_scratch(&mut device).await?;
                model.prepare_projection_owned_ffn(&mut device).await?;
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
        let mut cohort = StartupCohort::with_late_admission(
            startup_cohort::RECOVERY_ENABLED, startup_cohort::LATE_ADMISSION_ENABLED,
        );
        // Process-level diagnostics only; disabled unless explicitly requested.
        let chunk_trace_enabled = SCHEDULER_TRACE_ENABLED && chunk_trace::enabled();
        let mut active_chunk_trace = None::<chunk_trace::ChunkTrace>;
        let mut chunk_sequence = 0usize;
        loop {
            // Flush before any potentially blocking receive, including held Done/EOS.
            if let Some(reason) = cohort.release_reason(&sessions, Instant::now()) {
                cohort.release(
                    &mut sessions,
                    &mut reservations,
                    &mut free,
                    &mut rings,
                    Instant::now(),
                    reason,
                );
                if SCHEDULER_TRACE_ENABLED && sessions.is_empty() && pending.is_empty() {
                    trace.report();
                }
            }
            if sessions.is_empty() && pending.is_empty() {
                cohort.reset_idle();
            }
            let incoming: Vec<_> = if sessions.is_empty() && pending.is_empty() {
                // A fixed short idle window avoids launching an isolated P128
                // while the rest of an arriving batch is still being enqueued.
                // Receipt timestamps include this wait in the cohort budget.
                let Some(batch) = idle_batch::receive(&jobs, capacity) else { break };
                if chunk_trace_enabled {
                    eprintln!("SCHED_IDLE batch={} wait_ms={:.3} receipt_span_ms={:.3}",
                        batch.len(), batch[0].1.elapsed().as_secs_f64() * 1000.0,
                        batch.last().unwrap().1.duration_since(batch[0].1).as_secs_f64() * 1000.0);
                }
                batch
            } else {
                jobs.try_iter()
                    .take(capacity - sessions.len() - pending.len())
                    .map(|job| (job, Instant::now()))
                    .collect()
            };
            for (job, received) in incoming {
                request_id += 1;
                let timing =
                    SessionTiming::new(request_id, job.request.prompt_ids.len(), received);
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
                        delivery: StartupDelivery::default(),
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
                        delivery: StartupDelivery::default(),
                        generation: request.generation,
                        blocks: reservation.blocks.clone(),
                        reservation: Some(reservation),
                        events: request.events,
                        stream: request.stream,
                        timing: request.timing,
                    });
                }
            }
            if startup_cohort::ENABLED {
                // FIFO reservations are settled. Ready/expired release wins
                // before late joins and before any index-based work selection.
                cohort.after_admission(&mut sessions, &mut reservations, &mut free,
                    &mut rings, Instant::now());
            }
            if sessions.is_empty() {
                continue;
            }
            if startup_cohort::ENABLED {
                // A decoder-free interval can occur without an empty worker.
                // Preserve all active model state and the original receipt budget.
                cohort.rearm_prefill_only(&sessions);
            }
            let available: Vec<_> = sessions
                .iter()
                .map(|s| ready_input(s, cohort.allows(s)))
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
            let paired384128 = decode_width == 0 && scheduled.len() == 2
                && scheduled.iter().all(|s| matches!(s.count, 128 | 256 | 384))
                && scheduled.iter().map(|s| s.count).sum::<usize>() == crate::axes::P512::SIZE;
            let tail256 = decode_width == 0 && is_tail256(&scheduled);
            let tail384 = decode_width == 0 && is_tail384(&scheduled);
            let three_owner512 = decode_width == 0 && is_three_owner512(&scheduled);
            let cost_width = prompt_cost_width(&scheduled);
            let credited = CREDIT_SCHEDULER_ENABLED && credit_opportunity && decode_width == 0;
            let mut immediate_chunk_trace = None;
            if chunk_trace_enabled && (decode_width > 0 || active_work.is_empty()) {
                let snapshot = chunk_trace::ChunkTrace::capture(
                    chunk_sequence, &sessions, &scheduled, &available, cohort.active(),
                    ready, pending.len(), Instant::now(),
                );
                chunk_sequence += usize::from(snapshot.is_some());
                if decode_width == 0 { active_chunk_trace = snapshot; }
                else { immediate_chunk_trace = snapshot; }
            }
            if decode_width == 0 && active_work.is_empty() {
                active_work.clone_from(&scheduled);
            }
            let work_started = Instant::now();
            let mut prompt_phases = PromptPhases::default();
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
                    if grouped || paired384128 || three_owner512 || tail384 || tail256 {
                        if active_rings.is_empty() {
                            let mark = Instant::now();
                            if three_owner512 {
                                self.model.start_prefill_three_owner512(
                                    &mut self.device, &sequences, &mut self.workspace,
                                ).await?;
                            } else if tail384 {
                                self.model.start_prefill_tail384(
                                    &mut self.device, &sequences, &mut self.workspace,
                                ).await?;
                            } else if tail256 && sequences.len() == 1 {
                                self.model.start_prefill256(
                                    &mut self.device, &sequences[0], &mut self.workspace,
                                ).await?;
                            } else if tail256 {
                                self.model.start_prefill_pair256(
                                    &mut self.device, &sequences, &mut self.workspace,
                                ).await?;
                            } else if paired384128 {
                                self.model.start_prefill_pair512(
                                    &mut self.device, &sequences, &mut self.workspace,
                                ).await?;
                            } else {
                                self.model.start_prefill_group(
                                    &mut self.device, &sequences, &mut self.workspace,
                                ).await?;
                            }
                            prompt_phases.preparation = Some(mark.elapsed());
                        }
                        let outputs = if three_owner512 {
                            self.model.advance_prefill_three_owner512_measured(
                                &mut self.device, &mut self.workspace,
                                Some(&mut prompt_phases.advance),
                            ).await?
                        } else if tail384 {
                            self.model.advance_prefill_tail384_measured(
                                &mut self.device, &mut self.workspace,
                                Some(&mut prompt_phases.advance),
                            ).await?
                        } else if tail256 {
                            self.model.advance_prefill_tail256_measured(
                                &mut self.device, &mut self.workspace,
                                Some(&mut prompt_phases.advance),
                            ).await?
                        } else if paired384128 {
                            self.model.advance_prefill384_128_measured(
                                &mut self.device, &mut self.workspace,
                                Some(&mut prompt_phases.advance),
                            ).await?
                        } else {
                            self.model.advance_prefill_group_measured(
                                &mut self.device, &mut self.workspace,
                                Some(&mut prompt_phases.advance),
                            ).await?
                        };
                        let Some(outputs) = outputs else { return Ok(None); };
                        if outputs.len() != sequences.len() {
                            return Err("grouped output count does not match its requests".into());
                        }
                        let mark = Instant::now();
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
                        prompt_phases.read = mark.elapsed();
                        return Ok(Some(logits));
                    }
                    if sequences.len() == 1
                        && (active_rings.as_slice() == [sequences[0].blocks.sliding]
                            || (active_rings.is_empty() && sequences[0].inputs.len() > 1))
                    {
                        if active_rings.is_empty() {
                            let mark = Instant::now();
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
                            prompt_phases.preparation = Some(mark.elapsed());
                        }
                        let Some(output) = self
                            .model
                            .advance_prefill_measured(
                                &mut self.device,
                                &mut self.workspace,
                                Some(&mut prompt_phases.advance),
                            )
                            .await?
                        else {
                            return Ok(None);
                        };
                        let mark = Instant::now();
                        let logits = if sequences[0].inputs.last().unwrap().logits {
                            read_logits(&mut self.device, output).await?
                        } else {
                            Vec::new()
                        };
                        prompt_phases.read = mark.elapsed();
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
                        ForwardOutput::Batch16(output, live) => {
                            if live != selected.len() {
                                return Err("B16 output count does not match its requests".into());
                            }
                            let bans: Vec<_> = selected.iter().zip(&chunks)
                                .map(|(&index, (inputs, _))| {
                                    (inputs.last().unwrap().logits
                                        && sessions[index].generation.sampling_config().is_greedy())
                                        .then(|| sessions[index].generation.banned_tokens())
                                }).collect();
                            let tokens = self.greedy.select_b16(&mut self.device, output, &bans).await?;
                            let mut result = Vec::with_capacity(live);
                            for row in 0..live {
                                result.push(if !chunks[row].0.last().unwrap().logits {
                                    SampleOutput::Logits(Vec::new())
                                } else if let Some(token) = tokens[row] {
                                    SampleOutput::Greedy(token)
                                } else {
                                    SampleOutput::Logits(self.greedy.read_fallback_b16(
                                        &mut self.device, output, row).await?)
                                });
                            }
                            result
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
                if decode_width == 0 {
                    trace.prompt_phases(cost_width, &prompt_phases);
                    for session in &mut sessions {
                        if session.timing.progress.first.is_none()
                            && scheduled
                                .iter()
                                .any(|input| input.ring == session.blocks.sliding)
                        {
                            for (sum, elapsed) in
                                session.timing.phases.iter_mut().zip(prompt_phases.times())
                            {
                                *sum += elapsed.unwrap_or_default();
                            }
                        }
                    }
                }
                account_prefill_wait(
                    &mut sessions,
                    &scheduled,
                    decode_width,
                    work_started.elapsed(),
                    credited,
                );
            }
            if startup_cohort::ENABLED && matches!(&result, Ok(Ok(Some(_)))) {
                cohort.maybe_start(&mut sessions, &scheduled, Instant::now());
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
                    if let Some(snapshot) = &mut active_chunk_trace { snapshot.observe(elapsed); }
                    cadence.observe_prompt(prompt_phases.recurring_cost(elapsed), cost_width);
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
                    // Deliver only already-computed buffered events before the terminal
                    // worker error. Completed held members cannot receive a later Error.
                    cohort.release(
                        &mut sessions,
                        &mut reservations,
                        &mut free,
                        &mut rings,
                        Instant::now(),
                        "error",
                    );
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
            let completed_chunk = if decode_width == 0 {
                active_chunk_trace.take()
            } else {
                immediate_chunk_trace.take()
            };
            if let Some(mut snapshot) = completed_chunk {
                snapshot.observe(elapsed);
                snapshot.report(finished);
            }
            if SCHEDULER_TRACE_ENABLED {
                trace.work(decode_width, &scheduled, elapsed, credited);
                if sessions.is_empty() && pending.is_empty() {
                    trace.report();
                }
            }
            if decode_width > 0 {
                cadence.observe_decode(finished, decode_width, elapsed);
            } else {
                cadence.observe_prompt(prompt_phases.recurring_cost(elapsed), cost_width);
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
        if session.delivery.terminal()
            || (session.delivery.held() && !session.generation.is_prefilling())
            || input.count == 0
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
        let Session {
            generation,
            delivery,
            events,
            stream,
            timing,
            ..
        } = session;
        let emit = |delta: Delta<'_>| delivery.delta(delta, *stream, events, &mut timing.progress);
        let result = match &outputs[row] {
            SampleOutput::Logits(logits) => generation.advance_inputs(count, logits, emit),
            SampleOutput::Greedy(token) => generation.advance_inputs_greedy(count, *token, emit),
        };
        timing.progress.tokens = generation.completion_tokens();
        if !generation.is_prefilling() {
            delivery.ready(Instant::now());
        }
        let event = match result {
            Ok(None) => return true,
            Ok(Some(output)) => StreamEvent::Done(output),
            Err(error) => StreamEvent::Error(error.to_string()),
        };
        if delivery.finish(event, *stream, events, &mut timing.progress) {
            return true;
        }
        if SCHEDULER_TRACE_ENABLED {
            timing.report(Instant::now());
        }
        release_session(session, reservations, free, rings);
        false
    });
    Ok(true)
}

fn release_session(
    session: &mut Session,
    reservations: &mut Option<PageReservations>,
    free: &mut Vec<usize>,
    rings: &mut Vec<usize>,
) {
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
}

fn ready_input(session: &Session, allowed: bool) -> ReadyInput {
    if !allowed {
        return ReadyInput {
            ring: session.blocks.sliding,
            position: 0,
            count: 0,
            prompt512: false,
            prompt384: false,
            prompt256: false,
        };
    }
    ReadyInput {
        ring: session.blocks.sliding,
        position: session.generation.input().1,
        count: session.generation.input_count(PREFILL_CHUNK),
        prompt512: session.generation.input_count(crate::axes::P512::SIZE)
            == crate::axes::P512::SIZE,
        prompt384: session.generation.input_count(3 * crate::axes::P128::SIZE)
            == 3 * crate::axes::P128::SIZE,
        prompt256: session.generation.input_count(2 * crate::axes::P128::SIZE)
            == 2 * crate::axes::P128::SIZE,
    }
}

/// Choose a full512-row pair while preserving both owners' cyclic order.
/// Every advertised range comes from Generation::input_count, never padding.
fn prompt_pair_counts(first: ReadyInput, second: ReadyInput) -> Option<[usize; 2]> {
    if !PROMPT_PAIR512_ENABLED || [first, second].iter().any(|r|
        !r.groupable() || r.position % crate::axes::P128::SIZE != 0) {
        return None;
    }
    let quarters = |r: ReadyInput| if r.prompt384 {3} else if r.prompt256 {2} else {1};
    let a = quarters(first);
    let b = quarters(second);
    if a + b < 4 { return None; }
    Some([a * crate::axes::P128::SIZE, (4 - a) * crate::axes::P128::SIZE])
}

/// Padded execution is eligible only for two complete, aligned real quarters.
/// Selection checks full512 groups/pairs first and never skips an older short row.
fn tail256_pair_supported(first: ReadyInput, second: ReadyInput) -> bool {
    TAIL256_ENABLED && !first.prompt512 && [first, second].iter().all(|r|
        r.groupable() && r.position % crate::axes::P128::SIZE == 0)
}

fn is_tail256(inputs: &[ScheduledInput]) -> bool {
    matches!(inputs, [a] if a.count == 2 * crate::axes::P128::SIZE)
        || matches!(inputs, [a, b] if a.count == crate::axes::P128::SIZE
            && b.count == crate::axes::P128::SIZE)
}

/// The extra quarter belongs to the earliest eligible actual owner. No session
/// is skipped or repeated in the public scheduling list.
fn three_owner512_counts(rows: [ReadyInput; 3]) -> Option<[usize; 3]> {
    if !THREE_OWNER512_ENABLED || rows.iter().any(|r| !r.aligned_quarter()) {
        return None;
    }
    let double=rows.iter().position(|r| r.prompt256)?;
    let mut counts=[crate::axes::P128::SIZE;3];
    counts[double]*=2;
    Some(counts)
}

fn tail384_pair_counts(first: ReadyInput, second: ReadyInput) -> Option<[usize; 2]> {
    if !TAIL384_ENABLED || first.prompt512
        || [first,second].iter().any(|r| !r.aligned_quarter()) {
        return None;
    }
    if first.prompt256 {Some([256,128])}
    else if second.prompt256 {Some([128,256])}
    else {None}
}

fn tail384_triple_supported(rows: [ReadyInput; 3]) -> bool {
    TAIL384_ENABLED && !rows[0].prompt512 && rows.iter().all(|r| r.aligned_quarter())
}

fn is_three_owner512(inputs: &[ScheduledInput]) -> bool {
    inputs.len()==3 && inputs.iter().all(|r|matches!(r.count,128|256))
        && inputs.iter().map(|r|r.count).sum::<usize>()==512
}

fn is_tail384(inputs: &[ScheduledInput]) -> bool {
    (1..=3).contains(&inputs.len()) && inputs.iter().all(|r|matches!(r.count,128|256|384))
        && inputs.iter().map(|r|r.count).sum::<usize>()==384
}

/// Four independent requests keep priority. A validated two-owner selection
/// keeps both owner identities and fills512 rows; a singleton may use512. Active ranges are
/// copied exactly until the corresponding model checkpoint completes.
fn schedule_inputs(
    available: &[ReadyInput],
    selected: &[usize],
    active: &[ScheduledInput],
    prompt512_enabled: bool,
) -> Vec<ScheduledInput> {
    let pair = if active.is_empty() && selected.len() == 2 {
        prompt_pair_counts(available[selected[0]], available[selected[1]])
            .or_else(||tail384_pair_counts(available[selected[0]],available[selected[1]]))
    } else { None };
    let triple = if active.is_empty() && selected.len()==3 {
        three_owner512_counts([available[selected[0]],available[selected[1]],available[selected[2]]])
    } else {None};
    selected
        .iter()
        .enumerate()
        .map(|(row, &index)| {
            let ready = available[index];
            if let Some(input) = active.iter().find(|input| input.ring == ready.ring) {
                return *input;
            }
            ScheduledInput {
                ring: ready.ring,
                position: ready.position,
                count: if let Some(counts) = pair {
                    counts[row]
                } else if let Some(counts) = triple {
                    counts[row]
                } else if prompt512_enabled
                    && active.is_empty()
                    && selected.len() == 1
                    && ready.prompt512
                {
                    crate::axes::P512::SIZE
                } else if TAIL384_ENABLED && prompt512_enabled
                    && active.is_empty() && selected.len()==1
                    && ready.prompt384 && ready.aligned_quarter()
                {
                    3 * crate::axes::P128::SIZE
                } else if TAIL256_ENABLED && prompt512_enabled
                    && active.is_empty() && selected.len() == 1
                    && ready.prompt256 && ready.groupable()
                    && ready.position % crate::axes::P128::SIZE == 0
                {
                    2 * crate::axes::P128::SIZE
                } else {
                    ready.count
                },
            }
        })
        .collect()
}

/// Charge actual dense work: both padded tails execute four tensor quarters,
/// while only real owner positions attend and commit cache entries.
fn prompt_cost_width(inputs: &[ScheduledInput]) -> usize {
    if is_tail256(inputs) || is_tail384(inputs)
        || inputs.iter().map(|input| input.count).sum::<usize>() >= crate::axes::P512::SIZE {
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
    let pair = !group && group_enabled && pending.len() >= 2
        && prompt_pair_counts(available[pending[0]], available[pending[1]]).is_some();
    let first_three = (group_enabled && pending.len()>=3).then(||
        [available[pending[0]],available[pending[1]],available[pending[2]]]);
    let three_owner512 = !group && !pair
        && first_three.is_some_and(|rows|three_owner512_counts(rows).is_some());
    let tail_triple = !group && !pair && !three_owner512
        && first_three.is_some_and(tail384_triple_supported);
    let tail_pair = !group && !pair && !three_owner512 && !tail_triple
        && group_enabled && pending.len() >= 2
        && (tail384_pair_counts(available[pending[0]],available[pending[1]]).is_some()
            || tail256_pair_supported(available[pending[0]], available[pending[1]]));
    pending.truncate(if group {4} else if pair {2} else if three_owner512 || tail_triple {3}
        else if tail_pair {2} else {1});
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
            prompt384: false,
            prompt256: false,
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
            prompt384: false,
            prompt256: false,
            })
            .collect()
    }

    fn packed_ready(counts:&[usize])->Vec<ReadyInput> {
        counts.iter().enumerate().map(|(ring,&count)|ReadyInput{ring,position:0,
            count:count.min(128),prompt512:count>=512,prompt384:count>=384,prompt256:count>=256}).collect()
    }
    fn planned(counts:&[usize])->Vec<ScheduledInput> {
        let work=packed_ready(counts);let selected=pending_prompt(64,&work,0,&[],true);
        schedule_inputs(&work,&selected,&[],true)
    }

    #[test]
    fn real512_quad_pair_triple_single_all_precede_padded_routes() {
        for (input,expected) in [
            (vec![256,128,128,128],vec![128;4]),
            (vec![384,128,128],vec![384,128]),
            (vec![128,384,128],vec![128,384]),
            (vec![256,256,128],vec![256,256]),
            (vec![256,128,128],vec![256,128,128]),
            (vec![128,256,128],vec![128,256,128]),
            (vec![128,128,256],vec![128,128,256]),
            (vec![512],vec![512]),
        ] {
            let plan=planned(&input);
            assert_eq!(plan.iter().map(|r|r.count).collect::<Vec<_>>(),expected);
            assert_eq!(prompt_cost_width(&plan),4);
            assert!(!is_tail384(&plan));assert!(!is_tail256(&plan));
            assert_eq!(is_three_owner512(&plan),plan.len()==3);
        }
        let plan=planned(&[16,512,512,512]);
        assert_eq!(plan,vec![ScheduledInput{ring:0,position:0,count:16}]);
    }

    #[test]
    fn all_tail384_shapes_keep_real_owner_counts_and_dense_cost_four() {
        for counts in [vec![384],vec![256,128],vec![128,256],vec![128,128,128]] {
            let plan=planned(&counts);
            assert_eq!(plan.iter().map(|r|r.count).collect::<Vec<_>>(),counts);
            assert!(is_tail384(&plan));assert!(!is_three_owner512(&plan));
            assert!(!is_tail256(&plan));assert_eq!(prompt_cost_width(&plan),4);
        }
        for counts in [vec![256],vec![128,128]] {
            let plan=planned(&counts);assert!(is_tail256(&plan));assert_eq!(prompt_cost_width(&plan),4);
        }
        assert_eq!(prompt_cost_width(&planned(&[128])),1);
        for counts in [vec![384],vec![256,128],vec![128,128,256]] {
            let work=packed_ready(&counts);let selected=pending_prompt(64,&work,0,&[],false);
            assert_eq!(selected,vec![0]);
            assert_eq!(schedule_inputs(&work,&selected,&[],false)[0].count,128);
        }
    }

    #[test]
    fn larger_packing_keeps_cyclic_order_and_never_skips_a_short_owner() {
        let mut work=packed_ready(&[128,128,128,16]);
        for (r,ring) in work.iter_mut().zip([7,2,31,63]){r.ring=ring;}
        assert_eq!(pending_prompt(64,&work,3,&[],true),vec![0,2]);
        assert_eq!(pending_prompt(64,&work,60,&[],true),vec![3]);
        work[0].position=1;
        assert_eq!(pending_prompt(64,&work,3,&[],true),vec![0]);
        work[0].position=448;
        assert_eq!(pending_prompt(64,&work,3,&[],true),vec![0]);
        work[0].position=0;work[3].count=128;
        let mut next=3;let mut prefer=true;
        assert_eq!(select_work_in(64,&work,&mut prefer,&mut next,&[],true),vec![0,2,3,1]);
        assert_eq!(next,3);
    }

    #[test]
    fn three_owner_checkpoint_survives_decoder_compaction_and_cancelled_owner() {
        let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
        let (decoder,_decoder_rx)=session(pool.as_mut().unwrap(),1,1);
        let (mut a,a_rx)=session(pool.as_mut().unwrap(),2048,4);
        let (mut b,_b_rx)=session(pool.as_mut().unwrap(),2048,4);
        let (mut c,_c_rx)=session(pool.as_mut().unwrap(),2048,4);
        advance_prompt_to(&mut a,1792);advance_prompt_to(&mut b,1920);advance_prompt_to(&mut c,1920);
        let mut sessions=vec![decoder,a,b,c];let mut free=vec![];let mut rings=vec![];
        let available=ready_sessions(&sessions);let selected=pending_prompt(64,&available,0,&[],true);
        assert_eq!(selected,vec![1,2,3]);let active=schedule_inputs(&available,&selected,&[],true);
        assert_eq!(active.iter().map(|r|r.count).collect::<Vec<_>>(),vec![256,128,128]);
        let active_rings:Vec<_>=active.iter().map(|r|r.ring).collect();
        let mut prefer=false;let mut next=4;
        let selected=select_work_in(64,&available,&mut prefer,&mut next,&active_rings,true);
        assert_eq!(selected,vec![0]);
        complete_selected(&mut sessions,&selected,Some(&[SampleOutput::Greedy(3)]),&mut pool,&mut free,&mut rings).unwrap();
        let available=ready_sessions(&sessions);
        let resumed=select_work_in(64,&available,&mut prefer,&mut next,&active_rings,true);
        assert_eq!(resumed,vec![0,1,2]);assert_eq!(next,4);
        assert_eq!(schedule_inputs(&available,&resumed,&active,false),active);
        let pages=pool.as_ref().unwrap().available_pages();drop(a_rx);
        assert!(!complete_work(&mut sessions,&active,None,&mut pool,&mut free,&mut rings).unwrap());
        assert!(complete_work(&mut sessions,&active,Some(&[SampleOutput::Greedy(3),SampleOutput::Greedy(4)]),&mut pool,&mut free,&mut rings).is_err());
        assert_eq!(sessions.iter().map(|s|s.generation.input().1).collect::<Vec<_>>(),vec![1792,1920,1920]);
        assert_eq!(pool.as_ref().unwrap().available_pages(),pages);
        assert!(complete_work(&mut sessions,&active,Some(&[SampleOutput::Greedy(3),SampleOutput::Greedy(4),SampleOutput::Greedy(5)]),&mut pool,&mut free,&mut rings).unwrap());
        assert_eq!(sessions.iter().map(|s|s.blocks.sliding).collect::<Vec<_>>(),vec![active[1].ring,active[2].ring]);
        assert_eq!(sessions.iter().map(|s|s.generation.input().1).collect::<Vec<_>>(),vec![2048,2048]);
        assert_eq!(sessions.iter().map(|s|s.generation.input().0.token).collect::<Vec<_>>(),vec![4,5]);
        assert_eq!(pool.as_ref().unwrap().available_pages(),pages+Workspace::blocks_for(2052));
        assert!(free.is_empty()&&rings.is_empty());
    }

    #[test]
    fn tail384_shapes_complete_final_outputs_without_sampling_intermediate_ranges() {
        for counts in [vec![384],vec![256,128],vec![128,256],vec![128,128,128]] {
            let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
            let mut sessions=Vec::new();let mut receivers=Vec::new();
            for &count in &counts {let (mut s,rx)=session(pool.as_mut().unwrap(),2048,4);advance_prompt_to(&mut s,2048-count);sessions.push(s);receivers.push(rx);}
            let available=ready_sessions(&sessions);let selected=pending_prompt(64,&available,0,&[],true);
            let active=schedule_inputs(&available,&selected,&[],true);
            assert!(is_tail384(&active));assert_eq!(active.iter().map(|r|r.count).collect::<Vec<_>>(),counts);
            let outputs:Vec<_>=(0..counts.len()).map(|owner|SampleOutput::Greedy(3+owner)).collect();
            complete_work(&mut sessions,&active,Some(&outputs),&mut pool,&mut vec![],&mut vec![]).unwrap();
            for (owner,s) in sessions.iter().enumerate(){assert_eq!(s.generation.input().1,2048);assert_eq!(s.generation.input().0.token,3+owner);}
        }
        //384 of a448-token prompt is a nonfinal range, so no logits are sampled.
        let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
        let (a,_rx)=session(pool.as_mut().unwrap(),448,4);let mut sessions=vec![a];
        let available=ready_sessions(&sessions);let active=schedule_inputs(&available,&[0],&[],true);
        assert_eq!(active[0].count,384);
        complete_work(&mut sessions,&active,Some(&[SampleOutput::Logits(vec![])]),&mut pool,&mut vec![],&mut vec![]).unwrap();
        assert_eq!(sessions[0].generation.input().1,384);assert_eq!(sessions[0].generation.completion_tokens(),0);
        assert!(sessions[0].generation.is_prefilling());
    }

    #[test]
    fn exact_route_classifiers_reject_partial_and_ambiguous_shapes() {
        for counts in [vec![384,128,0],vec![256,256],vec![128,128,128,128],vec![255,128,129]] {
            let rows:Vec<_>=counts.iter().enumerate().map(|(ring,&count)|ScheduledInput{ring,position:0,count}).collect();
            assert!(!is_three_owner512(&rows));assert!(!is_tail384(&rows));
        }
        for counts in [vec![256,128,128],vec![128,256,128],vec![128,128,256],vec![384],vec![128,128,128]] {
            let rows:Vec<_>=counts.iter().enumerate().map(|(ring,&count)|ScheduledInput{ring,position:0,count}).collect();
            assert_eq!(prompt_cost_width(&rows),4);
            let start=Instant::now();let mut cadence=DecodeCadence::new();
            cadence.prompt_cost=[Duration::from_millis(2),Duration::from_millis(16)];
            cadence.observe_decode(start,1,Duration::from_millis(70));
            assert!(!cadence.prompt_fits(start+Duration::from_millis(5),1,prompt_cost_width(&rows)));
        }
    }

    #[test]
    fn three_owner_completion_keeps_unfinished_cohort_deadline_and_real_event_credit() {
        let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
        let mut sessions=Vec::new();let mut receivers=Vec::new();let start=Instant::now();
        for count in [256,128,128,16] {
            let (mut s,rx)=session(pool.as_mut().unwrap(),count,4);s.timing.received=start;
            sessions.push(s);receivers.push(rx);
        }
        let available=ready_sessions(&sessions);let selected=pending_prompt(64,&available,0,&[],true);
        let active=schedule_inputs(&available,&selected,&[],true);
        assert!(is_three_owner512(&active));
        let pages=pool.as_ref().unwrap().available_pages();
        let mut cohort=StartupCohort::default();cohort.maybe_start(&mut sessions,&active,start);
        assert!(cohort.active());
        complete_work(&mut sessions,&active,Some(&[SampleOutput::Greedy(3),SampleOutput::Greedy(4),SampleOutput::Greedy(5)]),&mut pool,&mut vec![],&mut vec![]).unwrap();
        assert!(sessions[..3].iter().all(|s|!cohort.allows(s)&&s.timing.progress.first.is_none()));
        assert!(cohort.allows(&sessions[3]));assert_eq!(pool.as_ref().unwrap().available_pages(),pages);
        let deadline=start+startup_cohort::RECEIPT_BUDGET;
        assert_eq!(cohort.release_reason(&sessions,deadline-Duration::from_nanos(1)),None);
        assert_eq!(cohort.release_reason(&sessions,deadline),Some("deadline"));
        cohort.release(&mut sessions,&mut pool,&mut vec![],&mut vec![],deadline,"deadline");
        for (owner,text) in ["a","b","c"].into_iter().enumerate(){
            assert!(matches!(receivers[owner].try_recv(),Ok(StreamEvent::Delta(t)) if t==text));
            assert_eq!(sessions[owner].timing.progress.first,Some(deadline));
        }
        assert!(sessions[3].generation.is_prefilling());
        assert_eq!(sessions[3].generation.input().1,0);
    }

    #[test]
    fn tail256_preserves_real512_priority_cyclic_order_and_disabled_fallback() {
        let mut work = ready_inputs(&[(7,128),(2,128),(31,128),(63,128)]);
        for r in &mut work { r.prompt256 = true; }
        let quad = pending_prompt(64,&work,3,&[],true);
        assert_eq!(quad,vec![0,2,3,1]);
        let scheduled = schedule_inputs(&work,&quad,&[],true);
        assert_eq!(scheduled.iter().map(|r|r.count).collect::<Vec<_>>(),vec![128;4]);
        assert!(!is_tail256(&scheduled));
        let pair = pending_prompt(64,&work[..2],3,&[],true);
        assert_eq!(pair,vec![0,1]);
        assert_eq!(schedule_inputs(&work,&pair,&[],true).iter().map(|r|r.count).collect::<Vec<_>>(),vec![256,256]);
        // A single request can still supply all512 rows before any padded tail.
        work[0].prompt512 = true;
        let single = pending_prompt(64,&work[..1],3,&[],true);
        assert_eq!(schedule_inputs(&work,&single,&[],true)[0].count,512);
        work[0].prompt512 = false;
        assert_eq!(schedule_inputs(&work,&single,&[],true)[0].count,256);
        assert_eq!(schedule_inputs(&work,&single,&[],false)[0].count,128);
        work[0].prompt256 = false; work[1].prompt256 = false;
        let tail = pending_prompt(64,&work[..2],3,&[],true);
        assert_eq!(tail,vec![0,1]);
        assert!(is_tail256(&schedule_inputs(&work,&tail,&[],true)));
        assert_eq!(pending_prompt(64,&work[..2],3,&[],false),vec![0]);
    }

    #[test]
    fn tail256_does_not_bypass_short_or_unaligned_oldest_prompt() {
        let mut work = ready_inputs(&[(7,128),(2,128),(31,128)]);
        work[0].position = 1920; work[1].position = 1920;
        assert_eq!(pending_prompt(64,&work,3,&[],true),vec![0,2,1]);
        work[0].count = 16;
        assert_eq!(pending_prompt(64,&work,3,&[],true),vec![0]);
        work[0].count = 128; work[0].position = 1; work[0].prompt256 = true;
        let single = pending_prompt(64,&work,3,&[],true);
        assert_eq!(single,vec![0]);
        assert_eq!(schedule_inputs(&work,&single,&[],true)[0].count,128);
        work[0].position = 448;
        assert_eq!(pending_prompt(64,&work,3,&[],true),vec![0]);
    }

    #[test]
    fn tail256_cost_reserves_padded_dense_work_and_keeps_decoders_responsive() {
        let one = [ScheduledInput{ring:7,position:1792,count:256}];
        let pair = [ScheduledInput{ring:7,position:1920,count:128},ScheduledInput{ring:2,position:1920,count:128}];
        assert_eq!(prompt_cost_width(&one),4);
        assert_eq!(prompt_cost_width(&pair),4);
        let start = Instant::now(); let mut cadence = DecodeCadence::new();
        cadence.prompt_cost = [Duration::from_millis(2),Duration::from_millis(16)];
        cadence.observe_decode(start,1,Duration::from_millis(70));
        let now = start + Duration::from_millis(5);
        assert!(cadence.prompt_fits(now,1,1));
        assert!(!cadence.prompt_fits(now,1,prompt_cost_width(&one)));
        let work = ready_inputs(&[(7,128),(9,1),(2,128)]);
        let mut prefer = false; let mut next = 3;
        assert_eq!(select_work_in(64,&work,&mut prefer,&mut next,&[7,2],true),vec![1]);
        assert_eq!(select_work_in(64,&work,&mut prefer,&mut next,&[7,2],true),vec![0,2]);
        assert_eq!(next,3,"resuming a layer cannot rotate the prompt queue");
        cadence.observe_prompt(Duration::from_millis(3),prompt_cost_width(&pair));
        assert_eq!(cadence.prompt_cost[0],Duration::from_millis(2));
        assert_eq!(cadence.prompt_cost[1],Duration::from_millis(16).mul_f64(0.9));
    }

    #[test]
    fn single_tail256_retains_range_after_decoder_compaction_and_cancel() {
        let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
        let (decoder,_rx)=session(pool.as_mut().unwrap(),1,1);
        let (mut prompt,prompt_rx)=session(pool.as_mut().unwrap(),2048,4);
        advance_prompt_to(&mut prompt,1792);
        let mut sessions=vec![decoder,prompt]; let mut free=vec![];let mut rings=vec![];
        let available=ready_sessions(&sessions);
        let selected=pending_prompt(64,&available,0,&[],true);
        let active=schedule_inputs(&available,&selected,&[],true);
        assert_eq!(active.len(),1);assert_eq!((active[0].position,active[0].count),(1792,256));
        let active_rings=[active[0].ring];let mut prefer=false;let mut next=2;
        let decode=select_work_in(64,&available,&mut prefer,&mut next,&active_rings,true);
        assert_eq!(decode,vec![0]);
        complete_selected(&mut sessions,&decode,Some(&[SampleOutput::Greedy(3)]),&mut pool,&mut free,&mut rings).unwrap();
        assert_eq!(sessions.len(),1);
        let available=ready_sessions(&sessions);
        let resumed=select_work_in(64,&available,&mut prefer,&mut next,&active_rings,true);
        assert_eq!(schedule_inputs(&available,&resumed,&active,false),active);
        let pages=pool.as_ref().unwrap().available_pages();drop(prompt_rx);
        assert!(!complete_work(&mut sessions,&active,None,&mut pool,&mut free,&mut rings).unwrap());
        assert!(complete_work(&mut sessions,&active,Some(&[]),&mut pool,&mut free,&mut rings).is_err());
        assert_eq!(sessions[0].generation.input().1,1792);
        assert_eq!(pool.as_ref().unwrap().available_pages(),pages);
        assert!(complete_work(&mut sessions,&active,Some(&[SampleOutput::Greedy(4)]),&mut pool,&mut free,&mut rings).unwrap());
        assert!(sessions.is_empty());assert_eq!(pool.as_ref().unwrap().available_pages(),256);
        assert!(free.is_empty()&&rings.is_empty());
    }

    #[test]
    fn paired_tail256_atomic_outputs_advance_only_actual_owners() {
        let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
        let (mut a,a_rx)=session(pool.as_mut().unwrap(),2048,4);
        let (mut b,_b_rx)=session(pool.as_mut().unwrap(),2048,4);
        advance_prompt_to(&mut a,1920);advance_prompt_to(&mut b,1920);
        let mut sessions=vec![a,b];let available=ready_sessions(&sessions);
        let selected=pending_prompt(64,&available,0,&[],true);
        let active=schedule_inputs(&available,&selected,&[],true);
        assert_eq!(active.iter().map(|r|r.count).collect::<Vec<_>>(),vec![128,128]);
        assert!(is_tail256(&active));let pages=pool.as_ref().unwrap().available_pages();drop(a_rx);
        assert!(!complete_work(&mut sessions,&active,None,&mut pool,&mut vec![],&mut vec![]).unwrap());
        assert!(complete_work(&mut sessions,&active,Some(&[SampleOutput::Greedy(3)]),&mut pool,&mut vec![],&mut vec![]).is_err());
        assert_eq!(sessions.iter().map(|s|s.generation.input().1).collect::<Vec<_>>(),vec![1920,1920]);
        assert_eq!(pool.as_ref().unwrap().available_pages(),pages);
        assert!(complete_work(&mut sessions,&active,Some(&[SampleOutput::Greedy(3),SampleOutput::Greedy(4)]),&mut pool,&mut vec![],&mut vec![]).unwrap());
        assert_eq!(sessions.len(),1);assert_eq!(sessions[0].blocks.sliding,active[1].ring);
        assert_eq!(sessions[0].generation.input().1,2048);assert_eq!(sessions[0].generation.input().0.token,4);
        assert_eq!(pool.as_ref().unwrap().available_pages(),pages+Workspace::blocks_for(2052));
    }

    #[test]
    fn tail256_intermediate_ranges_do_not_sample_nonfinal_owner() {
        let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
        let (a,_a_rx)=session(pool.as_mut().unwrap(),384,4);
        let mut sessions=vec![a];let available=ready_sessions(&sessions);
        // An already prepared Tail256 cannot expand when a larger route is enabled.
        let retained=[ScheduledInput{ring:available[0].ring,position:0,count:256}];
        let single=schedule_inputs(&available,&[0],&retained,true);
        assert_eq!(single[0].count,256);
        complete_work(&mut sessions,&single,Some(&[SampleOutput::Logits(vec![])]),&mut pool,&mut vec![],&mut vec![]).unwrap();
        assert!(sessions[0].generation.is_prefilling());
        assert_eq!(sessions[0].generation.input().1,256);
        assert_eq!(sessions[0].generation.completion_tokens(),0);
        let (b,_b_rx)=session(pool.as_mut().unwrap(),256,4);
        sessions.push(b);let available=ready_sessions(&sessions);
        let selected=pending_prompt(64,&available,0,&[],true);
        let retained:Vec<_>=selected.iter().map(|&i|ScheduledInput{ring:available[i].ring,
            position:available[i].position,count:128}).collect();
        let pair=schedule_inputs(&available,&selected,&retained,true);
        assert!(is_tail256(&pair));
        complete_work(&mut sessions,&pair,Some(&[SampleOutput::Greedy(3),SampleOutput::Logits(vec![])]),&mut pool,&mut vec![],&mut vec![]).unwrap();
        assert!(!sessions[0].generation.is_prefilling());
        assert_eq!(sessions[0].generation.input().1,384);
        assert_eq!(sessions[0].generation.input().0.token,3);
        assert!(sessions[1].generation.is_prefilling());
        assert_eq!(sessions[1].generation.input().1,128);
        assert_eq!(sessions[1].generation.completion_tokens(),0);
    }

    #[test]
    fn tail256_held_output_never_extends_cohort_receipt_deadline() {
        let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
        let (a,a_rx)=session(pool.as_mut().unwrap(),256,4);
        let (b,_b_rx)=session(pool.as_mut().unwrap(),128,4);
        let mut sessions=vec![a,b];let start=Instant::now();
        for s in &mut sessions{s.timing.received=start;}
        let active=[ScheduledInput{ring:sessions[0].blocks.sliding,position:0,count:256}];
        let mut cohort=StartupCohort::default();cohort.maybe_start(&mut sessions,&active,start);
        assert!(cohort.active());
        assert!(!complete_work(&mut sessions,&active,None,&mut pool,&mut vec![],&mut vec![]).unwrap());
        assert!(complete_work(&mut sessions,&active,Some(&[SampleOutput::Greedy(3)]),&mut pool,&mut vec![],&mut vec![]).unwrap());
        assert!(!cohort.allows(&sessions[0]));assert!(cohort.allows(&sessions[1]));
        assert!(sessions[0].timing.progress.first.is_none());
        let deadline=start+startup_cohort::RECEIPT_BUDGET;
        assert_eq!(cohort.release_reason(&sessions,deadline-Duration::from_nanos(1)),None);
        assert_eq!(cohort.release_reason(&sessions,deadline),Some("deadline"));
        cohort.release(&mut sessions,&mut pool,&mut vec![],&mut vec![],deadline,"deadline");
        assert!(cohort.allows(&sessions[0]));assert_eq!(sessions[0].generation.input().1,256);
        assert_eq!(sessions[0].timing.progress.first,Some(deadline));
        assert!(matches!(a_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));
    }
    #[test]
    fn two_owner384128_keeps_quad_priority_single512_fallback_and_short_tail_turn() {
        let mut work=ready_inputs(&[(7,128),(2,128),(31,128),(63,128)]);
        work[0].position=1664;work[0].prompt384=true;work[0].prompt512=true;
        work[1].position=1920;
        let quad=pending_prompt(64,&work,3,&[],true);
        assert_eq!(quad.len(),4);
        assert!(schedule_inputs(&work,&quad,&[],true).iter().all(|r|r.count==128));
        let selected=pending_prompt(64,&work[..2],3,&[],true);
        assert_eq!(selected,vec![0,1]);
        let scheduled=schedule_inputs(&work[..2],&selected,&[],true);
        assert_eq!(scheduled,vec![ScheduledInput{ring:7,position:1664,count:384},ScheduledInput{ring:2,position:1920,count:128}]);
        assert_eq!(prompt_cost_width(&scheduled),4);
        let single=pending_prompt(64,&work[..2],3,&[],false);
        assert_eq!(single,vec![0]);
        assert_eq!(schedule_inputs(&work[..2],&single,&[],true)[0].count,512);
        work[0].count=16;work[0].prompt384=false;work[0].prompt512=false;
        assert_eq!(pending_prompt(64,&work[..2],3,&[],true),vec![0]);
        work[0].count=128;work[0].prompt384=true;work[0].position=1665;
        assert_eq!(pending_prompt(64,&work[..2],3,&[],true),vec![0]);
    }

    #[test]
    fn pair384128_retains_both_ranges_through_decoder_removal_and_cancelled_owner() {
        let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
        let (decoder,_decoder_rx)=session(pool.as_mut().unwrap(),1,4);
        let (mut a,a_rx)=session(pool.as_mut().unwrap(),2048,4);
        let (mut b,_b_rx)=session(pool.as_mut().unwrap(),2048,4);
        advance_prompt_to(&mut a,1664);advance_prompt_to(&mut b,1920);
        let mut sessions=vec![decoder,a,b];let mut free=vec![];let mut rings=vec![];
        let work=ready_sessions(&sessions);
        let selected=pending_prompt(64,&work,0,&[],true);
        assert_eq!(selected,vec![1,2]);
        let active=schedule_inputs(&work,&selected,&[],true);
        assert_eq!(active.iter().map(|s|s.count).collect::<Vec<_>>(),vec![384,128]);
        let active_rings:Vec<_>=active.iter().map(|s|s.ring).collect();
        let mut prefer=false;let mut next=3;
        let decoder=select_work_in(64,&work,&mut prefer,&mut next,&active_rings,true);
        assert_eq!(decoder,vec![0]);
        complete_selected(&mut sessions,&decoder,Some(&[SampleOutput::Greedy(1)]),&mut pool,&mut free,&mut rings).unwrap();
        assert_eq!(sessions.len(),2);
        let work=ready_sessions(&sessions);
        let resumed=select_work_in(64,&work,&mut prefer,&mut next,&active_rings,true);
        assert_eq!(resumed,vec![0,1]);assert_eq!(next,3);
        assert_eq!(schedule_inputs(&work,&resumed,&active,true),active);
        let available=pool.as_ref().unwrap().available_pages();
        drop(a_rx);
        assert!(!complete_work(&mut sessions,&active,None,&mut pool,&mut free,&mut rings).unwrap());
        assert!(complete_work(&mut sessions,&active,Some(&[SampleOutput::Greedy(3)]),&mut pool,&mut free,&mut rings).is_err());
        assert_eq!(sessions.iter().map(|s|s.generation.input().1).collect::<Vec<_>>(),vec![1664,1920]);
        assert_eq!(pool.as_ref().unwrap().available_pages(),available);
        let outputs=[SampleOutput::Greedy(3),SampleOutput::Greedy(4)];
        assert!(complete_work(&mut sessions,&active,Some(&outputs),&mut pool,&mut free,&mut rings).unwrap());
        assert_eq!(sessions.len(),1);assert_eq!(sessions[0].blocks.sliding,active[1].ring);
        assert_eq!(sessions[0].generation.input().1,2048);
        assert_eq!(sessions[0].generation.input().0.token,4);
        assert_eq!(pool.as_ref().unwrap().available_pages(),available+Workspace::blocks_for(2052));
        assert!(free.is_empty()&&rings.is_empty());
    }

    #[test]
    fn pair384128_advances_nonfinal_a_without_sampling_it() {
        let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
        let (a,_a_rx)=session(pool.as_mut().unwrap(),512,4);
        let (b,_b_rx)=session(pool.as_mut().unwrap(),128,4);
        let mut sessions=vec![a,b];let work=ready_sessions(&sessions);
        let selected=pending_prompt(64,&work,0,&[],true);
        assert_eq!(selected,vec![0,1]);let scheduled=schedule_inputs(&work,&selected,&[],true);
        assert_eq!(scheduled.iter().map(|s|s.count).collect::<Vec<_>>(),vec![384,128]);
        let outputs=[SampleOutput::Logits(vec![]),SampleOutput::Greedy(4)];
        assert!(complete_work(&mut sessions,&scheduled,Some(&outputs),&mut pool,&mut vec![],&mut vec![]).unwrap());
        assert!(sessions[0].generation.is_prefilling());
        assert_eq!(sessions[0].generation.input().1,384);
        assert_eq!(sessions[0].generation.completion_tokens(),0);
        assert!(!sessions[1].generation.is_prefilling());
        assert_eq!(sessions[1].generation.input().1,128);
        assert_eq!(sessions[1].generation.completion_tokens(),1);
    }

    #[test]
    fn reversed_and_balanced_pairs_keep_oldest_owner_ranges_until_atomic_completion() {
        for counts in [[128,384],[256,256]] {
            let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
            let (mut a,a_rx)=session(pool.as_mut().unwrap(),2048,4);
            let (mut b,_b_rx)=session(pool.as_mut().unwrap(),2048,4);
            advance_prompt_to(&mut a,2048-counts[0]);advance_prompt_to(&mut b,2048-counts[1]);
            let mut sessions=vec![a,b];let ready=ready_sessions(&sessions);
            let selected=pending_prompt(64,&ready,0,&[],true);
            assert_eq!(selected,vec![0,1]);
            let work=schedule_inputs(&ready,&selected,&[],true);
            assert_eq!(work.iter().map(|w|w.count).collect::<Vec<_>>(),counts);
            assert_eq!(work.iter().map(|w|w.ring).collect::<Vec<_>>(),sessions.iter().map(|s|s.blocks.sliding).collect::<Vec<_>>());
            let available=pool.as_ref().unwrap().available_pages();drop(a_rx);
            assert!(!complete_work(&mut sessions,&work,None,&mut pool,&mut vec![],&mut vec![]).unwrap());
            assert!(complete_work(&mut sessions,&work,Some(&[SampleOutput::Greedy(3)]),&mut pool,&mut vec![],&mut vec![]).is_err());
            assert_eq!(pool.as_ref().unwrap().available_pages(),available);
            assert_eq!(sessions.iter().map(|s|s.generation.input().1).collect::<Vec<_>>(),vec![2048-counts[0],2048-counts[1]]);
            assert!(complete_work(&mut sessions,&work,Some(&[SampleOutput::Greedy(3),SampleOutput::Greedy(4)]),&mut pool,&mut vec![],&mut vec![]).unwrap());
            assert_eq!(sessions.len(),1);assert_eq!(sessions[0].blocks.sliding,work[1].ring);
            assert_eq!(sessions[0].generation.input().1,2048);assert_eq!(sessions[0].generation.input().0.token,4);
        }
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
            prompt384: false,
            prompt256: false,
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
            prompt384: false,
            prompt256: false,
            },
        );
        assert_eq!(pending_prompt(64, &work, 62, &[], true), vec![1]);
        assert_eq!(pending_prompt(64, &work, 0, &[], true), vec![3]);
        work.remove(3);
        work[0].position = 449; // Synthetic malformed chunk: the model must not group across a page.
        // The first three complete rows can execute together without including or
        // skipping over the malformed fourth row.
        assert_eq!(pending_prompt(64, &work, 62, &[], true), vec![1, 3, 2]);
        work[0].position = 384;
        assert_eq!(pending_prompt(64, &work, 62, &[], true), vec![1, 3, 2, 0]);
        assert_eq!(pending_prompt(64, &work, 62, &[], false), vec![1]);
        assert_eq!(pending_prompt(64, &work[..3], 62, &[], true), vec![1, 2, 0]);
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
        let id = reservation.blocks.sliding as u64;
        (
            Session {
                delivery: StartupDelivery::default(),
                generation: Generation::new(tokenizer(), request, CONTEXT).unwrap(),
                blocks: reservation.blocks.clone(),
                reservation: Some(reservation),
                events,
                stream: true,
                timing: SessionTiming::new(id, prompt, Instant::now()),
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
                prompt384: s.generation.input_count(3 * crate::axes::P128::SIZE)
                    == 3 * crate::axes::P128::SIZE,
                prompt256: s.generation.input_count(2 * crate::axes::P128::SIZE)
                    == 2 * crate::axes::P128::SIZE,
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
            ..TokenProgress::default()
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
            ..TokenProgress::default()
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
            ..TokenProgress::default()
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

    #[test]
    fn output_trace_records_overdue_average_and_gap_deadlines() {
        let start = Instant::now();
        let mut progress = TokenProgress::default();
        progress.emitted(start);
        assert_eq!(progress.deadline_overruns, 0);
        progress.tokens = 1;
        progress.emitted(start + Duration::from_millis(100));
        assert_eq!(progress.deadline_overruns, 1);
        assert_eq!(progress.max_deadline_overrun, Duration::from_millis(10));
        progress.tokens = 100;
        progress.emitted(start + Duration::from_millis(270));
        assert_eq!(progress.deadline_overruns, 2);
        assert_eq!(progress.max_deadline_overrun, Duration::from_millis(20));
        assert_eq!(progress.max_gap, Duration::from_millis(170));
    }

    #[test]
    fn preparation_and_finalization_do_not_train_the_recurring_layer_estimator() {
        let phases = PromptPhases {
            preparation: Some(Duration::from_millis(250)),
            advance: PrefillAdvanceTiming {
                layer: Some(Duration::from_millis(7)),
                finish: Some(Duration::from_millis(20)),
            },
            read: Duration::from_millis(2),
        };
        let elapsed = Duration::from_millis(280);
        let mut cadence = DecodeCadence::new();
        cadence.prompt_cost[1] = Duration::from_millis(7);
        cadence.observe_prompt(phases.recurring_cost(elapsed), 4);
        assert_eq!(cadence.prompt_cost[1], Duration::from_millis(7));
        assert_eq!(PromptPhases::default().recurring_cost(elapsed), elapsed);
        assert_eq!(
            phases.times(),
            [
                Some(Duration::from_millis(250)),
                Some(Duration::from_millis(7)),
                Some(Duration::from_millis(22)),
            ]
        );
    }

    #[test]
    fn phase_trace_preserves_whole_prompt_work_and_counts_each_phase_once() {
        let mut trace = SchedulerTrace::new();
        let input = [ScheduledInput {
            ring: 0,
            position: 0,
            count: 512,
        }];
        let phases = PromptPhases {
            preparation: Some(Duration::from_millis(250)),
            advance: PrefillAdvanceTiming {
                layer: Some(Duration::from_millis(7)),
                finish: None,
            },
            read: Duration::ZERO,
        };
        trace.prompt_phases(4, &phases);
        trace.work(0, &input, Duration::from_millis(258), true);
        let final_phase = PromptPhases {
            preparation: None,
            advance: PrefillAdvanceTiming {
                layer: Some(Duration::from_millis(7)),
                finish: Some(Duration::from_millis(5)),
            },
            read: Duration::from_millis(1),
        };
        trace.prompt_phases(4, &final_phase);
        trace.work(0, &input, Duration::from_millis(14), false);
        assert_eq!(trace.prompt[1].total, Duration::from_millis(272));
        assert_eq!(trace.phases[0][1].calls, 1);
        assert_eq!(trace.phases[0][1].total, Duration::from_millis(250));
        assert_eq!(trace.phases[1][1].calls, 2);
        assert_eq!(trace.phases[1][1].total, Duration::from_millis(14));
        assert_eq!(trace.phases[2][1].calls, 1);
        assert_eq!(trace.phases[2][1].total, Duration::from_millis(6));
        assert_eq!(trace.prompt_dispatches[1], 2);
        assert_eq!(trace.credited.total, Duration::from_millis(258));
    }

    fn cohort_work(session: &Session) -> ScheduledInput {
        ScheduledInput {
            ring: session.blocks.sliding,
            position: session.generation.input().1,
            count: session.generation.input_count(PREFILL_CHUNK),
        }
    }

    fn cohort_complete(
        sessions: &mut Vec<Session>,
        ring: usize,
        token: usize,
        pool: &mut Option<PageReservations>,
    ) {
        let s = sessions.iter().find(|s| s.blocks.sliding == ring).unwrap();
        let work = [cohort_work(s)];
        assert!(
            complete_work(
                sessions,
                &work,
                Some(&[SampleOutput::Greedy(token)]),
                pool,
                &mut vec![],
                &mut vec![]
            )
            .unwrap()
        );
    }

    #[test]
    fn exact_remaining_prompt_lengths_survive512_tail_decode_and_terminal_phases() {
        let mut pool = PageReservations::new(64, 256, CONTEXT).unwrap();
        let (mut session, _receiver) = session(&mut pool, 1025, 2);
        assert_eq!(session.generation.remaining_prompt_inputs(), Some(1025));
        assert!(session.generation.advance_inputs(512, &[], |_| true).unwrap().is_none());
        assert_eq!(session.generation.remaining_prompt_inputs(), Some(513));
        assert!(session.generation.advance_inputs(512, &[], |_| true).unwrap().is_none());
        assert_eq!(session.generation.remaining_prompt_inputs(), Some(1));
        assert!(session.generation.advance_inputs_greedy(1, 3, |_| true).unwrap().is_none());
        assert_eq!(session.generation.remaining_prompt_inputs(), None);
        assert!(session.generation.advance_inputs_greedy(1, 4, |_| true).unwrap().is_some());
        assert_eq!(session.generation.remaining_prompt_inputs(), None);
    }

    #[test]
    fn physical32_slot31_checkpoint_survives_compaction_and_cancel_before_release() {
        let mut pool = Some(PageReservations::new(32,256,CONTEXT).unwrap());
        let mut sessions = Vec::new();
        let mut receivers = Vec::new();
        for _ in 0..31 {
            let (s,r) = session(pool.as_mut().unwrap(),1,1);
            sessions.push(s); receivers.push(r);
        }
        let (last,last_rx) = session(pool.as_mut().unwrap(),513,1);
        assert_eq!(last.blocks.sliding,31);
        sessions.push(last); drop(last_rx);
        let active = [ScheduledInput{ring:31,position:0,count:512}];
        assert!(pool.as_mut().unwrap().try_reserve(1).unwrap().is_none());
        assert!(!complete_work(&mut sessions,&active,None,&mut pool,&mut vec![],&mut vec![]).unwrap());
        assert_eq!(sessions[31].generation.input().1,0);
        let first = cohort_work(&sessions[0]);
        assert!(complete_work(&mut sessions,&[first],Some(&[SampleOutput::Greedy(3)]),
            &mut pool,&mut vec![],&mut vec![]).unwrap());
        assert_eq!(sessions.len(),31);
        assert_eq!(sessions[30].blocks.sliding,31);
        let ready = ready_sessions(&sessions);
        assert_eq!(schedule_inputs(&ready,&[30],&active,true),active);
        let new = pool.as_mut().unwrap().try_reserve(1).unwrap().unwrap();
        assert_eq!(new.blocks.sliding,0); // Slot31 is still owned despite receiver cancellation.
        pool.as_mut().unwrap().release(new).unwrap();
        assert!(complete_work(&mut sessions,&active,Some(&[SampleOutput::Logits(vec![])]),
            &mut pool,&mut vec![],&mut vec![]).unwrap());
        assert_eq!(sessions[30].generation.input().1,512);
        let finish = [ScheduledInput{ring:31,position:512,count:1}];
        assert!(complete_work(&mut sessions,&finish,Some(&[SampleOutput::Greedy(3)]),
            &mut pool,&mut vec![],&mut vec![]).unwrap());
        assert_eq!(sessions.len(),30);
        let next = pool.as_mut().unwrap().try_reserve(8192).unwrap().unwrap();
        assert_eq!(next.blocks.sliding,31);
        assert_eq!(next.blocks.full.len(),16);
    }

    #[test]
    fn chunk_trace_is_read_only_and_survives_cancelled_receivers_and_session_reordering() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (a, ar) = session(pool.as_mut().unwrap(), 1025, 2);
        let (mut b, _br) = session(pool.as_mut().unwrap(), 1, 3);
        b.generation.advance_inputs_greedy(1, 3, |_| true).unwrap();
        drop(ar); // Diagnostics never probe the channel or release its reservation.
        let mut sessions = vec![a, b];
        let available: Vec<_> = sessions.iter().map(|s| ready_input(s, true)).collect();
        let work = [ScheduledInput { ring: sessions[0].blocks.sliding, position: 0, count: 512 }];
        let now = Instant::now();
        let snapshot = chunk_trace::ChunkTrace::capture(3, &sessions, &work, &available,
            false, 1, 0, now).unwrap();
        let line = snapshot.compact(now);
        assert!(line.contains("owners=1 tokens=512"));
        assert!(line.contains(&format!("rows={}:{}:0:512:1025",
            sessions[0].timing.id, sessions[0].blocks.sliding)));
        assert!(!complete_work(&mut sessions, &work, None, &mut pool,
            &mut vec![], &mut vec![]).unwrap());
        assert_eq!(sessions[0].generation.remaining_prompt_inputs(), Some(1025));
        assert!(sessions[0].reservation.is_some());
        sessions.swap(0, 1);
        assert_eq!(snapshot.compact(now), line);
        let decoder = [ScheduledInput { ring: sessions[0].blocks.sliding,
            position: sessions[0].generation.input().1, count: 1 }];
        let available: Vec<_> = sessions.iter().map(|s| ready_input(s, true)).collect();
        assert!(chunk_trace::ChunkTrace::capture(4, &sessions, &decoder, &available,
            false, 1, 0, now).is_none());
        assert!(complete_work(&mut sessions, &work, Some(&[SampleOutput::Logits(vec![])]),
            &mut pool, &mut vec![], &mut vec![]).unwrap());
        assert_eq!(sessions[1].generation.remaining_prompt_inputs(), Some(513));
        assert_eq!(snapshot.compact(now), line); // Captured remaining is the scheduling-time value.
    }

    #[test]
    fn startup_cohort_holds_actual_first_delta_without_credit_or_extra_decode() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (a, ar) = session(pool.as_mut().unwrap(), 128, 4);
        let (b, _br) = session(pool.as_mut().unwrap(), 128, 4);
        let mut sessions = vec![a, b];
        let first = cohort_work(&sessions[0]);
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[first], Instant::now());
        assert!(cohort.active());
        cohort_complete(&mut sessions, first.ring, 3, &mut pool);
        assert!(matches!(ar.try_recv(), Err(mpsc::TryRecvError::Empty)));
        assert_eq!(sessions[0].generation.completion_tokens(), 1);
        assert_eq!(sessions[0].timing.progress.tokens, 1);
        assert!(sessions[0].timing.progress.first.is_none());
        assert!(
            sessions[0]
                .timing
                .progress
                .deadline(DECODE_CREDIT_TARGET, DECODE_GAP_SOFTCAP)
                .is_none()
        );
        assert!(!cohort.allows(&sessions[0]));
        assert!(cohort.allows(&sessions[1]));
        let available: Vec<_> = sessions
            .iter()
            .map(|s| ready_input(s, cohort.allows(s)))
            .collect();
        assert_eq!(
            select_work_in(64, &available, &mut false, &mut 0, &[], true),
            vec![1]
        );
        let second = sessions[1].blocks.sliding;
        cohort_complete(&mut sessions, second, 4, &mut pool);
        let release = Instant::now();
        assert_eq!(cohort.release_reason(&sessions, release), Some("ready"));
        cohort.release(
            &mut sessions,
            &mut pool,
            &mut vec![],
            &mut vec![],
            release,
            "ready",
        );
        assert!(matches!(ar.try_recv(),Ok(StreamEvent::Delta(text)) if text=="a"));
        assert_eq!(sessions[0].timing.progress.first, Some(release));
        assert_eq!(
            sessions[0]
                .timing
                .progress
                .deadline(DECODE_CREDIT_TARGET, DECODE_GAP_SOFTCAP),
            Some(release + DECODE_CREDIT_TARGET)
        );
        assert!(sessions.iter().all(|s| cohort.allows(s)));
    }

    #[test]
    fn startup_cohort_one_token_and_eos_keep_done_and_reservation_until_idle_flush() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (a, ar) = session(pool.as_mut().unwrap(), 128, 1);
        let (b, br) = session(pool.as_mut().unwrap(), 128, 4);
        let mut sessions = vec![a, b];
        let before = pool.as_ref().unwrap().available_pages();
        let first = cohort_work(&sessions[0]);
        let second = sessions[1].blocks.sliding;
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[first], Instant::now());
        cohort_complete(&mut sessions, first.ring, 3, &mut pool);
        cohort_complete(
            &mut sessions,
            second,
            crate::host::generate::EOS_TOKEN,
            &mut pool,
        );
        assert_eq!(sessions.len(), 2);
        assert!(
            sessions
                .iter()
                .all(|s| s.delivery.terminal() && !cohort.allows(s))
        );
        assert_eq!(pool.as_ref().unwrap().available_pages(), before);
        assert!(matches!(ar.try_recv(), Err(mpsc::TryRecvError::Empty)));
        assert!(matches!(br.try_recv(), Err(mpsc::TryRecvError::Empty)));
        // All work is blocked, but this opens before any recv or model call.
        assert_eq!(
            cohort.release_reason(&sessions, Instant::now()),
            Some("ready")
        );
        cohort.release(
            &mut sessions,
            &mut pool,
            &mut vec![],
            &mut vec![],
            Instant::now(),
            "ready",
        );
        assert!(sessions.is_empty());
        assert_eq!(pool.as_ref().unwrap().available_pages(), 256);
        assert!(matches!(ar.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));
        assert!(
            matches!(ar.try_recv(),Ok(StreamEvent::Done(o)) if o.text=="a" && o.completion_tokens==1)
        );
        assert!(
            matches!(br.try_recv(),Ok(StreamEvent::Done(o)) if o.text.is_empty() && o.completion_tokens==0)
        );
        assert!(matches!(
            ar.try_recv(),
            Err(mpsc::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn startup_snapshot_does_not_grow_and_oldest_receipt_bounds_release() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (a, ar) = session(pool.as_mut().unwrap(), 128, 4);
        let (b, _br) = session(pool.as_mut().unwrap(), 512, 4);
        let mut sessions = vec![a, b];
        let start = Instant::now();
        for s in &mut sessions {
            s.timing.received = start;
        }
        let first = cohort_work(&sessions[0]);
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[first], start + Duration::from_millis(100));
        cohort_complete(&mut sessions, first.ring, 3, &mut pool);
        let (late, _lr) = session(pool.as_mut().unwrap(), 128, 4);
        sessions.push(late);
        assert!(!cohort.allows(&sessions[2]));
        let deadline = start + startup_cohort::RECEIPT_BUDGET;
        assert_eq!(
            cohort.release_reason(&sessions, deadline - Duration::from_nanos(1)),
            None
        );
        assert_eq!(cohort.release_reason(&sessions, deadline), Some("deadline"));
        cohort.release(
            &mut sessions,
            &mut pool,
            &mut vec![],
            &mut vec![],
            deadline,
            "deadline",
        );
        assert!(matches!(ar.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));
        assert_eq!(sessions[0].timing.progress.first, Some(deadline));
        assert!(sessions.iter().all(|s| cohort.allows(s)));
        let later = cohort_work(&sessions[2]);
        cohort.maybe_start(&mut sessions, &[later], deadline);
        assert!(!cohort.active(), "one finite cohort per busy wave");
    }

    #[test]
    fn startup_visible_decoder_prevents_barrier_and_partial_prompt_cannot_start_one() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (a, _ar) = session(pool.as_mut().unwrap(), 128, 4);
        let (b, _br) = session(pool.as_mut().unwrap(), 256, 4);
        let mut sessions = vec![a, b];
        let partial = cohort_work(&sessions[1]);
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[partial], Instant::now());
        assert!(!cohort.active());
        let first = sessions[0].blocks.sliding;
        cohort_complete(&mut sessions, first, 3, &mut pool);
        assert!(sessions[0].timing.progress.first.is_some());
        let next = cohort_work(&sessions[1]);
        cohort.maybe_start(&mut sessions, &[next], Instant::now());
        assert!(!cohort.active());
        assert!(sessions.iter().all(|s| cohort.allows(s)));
    }

    fn stop_prefix_session(
        pool: &mut PageReservations,
        stream: bool,
    ) -> (Session, mpsc::Receiver<StreamEvent>) {
        let (mut s, r) = session(pool, 128, 4);
        s.stream = stream;
        s.generation = Generation::new(
            tokenizer(),
            crate::host::generate::GenerationRequest {
                prompt_ids: vec![2; 128],
                soft_at: vec![None; 128],
                image: None,
                audio: None,
                max_new_tokens: 4,
                sampling: crate::host::sampling::SamplingConfig::DEFAULT,
                stop_strings: vec!["ab".into()],
                separate_reasoning: false,
                ignore_eos: false,
            },
            CONTEXT,
        )
        .unwrap();
        (s, r)
    }

    #[test]
    fn startup_invisible_decoder_is_ready_without_text_and_does_not_deadlock() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (a, ar) = stop_prefix_session(pool.as_mut().unwrap(), true);
        let (b, br) = session(pool.as_mut().unwrap(), 128, 1);
        let mut sessions = vec![a, b];
        let first = sessions[0].blocks.sliding;
        cohort_complete(&mut sessions, first, 3, &mut pool); // "a" is held by stop-prefix logic.
        assert!(!sessions[0].generation.is_prefilling());
        assert!(sessions[0].timing.progress.first.is_none());
        let second = cohort_work(&sessions[1]);
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[second], Instant::now());
        assert!(!cohort.allows(&sessions[0]));
        cohort_complete(&mut sessions, second.ring, 4, &mut pool);
        assert_eq!(
            cohort.release_reason(&sessions, Instant::now()),
            Some("ready")
        );
        cohort.release(
            &mut sessions,
            &mut pool,
            &mut vec![],
            &mut vec![],
            Instant::now(),
            "ready",
        );
        assert_eq!(sessions.len(), 1);
        assert!(cohort.allows(&sessions[0]));
        assert!(sessions[0].timing.progress.first.is_none());
        assert!(matches!(ar.try_recv(), Err(mpsc::TryRecvError::Empty)));
        assert!(matches!(br.try_recv(),Ok(StreamEvent::Delta(t)) if t=="b"));
        cohort_complete(&mut sessions, first, 4, &mut pool); // matches stop, no fabricated delta.
        assert!(sessions.is_empty());
        assert!(matches!(ar.try_recv(),Ok(StreamEvent::Done(o)) if o.text.is_empty()));
    }

    #[test]
    fn startup_disconnect_releases_once_and_preserves_other_members() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (a, ar) = session(pool.as_mut().unwrap(), 128, 4);
        let (b, _br) = session(pool.as_mut().unwrap(), 128, 4);
        let mut sessions = vec![a, b];
        let first = cohort_work(&sessions[0]);
        let second = sessions[1].blocks.sliding;
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[first], Instant::now());
        drop(ar);
        cohort_complete(&mut sessions, first.ring, 3, &mut pool);
        cohort_complete(&mut sessions, second, 4, &mut pool);
        assert_eq!(pool.as_ref().unwrap().available_pages(), 254);
        cohort.release(
            &mut sessions,
            &mut pool,
            &mut vec![],
            &mut vec![],
            Instant::now(),
            "ready",
        );
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].blocks.sliding, second);
        assert_eq!(pool.as_ref().unwrap().available_pages(), 255);
        cohort.release(
            &mut sessions,
            &mut pool,
            &mut vec![],
            &mut vec![],
            Instant::now(),
            "ready",
        );
        assert_eq!(pool.as_ref().unwrap().available_pages(), 255);
    }

    #[test]
    fn startup_nonstream_buffers_only_done_and_done_never_precedes_later_delta() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (mut a, ar) = session(pool.as_mut().unwrap(), 128, 1);
        a.stream = false;
        let (b, _br) = session(pool.as_mut().unwrap(), 128, 1);
        let mut sessions = vec![a, b];
        let first = cohort_work(&sessions[0]);
        let second = sessions[1].blocks.sliding;
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[first], Instant::now());
        cohort_complete(&mut sessions, first.ring, 3, &mut pool);
        assert!(sessions[0].timing.progress.first.is_none());
        // A held terminal can never be sampled or advanced a second time.
        assert!(
            complete_work(
                &mut sessions,
                &[first],
                Some(&[SampleOutput::Greedy(4)]),
                &mut pool,
                &mut vec![],
                &mut vec![]
            )
            .is_err()
        );
        cohort_complete(&mut sessions, second, 4, &mut pool);
        cohort.release(
            &mut sessions,
            &mut pool,
            &mut vec![],
            &mut vec![],
            Instant::now(),
            "ready",
        );
        assert!(sessions.is_empty());
        assert!(matches!(ar.try_recv(),Ok(StreamEvent::Done(o)) if o.text=="a"));
        assert!(matches!(
            ar.try_recv(),
            Err(mpsc::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn startup_terminal_error_and_offline_flush_preserve_computed_event_order() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (a, ar) = session(pool.as_mut().unwrap(), 128, 1);
        let (b, br) = session(pool.as_mut().unwrap(), 128, 4);
        let mut sessions = vec![a, b];
        let first = cohort_work(&sessions[0]);
        let second = sessions[1].blocks.sliding;
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[first], Instant::now());
        cohort_complete(&mut sessions, first.ring, 3, &mut pool);
        cohort_complete(&mut sessions, second, crate::axes::W::SIZE, &mut pool);
        cohort.release(
            &mut sessions,
            &mut pool,
            &mut vec![],
            &mut vec![],
            Instant::now(),
            "error",
        );
        assert!(sessions.is_empty());
        assert_eq!(pool.as_ref().unwrap().available_pages(), 256);
        assert!(matches!(ar.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));
        assert!(matches!(ar.try_recv(),Ok(StreamEvent::Done(o)) if o.text=="a"));
        assert!(matches!(
            ar.try_recv(),
            Err(mpsc::TryRecvError::Disconnected)
        ));
        assert!(matches!(br.try_recv(), Ok(StreamEvent::Error(_))));
        assert!(matches!(
            br.try_recv(),
            Err(mpsc::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn startup_mixed_inflight_invisible_decode_batch_consumes_every_computed_output() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (a, ar) = stop_prefix_session(pool.as_mut().unwrap(), true);
        let (b, br) = session(pool.as_mut().unwrap(), 1, 4);
        let mut sessions = vec![a, b];
        let first = sessions[0].blocks.sliding;
        cohort_complete(&mut sessions, first, 3, &mut pool);
        let work: Vec<_> = sessions.iter().map(cohort_work).collect();
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &work, Instant::now());
        assert!(!cohort.active());
        assert!(
            complete_work(
                &mut sessions,
                &work,
                Some(&[SampleOutput::Greedy(5), SampleOutput::Greedy(4)]),
                &mut pool,
                &mut vec![],
                &mut vec![]
            )
            .unwrap()
        );
        assert_eq!(sessions[0].generation.completion_tokens(), 2);
        assert_eq!(sessions[1].generation.completion_tokens(), 1);
        assert!(matches!(ar.try_recv(),Ok(StreamEvent::Delta(t)) if t=="ac"));
        assert!(matches!(br.try_recv(),Ok(StreamEvent::Delta(t)) if t=="b"));
    }

    #[test]
    fn startup_channel_marker_without_first_text_releases_as_ready() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (mut a, ar) = session(pool.as_mut().unwrap(), 128, 4);
        a.generation = Generation::new(
            tokenizer(),
            crate::host::generate::GenerationRequest {
                prompt_ids: vec![2; 128],
                soft_at: vec![None; 128],
                image: None,
                audio: None,
                max_new_tokens: 4,
                sampling: crate::host::sampling::SamplingConfig::DEFAULT,
                stop_strings: vec![],
                separate_reasoning: true,
                ignore_eos: false,
            },
            CONTEXT,
        )
        .unwrap();
        let (b, _br) = session(pool.as_mut().unwrap(), 128, 1);
        let mut sessions = vec![a, b];
        let first = cohort_work(&sessions[0]);
        let second = sessions[1].blocks.sliding;
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[first], Instant::now());
        cohort_complete(
            &mut sessions,
            first.ring,
            crate::host::generate::CHANNEL_OPEN_TOKEN,
            &mut pool,
        );
        cohort_complete(&mut sessions, second, 3, &mut pool);
        assert_eq!(
            cohort.release_reason(&sessions, Instant::now()),
            Some("ready")
        );
        cohort.release(
            &mut sessions,
            &mut pool,
            &mut vec![],
            &mut vec![],
            Instant::now(),
            "ready",
        );
        assert_eq!(sessions.len(), 1);
        assert!(!sessions[0].generation.is_prefilling());
        assert!(cohort.allows(&sessions[0]));
        assert!(sessions[0].timing.progress.first.is_none());
        assert!(matches!(ar.try_recv(), Err(mpsc::TryRecvError::Empty)));
    }

    #[test]
    fn startup_does_not_delay_a_lone_request_or_an_already_expired_snapshot() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (a, _ar) = session(pool.as_mut().unwrap(), 128, 4);
        let mut sessions = vec![a];
        let first = cohort_work(&sessions[0]);
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[first], Instant::now());
        assert!(!cohort.active());
        assert!(!sessions[0].delivery.held());
        let (b, _br) = session(pool.as_mut().unwrap(), 128, 4);
        sessions.push(b);
        let now = Instant::now();
        sessions[0].timing.received = now - startup_cohort::RECEIPT_BUDGET;
        let mut expired = StartupCohort::default();
        expired.maybe_start(&mut sessions, &[first], now);
        assert!(!expired.active());
        assert!(sessions.iter().all(|s| !s.delivery.held()));
    }

    #[test]
    fn startup_rearms_after_visible_completion_with_pending_prefill_ownership_intact() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (old, old_events) = session(pool.as_mut().unwrap(), 128, 2);
        let (a, a_events) = session(pool.as_mut().unwrap(), 128, 4);
        let (b, _b_events) = session(pool.as_mut().unwrap(), 512, 4);
        let mut sessions = vec![old, a, b];
        let old_ring = sessions[0].blocks.sliding;
        cohort_complete(&mut sessions, old_ring, 3, &mut pool);
        assert!(matches!(old_events.try_recv(), Ok(StreamEvent::Delta(t)) if t == "a"));
        let first = cohort_work(&sessions[1]);
        let active_prompt = ScheduledInput {
            ring: sessions[2].blocks.sliding,
            position: 0,
            count: 512,
        };
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[first], Instant::now());
        assert!(!cohort.active());
        assert!(!cohort.rearm_prefill_only(&sessions));

        cohort_complete(&mut sessions, old_ring, 4, &mut pool);
        assert!(matches!(old_events.try_recv(), Ok(StreamEvent::Delta(t)) if t == "b"));
        assert!(matches!(old_events.try_recv(), Ok(StreamEvent::Done(_))));
        assert_eq!(sessions.len(), 2);
        let available_pages = pool.as_ref().unwrap().available_pages();
        let blocks: Vec<_> = sessions.iter().map(|s| s.blocks.clone()).collect();
        let receipts: Vec<_> = sessions.iter().map(|s| s.timing.received).collect();
        assert!(cohort.rearm_prefill_only(&sessions));
        assert!(!complete_work(
            &mut sessions, &[active_prompt], None, &mut pool, &mut vec![], &mut vec![]
        ).unwrap());
        cohort.maybe_start(&mut sessions, &[first], Instant::now());
        assert!(cohort.active());
        cohort_complete(&mut sessions, first.ring, 3, &mut pool);
        assert!(matches!(a_events.try_recv(), Err(mpsc::TryRecvError::Empty)));
        assert_eq!(pool.as_ref().unwrap().available_pages(), available_pages);
        for (index, s) in sessions.iter().enumerate() {
            assert_eq!(s.blocks.sliding, blocks[index].sliding);
            assert_eq!(s.blocks.full, blocks[index].full);
            assert_eq!(s.timing.received, receipts[index]);
            assert!(s.reservation.is_some());
        }
        assert_eq!(sessions[1].generation.input().1, active_prompt.position);
        assert_eq!(sessions[1].generation.inputs(active_prompt.count).0.len(), 512);
    }

    #[test]
    fn startup_rearm_rejects_invisible_decode_phase() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (old, _old_events) = stop_prefix_session(pool.as_mut().unwrap(), true);
        let (a, _a_events) = session(pool.as_mut().unwrap(), 128, 4);
        let mut sessions = vec![old, a];
        let old_ring = sessions[0].blocks.sliding;
        cohort_complete(&mut sessions, old_ring, 3, &mut pool);
        assert!(sessions[0].timing.progress.first.is_none());
        assert!(!sessions[0].generation.is_prefilling());
        let mixed: Vec<_> = sessions.iter().map(cohort_work).collect();
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &mixed, Instant::now());
        assert!(!cohort.active());
        assert!(!cohort.rearm_prefill_only(&sessions));
        let first = cohort_work(&sessions[1]);
        cohort.maybe_start(&mut sessions, &[first], Instant::now());
        assert!(!cohort.active());
        assert!(sessions.iter().all(|s| !s.delivery.held()));
    }

    #[test]
    fn startup_rearm_cannot_expand_an_active_snapshot() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (a, _a_events) = session(pool.as_mut().unwrap(), 128, 4);
        let (b, _b_events) = session(pool.as_mut().unwrap(), 128, 4);
        let mut sessions = vec![a, b];
        let first = cohort_work(&sessions[0]);
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[first], Instant::now());
        assert!(sessions.iter().all(|s| s.generation.is_prefilling()));
        let (late, _late_events) = session(pool.as_mut().unwrap(), 128, 4);
        sessions.push(late);
        assert!(!cohort.rearm_prefill_only(&sessions));
        let late_work = cohort_work(&sessions[2]);
        cohort.maybe_start(&mut sessions, &[late_work], Instant::now());
        assert!(cohort.active());
        assert!(!cohort.allows(&sessions[2]));
        assert!(!sessions[2].delivery.held());
    }

    #[test]
    fn startup_rearm_keeps_expired_original_receipt_budget() {
        let mut pool = Some(PageReservations::new(64, 256, CONTEXT).unwrap());
        let (a, _a_events) = session(pool.as_mut().unwrap(), 128, 4);
        let (b, _b_events) = session(pool.as_mut().unwrap(), 128, 4);
        let mut sessions = vec![a, b];
        let now = Instant::now();
        let received = now - startup_cohort::RECEIPT_BUDGET;
        sessions[0].timing.received = received;
        let first = cohort_work(&sessions[0]);
        let mut cohort = StartupCohort::default();
        cohort.maybe_start(&mut sessions, &[first], now);
        assert!(!cohort.active());
        assert!(cohort.rearm_prefill_only(&sessions));
        cohort.maybe_start(&mut sessions, &[first], now);
        assert!(!cohort.active());
        assert_eq!(sessions[0].timing.received, received);
        assert!(sessions.iter().all(|s| !s.delivery.held()));
    }
    include!("recovery_cohort_tests.rs");
    include!("recovery_late_admission_tests.rs");

}
