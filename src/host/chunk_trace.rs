//! Opt-in completion records for prompt packing decisions. This module never
//! selects work or changes request/Model state. Snapshots describe scheduling
//! time; one record is printed only after that exact model chunk completes.
use super::*;

pub(super) fn enabled() -> bool {
    std::env::var("MOA_SCHEDULER_CHUNK_TRACE").is_ok_and(|value| value == "1")
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Chosen {
    id: u64,
    ring: usize,
    position: usize,
    count: usize,
    remaining: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct PendingPrompt {
    id: u64,
    ring: usize,
    position: usize,
    remaining: usize,
    ready_count: usize,
}

pub(super) struct ChunkTrace {
    sequence: usize,
    cohort: bool,
    ready_singletons: usize,
    visible_decoders: usize,
    queued: usize,
    batch: usize,
    chosen: Vec<Chosen>,
    pending: Vec<PendingPrompt>,
    started: Instant,
    compute: Duration,
    quanta: usize,
}
impl ChunkTrace {
    pub(super) fn capture(
        sequence: usize,
        sessions: &[Session],
        scheduled: &[ScheduledInput],
        available: &[ReadyInput],
        cohort: bool,
        ready_singletons: usize,
        queued: usize,
        now: Instant,
    ) -> Option<Self> {
        let chosen: Vec<_> = scheduled.iter().filter_map(|input| {
            let session = sessions.iter().find(|s| s.blocks.sliding == input.ring)?;
            let remaining = session.generation.remaining_prompt_inputs()?;
            Some(Chosen { id: session.timing.id, ring: input.ring,
                position: input.position, count: input.count, remaining })
        }).collect();
        if chosen.is_empty() { return None; }
        let pending = sessions.iter().zip(available).filter_map(|(session, ready)| {
            let remaining = session.generation.remaining_prompt_inputs()?;
            Some(PendingPrompt { id: session.timing.id, ring: session.blocks.sliding,
                position: session.generation.input().1, remaining, ready_count: ready.count })
        }).collect();
        Some(Self {
            sequence, cohort, ready_singletons, queued, batch: scheduled.len(), chosen, pending,
            visible_decoders: sessions.iter().filter(|s| !s.delivery.terminal()
                && !s.generation.is_prefilling() && s.timing.progress.first.is_some()).count(),
            started: now, compute: Duration::ZERO, quanta: 0,
        })
    }
    pub(super) fn observe(&mut self, elapsed: Duration) {
        self.compute += elapsed;
        self.quanta += 1;
    }
    pub(super) fn compact(&self, now: Instant) -> String {
        // A worker ID disambiguates a physical ring reused by a later request.
        let chosen = self.chosen.iter().map(|r| format!("{}:{}:{}:{}:{}",
            r.id, r.ring, r.position, r.count, r.remaining)).collect::<Vec<_>>().join(",");
        let pending = self.pending.iter().map(|r| format!("{}:{}:{}:{}:{}",
            r.id, r.ring, r.position, r.remaining, r.ready_count)).collect::<Vec<_>>().join(",");
        format!("SCHED_CHUNK seq={} cohort={} ready1={} visible={} queued={} batch={} owners={} tokens={} quanta={} work_ms={:.3} span_ms={:.3} rows={} pending={}",
            self.sequence, usize::from(self.cohort), self.ready_singletons, self.visible_decoders,
            self.queued, self.batch, self.chosen.len(), self.chosen.iter().map(|r| r.count).sum::<usize>(),
            self.quanta, self.compute.as_secs_f64()*1000.0,
            now.saturating_duration_since(self.started).as_secs_f64()*1000.0, chosen, pending)
    }
    pub(super) fn report(self, now: Instant) {
        eprintln!("{}", self.compact(now));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn format_preserves_exact_lengths_range_and_snapshot_across_yields() {
        let now=Instant::now();
        let mut trace=ChunkTrace {
            sequence:4, cohort:true, ready_singletons:3, visible_decoders:2, queued:1, batch:1,
            chosen:vec![Chosen{id:19,ring:63,position:896,count:512,remaining:1152}],
            pending:vec![PendingPrompt{id:19,ring:63,position:896,remaining:1152,ready_count:128},
                PendingPrompt{id:20,ring:31,position:0,remaining:777,ready_count:0}],
            started:now, compute:Duration::ZERO, quanta:0,
        };
        trace.observe(Duration::from_millis(3));
        trace.observe(Duration::from_millis(4));
        let line=trace.compact(now+Duration::from_millis(60));
        assert!(line.contains("owners=1 tokens=512 quanta=2 work_ms=7.000 span_ms=60.000"));
        assert!(line.contains("rows=19:63:896:512:1152"));
        assert!(line.contains("pending=19:63:896:1152:128,20:31:0:777:0"));
    }
}
