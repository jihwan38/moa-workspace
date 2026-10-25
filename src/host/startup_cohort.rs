//! Bounded first-output barrier for one finite snapshot of admitted requests.
//! Only actual computed events are retained. No tokens execute after a member
//! finishes prefill until the barrier opens; external TTFT includes this delay.
use super::*;

pub(super) const ENABLED: bool = true;
pub(super) const RECEIPT_BUDGET: Duration = Duration::from_millis(3250);

#[derive(Default)]
pub(super) struct StartupDelivery {
    buffered: Option<Vec<StreamEvent>>,
    terminal: bool,
    ready_at: Option<Instant>,
}
impl StartupDelivery {
    pub(super) fn held(&self) -> bool {
        self.buffered.is_some()
    }
    pub(super) fn terminal(&self) -> bool {
        self.terminal
    }
    fn hold(&mut self, ready: bool, now: Instant) {
        assert!(self.buffered.is_none() && !self.terminal);
        self.buffered = Some(Vec::new());
        self.ready_at = ready.then_some(now);
    }
    pub(super) fn ready(&mut self, now: Instant) {
        if self.held() {
            self.ready_at.get_or_insert(now);
        }
    }
    pub(super) fn delta(
        &mut self,
        delta: Delta<'_>,
        stream: bool,
        events: &mpsc::Sender<StreamEvent>,
        progress: &mut TokenProgress,
    ) -> bool {
        assert!(!self.terminal, "terminal requests cannot emit later deltas");
        if !stream {
            return true;
        }
        let event = match delta {
            Delta::Content(text) => StreamEvent::Delta(text.to_owned()),
            Delta::Reasoning(text) => StreamEvent::ReasoningDelta(text.to_owned()),
        };
        if let Some(buffer) = &mut self.buffered {
            buffer.push(event);
            true
        } else {
            send_event(events, progress, event, stream, Instant::now())
        }
    }
    /// Returns true while a completed request must retain its allocation.
    pub(super) fn finish(
        &mut self,
        event: StreamEvent,
        stream: bool,
        events: &mpsc::Sender<StreamEvent>,
        progress: &mut TokenProgress,
    ) -> bool {
        assert!(!self.terminal);
        self.terminal = true;
        if let Some(buffer) = &mut self.buffered {
            buffer.push(event);
            self.ready_at.get_or_insert_with(Instant::now);
            true
        } else {
            let _ = send_event(events, progress, event, stream, Instant::now());
            false
        }
    }
    /// All buffered events precede Done/Error. A disconnected receiver releases
    /// its reservation at this real send, without a dummy cancellation probe.
    fn flush(
        &mut self,
        session: &mut SessionTiming,
        events: &mpsc::Sender<StreamEvent>,
        stream: bool,
        now: Instant,
    ) -> bool {
        let Some(buffer) = self.buffered.take() else {
            return !self.terminal;
        };
        if let Some(ready) = self.ready_at.take() {
            session.cohort_hold += now.saturating_duration_since(ready);
        }
        for event in buffer {
            if !send_event(events, &mut session.progress, event, stream, now) {
                return false;
            }
        }
        !self.terminal
    }
}

fn send_event(
    events: &mpsc::Sender<StreamEvent>,
    progress: &mut TokenProgress,
    event: StreamEvent,
    stream: bool,
    now: Instant,
) -> bool {
    let visible = match &event {
        StreamEvent::Delta(text) | StreamEvent::ReasoningDelta(text) => !text.is_empty(),
        StreamEvent::Done(output) if !stream => {
            !output.text.is_empty() || !output.reasoning.is_empty()
        }
        _ => false,
    };
    let sent = events.send(event).is_ok();
    if sent && visible {
        progress.emitted(now);
    }
    sent
}

struct Snapshot {
    members: Vec<u64>,
    started: Instant,
    oldest_receipt: Instant,
    deadline: Instant,
}

#[derive(Default)]
pub(super) struct StartupCohort {
    attempted: bool,
    active: Option<Snapshot>,
}
impl StartupCohort {
    pub(super) fn reset_idle(&mut self) {
        assert!(self.active.is_none());
        self.attempted = false;
    }
    #[cfg(test)]
    pub(super) fn active(&self) -> bool {
        self.active.is_some()
    }
    pub(super) fn allows(&self, session: &Session) -> bool {
        !session.delivery.terminal()
            && !(session.delivery.held() && !session.generation.is_prefilling())
            && self
                .active
                .as_ref()
                .is_none_or(|s| s.members.contains(&session.timing.id))
    }
    /// Snapshot at the first completed prompt's sampling boundary. Arrivals
    /// during its prefill may join; subsequent arrivals cannot extend the wait.
    pub(super) fn maybe_start(
        &mut self,
        sessions: &mut [Session],
        scheduled: &[ScheduledInput],
        now: Instant,
    ) {
        if self.attempted || self.active.is_some() {
            return;
        }
        if sessions.iter().any(|s| s.timing.progress.first.is_some()) {
            self.attempted = true;
            return;
        }
        let samples_first = scheduled.iter().any(|input| {
            sessions.iter().any(|s| {
                s.blocks.sliding == input.ring && s.generation.finishes_prompt(input.count)
            })
        });
        if !samples_first {
            return;
        }
        self.attempted = true;
        // Every already-computed output must be consumed once. Do not freeze a
        // decode row that is itself part of this completed mixed batch.
        if scheduled.iter().any(|input| {
            sessions
                .iter()
                .any(|s| s.blocks.sliding == input.ring && !s.generation.is_prefilling())
        }) {
            return;
        }
        let oldest = sessions.iter().map(|s| s.timing.received).min().unwrap();
        let deadline = oldest + RECEIPT_BUDGET;
        if sessions.len() < 2 || now >= deadline {
            return;
        }
        let members = sessions.iter().map(|s| s.timing.id).collect();
        for session in sessions {
            session
                .delivery
                .hold(!session.generation.is_prefilling(), now);
        }
        self.active = Some(Snapshot {
            members,
            started: now,
            oldest_receipt: oldest,
            deadline,
        });
    }
    pub(super) fn release_reason(
        &self,
        sessions: &[Session],
        now: Instant,
    ) -> Option<&'static str> {
        let snapshot = self.active.as_ref()?;
        if snapshot.members.iter().all(|id| {
            sessions
                .iter()
                .find(|s| s.timing.id == *id)
                .is_none_or(|s| s.delivery.terminal() || !s.generation.is_prefilling())
        }) {
            Some("ready")
        } else if now >= snapshot.deadline {
            Some("deadline")
        } else {
            None
        }
    }
    pub(super) fn release(
        &mut self,
        sessions: &mut Vec<Session>,
        reservations: &mut Option<PageReservations>,
        free: &mut Vec<usize>,
        rings: &mut Vec<usize>,
        now: Instant,
        reason: &str,
    ) {
        let Some(snapshot) = self.active.take() else {
            return;
        };
        if SCHEDULER_TRACE_ENABLED {
            eprintln!(
                "SCHED_COHORT members={} reason={} hold_ms={:.1} oldest_ms={:.1}",
                snapshot.members.len(),
                reason,
                now.saturating_duration_since(snapshot.started)
                    .as_secs_f64()
                    * 1000.0,
                now.saturating_duration_since(snapshot.oldest_receipt)
                    .as_secs_f64()
                    * 1000.0
            );
        }
        sessions.retain_mut(|s| {
            if !snapshot.members.contains(&s.timing.id) {
                return true;
            }
            if s.delivery.flush(&mut s.timing, &s.events, s.stream, now) {
                return true;
            }
            if SCHEDULER_TRACE_ENABLED {
                s.timing.report(now);
            }
            release_session(s, reservations, free, rings);
            false
        });
    }
}
