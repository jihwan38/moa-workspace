//! Bounded first-output barrier for one finite snapshot of admitted requests.
//! Only actual computed events are retained. No tokens execute after a member
//! finishes prefill until the barrier opens; external TTFT includes this delay.
use super::*;

pub(super) const ENABLED: bool = true;
/// Process-wide policy selection; isolated candidate activation changes this
/// constant, never request content, evaluation flags, or the receipt budget.
// The fixed-membership recovery candidate regressed the actual overlap probe
// (r1130 versus r1129). Retain its explicit lifecycle tests, but keep serving
// on the established cohort policy pending a distinct measured repair.
pub(super) const RECOVERY_ENABLED: bool = false;
pub(super) const RECEIPT_BUDGET: Duration = Duration::from_millis(3750);

#[derive(Default)]
pub(super) struct StartupDelivery {
    buffered: Option<Vec<StreamEvent>>,
    terminal: bool,
    ready_at: Option<Instant>,
    // Request-owned history survives release and Session-vector compaction.
    // A newly admitted Session starts false even when its physical ring is reused.
    once_participated: bool,
}
impl StartupDelivery {
    pub(super) fn held(&self) -> bool {
        self.buffered.is_some()
    }
    pub(super) fn terminal(&self) -> bool {
        self.terminal
    }
    pub(super) fn once_participated(&self) -> bool {
        self.once_participated
    }
    fn hold(&mut self, ready: bool, now: Instant) {
        assert!(self.buffered.is_none() && !self.terminal);
        self.once_participated = true;
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
    recovery: bool,
    attempted: bool,
    active: Option<Snapshot>,
}
impl StartupCohort {
    pub(super) fn new(recovery: bool) -> Self {
        Self { recovery, ..Self::default() }
    }
    pub(super) fn reset_idle(&mut self) {
        assert!(self.active.is_none());
        self.attempted = false;
    }
    /// A previous visible stream may finish while other admitted prompts remain.
    /// Rearm only before any remaining generation enters Decode, including a
    /// decoder whose text is still hidden by channel or stop-prefix handling.
    /// This changes eligibility only: receipt times, allocations and any active
    /// model prompt checkpoint remain untouched. An existing snapshot never grows.
    pub(super) fn rearm_prefill_only(&mut self, sessions: &[Session]) -> bool {
        if !self.recovery
            && self.attempted
            && self.active.is_none()
            && !sessions.is_empty()
            && sessions.iter().all(|s| {
                s.generation.is_prefilling()
                    && s.timing.progress.first.is_none()
                    && !s.delivery.terminal()
                    && !s.delivery.held()
            })
        {
            self.attempted = false;
            true
        } else {
            false
        }
    }
    pub(super) fn active(&self) -> bool {
        self.active.is_some()
    }
    pub(super) fn allows(&self, session: &Session) -> bool {
        !session.delivery.terminal()
            && !(session.delivery.held() && !session.generation.is_prefilling())
            && self
                .active
                .as_ref()
                .is_none_or(|s| {
                    s.members.contains(&session.timing.id)
                        || (self.recovery
                            && (!session.generation.is_prefilling()
                                || session.delivery.once_participated()))
                })
    }
    /// Snapshot at the first completed prompt's sampling boundary. Arrivals
    /// during its prefill may join; subsequent arrivals cannot extend the wait.
    pub(super) fn maybe_start(
        &mut self,
        sessions: &mut [Session],
        scheduled: &[ScheduledInput],
        now: Instant,
    ) {
        if self.recovery {
            self.maybe_start_recovery(sessions, scheduled, now);
            return;
        }
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
    fn fresh_member(session: &Session) -> bool {
        session.generation.is_prefilling()
            && session.timing.progress.first.is_none()
            && !session.delivery.terminal()
            && !session.delivery.held()
            && !session.delivery.once_participated()
    }
    fn maybe_start_recovery(
        &mut self,
        sessions: &mut [Session],
        scheduled: &[ScheduledInput],
        now: Instant,
    ) {
        if self.active.is_some() {
            return;
        }
        // Capture before complete_work advances the actually computed final
        // prompt row. Existing decode rows in a mixed result remain unheld.
        if !scheduled.iter().any(|input| sessions.iter().any(|session| {
            session.blocks.sliding == input.ring
                && Self::fresh_member(session)
                && session.generation.finishes_prompt(input.count)
        })) {
            return;
        }
        let members: Vec<_> = sessions.iter().filter(|s| Self::fresh_member(s))
            .map(|s| s.timing.id).collect();
        if members.len() < 2 {
            return;
        }
        let oldest_receipt = sessions.iter().filter(|s| Self::fresh_member(s))
            .map(|s| s.timing.received).min().unwrap();
        let deadline = oldest_receipt + RECEIPT_BUDGET;
        // An expired otherwise-eligible older request cannot be omitted to
        // make a younger subset fit. Admission never changes received.
        if now >= deadline {
            return;
        }
        for session in sessions.iter_mut().filter(|s| members.contains(&s.timing.id)) {
            assert!(Self::fresh_member(session));
            session.delivery.hold(false, now);
        }
        self.active = Some(Snapshot { members, started: now, oldest_receipt, deadline });
        self.trace_recovery(sessions, now, "start");
    }
    fn trace_recovery(&self, sessions: &[Session], now: Instant, action: &str) {
        if !self.recovery || !SCHEDULER_TRACE_ENABLED {
            return;
        }
        let Some(snapshot) = &self.active else { return; };
        let members: Vec<_> = snapshot.members.iter().map(|id| {
            sessions.iter().find(|s| s.timing.id == *id).map_or_else(
                || format!("{id}:gone"),
                |s| format!("{}:{}:{}:{}", id, s.blocks.sliding,
                    s.generation.remaining_prompt_inputs().unwrap_or(0),
                    usize::from(s.delivery.terminal() || !s.generation.is_prefilling())))
        }).collect();
        let passthrough: Vec<_> = sessions.iter().filter(|s| {
            !snapshot.members.contains(&s.timing.id) && self.allows(s)
        }).map(|s| format!("{}:{}:{}",s.timing.id,s.blocks.sliding,
            s.generation.remaining_prompt_inputs().unwrap_or(0))).collect();
        eprintln!("SCHED_RECOVERY action={} members={} passthrough={} oldest_ms={:.3} left_ms={:.3}",
            action,members.join(","),passthrough.join(","),
            now.saturating_duration_since(snapshot.oldest_receipt).as_secs_f64()*1000.0,
            snapshot.deadline.saturating_duration_since(now).as_secs_f64()*1000.0);
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
        self.trace_recovery(sessions, now, reason);
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
