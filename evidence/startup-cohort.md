# Bounded startup cohort candidate

Base: root `4de353c`. Implementation is isolated in `round2/startup-cohort`.
`host/startup_cohort.rs::ENABLED` defaults to false. No API, model arithmetic,
request sampling policy, or benchmark flag is changed.

The existing worker starts decode deadlines as soon as an early request emits
text. Its decode rounds then compete with other admitted prefills. This candidate
tests a general latency tradeoff: retain the first actually computed output of
each member of one finite admitted snapshot, finish the other snapshot prefills,
then expose their outputs together and start decode cadence from actual emission.

## Lifecycle

The snapshot is selected immediately before the first completed prompt batch
advances Generation. At least two admitted requests must exist, no session may
already have a visible output, and the oldest worker receipt must be younger than
3250 ms. A mixed completed batch already containing a decode-phase row stays on
the normal route so every computed output is consumed exactly once.

Membership uses stable request IDs and never expands. Subsequent arrivals are
admitted normally when capacity permits, but do not execute until the snapshot
opens. Each member executes its original prompt, samples normally, and buffers
only actual Content/Reasoning/Done/Error events. Once its prompt finishes, its
decode pauses. No future token is sampled and no dummy event is sent.

The barrier opens once every member has finished prefill, or when the oldest
worker receipt reaches 3250 ms. This check runs before every potentially blocking
receive. A generation with only a channel marker or held stop-prefix is explicitly
in decode phase and therefore ready even when no text exists. A terminal EOS,
one-token completion, or generation error remains in Session with its reservation
until its retained events are delivered. Complete generations never enter
`Generation::input` again.

Release preserves each request's event order. Terminal or disconnected members
release their reservations exactly once; unfinished members resume normal decode.
An offline error first flushes already-computed retained events, removes completed
members, then sends the worker error only to remaining requests. A Done cannot be
followed by another Delta/Error for the same retained member. Zero-token requests
keep the existing immediate Done path and never acquire a cohort reservation.

The worker's existing resumable Prompt512/group allocations remain owned until
their full layers finish. Opening the barrier does not cancel or restart an active
prompt. Only generations that have completed model work can be freed on release.
One snapshot is attempted per busy wave; once sessions and pending requests empty,
the next wave becomes eligible. Existing visible decoders always retain normal
cadence.

## Timing and limitations

`TokenProgress` is updated only after a successful real send of nonempty stream
text. Buffered samples have no first-output timestamp and earn no future credit.
Nonstream requests send only Done, so their first-visible timestamp is assigned
at that actual response rather than at an internal callback. Existing completion
usage and sampling/EOS/stop behavior are unchanged.

The 3250 ms clock starts from Worker receipt, including known pending-admission
wait. The fixed API does not expose HTTP arrival time or time spent in its incoming
mpsc queue. External TTFT and elapsed throughput still measure the real delayed
response; no reported evaluator timestamp is rewritten. The deadline is checked
between atomic model operations, so media preparation or a model quantum can
overshoot it. This is a bounded scheduling policy, not a proof of a hard 4 s SLO.

The expected C6 mechanism is fewer decode rounds during the initial finite cohort's
prefill and more aligned decode batches after release. Early requests deliberately
wait longer; late requests may wait less. This is an unmeasured hypothesis. Its
benefit may be limited with continuously arriving or unequal-length requests,
because an already-visible decoder prevents a new startup cohort. Trace actual
cohort counts and membership before attributing a full-run improvement to it.

`SCHED_COHORT` records member count, ready/deadline/error reason, barrier duration,
and oldest receipt age. `SCHED_REQ` adds per-request ready-but-held time. Existing
decode-width, prompt-width, complete-stage, output-gap and deadline-overrun traces
remain available. A candidate must be compared to the same source with ENABLED
false, with real HTTP TTFT/TPOT/throughput and public quality gates.

## Validation before NPU execution

`cargo check --locked --offline --lib` passed. The complete host suite passed
58 tests, including 11 new cohort tests for withheld actual output/credit, one-token
and EOS Done retention, immutable membership/deadline, visible decoder exclusion,
invisible stop-prefix/channel-marker phases, disconnection, nonstream delivery,
terminal error ordering, mixed in-flight output consumption, and lone/expired
snapshot fallback. Existing generation, admission, group/Prompt512 yield and
ownership tests also passed.

Batch-agent read-only review found no terminal, active ownership, late-arrival,
or clock-update defect; its mixed-batch edge was made an explicit tested fallback.
Server compile and actual HTTP gates are separate requirements. Only root dispatches
NPU jobs and decides whether to activate this candidate.
