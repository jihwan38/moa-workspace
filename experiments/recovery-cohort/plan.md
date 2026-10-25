# Default-off finite recovery cohort

Base: root a8b4d0f, reviewed proposal fdadb79/e087bb9 and independent review8b8e724.
No kernels, model computation, sampling, admission capacity or budgets change.
Existing policy remains the default. `RECOVERY_ENABLED=false` is ordinary
process-wide source configuration; Worker constructs one StartupCohort with it.
Tests use explicit new(true), independent of default activation. Candidate
activation is a single constant flip followed by a plain source build.

The recovery path captures all fresh un-emitted admitted Prefill requests at an
actually finishing fresh row's sampling boundary. At least two are required;
the oldest original receipt fixes the unchanged3750ms deadline. Every capture,
including legacy, records request-owned once_participated in StartupDelivery.
Recovery never captures a previous participant. Existing Decode (even invisible)
and released participants still in Prefill bypass later barriers. Only fresh
late Prefill waits; no membership/deadline extension, hidden extra decode or
synthetic event occurs. Existing90ms credit/150ms soft-gap rules stay intact.
Default-off legacy selection remains unchanged; its participation bookkeeping
is new but is not consulted by that old selection path.

Active member readiness, terminal Done/EOS/error, cancellation and reservation
release reuse the original real-event buffering and Worker top-loop/fatal paths.
No Session vector index becomes persistent state. Diagnostics contain IDs,
physical rings, remaining counts and deadline offsets only, never prompt text.

Gates: all existing unfiltered library tests plus actual complete_work-based
regressions for staggered receipts/no renewed hold, previous Prompt512 checkpoint
passthrough under later cohorts and decoder compaction, mixed old visible and
invisible decoder outputs, no extra token/credit, late arrivals/expired oldest,
EOS/one-token/nonstream/error/cancellation/EOF, ring reuse and arbitrary cohort
size. Plain server check. Independent source/lifecycle review before handoff.
No device build or Arena dispatch in this task. Real HTTP timing and SLO remain
root-owned future gates; scheduler eligibility is not a hard TPOT guarantee.
