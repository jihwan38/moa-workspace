# Optional prompt chunk completion records

Base42978e1. This change only observes existing decisions; the3750ms receipt
budget, rearm guard, prompt choice, model calls and token/event semantics stay
unchanged. The process opt-in is `MOA_SCHEDULER_CHUNK_TRACE=1`, additionally
requiring the existing `SCHEDULER_TRACE_ENABLED` master gate. The default emits
no new records and performs no snapshot allocations or extra clock reads.

`SCHED_CHUNK` prints once when a model input chunk completes, never per layer or
ordinary decode step. A paused chunk retains one immutable scheduling-time
snapshot across independent decode and session-vector changes. Single-token
prompt tails mixed into a decode dispatch are recorded separately; pure decode
is omitted. Failed/incomplete model chunks produce no completion record.

Fields: `seq` is a worker-local chunk sequence; `cohort`, `ready1`, `visible`,
`queued` describe scheduling-time constraints. `batch` counts all scheduled rows;
`owners` and `tokens` count prompt rows/inputs only. `quanta` and `work_ms` count
the dispatched work containing this chunk; `span_ms` also includes intervening
work. A mixed singleton batch includes its other decoder rows in work_ms.
`rows` contains `worker_id:ring:position:count:remaining_before`; `pending` lists
all admitted prompt sessions as `worker_id:ring:position:remaining:ready_count`.
A zero ready_count denotes an ineligible session (e.g. an arrival excluded from
a finite cohort). Remaining lengths are exact Generation phase values before
page/chunk caps, not inferred from the chosen128/512 count. Terminal/decoding
sessions return no remaining prompt value and are safe to query.

The next experiment is an offline reconstruction of existing chunk packing:
quantify which singleP128 chunks had other eligible prompt work and whether
keeping four owners alive longer could remove tail underfilling. An actual
scheduling change requires that evidence; this diagnostic does not implement
length-based priority or extend a cohort. Logs contain IDs/counts only, no
prompt text, tokens, or generated content. Existing request/client IDs remain
separate; do not join them by numeric equality.

Preserved a037 compile failure: new tests supplied unit-returning callbacks to
Generation's bool callback API. Only test callbacks changed to true for a038.

Validation: a038 passed all 66 host tests; a039 passed the plain server cargo check.
The parser smoke test covered cohort-excluded arrivals, page boundaries and
immutable remaining-input snapshots. The independent read-only lifecycle audit
found no scheduler, reservation or generation-state mutation, including resumed
prompts, session reorder, mixed singleton tails and fatal error paths.
No device execution or scheduling experiment is claimed by these checks.
