# C071 / q016 C16 phase failure: read-only evidence

No scheduler, deadline, model, or harness change was made. `analyze.py` reduces
the three immutable Arena logs into `analysis.json`, omitting prompt/output text.
The C071 C15/C16 executable is identical; q016 changes only grouped sliding O.
Both scheduler source hashes also match q016. The frozen source is e2aeb0b for
C071 and cc503bf for q016.

| Run | Measured requests | Throughput | TTFT p90 ms | TPOT p90 ms | Measured TTFT >4s |
|---|---:|---:|---:|---:|---:|
| r971 / 319706 C071 C15 | 105 | 148.699 | 3795.520 | 43.404 | 0 |
| r972 / 319705 C071 C16 | 96 | 136.434 | 4580.981 | 89.918 | 15 |
| r981 / 319723 q016 C16 | 96 | 143.739 | 4249.423 | 89.731 | 14 |

Correction to the initial warmup impression: C071 C16 warm TTFT p90 is
4052.192ms, with two of sixteen above 4s. q016 warm is genuinely within the
threshold (3761.638ms p90; every request below 4s).

## What is observed

C15 has eight 15-member cohort releases, all at the receipt deadline. C16 has
only four 16-member releases, then no more. q016's first two releases are `ready`
at 3753.6/3747.3ms oldest age; its next two are `deadline` at 3752.2/3750.7ms.
All snapshots report sixteen members, equal to the entire client concurrency.
These logs do not support the hypothesis that those initial snapshots omitted
some of their sixteen outstanding requests. They do not log explicit member IDs
or incoming-queue occupancy, so this is not a proof about arbitrary traffic.

The workload of successive admission-order bands differs. C15 remains 13,824
input tokens per fifteen admissions. C16 warm and first measured bands contain
14,080, then the next two contain 14,848 (+5.5%) and 15,872 (+12.7%). These are
observed lengths, not proposed scheduling categories. The public harness cycles
its five inputs, and replaces each completed request immediately:
`scripts/stage2/performance.py::select_inputs` and
`scripts/stage2/accuracy.py::remote` lines222–235. Consequently a concurrency
that is not a multiple of five gets different work in later admission bands.

Both C071 widths have essentially the same device service rates: recurring
prompt layer 2.69 versus2.70ms, full decoder round41.29 versus41.28ms. There is
no R32 transition at C16. The failure is not evidence of a kernel cost cliff.

q016 makes the initial sixteen complete prefills together, but the next band
splits: thirteen requests have positive hold, and three 2048-token requests
finish after release. Two of these, worker IDs42/43, have first output at
4117.4/4117.7ms from admission, including163.5ms across four decoder rounds.
The following band inherits123.3ms of old-decoder work before its later cohort;
four long prompts finish at~4623ms, with483.2ms across twelve decoder rounds.
After worker ID64 no remaining request has a positive cohort hold. Later long
requests69/70 wait2570.6ms across65 decoder rounds before their first output.

The actual HTTP timeline shows the feedback:

| q016 measured request-index band | Arrival span ms | First-output span ms | Completion span ms |
|---|---:|---:|---:|
| 1–16 | 6.7 | 2.3 | 27.5 |
| 17–32 | 27.5 | 386.2 | 390.7 |
| 33–48 | 390.7 | 873.2 | 1095.9 |

C071 expands faster: corresponding completion spans325.5,933.0,1799.4ms.
These bands are client/admission order groupings, not explicit logged snapshot
member lists. Do not equate HTTP request index with Worker ID inside a band.

## Verified control flow

`src/host/startup_cohort.rs::maybe_start` snapshots all currently admitted
sessions only after a scheduled prompt reaches its first-sampling boundary. It
does not snapshot before an awaited first prefill; newly arrived jobs not yet
dequeued remain outside until the next worker loop. `allows` excludes later
admissions while that finite snapshot is active. `release_reason` first checks
all members ready, then the3750ms original-receipt deadline. Release preserves
unfinished prefills and emits only buffered, already-computed events.

`rearm_prefill_only` requires *every* live session to remain in Prefill, with no
prior visible output or held/terminal delivery. Even one continuing decoder
prevents rearm. This accounts for why the completion-skewed later stream never
returns to a decoder-free cohort. The credit scheduler then fairly supplies
both decoder and prefill work; the observed TPOT converges to~90ms. The held
phase does not bank fictitious output credit.

`Worker::account_prefill_wait` charges the entire selected operation to every
selected prompt owner. Thus `own`/`ph` cannot be summed across owners as total
hardware time; `pw` also includes time spent held while other prefills execute.
`first` is measured from worker admission, not HTTP dispatch. Keep the separate
HTTP timings as the externally observed TTFT.

## 3850/3900ms assessment

A longer finite wait is a legitimate general tradeoff, but the logs do not
establish either proposed value as sufficient. Under the *same observed work
order*, subtracting all163.5ms decoder work from q016 IDs42/43 still gives
~3954ms from their own admission, before their offset from the oldest receipt
and HTTP-to-worker delay. This is a rough replay bound, not a new execution:
changed release timing can also change packing. In C071 the corresponding
deficit is larger. Holding until3900 may make one additional long request ready
and reduce output skew, while still releasing the last two unfinished. It can
also shift the majority's TTFT close to4s, with only100ms nominal margin for
receipt delay and an atomic preparation/finalization operation.

Therefore neither a deadline increase nor snapshot extension is justified as
a proven fix. q016 demonstrates that removing enough compute to make the first
band ready is insufficient for the heavier later bands. Preserve3750 while
obtaining the missing causal trace; do not run a budget grid.

## Minimal next trace question

Under the existing opt-in process diagnostic gate, emit a compact snapshot-start
and snapshot-release record, without token contents: cohort sequence, worker
elapsed time, admitted/queued count, member IDs, ready/prefilling/visible/held
counts, oldest receipt age, and each unfinished member's logical position and
remaining input count. Include live nonmember count to test exclusion directly.
At release report the current active chunk owners/counts and completed layer
quanta (already tracked by `ChunkTrace`) so remaining execution is observable.
Enabling the existing `MOA_SCHEDULER_CHUNK_TRACE` alongside this suffices to map
the last incomplete chunk to its subsequent work time. Log a rearm transition
or reason-blocked aggregate, not every loop. No policy modification is needed.

The decisive questions are: (1) exactly which work remains at the first failing
deadline, (2) whether extra3850/3900 wait would cover that work before4s from HTTP
dispatch, and (3) whether any admitted nonmember actually waits while a snapshot
is active. The current traces answer the overall feedback mechanism, but not
those exact remaining-layer or queue questions.
