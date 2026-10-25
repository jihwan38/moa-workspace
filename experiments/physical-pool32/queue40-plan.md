# Bounded actual HTTP capacity/FIFO diagnostic

Use the frozen c080 server unchanged. This is a capacity/stream-integrity
experiment, not the official mixed performance score or a new scheduling policy.

Prepare one public performance wave: ROUNDS=1, concurrency40, --test256,
no warmup. The existing input stays exactly256 prompt tokens,64 generated tokens,
ignore_eos=true, streaming+usage. No prompt or response substitutions are needed.
Every accepted request is independently computed. Forty requests exceed32
resident sliding slots, but each reserves only one512-token full page
(256+64=320), so full-page capacity256 cannot explain admission blocking.
The complete work is10,240 prompt tokens+2,560 generated tokens. A70-second job
is a bounded first test, not a promise about its duration or latency percentiles.

Preparation using existing tools (root dispatch only):

    python3 tools/make_evaluation.py --candidate c080-physical32 \
      --name c080-physical32-c40-queue --rounds 1 --log-tail-bytes 1000000

In that copied POSIX entrypoint, export MOA_SCHEDULER_CHUNK_TRACE=1 before the
Python launch; record the changed entrypoint SHA and unchanged server SHA in the
evaluation manifest. This process-only switch already exists in
host/chunk_trace.rs; it adds completion logs and does not affect policy.

Then use the standard bundle entrypoint:

    RNGD_SUBMIT_TIMEOUT=70 scripts/stage2/performance_test.sh \
      --test 256 --concurrency 40 --no-build --no-wait

The Python harness already permits concurrency1..100 and starts the40 workers
behind one actual barrier. Preserve all40 per-request records and the1MB server
tail. No shortened output limit, timeout relaxation or benchmark-routing branch.

Acceptance/analysis:

1. All40 HTTP records must pass text/SSE/usage/[DONE] protocol checks, unique
   request_index1..40, prompt_tokens256, completion_tokens64, and no error/drop.
   Their actual TTFT includes waiting before Worker dequeue. SLO failure, if any,
   is reported; this diagnostic is not presented as a ranked mixed score.
2. Confirm the request dispatch span precedes the first completed response,
   demonstrating >32 requests outstanding end-to-end. A width32 decode or a
  32-member snapshot establishes actual resident saturation if observed.
   If arrival/trace evidence does not show saturation, label the queue exercise
   inconclusive rather than asserting all8 extras waited in mpsc.
3. Require exactly40 completed SCHED_REQ IDs and40 complete prompt histories,
   all ring IDs0..31, and no reused ring for a new traced owner before its prior
   owner's SCHED_REQ completion. Include SCHED_RECOVERY membership/join/release
   and SCHED_CHUNK rows in the ownership reconstruction. A completed chunk may
   reveal an owner later than actual admission, so reconstructed occupancy is a
   lower bound; do not invent admission timestamps.
4. Reconcile SCHED_TOTAL decode row counts with actual request usage and the
   prompt's first generated tokens; report real widths/calls. The last8 requests
   must finish after slots are reused, with no fatal/offline/invalid-slot error.
5. Source establishes FIFO at the actual worker queue: idle_batch caps its
   collection atcapacity; busy receive takes onlycapacity-sessions-pending;
   uncollected jobs remain in std::sync::mpsc, and PageReservations admits only
   pending.front beforepop_front. Completion order is not FIFO and must not be
   asserted. Client thread indices are not server receive IDs: the API carries
   no correlation ID intoJob, so this unmodified harness cannot independently
   establish exact client-index FIFO order.
6. SCHED_REQ q starts at worker dequeue and omits upstream queue wait. Never use
   its smallq to claim the queued requests had no waiting. Use HTTP TTFT/latency
   for their externally visible cost. Existing CPU tests cover exact FIFO queue
   semantics, cancellation/reservation preservation and stale-epoch rejection.

The C16 matched public performance run remains the throughput comparison. Only
this separate capacity diagnostic establishes actual >32 load completion. No
public task/quality result is inferred from it.
