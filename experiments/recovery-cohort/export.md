# Recovery cohort: clean candidate export

Base c075 `0dc80b7`. Apply in order:

1. `bffae2c`: default-off implementation and design plan.
2. `7b8a4be`: seven actual-completion lifecycle regressions.
3. `e2ef734`: preserved original default-off118-test evidence.
4. `7623b74`: separate authorized candidate activation, only
   `RECOVERY_ENABLED=false` -> `true`.

No conflicts occurred. The ordinary Rust constant is read once by Worker when
constructing StartupCohort; tests use explicit `new(true)`. No request value,
benchmark flag, environment feature or model precision selector activates it.
The original default-off tree remains frozen at recovery-cohort/3cd2303.
Legacy selection is unchanged when recovery is off; both capture paths record
once-participated request history, but only recovery uses it for eligibility.

Only startup_cohort.rs, Worker construction/test inclusion, and the new CPU test
file change under src. Hash proof verifies c075 Model, grouped QKV/prefill,
B16 UP/prefix/FFN, Generation, admission and sampling bytes remain unchanged.
The3750ms receipt,90ms earned-token and150ms soft-gap policies are unchanged.

Both source configurations pass118 unfiltered library tests (0 ignored,
0 filtered,0 failed). The clean c075+ON source7623b74 also passes plain server
check in13.58s. a002/a003 record exact commands and source hashes; all385 hashes
are verified against the final source. CPU package debug0 and incremental0 were
used with the owned prefill target; no root target, device build or Arena call.

The seven new tests exercise actual Session/Generation/complete_work behavior:
visible and invisible old decoders in one mixed completion; no hidden extra
step/credit; staggered receipts without renewed holding; prior Prompt512 owner
passthrough under a later cohort despite decoder compaction and cancellation;
late arrivals and expired-oldest rejection; EOS/nonstream/idle EOF; silent first
tokens; terminal errors and fatal flush ordering; fresh physical-ring reuse;
and a33-member cohort with no fixed benchmark batch quota. Existing legacy,
admission, sampling, media and model-shape tests remain active.

Independent round2_path review passed the functional source and root reviewed
the source/tests before export. `SCHED_RECOVERY` logs capture/release membership,
remaining prompt inputs, passthrough owners and original-deadline offsets without
prompt text. Existing release/EOF/fatal cleanup stays unchanged. Prior participants
still in Prefill bypass later barriers; current held-ready members never bypass.

This is a source/CPU handoff for root's matched serving test, not a measured
throughput or SLO improvement. Deliberate holding remains bounded by the fixed
receipt deadline plus existing atomic-work granularity. Old decoders retain
normal eligibility and credit accounting; actual TPOT/gaps require HTTP evidence.

Peer review found that the old checkpoint test removed a decoder positioned after B, so its stated compaction was not exercised. Test-only `1a1b05f` places the old decoder before B and explicitly asserts B index1→0 while preserving the saved ring/range and subsequent completion checks. Full unfiltered `a004-recovery-compaction-test` passes118/118 (zero ignored/filtered). Production source is unchanged; this does not alter the frozen policy candidate.
