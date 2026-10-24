The c014 HTTP runs exposed a scheduling cost beyond kernel time: 256-token C4
public TTFT p90 was7243.424ms (job318037), while2048-token C2 was5567.471ms
(job318038). Neither meets the official4s TTFT gate, although both test harnesses
reported successful requests. The scheduler alternated each complete prompt
layer with a complete decoder batch, so one128-token prompt chunk could wait
through48 full decoder rounds. This diagnosis follows select_work_in and the
measured2–5ms prompt layer versus30ms+ decoder round; it remains an E2E
hypothesis until the changed HTTP run completes.

DecodeCadence now records elapsed complete decode work including host transfer,
sampling, and stream-event sends. For the same ready width, it allows extra
complete prompt layers while elapsed time since the prior round plus the observed
prompt quantum and observed next decode cost fit90ms. The remaining10ms is a
measurement/dispatch margin relative to the100ms requirement, not a guarantee.
Unseen widths run before using an estimate. Estimates take the maximum of the
latest cost and a90%-decayed previous peak. Existing alternation always grants
one prompt layer after a decode, preventing starvation when compute alone is
already over budget. Stable-ring chunk rotation and all-ready decode batching
are unchanged. No request text, benchmark identity or performance flag affects
this scheduling decision.

All23 host tests pass after the change, including deadline arithmetic, width
changes, costly prompt quanta, high canonical slots, active-prompt ownership,
round-robin removal, and Generation transitions. Exact interleaving proofs are
separate NPU evidence. Official throughput and latency are still unverified.
