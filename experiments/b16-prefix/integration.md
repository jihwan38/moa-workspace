# Exact B16 prefix activation

Base: 6b55cb0 (R32 automatic17..32 + packed tails + B16 epilogue).
The selector defaults true only inside the existing canonical+blocked guard.
`set_b16_prefix_fused(false)` keeps the original four-call diagnostic control.
No Dispatch, R32, Worker, generation, admission, KV or attention code changes.

Source evidence: micro319330 compared4,399,488 stage bytes exactly; model319380
compared21,757,952 logits and18 complete cache pairs, with0 mismatches.
The model included48 active prompt quanta,512-token histories,17->2 regrouping,
position0 reset,16+15 canonical narrow scratch reuse, unblocked output fallback,
and re-entry into the fused prefix. Both arms enabled the same B16 epilogue.

Warmed sequential model comparisons at widths2/8/10/16 saved2.247–3.260ms
(mean2.639ms); these are diagnostic timings, not HTTP/SLO evidence.
Scalar width1 and unchanged R64 width17 timing differences are excluded.

Integration audit: R32 capacity32 is local to each Decode64 call and only
Automatic or explicit Decode32Diagnostic enables it. New prefix state belongs
to Prefill and does not change either route. Upper32 stale R64 rows are never
requested at width<=32 and are overwritten before subsequent wider use.
Tail384/ThreeOwner512 retain their own row/residual state; active_allocations
protects every real ring/page before either decode route. Worker retained ranges
remain keyed by ring through session-vector removals.

The component model gate predates the combined R32+packed-tail tree; combined
HTTP/active-tail qualification remains the root's separate candidate gate.
