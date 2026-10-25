# Exact B16 device greedy prototype

Base30038f5. The kernel preserves the proven R64 integer ordering, ban words,
lowest-ID tie reduction and CPU fallback for nonfinite or zero/subnormal maxima.
Only placement changes: canonical BF16[B,W], B/8 clusters, eight local query rows;
no outer R64 sixteen-row group loop. Ban words are[W/32,B], results[B,8].

The probe runs all64 original special cases as four disjoint groups of16, with
independent total_cmp CPU oracles and real row extraction/read_logits/sample on
fallback. It covers all finite BF16 ordering patterns, signedzero/subnormal/NaN/
infinity, ties, arbitrary masks, all-banned fallback and extreme vocabulary IDs.
Inclusive timing measures ban packing/upload, kernel and512-byte result read at
live8and16 against actual existing read_logits+sample. Baseline row extraction
is performed before timing, so this does not overstate savings from removing it.
No serving or ForwardOutput changes before actual microkernel gates.


Actual micro gate: Arena 318780, runs/r610-b16-greedy-submit/arena.log (root
collected): all64 cases including real CPU fallback passed. Warm inclusive C8
selector0.449–0.481ms versus CPU8.07–8.19ms; an early device result read produced a
2.64ms outlier. C16 selector0.415–0.509ms versus CPU15.75–16.03ms. This is a
sampling-boundary measurement, not a full-server throughput result.

Host integration retains native B16 logits only for pure independent singleton
batches of2..16 with requested logits. C1, mixed prompt/decode and the ordinary
forward_sequences API keep scalar rows. The common selector's IEEE fallback
policy is unchanged; stochastic and exceptional rows are extracted individually.
No persistent mode or new model-weight/cache ownership is introduced.

The model probe compares materialized versus retained full vocabulary bits at
widths2/8/9/16, normal and suppressed winners, seeded stochastic sampling and
unrequested rows. It also verifies C1, mixed prompt/singleton output ownership,
active narrow-prefill interleaving and active-cache alias rejection. A passed
microkernel is not sufficient evidence for these host contracts.

Host validation: a035 passed all63 tests; cargo check --bins passed. The bounded
independent host audit found no retained-buffer, output-order or policy defect.
Actual-model verification is still pending.
