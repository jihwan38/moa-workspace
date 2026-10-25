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
