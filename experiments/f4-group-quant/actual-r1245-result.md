# Supplied-weight scale fitting: PASS, weight-space result only

Root job321289 / r1245 completed in6.561s using frozen source8e7ce99e. All12
UP/GATE/DOWN projections in layers0/5/6/47 were processed:707,788,800 current E4
values across737,280 groups. Every group's fitted SSE was no worse;737,277
groups improved strictly. Independent per-value F64 decoders checked737,280
values across768 complete sampled groups, including block boundaries.

Across the12 projections, ceil-max F4 relative L2 against current E4 was
0.14087–0.18328; fitted scales reduced it to0.10270–0.12202. With the unchanged
row dequantizer included, fitted L2 was0.10303–0.12931. Complete per-projection
metrics are retained in `actual-r1245.json`; these ranges are not a pooled L2.

Actual host D2H took59.74–63.12ms per59M-value matrix. Packed-only ceil-max
encoding took16.85–17.53ms, versus111.01–124.73ms for fitted encoding plus its
histogram error report. Timers are separate. The initial model load was4.018s;
no new F4 upload was performed. Process VmHWM9,140,628KiB includes that load;
live RSS after each CPU projection was141,140KiB. Neither is an isolated encoder
peak, unlike the separate standalone benchmark.

This proves the static encoder and supplied current-E4 weight SSE property.
Original NVFP4 reconstruction error, F4 device arithmetic with the new dyadic
scales, activation/residual drift, selections and task quality remain separate.
The unchanged complete B16 graph is being gated in `b16-f4-mse-micro`; nothing
is enabled in serving by this evidence.
