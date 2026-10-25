Base c53ae8a, whose Q4 postattention normalizer passed micro318898.
Production control is ops_norm_fused::post_feedforward_four: four sequential
quarter reads/normalizations/residual adds/gates inside one device launch.
Candidate reads the complete existing P512/H240 blocked output directly into
DM with P512/64%2 clusters, P512/2%32,H/480 slices, P512/128,P512%2,H%480 elements.
After real named-axis redistribution, factor-only relabel maps P512=128*Q4+P128.
Original H480 RMS arithmetic and BF16 boundaries remain, Q4 is never reduced.
Residual VRF handling reuses the validated two-quarter7680-byte tiles. All four
independently owned HBM residuals are written before the launch completes.

Expected removed work: four separate blocked-HBM ingress schedules and repeated
RMS/gate pipelines become one distributed ingress and Q4-time streams. No cache,
request grouping, model arithmetic, host integration or production defaults change.
Gate: all512 rows exact against current fused control and four original P128
controls; CPU physical packing/independent F64 residual oracle; zero/tiny/random,
quarter permutations and positive/negative/zero gates. Complete launch ABBA with
residual reset outside timing, actual NPU spans, root-only dispatch. Compile does
not establish correctness or speed. Reject unless actual inclusive timing wins.
