# Current P512 ceiling audit: no supported new30% mechanism established

This read-only audit does **not** justify another kernel build or geometry sweep.
The current bottleneck is useful dense work, not substantial prompt padding.
I found no untried, supported SDK0.9 mechanism with a defensible30% completed
P512 reduction. The remaining high-ceiling arithmetic route requires a public
full-rate INT4 contraction that has not been demonstrated; treating its hardware
peak as available software would repeat an already falsified assumption.

## Actual current cost and packing

`bounds.json` binds the source and log hashes. The physical32 whole-model gate
`r1212`, warm single-owner512 route, reports118.286ms including preparation/final
logits and excluding diagnostic downloads/interleaves/cache comparisons:

| Completed stage | ms | Fraction of whole chunk |
|---|---:|---:|
| Composed FFN including post-attention/pre-FFN/post-FFN norms and quantization |52.983|44.79%|
| Sliding headed attention and commit |20.321|17.18%|
| Four output projections |13.761|11.63%|
| Four QKV/QK projections |12.580|10.64%|
| QKV norm/quantization |3.652|3.09%|
| Full attention |3.932|3.32%|
| Full heads plus commits |2.818|2.38%|
| Final vocabulary projection |2.176|1.84%|

The remainder includes preparation and work not covered by these stage timers.
It is not evidence of idle time. Start positions/page counts affect attention;
this one model gate is a concrete current-math example, not a workload average.

Actual matched HTTP `r1302` c081 versus `r1301` q020 has the same measured96
requests/87,808 prompt tokens/6,144 outputs after a real16-request warmup. All
prompt quanta are cost4; their48-layer counts reconcile exactly with chunk
counts. Warm-subtracted totals are:

| Candidate | Dense512 chunks | Computed rows | Useful prompt tokens | Extra rows |
|---|---:|---:|---:|---:|
| c081 |175|89,600|87,808|1,792 (2.00%)|
| q020 |173|88,576|87,808|768 (0.87%)|

These logs do not enable `SCHED_CHUNK`, so they do not identify which individual
chunks were256/384 tails. They do establish the aggregate padding ceiling. The
`p1/p2/p4` fields count real owners, not real quarters; they must not be used to
infer padding. `prepare_tail256`/`prepare_tail384` in
`src/host/prefill_group.rs` initialize tensor-only padding, then `advance_rows`
still executes four-row norm/QKV/O/FFN. The actual useful-row count comes from
request usage, independent of owner count. Even eliminating every padded row
cannot supply a30% reduction in this workload.

The first16-request wave contains14,080 prompt tokens and28 dense chunks. At
the observed2.33ms/layer this accounts for3,131.52ms recurring work, before
roughly230ms preparation and44ms finalization. The current admission/cohort
limit is therefore supported by actual work, not a missing large batch.
Measured decoder rows reconcile to96×63=6,048 in both runs. Decoder calls are
381(c081) and379(q020), already close to the lower bound378 at width16.

## Arithmetic bound rejects a30% FFN-only promise

The code uses H3840, L15360,48 layers and three dense FFN matrices. Their512-row
work is `2*512*3840*15360*3*48 = 8.6973087744e12` floating-point operations.
At the advertised full-chip512FP8TFLOPS, the multiplication-only lower bound is
16.987ms. This is a peak specification, not a measurement of Arena clock or
utilization. [Official RNGD specifications](https://developer.furiosa.ai/latest/en/overview/rngd.html)
give512FP8TFLOPS,256BF16TFLOPS,1024INT4TOPS and1.5TB/s HBM at1GHz.

Changing only FFN would need to reduce its entire52.983ms composed chain to
**17.497ms** to save30% of the118.286ms chunk. That leaves only0.510ms beyond the
ideal dense multiplication floor for all48 layers' normalization, GELU, global
and row scales, reductions, quantization, output movement and waits. A merely
2× completed FFN improvement saves22.40% of this chunk. No current source or
trace establishes the almost-peak complete graph needed for30%.

The current one-launch composition is already explicit in
`ops_group_ffn_projection_storegroup::{sliding_projection_owned,full_projection_owned}`:
post-attention/norm/quant stage → projection-owned FFN → blocked residual tail.
`ops_ffn_projection_clusters::stage_upgate` retains one UP or GATE weight epoch
per projection-owned cluster and streams all512 rows through8 P64 tiles. Merely
renaming this as whole-FFN fusion or weight reuse does not remove new work.

## Closed branches remain closed

| Mechanism | Actual evidence or resource limit | Consequence |
|---|---|---|
| Current projection-owned P32 | r1182 exact complete chain,1.06–1.07→1.18–1.21ms. Concurrent refills lengthen; UP/GATE envelope explains almost all Task regression. | Current-owner graph was actually tested; favorable static estimate did not transfer. |
| P32 cache order L30/L4 | Isolated fixed resource gates rejected; physical contiguous weight repair did not remove dot cost. | No accumulator/cache-shape sweep. |
| Full-H L32 weight-stationary UP | r1089 all6 exact projection gates;~0.298→0.504ms including required output join. | A proven arithmetic/layout identity is not a serving win. |
| Native L120/shared/column consumer | r637 and r907 complete FFN regressions; tiny producer savings lost in consumer/TU movement. | No rebranding the same contiguous-L120 idea. |
| K240→K256/P256 dual producer | r799 intended arithmetic passed;1.264–1.266→1.405–1.408ms complete. | Column consumer/movement cost remains counted. |
| P512 F4 inline lookup | r1225–1228 complete1.06116→0.98010ms (7.34–7.81%); residual drift17.54–20.98%. | If fully transferred, only~3.42% whole-chunk reduction here; no current host adoption. |
| P512 F4 one-time DM decode | b6031192 resource HOLD:954729 versus inline917695, despite fewer table loads; no DramStack. | Already tried, not a new large-gain proposal. |
| Four-P128 attention graph | Runtime alias hazard at ring wrap, plus no completed latency gain. | Sequential Rust mutable calls did not enforce compiled old-ring read ordering. |
| Mixed112+16 | Exact model/selection primitive~102ms; fails equal-progress break-even versus current512 route and loses128 alignment. | Does not improve decoder-free startup work; no Worker port. |

Sources for the individual bounds are the preserved local reports:
`ffn-projection-p32/experiments/ffn-projection-p32/actual-decision.md`,
`ffn-trf-cache/experiments/ffn-trf-cache/decision.md`,
`ffn-up-weight-stationary/experiments/ffn-up-weight-stationary/actual-r1089.json`,
`p512-f4-partialscale-micro/experiments/p512-f4-partialscale-micro/actual-result.md`,
`p512-f4-predecode/experiments/p512-f4-predecode/result.md`, and
`mixed-prefill-decode/experiments/mixed-prefill-decode/model-and-serving-cost.md`.

## Two remaining directions, with their real limits

1. **Full-rate low-bit arithmetic is the only identified hardware ceiling that
   materially changes the above bound.** A4W4 at the advertised1024TOPS halves
   the ideal FFN arithmetic floor, leaving room for a30% complete-chunk target.
   However public SDK0.9 has no demonstrated full-rate legal path in our audit:
   the exact paddedK128 precedent fails `RawInt5/Flit2`,16 TRF lanes are illegal,
   and legalK64 actual dot/reduction time is unchanged from FP8. Public raw I32
   buffer binding changes packing, not that engine configuration. This remains
   a **software-legality HOLD**, not a claim that hardware cannot do it. The
   prerequisite is a concrete supported public compiled example or new backend
   evidence, followed by an intended integer oracle, inclusive quantization and
   actual supplied-model quality. No repeat compilation or private API escape is
   justified. The pinned primary-source proof is in
   `physical-pool32-export/experiments/int4-full-rate-audit/decision.md`.

2. **Real compact256 dense kernels can remove wasted work on a particular
   Tail256 chunk**, retaining its existing two-state ownership and current
   arithmetic. Half of its four-row dense work is padding, so a local30% saving
   has a plausible ceiling. But current measured aggregate padding is0.87–2%,
   and actual smaller-shape matrix efficiency is not established. This is a
   bounded tail optimization, not a route to30% P512 or severalfold throughput;
   no resource build is recommended for the present strategic target.

Larger token cohorts can amortize remaining weight delivery, but cannot erase
the dense FLOPs, and the current projection-owned code already removes the
duplicate UP/GATE weight epoch. A broad P1024/P2048 rewrite without a measured
new critical-path saving would not meet this audit's threshold. No benchmark
routing, omitted model layers, cached cross-request computation, new source
kernel, or Arena submission was introduced.
