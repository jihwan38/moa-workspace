# HOLD: fixed-E4 native QK does not establish the required 10% complete gain

This source/actual-trace screen uses the fixed-unscaled E4 complete B16 graph
`4bf039b2`, measured by the unchanged micro in `b355d6d8`. The measured control
for this proposal is **201.287µs** append + E4 attention + supplied-weight O,
not the 238.778µs original BF16-cache graph. No resource build, kernel change,
host port, model gate or Arena dispatch was performed for this follow-up.

The proposed operation is legal in principle: preserve fixed-E4 K in native E4
through the shared R4 K-TRF mapping, dynamically quantize each query/head to E4,
multiply each native F32 score by its query scale before the BF16 score boundary,
then retain the current softmax, BF16 V/PV and O. It introduces query precision
and F32 contraction-association changes, so would require independent numerical
and task gates. Fixed key scale1 removes the old per-key scale gather/VRF/multiply
path; it does not remove query quantization or query rescaling.

## Current complete-path budget

Actual r1289/321461 and r1290/321462 cover four cases, both clusters and two B8
groups per cluster. Mean complete Task is386,793.625 raw cycles. Source/resource
topology and the repeated span signatures identify:

| Work per complete cluster | Mean actual cycles | Fraction of Task |
| --- | ---: | ---: |
| Two K E4→BF16 decode passes | 23,179 | 5.99% |
| Entire two BF16 QK contractions | 2,306 | 0.60% |
| Two K lookup-table HBM transfers | 4,197.25 | 1.09% |
| K table-load portion outside K gather | 352.75 | 0.09% |

Crediting **all** of those costs as removed gives only7.77% of the current Task
(case/cluster range7.66–7.96%), before any native replacement contraction,
query amax/scale/cast, query-scale load or score multiplication. Some credited
work already overlaps other operations, so this deliberately overcredits the
bounded removal. This is a screening budget, **not** an additive prediction of
wall time or a universal performance upper bound.

The much larger K→TRF spans cannot simply be added to that budget: each of all16
observed K→TRF spans is entirely overlapped by V gather and finishes1,911–3,781
cycles before V gather finishes. K table loads overlap K gather by89.3–98.9%.
Native K has half the TRF byte footprint (32KiB versus64KiB per slice), but
shortening an already-overlapped load is not automatically a critical-path win.
There is no separate K TU transpose in this current reader; K uses a swizzled
Sub fetch/collect→TRF. The explicit large V transpose remains unchanged.

Changing instruction order could affect FIFO/resource availability, so these
observations do not prove no possible rescheduling gain. There is currently no
source-backed scheduling mechanism establishing the extra gain needed to cross
10%, and the new query work has a positive cost. Under the predeclared gate,
stop here rather than compile another speculative attention variant.

`analyze.py` reads preserved logs and the existing same-graph resource schedule,
asserts the46-TU/four-codec/two-K-TRF signatures, records all matched intervals,
and emits `result.json`. Actual spans carry no source IDs; attribution uses
matching source/resource structure and unique repeated signatures, **not** a
DMA ordinal copied between unrelated compiler graphs. Raw cycles are not
converted to nanoseconds or a marketing clock rate.

## Prior experiments retained in the decision

- `native-attention-qk/experiments/native-attention-qk/r776-r779-results.json`:
  old scaled-cache R64 complete attention including query codec was approximately
  617µs reconstructed E4 versus551–577µs native QK. Conservative after-other-path
  improvement was6.7–7.3%; append/metadata were excluded. This is not a B16
  complete-chain or task-quality gate.
- `r781-results.json`: quantizing before query broadcast was rejected.
  Candidate595–597µs versus574–577µs added16 DMA operations; DMA union increased
  848k→919k while TU union changed538k→533k. Removing logically repeated codec
  work did not repay redistribution. The proven shared-R4 native K-TRF mapping
  remains a legality precedent, not a speed prediction here.
- `p128-native-qk-bound/experiments/p128-native-qk-bound/result.md`: complete
  fused P128 resource123,624→123,627 cycles, no DramStack. Its QK4,364→2,316
  reduction was offset by Q/K casts and downstream work. That experiment still
  read BF16 K, so the fixed-E4 cache is a real difference; nevertheless, it rules
  out treating a native contraction name or faster isolated QK as a complete win.

Relevant source: `ops_b16_fixed_e4_reader.rs:166` K decoding, `:173` K TRF,
`:179` QK, `:219` V decoding and `:225` unchanged V transpose;
`ops_request_sliding_r64_native_qk.rs:597` query quantization and `:847` native
shared-R4 scores. No additional source or artifact was altered.

Reproduce (from this worktree):

```bash
python3 experiments/fixed-e4-native-qk-audit/analyze.py > /tmp/fixed-e4-native-qk-audit.json
```

All fixed-E4 model/public quality gates remain separate and pending. This HOLD
neither adopts fixed-E4 cache storage nor weakens its existing quality guards.
