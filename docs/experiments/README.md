# Experiment ledger

All rows in the main performance table use the unchanged official three-kernel, three-input harness.
An Arena result is a development measurement; official ranking requires `moa-submitter` evaluation.
Baseline checkpoint: workspace commit `2890176`, official kernel source `8559916`.
E31 is a repeated checkpoint: all 15 official-harness comparisons and 30 separate
additional-input comparisons pass. Three independent same-package E31 jobs also pass; their median-of-medians is97085/45355/389335.
E36 has passed30 additional comparisons. E37 is the latest passing combined screen (94094/45796/376040); its additional-input and repeated-package checks are pending.
E29 remains a passing earlier screen; E30 is a passing paired QKV control. E28/E32 timeout,
and E33 has substantial FFN correctness failures. Diagnostic jobs are separated below.

| Variant | Arena jobs | Accuracy | QKV cycles | O cycles | FFN cycles | Decision |
|---|---|---|---:|---:|---:|---|
| Original | 69783, 69913, 69931 | Pass | 244051 | 404222 | 4264297 | Baseline; medians across jobs |
| E5 direct-layout, two-row commits | 69992 | FFN fail | 244661 | 402884 | 1249479 | Rejected; speed is not a valid optimization result |
| E4 two-row only | 70000 | FFN fail | 244172 | 404030 | 3241077 | Rejected; isolate common two-row path |
| E2 FP8 lookup/contraction fusion | 70009 | Pass | 227840 | 334457 | 4265795 | Retain for subsequent combinations; single-job screen |
| E3 256-slice output + QKV fusion | 70014 | Pass | 228450 | 195210 | 4263461 | Retain; output 2.071× baseline, single-job screen |
| E3 direct HBM replication + E3 rows + E2 QKV | 70022 | Pass | 227338 | 143976 | 4267115 | Retain; single-job screen |
| E5 four-row direct-layout + preceding projections | 70029 | Pass | 227578 | 154254 | 1139366 | Retain; completed package repeats in next row |
| E5 same package repeated | 70029, 70038, 70046 | Pass | 227916 | 146508 | 1139366 | Medians across 3 jobs; FFN range 0.063%, O 6.600% |
| E1 QKV HBM relay + preceding optimizations | 70039 | Pass | 188573 | 145998 | 1140479 | Improves QKV; hierarchical copy candidate next |
| E1 hierarchical QKV/O replication | 70055 | Pass | 166416 | 116132 | 1139255 | Retain; single-job screen |
| E6 MLP and residual loop expansion | 70070 | Pass | 165134 | 113136 | 820226 | Retain; FFN 5.199× original, single-job screen |
| E7 two-cluster QKV + FFN hierarchical input | 70074 | Pass | 133369 | 112661 | 828972 | QKV improves; FFN broadcast is 1.07% slower than previous screen |
| E8 ring32 QKV + two-cluster O + blockscale up/gate | 70107 | Pass | 129964 | 76147 | 921693 | Retain QKV/O; blockscale FFN is 12.37% slower than E6 |
| E9 replicated QKV norm + parallel O residual + two-cluster up/gate | 70118 | Pass | 130496 | 69651 | 657198 | O and FFN improve; QKV change is within observed variance |
| E10 all FFN matrices on two clusters | 70135 | Pass | 126444 | 68813 | 479755 | Retain; FFN 8.889× original, 27% faster than E9 |
| E11 up/gate8 and down12 row tiles | 70149 | Pass | 125967 | 68677 | 448221 | Retain; FFN 6.57% faster than E10 |
| E12 blockscale up/gate with reused unscaled BF16 weights | 70157 | Pass | 125619 | 68945 | 436380 | Retain for further tests; FP32 sum order changes |
| E13 fused FP4 LUT→F32→BF16 materialization | 70162 | FFN fail/timeout | — | — | — | Rejected; run0 only30.96% within tolerance, run1 launch timeout |
| E14 head-parallel QKV norm and eight-column O | 70174 | Pass | 118881 | 65694 | 436791 | Retain QKV/O, all official input comparisons pass |
| E15 blockscale down with two-stage decode | 70181 | Pass | 115780 | 65476 | 397675 | Retain; extra run3..8 also pass in job70186; repeats tracked separately |
| E16 parallel FFN residual and layer scale | 70204 | Pass | 117637 | 65217 | 388813 | Retain; FFN 2.23% faster than E15 screen |
| E17 two-column down with HBM relay + E16 tail | 70213 | Pass | 116762 | 66323 | 416500 | Reject; FFN 7.12% slower than E16 |
| E18 QKV eight-way row interleave + E16 | 70248 | Pass | 105502 | 65139 | 389781 | Retain; QKV 8.88% below E15 repeat median, repeats below |
| E19 QKV runtime FP8 hi/lo, two independent dots | 70261 | Pass | 110479 | 66086 | 388705 | QKV slower than E18; prototype for two-lane native FP8 |
| E20 QKV native FP8 two-lane + fused O norm | 70290 | Q/K pass, V fail | — | 65919 | 391667 | Reject QKV; V sparse large errors, root cause under investigation. O/FFN individually pass |
| E21 row-interleaved FP8 QKV + native FP8 O | 70300 | Q/K pass, V fail; O/FFN pass | — | 59016 | 389251 | QKV rejected; O retained for standalone-valid package and extra inputs |
| E22 safe QKV + native FP8 up/gate + native O | 70321 | QKV/O pass; FFN launch timeout | 103681 | 58903 | — | Reject native FP8/F32-block FFN path |
| E23 KV register reload + streamed BF16 scale contraction FFN | 70327 | V fail; O/FFN pass | — | 59023 | 384728 | KV reload does not fix failure; FFN passes with extra BF16 block rounding |
| E24 native FP8 O with verified E18 QKV/FFN | 70339 | Pass | 105045 | 59027 | 386021 | Retain; additional30 comparisons pass in70353 |
| E25 native FP8 FFN + BF16 block/scale DPE | 70354 | QKV/O pass; FFN launch timeout | 105623 | 58638 | — | Reject pending common-path failure diagnosis |
| E26 O channel scale/RMS/residual on eight slices | 70361 | Pass | 104895 | 52579 | 389949 | Retain; O10.93% below E24 screen, additional30 comparisons pass70381 |
| E27 Q/K two-lane, V separate hi/lo dots + E26 O | 70382 | Pass | 97247 | 52543 | 386729 | Extended-check checkpoint; extra30 pass70409; V workaround does not establish root cause |
| E28 explicit ×1 on pair group0, native FP8 FFN | 70410 | QKV/O pass; first FFN launch timeout | 98428 | 53363 | — | Reject; exit101/rank0, no FFN output or cycles; no valid combined score |
| E29 sharded QKV input + direct-packed O + safe FFN | 70428 | Pass | 94491 | 49352 | 389199 | Latest passing screen; QKV/O2.83%/6.07% below E27 screen; extra inputs and repeats outstanding |
| E30 all-pair skip control + safe FFN | 70440 | Pass | 98599 | 52113 | 389137 | Direct-packed QKV passes without explicit ×1; identity is not required for these inputs |
| E31 Switch gather/local RMS sum + E29 | 70441 | Pass | 98797 | 45355 | 389335 | O improves; QKV screen slower, repeat required; extra30 pass70448 |
| E31 same package repeated | 70441, 70459, 70465 | Pass | 97085 | 45355 | 389335 | Four-file package hashes identical; job-median ranges2.714%/1.965%/0.177% |
| E32 pair-free native FP8 FFN diagnostic | 70447 | QKV/O pass; first FFN launch timeout | 99692 | 52233 | — | Reject; removing Sequential pair does not resolve common native packet4 path |
| E33 BF16 down block/scale DPE + E31 | 70455 | QKV/O pass; FFN fail | 96585 | 45738 | — | Reject; only12.94–16.17% FFN values within tolerance; causes unresolved |
| E34 O row2 interleave + E31 packed/parallel | 70472 | Pass | 97092 | 47308 | 389291 | Reject for performance; O4.31% above E31 repeat median |
| E35 native FP8 full256 live rows + real zero tails | 70485 | Pass | 97301 | 45799 | 811219 | Diagnostic PASS; slower, establishes packet4+pair can execute in this configuration |
| E36 FFN H480×8 epilogue + E31 | 70498 | Pass | 97052 | 45141 | 378021 | Retain; FFN2.91% below E31 repeat median; extra30 pass70506 |
| E37 kernel-specific RMS + E36 FFN epilogue | 70519 | Pass | 94094 | 45796 | 376040 | Latest combined screen; each device image identical to its passing parent; repeats/extra pending |

Cycles shown for passing kernels inside a failed package are diagnostic observations from the raw log.
They do not constitute a valid combined result and must not be mixed with other jobs' FFN cycles.
No same-package repeat has yet been completed for E27 or E29; their JSON `relative_range=0` reflects one job,
not demonstrated zero variance. The older public-reference `indicative_score` of 5.748681 is an estimate,
not an official score or current rank. See [E27 measurement](e27-qk-pair-v-two-dot.json),
[E28 failure](e28-explicit-pair-identity.json), and [E29 measurement](e29-sharded-qkv-packed-output.json).

The failed E4 and E5 experiments have identical reported FFN max/mean errors and tolerance pass rates
for all three inputs. Their common two-row contraction/reduction/commit/copy path is the current
suspect; the additional column distribution is not isolated as the failure cause. This is a
working diagnosis, not a proof. The four-row candidate restored the original output path while
retaining direct column placement and passed all official input comparisons. The precise mechanism
of the two-row failure has not been established. O cycles differed by about 7% between the two
last jobs even though its source was unchanged; the identical package repetitions also show a 6.600% range for O, versus 0.225%
for QKV and 0.063% for FFN. Small O improvements require repeated comparison.

`*-provenance.json` files record sources and artifact hashes. `*.json` measurement records retain
job metadata, the actual submitted package hashes, validity and, for passing runs, cycle aggregates.
Raw logs, binary snapshots and large schedules are under Git-ignored `moa-dev/logs` and
`moa-dev/experiments/artifacts`. Failed candidates and compile-time comparisons are preserved under
`moa-dev/experiments/candidates`; static schedule predictions do not establish correctness or speedup.

The `capture.py` helper snapshots current source and an existing binary independently. The root
experiment sequence builds successfully and captures immediately. It either submits before the next
edit or freezes an upload package from that snapshot, verifies its hashes, and submits the frozen files.
The uploaded package is downloaded and checked against the captured binary/fixture after each job.
The helper itself does not claim to prove that a binary was built from the current source.

Additional development seeds run3..8 passed all 30 output comparisons for the E5 four-row
snapshot (job70057). Binary, fixture and entrypoint matched the submitted package. This result
is in `e5-four-row-extra-correctness.json` and is separate from the official three-input score.

E10 also passed all30 additional run3..8 comparisons in job70150 (`e10-all-two-cluster-extra-correctness.json`). This predates the blockscale summation-order change and does not validate E12 or later candidates.

E15 all-blockscale passed all30 additional run3..8 comparisons in job70186 (`e15-all-blockscale-extra-correctness.json`). Package hashes matched the independent build. This includes QKV head normalization, O eight-column FP32 reduction and all FFN blockscale operations.

E15 same-package repeats70181/70194/70200 allpassed; aggregate medians115780/65476/397675. Relative job-median ranges: QKV4.249%, O2.500%, FFN0.266%. All four uploaded package file hashes matched. See `e15-best-repeats.json`.

E18 same-package repeats70248/70271/70285 all passed; aggregate medians105462/65662/392608. Relative job-median ranges: QKV1.599%, O1.188%, FFN0.792%. All four uploaded file hashes matched. See `e18-best-repeats.json`. Compared with original repeat medians, speedups are2.314×/6.156×/10.861×.

E18 also passed all30 additional run3..8 output comparisons in job70309 (`e18-row-interleaved-extra-correctness.json`). This validates the E18 row layout and E16 residual changes on six additional configurations; it does not validate subsequent FP8 prototypes.

E24 additional run3..8 passed all30 comparisons in70353 (`e24-native-output-extra-correctness.json`), covering column-local FP8 input decomposition on six additional configurations. This does not establish all-input tolerance; an explicit outlier/cancellation counterexample is recorded in the independent candidate review.

E26 additional run3..8 passed all30 comparisons in70381 (`e26-output-parallel-extra-correctness.json`).

E27 additional run3..8 passed all30 comparisons in job70409
([record](e27-qkv-fp8-extra-correctness.json)). Its candidate reference points to the frozen E27 source
and official binary SHA256 `c87d0e77c181610407f2b456f35679700df2eaf88524e272ed29dcddd210e96a`.
The extra harness has its own binary/fixture and verified uploaded package; its cycles are diagnostic
only and are not aggregated with job70382. Nine tested configurations do not prove all-input tolerance,
and this result does not validate E28 or E29.

The instrumented QKV diagnosis ran as job70363
([job record](qkv-fp8-two-lane-diagnostics.json),
[coordinate analysis](../../moa-dev/experiments/candidates/qkv-fp8-two-lane-diagnostics/analysis-job-70363.md)).
It preserved the E20 kernel binaries, input generator and comparisons while adding host output of V
coordinates/values. Q/K passed and V failed; the largest direct errors concentrate at local row0 in
each cluster's first eight canonical slices. Other affected coordinates are consistent with the
changed head RMS scale. This narrows the diagnosis but does not prove an allocator, producer or pair
arithmetic bug. Instrumented performance is excluded from the main table.

Scratch-only candidates also remain separate from runtime evidence: O row interleave
[2-way](../../moa-dev/experiments/candidates/output-fp8-row-interleave-two/README.md) and
[4-way](../../moa-dev/experiments/candidates/output-fp8-row-interleave-four/README.md) compile at
33,475 / 34,435 static cycles versus the same-source E24 control's 32,252. DMA estimates are unchanged;
Switch overhead increases. [BF16 FFN row interleave](../../moa-dev/experiments/candidates/ffn-bf16-row-interleave/README.md)
also compiles but regresses 8.31% against its own E15-core static control. None of these comparisons
establishes NPU correctness or runtime performance.

The [Arena version probe](../../moa-dev/experiments/candidates/arena-version-probe/README.md) is an
environment observation, not a kernel benchmark. Driver/firmware query failures must remain unknown;
they do not prove a mismatch or explain the FP8 failures. Follow-up measurement priorities are tracked
in [the optimization plan](../OPTIMIZATION_PLAN.md#초기-실험-계보와-현재-우선순위).

E31 additional run3..8 passed all30 comparisons in job70448 ([record](e31-rms-extra-correctness.json)).
The eight-way RMS sum order changes; passing nine configurations is not an all-input proof.
E30 has a clean paired control audit: the E28/E30 QKV binaries differ in exactly24 bytes,
while their instruction fields, allocation, O kernel and QKV entry point match. Both QKV paths pass.
See [control audit](../../moa-dev/experiments/candidates/qkv-fp8-explicit-hi-identity/control-audit.md).
Direct packing also changed between earlier failing E21 and E28/E30, so a compiler defect or its
precise mechanism remains unproven. E33 FFN runtime408287 is invalid performance because accuracy failed.

E31 same-package repeat evidence: [e31-rms-repeats.json](e31-rms-repeats.json). Original-repeat speedups are2.514×/8.912×/10.953×. These are Arena cycles, not official ranking.

E36 additional run3..8 passed all30 comparisons in job70506 ([record](e36-ffn-epilogue-extra-correctness.json)).
Its QKV and O device binaries match E31; the changed FFN retains existing BF16 boundaries and uses
H480×8 for down output/global scale/postnorm/residual/store. It has not yet completed package repeats.

E29 control repeats70428/70482/70493 allpass: median94491/49228/388311, ranges1.812%/1.479%/0.476%.
This motivates using its byte-identical QKV image in E37 while keeping E36 output/FFN.

E31 cache-preservation job70507 passes15 original outputs plus12 sentinel checks. All12,570,624
non-target cache elements checked after launch match exactly; upload binary/fixture/runner hashes match.
[Record](e31-cache-preservation-correctness.json). This custom harness is diagnostic and establishes
preservation for E31's device code and those3target slots, not arbitrary input or later QKV binaries.
