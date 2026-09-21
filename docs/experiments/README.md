# Experiment ledger

All rows in the main performance table use the unchanged official three-kernel, three-input harness.
An Arena result is a development measurement; official ranking requires `moa-submitter` evaluation.
Baseline checkpoint: workspace commit `2890176`, official kernel source `8559916`.
Current adopted source: E55, which combines the BF16 QKV dot and F32 channel multiply in one Main
pipeline before the first BF16 cast and row Switch. Its three same-package jobs71027/71057/71072
each pass all15 official-harness comparisons; the median-of-job-medians is102878/51249/380339.
The first screen remains a separate historical row. E54 is the previous correctness checkpoint.
E55's 30 extra-input comparisons and two QKV boundary cases also pass.
O/FFN device images are unchanged from E54. The earlier E53 O-boundary job tests that identical O
image; it is inherited evidence, not a newly executed E55 O-boundary suite.
The [official CLI local preflight](e55-official-submit-preflight.json) captured three
byte-identical uploads containing the exact23 allowed E55 source files. It tested
packaging against a loopback receiver; no official submission, authentication,
server-contract check, or new NPU measurement was performed.
The [model-host build](e55-model-host-build.json) separately compiled `gemma4`
and `server` and verified their14-entry ELF kernel registries. Both include the
exact three measured E55 images. No model assets were loaded and no end-to-end
accuracy or throughput test was run. The [interleaved K/V candidate](../../moa-dev/experiments/candidates/qkv-bf16-interleaved-kv-e55/README.md)
failed its first exact compile at an axis-identity check; it has no valid image,
schedule, or NPU result and does not replace E55.
The later [Arena asset probe](e55-arena-model-assets.json) found the actual model,
tokenizer, and chat template in the configured third loader candidate. The
[standalone Rust identity job71224](e55-arena-model-identity.json) completed full
weight/metadata hashes and all1149 required header contracts; its partial status
reflects only the missing optional JSON template. The
[unmodified E55 model smoke](e55-arena-model-smoke.json) then timed out after70
seconds before its model-loaded log; no generated token or E2E metric was obtained.
Device initialization and loading are not distinguished by that silent interval.
A separate [fixed-prefix diagnostic](../../moa-dev/experiments/candidates/model-fixed-prefix-e55/README.md)
has two built hosts,47 verified inputs each,10 verified text images, and explicit
initialization progress markers. The [E55-only job71198](e55-arena-model-progress.json)
completed the real14-token prefix and produced262144 finite logits in one run.
That run is neither an original/E55 comparison nor an official model benchmark.
The subsequent [paired job71207](e55-model-fixed-prefix-pair.md) compares all262144
finite logits: max abs1.5, mean abs0.194487, RMSE0.251039, and identical raw top1.
Its accuracy verdict remains null; one greeting and one uncontrolled timing do
not establish model accuracy or official performance.
The [four-prefix job71239](e55-model-prefix-suite.md) extends this to82 input
tokens and1048576 finite logits with four matching raw top1 IDs. It records
full weight/metadata hashes before execution and unchanged weight stat/header
and metadata hashes afterward. Accuracy acceptance remains null; the observed
prefix/read ratios4.837–4.866 are sequential single-job diagnostics.
The [single-kernel controls](e55-model-single-kernel-controls.md) restore original
QKV, O, FFN, or full O independently while preserving the other nine E55 images.
All four jobs71275/71281/71282/71290 completed. The full-O pair is bit-identical
to E55 across the four full vectors; QKV/O/FFN original-reference differences
are not additive kernel error contributions.
The unchanged live E55 remains the accepted checkpoint.
Earlier adoption decisions describe the checkpoint retained at that experiment's date.
E31 is a repeated checkpoint: all 15 official-harness comparisons and 30 separate
additional-input comparisons pass. Three independent same-package E31 jobs also pass; their median-of-medians is97085/45355/389335.
E36 has passed30 additional comparisons. E37 is a previous repeated checkpoint: jobs70519/70528/70541
and30 additional-input comparisons all pass; its median-of-job-medians is94195/45748/379741.
E29 also completed three passing control jobs70428/70482/70493; E27 remains a single official-package screen
with separate additional-input coverage. E30 is a passing paired QKV control. E41 passes all15 comparisons
but its native FFN remains slower than the adopted checkpoint. E42 also passes and is slower.
E28/E32 time out, E33 has substantial FFN correctness failures, and E40 produces non-finite FFN output.
Diagnostic jobs are separated below.
E44/E45/E46 pass the official screen but are slower in their changed kernel and are not adopted.
E47 is a previous performance checkpoint with a subsequently confirmed boundary-input limitation: jobs70664/70681/70688 all pass, median94662/44588/377564;
additional30 comparisons pass70694. Only O changes from E37; QKV/FFN device images are identical.
E48 up-only fused decode passes its first screen, without a demonstrated performance gain.

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
| E29 sharded QKV input + direct-packed O + safe FFN | 70428 | Pass | 94491 | 49352 | 389199 | Initial screen; QKV/O2.83%/6.07% below E27 screen; completed control repeats below |
| E29 same package control repeated | 70428, 70482, 70493 | Pass | 94491 | 49228 | 388311 | Medians across3 jobs; QKV image retained in E37; distinct from E27 single-screen evidence |
| E30 all-pair skip control + safe FFN | 70440 | Pass | 98599 | 52113 | 389137 | Direct-packed QKV passes without explicit ×1; identity is not required for these inputs |
| E31 Switch gather/local RMS sum + E29 | 70441 | Pass | 98797 | 45355 | 389335 | Initial screen; completed package repeats below; extra30 pass70448 |
| E31 same package repeated | 70441, 70459, 70465 | Pass | 97085 | 45355 | 389335 | Four-file package hashes identical; job-median ranges2.714%/1.965%/0.177% |
| E32 pair-free native FP8 FFN diagnostic | 70447 | QKV/O pass; first FFN launch timeout | 99692 | 52233 | — | Reject; removing Sequential pair does not resolve common native packet4 path |
| E33 BF16 down block/scale DPE + E31 | 70455 | QKV/O pass; FFN fail | 96585 | 45738 | — | Reject; only12.94–16.17% FFN values within tolerance; causes unresolved |
| E34 O row2 interleave + E31 packed/parallel | 70472 | Pass | 97092 | 47308 | 389291 | Reject for performance; O4.31% above E31 repeat median |
| E35 native FP8 full256 live rows + real zero tails | 70485 | Pass | 97301 | 45799 | 811219 | Diagnostic PASS; slower, establishes packet4+pair can execute in this configuration |
| E36 FFN H480×8 epilogue + E31 | 70498 | Pass | 97052 | 45141 | 378021 | Retain; FFN2.91% below E31 repeat median; extra30 pass70506 |
| E37 kernel-specific RMS + E36 FFN epilogue | 70519 | Pass | 94094 | 45796 | 376040 | Initial screen; each device image identical to its passing parent; completed repeats/extra below |
| E37 same package repeated | 70519, 70528, 70541 | Pass | 94195 | 45748 | 379741 | Previous checkpoint; four uploaded file hashes identical; extra30 pass70544 |
| E38 full256 direct weight/scale writes | 70525 | Pass | 96579 | 45101 | 482511 | Faster than E35, slower than adopted BF16 FFN; reject as performance replacement |
| E39 native FP8 with actual2-row tail initialization | 70551 | Pass | 97459 | 45220 | 415457 | Improves native branch but still slower than adopted BF16 FFN |
| E40 native FP8 row2 mapping + tail-only initialization | 70552 | QKV/O pass; FFN non-finite | 96455 | 45927 | — | Reject; all3 FFN runs non-finite at output0; recorded432862 cycles invalid |
| E41 native FP8 actual30-row view | 70562 | Pass | 96917 | 46477 | 426442 | Single-job screen; native branch passes but FFN is slower than E39 and adopted E37 |
| E42 actual30 rows + row256 mapping | 70576 | Pass | 98728 | 45794 | 456849 | Reject for performance; slower than E37 |
| E44 E37 with paired V | 70625 | Pass | 95892 | 46090 | 378002 | Do not adopt; QKV screen1.80% above E37 repeat median |
| E45 coalesced down-scale HBM staging | 70641 | Pass | 94492 | 45842 | 394044 | Do not adopt; FFN3.77% above E37 repeat median despite static -0.43% |
| E46 native FP8 with F32 block partials/VE reduce | 70652 | Pass | 97797 | 45852 | 434284 | Do not adopt; FFN14.36% above E37,1.84% above direct parent E41 |
| E47 sixteen-column O projection | 70664 | Pass | 94662 | 44309 | 377274 | Initial screen; completed repeats and extra inputs below |
| E47 same package repeated | 70664, 70681, 70688 | Pass | 94662 | 44588 | 377564 | Previous performance checkpoint; later O boundary failure70785, superseded by E53 |
| E48 up-only fused FP4 decode + E37 QKV/O | 70683 | Pass | 94545 | 47079 | 381532 | First screen; FFN0.47% above E37 repeat median; no adoption or general fused-FFN claim |
| E50 down-only fused FP4 decode + E37 QKV/O | 70700 | Pass | 94030 | 46072 | 376184 | Initial screen; apparent gain was not reproduced by E51 repeats |
| E51 E47 O16 + E50 down fused, repeated | 70712, 70722, 70736 | Pass | 94451 | 44876 | 381286 | Reject; FFN0.99% above E47 median; extra30 pass70719 |
| E52 QKV sixteen-way row interleave | 70767 | Pass | 94674 | 45029 | 379774 | Do not adopt; QKV same as E47 within observed variation, static improvement did not translate |
| E53 BF16 O activation and F32 scale-before-cast | 70806 | Pass | 93987 | 50882 | 376814 | Previous O correctness fix; extra30 pass70814 and O boundary8 pass70810; performance repeats separately |
| E53 same package repeated | 70806, 70818, 70827 | Pass | 95621 | 50882 | 377880 | Previous checkpoint, superseded by E54 QKV correction; O14.12% above E47, original speedups2.552×/7.944×/11.285× |
| E54 BF16 QKV input and F32 channel-scale boundary | 70933 | Pass | 107845 | 50590 | 382182 | First screen; completed repeats below; extra30 pass70959, QKV boundary2 cases pass70958 |
| E54 same package repeated | 70933, 70974, 70987 | Pass | 108700 | 50970 | 382268 | Previous correctness checkpoint; QKV13.68% above E53; original speedups2.245×/7.931×/11.155×; job-median ranges1.262%/1.173%/0.081% |
| E55 fused BF16 QKV dot and F32 channel scale | 71027 | Pass | 103478 | 50725 | 381093 | First screen; completed repeats below; extra30 pass71052, QKV boundary2 cases pass71032 |
| E55 same package repeated | 71027, 71057, 71072 | Pass | 102878 | 51249 | 380339 | Current checkpoint; QKV5.36% below E54 and7.59% above lossy E53; original speedups2.372×/7.887×/11.212×; ranges0.976%/1.169%/1.036% |

Cycles shown for passing kernels inside a failed package are diagnostic observations from the raw log.
They do not constitute a valid combined result and must not be mixed with other jobs' FFN cycles.
E27 has one official-package screen; its JSON `relative_range=0` reflects one job, not demonstrated zero variance.
Its separate additional-input job is a correctness check, not a same-package performance repeat.
E29 completed three same-package control jobs, recorded separately from its original single-job JSON in
[E29 control repeats](e29-qkv-input-control-repeats.json). See also [E27 measurement](e27-qk-pair-v-two-dot.json),
[E28 failure](e28-explicit-pair-identity.json), and [E29 initial measurement](e29-sharded-qkv-packed-output.json).
Public-reference `indicative_score` values remain estimates, not official scores or current ranks.

[E41 job70562](e41-native-row30.json) passed all15 official-harness comparisons.
[E42 native row30/row256 layout](e42-native-row30-row256.json), job70576, also passed all15 comparisons; FFN456849 cycles is slower than E37.

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
is in `e5-four-row-extra-correctness.json` and is separate from the Arena performance
measurements using the official three-input harness.

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
in [the optimization plan](../OPTIMIZATION_PLAN.md#다음-작업과-실험을-시작할-조건).

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

[E29 control repeats](e29-qkv-input-control-repeats.json)70428/70482/70493 allpass: median94491/49228/388311, ranges1.812%/1.479%/0.476%.
This motivates using its byte-identical QKV image in E37 while keeping E36 output/FFN.

E31 cache-preservation job70507 passes15 original outputs plus12 sentinel checks. All12,570,624
non-target cache elements checked after launch match exactly; upload binary/fixture/runner hashes match.
[Record](e31-cache-preservation-correctness.json). This custom harness is diagnostic and establishes
preservation for E31's device code and those3target slots, not arbitrary input or later QKV binaries.

E37 repeats70519/70528/70541 allpass, median94195/45748/379741. Job-median ranges0.143%/1.272%/0.984%.
[Repeated package](e37-combined-repeats.json), [extra30 PASS job70544](e37-combined-extra-correctness.json).
Nine independent configurations are tested: official runs0–2 repeated in3jobs, plus6development runs3–8.
The repeated official inputs are not counted as nine different configurations. Public-reference indicative score6.121695 is not an official score or rank.

E37 also passed a separate non-target KV cache sentinel check in job70589
([record](e37-cache-preservation-correctness.json)). All three device images match the adopted E37 binary exactly.
The original15 output comparisons and12 cache checks pass; all12,570,624 non-target BF16 elements
across the three tested slots are preserved bit-for-bit. This custom host harness is correctness-only evidence;
its diagnostic cycles are not included in the E37 repeated performance aggregate.


Follow-up diagnostics after the E37 checkpoint:

- [E43 down activation broadcast](../../moa-dev/experiments/candidates/ffn-down-activation-broadcast-four/README.md): four compile failures were followed by a passing explicit-anonymous4 + CustomBroadcast32 mapping. DMA decreases3,889 cycles, but Main grows4,103 and total static schedule162,209→166,181. NPU not run; no speedup claim.
- [O4×1024](../../moa-dev/experiments/candidates/output-fp8-four-columns/README.md): exact compile passes but regresses24,997→25,873(+3.50%). DMA decreases146 while Main increases1,336. Four valid columns remain padded to eight physical slots; the physical global-reduce ratio remains8. Weight DMA uses256B inner packets in both layouts, not a demonstrated1KB hardware burst. NPU not run.
- [FP4 decode probe job70620](fp4-decode-probe.json): both legacy two-pass and fused conversion pass bitwise comparisons for all25,165,824 BF16 values across three patterns. Scope is one cluster,256 live slices,32×512 local values. This does not reproduce the E13 full-FFN shape, resource lifetimes or failure.
- [Tagged fullshape decode job70658](fp4-decode-probe-fullshape-tagged.json): both paths pass all117,964,800 BF16 value comparisons with zero mismatches. Scope is two clusters,240 valid/256 physical slices each,32×3840 local values, and input sequences distinguishing all15,360 rows/480 cluster-slice locations. This covers E37 up/gate decoder shape, but excludes DPE, scales, activation quantization and full-FFN buffer lifetimes. It neither proves E13's root cause nor validates a whole fused FFN.
- [E45 job70641](e45-down-scale-coalesced.json) passes all15 official comparisons, but FFN394,044 is3.77% slower than E37's repeated379,741. Static scale HBM transfer decreases while the new SRAM scatter raises total DMA by1,898. That agrees with the direction of hardware regression without proving its complete cause; the candidate is not adopted.
- [E46 job70652](e46-native-f32-block-reduce.json) passes all15 comparisons, but FFN434,284 is14.36% above E37 and1.84% above E41. Removing the intermediate BF16 block rounding does not establish better overall accuracy or speed. No extra-input or repeated-job evidence is claimed for this candidate.
- [E47 repeats](e47-output-sixteen-columns-repeats.json): jobs70664/70681/70688 all15 PASS, medians94662/44588/377564. O job-median range0.794%, repeated improvement2.54% over E37. [Extra job70694](e47-output-sixteen-columns-extra-correctness.json) passes all30 comparisons across six additional configurations. Initially adopted after both checks, then superseded by E53 after the boundary failure below; the first screen44309 is not substituted for the repeat aggregate.
- [E48 up-only fused decode job70683](e48-up-only-fused-decode.json) passes all15 comparisons, Q94,545/O47,079/FFN381,532. FFN is0.47% above E37's repeated median, so this first screen does not demonstrate a gain. It changes only up decode relative to safe E37 FFN, retaining legacy gate/down; QKV/O sources match E37 rather than E47. Correctness scope is these three official inputs for this exact combined kernel, not all-input or all-fused FFN equivalence.

E47 was adopted at this stage. Its [source/build audit](e47-source-build-audit.json) reproduces the
measured host binary exactly; all three device images also match the independent extra-input build.
Earlier E37 restore/clean-build audits remain evidence about their captured artifacts. E48/E49/E50
are separate FFN experiments and were not part of that frozen E47 package.

[E49 up+gate fused](../../moa-dev/experiments/candidates/ffn-up-gate-fused-decode-e37/README.md)
reduces Main work but delays pre-RMS and removes overlap of the second weight load with the first
projection. Static makespan rises157866→189967 versus E48; no full build or NPU run was made.
[E50 down-only fused](e50-down-fused-decode.json) passes15 comparisons in job70700 with FFN376184.
The separate E51 package combines E47 O16 and E50 FFN. [All3 repeated jobs](e51-o16-down-fused-repeats.json)
and [extra30 comparisons](e51-o16-down-fused-extra-correctness.json) pass, but FFN median381286 is0.99%
above E47 377564. Its observed job-median range377392..384033 overlaps E47 377274..377606.
The initial E50 apparent speedup was not reproduced; E47 was retained at this stage, before the E53 correction below. [Combination audit](e51-combination-audit.json).

[O32×128 compile-only candidate](../../moa-dev/experiments/candidates/output-fp8-thirtytwo-columns/README.md)
regresses24621→37344 compared with E47 O16. Weight DMA rises13383→26218 while Main grows only171.
No NPU execution or correctness claim is made; this candidate is not adopted.

[Up-only eight-row staging](../../moa-dev/experiments/candidates/ffn-up-tiled-fused-decode-e47/README.md)
is compile-only: in-place slots failed an abstract-buffer owner check; the owned-tensor correction
compiled but regressed162209→162644. Main work decreases10994 while DMA grows3321 and Main/DMA
overlap shrinks7340. The schedule memory-occupancy proxy falls8.97%, not a measured runtime peak.
The final schedule has four simultaneously live FP4 tiles, so it is not a forced physical two-slot buffer.
No NPU run or adoption was made.

E52 only changes QKV row interleave8→16. [Official-input screen](e52-qkv-sixteen-row.json) passes15 comparisons, but Q94674 does not improve E47 repeat median94662. The static50364→49811 reduction is not a hardware speedup. The full E47 binary was rebuilt byte-identically after isolating this candidate.

[Shared caller audit](e47-shared-caller-audit.md) traces the unchanged full-attention normalize and changed shared residual loop expansion. Two isolated exact compiles establish buildability and static schedule differences only; neither full-attention hardware correctness nor end-to-end Gemma throughput was measured.

[Separate E47 O boundary probe](e47-output-boundary-probe.json), job70785, completes8 cases with all40 input checks and uploaded hashes verified. Cases0..6 pass; case7 fails all3840 outputs: expected0.609375, actual0.5, absolute error0.109375. No non-finite values or timeout occur. This diagnostic does not replace the official fixtures or reinterpret existing9-configuration passes, but it establishes a numerical limitation and motivates the BF16-input correction.

E53 was adopted at this stage and was later superseded by E54's QKV correction. Its O projection preserves BF16 inputs and applies channel scale in F32 before the first BF16 cast. [Original/E47/E53 boundary comparison](output-boundary-comparison.md) isolates the same host/payload and verifies all uploaded hashes/images: original and E53 pass all8 cases, while E47 fails3840 outputs in case7. E53 also passes [official15 comparisons](e53-bf16-output.json) and [extra30](e53-bf16-output-extra-correctness.json). This is a correctness correction with a measured performance cost, not an O speed improvement over E47. The cleaned E53 source reproduces its entire measured host byte-identically; [source/build audit](e53-source-build-audit.json). E53 QKV/FFN images were unchanged from E47.

An earlier [BF16 QKV draft](../../moa-dev/experiments/candidates/qkv-bf16-sharded-rms-e47/README.md) combines the previously tested E18 BF16 helpers with E47 sharded pre-RMS and compiles at52282 vs50364. That frozen draft has no NPU validation or adoption. Its full source contains the old E47 O and a BF16-before-channel-scale boundary; it must not replace the current source wholesale. E54 is a subsequent implementation preserving F32 through channel scale, with separate NPU evidence below. The motivating [QKV CPU counterexample and later device checks](../research/qkv-boundary-design-audit.md) are outside the current official QKV input generator support.

[E53 same-package repeats](e53-bf16-output-repeats.json) all pass, median95621/50882/377880, relative ranges2.207%/1.254%/0.374%. All submitted file hashes match the frozen host. O is14.12% slower than E47 but passes the newly demonstrated boundary failure; QKV/FFN are unchanged images, so their cycle shifts are not source changes. The public-reference indicative score5.8886 is diagnostic only, not an official score or rank.

E53 follow-up static-only checks left E53 unchanged at that stage. [BF16 O8×512](../../moa-dev/experiments/candidates/output-bf16-eight-columns-e53/README.md) compiles once but regresses27239→27752(+1.88%): Main is unchanged and DMA rises513. [FFN sharded pre-RMS](../../moa-dev/experiments/candidates/ffn-sharded-input-rms-e53/README.md) compiles two variants at162735/164216 vs162209. Both reduce Main work but lose overlap with DMA; neither was sent to the NPU. No official-input accuracy or hardware speedup is inferred from these compiles.

The E53 [fused F32 channel-scale O](../../moa-dev/experiments/candidates/output-bf16-fused-channel-scale-e53/README.md) compiles on its second attempt but predicts27393 vs27239(+154/+0.57%). A per-row scalar VRF is supported and the graph preserves F32 column sum→channel multiply→BF16 cast. Moving the scale into the dot pipeline adds a579-cycle precondition before dot; removing the separate401-cycle Main and saving24 in result DMA gives net+154. The candidate was not run on NPU and was not adopted.

E54 was the previous adopted correctness checkpoint. Its `ops::sliding_project_qkv` keeps the pre-RMS
activation BF16, gathers the H480 shards and broadcasts via ring32. Its `project_query` and
`project_one_kv_matrix` look up FP8 weights as BF16, contract against BF16 input, retain F32 raw dots
through `InterTranspose`, apply channel scale in F32, then cast to BF16 before head norm/RoPE/cache writes. Removed FP8
helpers belong to the [frozen E53 projection](../../moa-dev/experiments/candidates/qkv-bf16-f32-scale-e53/projection.rs.original), not the current execution path.

[E54 job70933](e54-bf16-qkv-f32-scale.json) passes all15 official-harness comparisons with
Q107845/O50590/FFN382182; these remain its **single-screen** values. Completed
[same-package repeats](e54-bf16-qkv-repeats.json)70933/70974/70987 each pass all15 comparisons,
with median-of-job-medians **Q108700/O50970/FFN382268** and relative job-median ranges
1.262%/1.173%/0.081%. All four submitted package file hashes match across the three jobs.
Original-repeat speedups are2.245×/7.931×/11.155×. QKV is13.68% slower than E53's repeated95621;
the change was retained at that stage for general-input correctness, not as a performance gain. O/FFN images
are identical to E53, so their cycle shifts do not indicate source changes. These Arena measurements
are not official ranking or end-to-end Gemma throughput. [Extra job70959](e54-bf16-qkv-extra-correctness.json)
passes all30 comparisons for runs3..8; its cycles are diagnostic and are not added to the performance aggregate.

[E54 boundary job70958](e54-qkv-boundary-probe.json) passes both QKV cases and all6 output comparisons.
The identical-probe [three-way comparison](qkv-boundary-comparison.md) records E53 job70943's8072
failing values and original job70944's PASS. Each job passes38 input checks and preserves8,380,416
non-target cache BF16 elements. These two diagnostic inputs are **outside the official QKV
generator's support**; E53's prior official-input PASS remains a fact. Matching original/E54
result summaries do not prove identical output arrays. No all-input equivalence is claimed.

[E54 build audit](e54-source-build-audit.json) verifies the clean-source rebuild is byte-identical
to the measured host, the independent extra host contains each same device image once, and O/FFN
images are unchanged from E53. The existing O8-boundary PASS in E53 job70810 therefore tests the
same O image; it is inherited evidence, **not a new E54 O-boundary job**. [Source contract audit](e54-source-contract-audit.json)
confirms the43-source scope,10 unchanged public signatures and6 unchanged official harness files.
Official submission, score and rank remain unconfirmed, and the top-three goal has not been achieved.

E55 now replaces E54. The BF16 pre-RMS, full-H gather and ring32 broadcast remain unchanged.
Canonical channel scales are loaded contiguously, converted to F32 and routed with inverse row
Switch into VRF. The BF16 DPE completes the entire F32 dot, multiplies that result by the channel
scale in the **same Main pipeline**, then casts to BF16. Row Switch therefore moves BF16 values
before the unchanged head RMS/RoPE/cache path. Only `project_query` and `project_one_kv_matrix`
bodies change from E54; no FP8 activation approximation or earlier BF16 rounding is introduced.

[First screen job71027](e55-fused-qkv-screen.json) passes15 comparisons at103478/50725/381093.
Completed [same-package repeats](e55-fused-qkv-repeats.json)71027/71057/71072 each pass all15,
with median-of-job-medians **Q102878/O51249/FFN380339**. Relative job-median ranges are
0.976%/1.169%/1.036%, and all four submitted package file hashes match across the jobs.
QKV cycles are5.36% lower than E54's108700, while still7.59% higher than E53's lossy95621.
Original-repeat speedups are2.372×/7.887×/11.212×. O/FFN device images are identical to E54,
so their cycle changes are not attributed to source changes. Static QKV52282→51293(−1.89%) is
a compiler schedule comparison, separate from this repeated hardware result.

[Extra job71052](e55-fused-qkv-extra-correctness.json) passes all30 comparisons for runs3..8.
[Boundary job71032](e55-qkv-boundary-probe.json) passes both cases/all6 QKV comparisons,38 input
checks and8 cache checks, preserving8,380,416 non-target BF16 elements. The [same-probe comparison](qkv-boundary-comparison.md)
retains E53's failure and the original/E54/E55 passing summaries without inferring identical output
arrays. Those diagnostic inputs remain outside the official QKV generator support; nine seeded
configurations plus two boundary inputs do not prove all-input correctness. O8-boundary evidence
is still E53 job70810 on the identical O image, not a new E55 O-boundary execution.

[E55 source/build audit](e55-source-build-audit.json) verifies all46 frozen source/Cargo/toolchain
inputs, byte-identical isolated/live full hosts, exact QKV image identity and the three same images
in the independent extra host. [Source contract audit](e55-source-contract-audit.json) verifies43
source files,10 unchanged public signatures and6 unchanged official harness files; E54's other42
source files and the entire `ops.rs` remain byte-identical. E55's current adoption follows the
completed screen, repeats, extra-input and boundary checks. These are Arena development results;
official submission/authentication and top-three ranking remain unresolved.
