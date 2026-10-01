# E153: quantization consumes original pre-RMS shards

Actual start `2a49667318718b2f0e2fda00b7feb8a3f3727515`, branch `opt/ffn-prerms-sharded-consumer`, source baseline-e151/8adbbae and separate performance control baseline/e151.json. All completed E150/E151/E152 records are integrated before this series; the rejected E152 gate split is absent.

Keep the original pre-RMS arithmetic and normalized F32 storage, but finish with local BF16 H480 shards instead of final F32 gather. A cluster0 sharded quantizer consumes them directly, using local/global finite Max plus original clamp/power-of-two scale, Sub sum and exact three480B copies. Replicate the finished packed terms late to two clusters (23040B), then use original184320B leader distribution/ring32 and current native/weighted-down/E151scale path.

This removes early canonical BF16 gather/cluster replication/reshard dependencies but adds global Max/late packed DMA and changes lifetimes. The old41k overlapping interval is not a saving claim. M06v4 timeout cannot be explained by E140 typed-copy repair: v4 already wrote packed terms directly. The new complete path is validated anew. Source3files/43unchanged and QKV/O byte identity, actual shard/Max/term bounds/joins/8KiBVRF, whole NPU and changed-boundary numerical checks are required. See sharded-v1/recipe.json for separate adoption/tuning/research decisions.

## First whole screen and bounded follow-up

141557 passes45 original comparisons/actual upload. FFN298332 versus E151294978/294929
(bracket294953.5) regresses1.145434789%. E153 is not adopted; E151 remains runtime.
Raw8/fullFFN packages and original references/actual descriptors are verified but NPU0.
Stop parameter tuning and adoption repeats. One frozen-image E151/E153 names-only
readiness comparison is selected to identify a concrete supply/dependency repair.
No exclusive physical cause or family-wide rejection follows from this first loss.
[Decision](decision.json) · [Boundary scope](boundary-results.json).

## Current-image readiness result

141612 passes57 comparisons and repeats the whole regression:299012 versus
296324/294082, bracket295203 (+1.290298540%). Names-only E151/E153 images each
carry142 checked markers with original instructions, addresses and IO preserved.
Same-clock scale-finish→quantVRF-finish median is2256cycles(E151) versus37235(E153).
Candidate hi starts6679.5cycles earlier, but gate native starts7282.5cycles later.
These overlapping intervals are not savings or proof of exclusive DMA contention.
E154 therefore tests replicated scalar output/direct local scaleVRF while retaining
the remaining complete path; it has its own branch and new exact/NPU checks.
[Analysis](readiness/profile-analysis.json) · [Decision](decision.json).
