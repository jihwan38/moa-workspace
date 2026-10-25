Frozen e759c4fc/ELF fc1b1c6d passed both full-model host-isolation probes:
r1187/job321134 (512 histories,18.15s) and r1188/job321135 (1024,23.33s).
Each checks22,544,384 logits;17 paired owners plus the active prompt compare
3,321,888,768 BF16 KV values (6,643,777,536bytes), all exact. Across both runs,
unchanged scalar/R32/prompt paths total9,961,472 exact logits. Native B16 changes
are approximate and were not counted as equality passes.

Each run measures67 actual B16 rows (17,563,648 logits). With512 history,14,471,546
BF16 logits differ, maxabs1.0625, median per-row relativeL2 1.24%, worst3.94%.
With1024 history,15,384,753 differ, maxabs1.3125, median2.43%, worst5.22%.
Default greedy choices change1/67 and4/67 respectively (same counts with empty
bans). Winner-masked greedy changes5/67 and9/67. Seeded stochastic outcomes also
change:13/67 and17/67 with empty/default bans,15/67 and22/67 winner-masked.
These are actual-model approximation statistics; no task-accuracy pass follows.
Own-logit device greedy, all-banned fallback and None policies pass both variants.
Teacher forcing is identical, so all cache histories remain independently exact.

Warm LM stage ranges (excluding first per-arm B16 invocation) are2.361–2.407ms
original versus1.147–1.198ms FP8 at512, and2.360–2.394 versus1.149–1.185ms at1024.
Example warm C16 full-model calls are37.770→36.026ms and38.262→36.940ms. These
model passes are sequential control-first, not ABBA: unchanged scalar/R32 timings
also improve on the second pass, so no causal whole-model percent is claimed.
The separate micro ABBA establishes the inclusive LM reduction.

Current original weights, canonical64 KV, joined projection-owned FFN, group and
greedy scratch are present. Shadow startup4.468/4.454ms allocates1,007,732,480bytes,
repeated preparation allocates nothing, and after all warm candidate routes
2,507,943,168bytes remain (largest block≈2,506,860,000). This validates capacity
for declared current model/pools and exercised warm routes; future allocations
still require their own capacity review.

Root authorizes isolated q018-b16-lm-fp8 quality candidate only: source-level
Automatic B16 activation, original BF16 embeddings retained, other widths/heads,
Worker policies and selection APIs unchanged. Fresh six-suite quality plus actual
multi-token B16 decode traces are required; no root production integration or
official submission is authorized by these numerical/isolation results alone.
