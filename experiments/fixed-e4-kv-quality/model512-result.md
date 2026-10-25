# Corrected fixed E4 model512 numerical result

Use fresh OFF a007 and verified ON a005, not quarantined OFF a006. OFF head/scalar negative controls r1330/321635 and r1331/321636 pass13,107,200 and830,529,024 bytes. The per-function ELF audit confirms all12 production precision-sensitive callers (plus one diagnostic) have the intended graph;297 unaffected kernels match.

Corrected paired model runs r1334 OFF and r1335 ON both pass **18,874,368 all-vocabulary replay logits** and **2,952,790,016 BF16 cache values** (5,905,580,032 bytes, one side). Each program recomputes real512-token histories for two disjoint16-owner sets, interleaves one activeP128, exercises scalar/B16 widths/fallback/reentry/reset, and validates device selection plus deterministic sampling from its own outputs. Runtime owning-buffer identity is32 physical slots,64 query rows,256 full pages and12,884,901,888 KV bytes. All logged logits are finite.

Across OFF/ON's72 matched output rows:

- 34 unrestricted top1 tokens change;37 seeded stochastic selections change.
- None of the full-vocabulary FNV64 checksums match. These are diagnostic checksums, not cryptographic exactness proofs.
- The **128-sample-only** relative L2 mean is0.1866877674 and maximum0.8590307006; sampled maximum absolute difference16.90625. These are not all-element error norms.
- Full sum/L1/squared-L2/min/max are computed over all262,144 logits independently on each side, but cannot reconstruct the full-vector difference norm.

The corrected ON run reproduces all72 complete printed row records from the earlier ON r1313 exactly. The invalid OFF control, not an observed ON nondeterminism, caused the first misleading zero-drift comparison.

This is a substantial perturbation. Same-candidate replay establishes deterministic intended execution and cache ownership, not accuracy equivalence to the provided model. Remaining R32/tiled/packed owner routes and six public quality suites remain required before adoption. Any eventual public candidate must remain a separate numerical research artifact; no compressed storage host port is justified by this result alone. Surrogate conversion over BF16 pools is not a compression speed measurement.

`model512-corrected.json` contains every row's bounded comparison; `a007-actual-gates.json` records log hashes and exact gate markers. Reproduce with compare_reports.py and the two root-owned arena.log files. Sources/artifacts remain frozen and source-only dependencies unchanged.
