# Exact grouped sliding QKV export

All eleven original/candidate owner-route jobs passed, r1099–r1109. Totals are68,943,872 all-vocabulary logit values,9,437,184,000 owner-cache BF16 values and4,152,360,960 untouched-sentinel BF16 values, with zero bit mismatches and no nonfinite logits. The comparison includes full reserved caches, reversed physical mappings, true histories, reset, NaN padded quarters, post-prompt decode and candidate-only B16/R32 interleaves. The isolated hardware gate uses functional3deaf9e/frozenfc3f009 and ELF SHA256 `71cf54b68bc6ef12c305f8f7b003a7356ee3f3f6919a4f74650fb7ee6fc6b182`.

| Case | Real owner counts | Warm original ms | Warm candidate ms |
|---|---|---:|---:|
| 0 | 128+128+128+128 | 122.801 | 121.686 |
| 1 | 512 | 123.378 | 122.393 |
| 2 | 256+256 | 125.705 | 124.631 |
| 3 | 256 | 109.150 | 107.853 |
| 4 | 128+128 | 107.792 | 106.612 |
| 5 | 384 | 115.985 | 114.627 |
| 6 | 128+256 | 117.558 | 116.754 |
| 7 | 128+128+128 | 115.495 | 113.840 |
| 8 | 256+128+128 | 126.204 | 123.566 |
| 9 | 128+256+128 | 124.390 | 123.165 |
| 10 | 128+128+256 | 123.715 | 121.973 |

These are two-round model diagnostics, with order reversed on reset; the table uses the second round. Median warm whole-prompt savings1.225ms, range0.804–2.638ms. Warm `group.qkv.projections` (all40 sliding plus8 unchanged full layers) changes from roughly13.97–14.07ms to12.62–12.74ms. Totals include preparation and layer/finalization work, exclude logit downloads, interleaved decode and cache comparisons. No HTTP throughput or statistical repeated-ABBA claim is made.

The clean export is based on current roota8b4d0f, including exact B16 UP ingress:

- `2058412`: scoped default-off QKV module, host selector and explicit false/true probe. Only `lib.rs` append conflicted; both B16 ingress exports were retained.
- `24a2ef9`: separate authorized activation of four Group/Tail execution defaults. Probe explicitly sets both arms, so its original producer control survives activation.

`source-audit.json` proves the new module is byte-identical to the all-eleven gated kernel and the host differs from that gate only by four defaults plus inherited B16 changes. Worker, cohort, B16 prefill/decode, R32, Wide, both stationary O modules and projection-owned FFN modules are byte-identical to a8b4d0f. Removing the added local QKV setter from Model yields the exact a8b4d0f Model, preserving all B16UP fields and dispatch. FullQK, norms, attention, KV, cursor advancement and real-owner layout are unchanged. There are no persistent mode or request-level selectors and no extra scratch/weight allocations.

`model-results.json` retains per-job source/log hashes, exact counts, both rounds and stage counters. Recreate with `python3 collect_results.py /home/jihwan38/moa-worktrees/round2`. Fresh server compilation and HTTP gates remain root-owned; this export did not dispatch Arena.

Final activated-source validation: a141 plain server+oracle typecheck passed10.31s; a142 all111 library tests passed,0failed/ignored/filtered (package-only test debug0, CARGO_INCREMENTAL=0). Exact records and log hashes are in `checks.json`. No additional device build or Arena job was run for this export.
