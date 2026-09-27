# E133 full E5 native down

The frozen `down-e5-v2` candidate improves whole FFN in three contemporary E128 brackets: −3.8264%, −3.4336%, and −3.8376%. The median paired change is **−3.8264%**. Aggregate medians are 316,668 candidate cycles and 328,744 control cycles (−3.6734%). These are three independent allocations of the same measured bytes; the before/after stages and input repetitions are not additional independent allocations.

All 135 standard-stage comparisons passed. Extra6 job 109759 passed all 30 comparisons with identical device images, giving nine distinct standard inputs. The actual-down boundary oracle passed seven exact mapping cases (26,880 outputs), and its adversarial case lost 50% of the output. A source-grounded CPU construction using component-wise generator-range inputs also predicts full-FFN error above the public tolerance. It has not been replayed as a complete PyTorch fixture or on NPU and is not a discovered failing PRNG seed. **Root deferred E133 adoption, including Stage1, pending a concrete precision repair. No live source or baseline has been changed.** A gain below 5% qualifies for bounded adoption validation; parameter tuning has stopped.

| Evidence | Result |
|---|---|
| Whole 109740 | 329,391 → 316,668 → 329,143; 45 comparisons PASS |
| Whole 109750 | 328,483 → 317,411 → 328,911; 45 comparisons PASS |
| Whole 109751 | 329,021 → 316,128 → 328,467; 45 comparisons PASS |
| Extra6 109759 | Run3..8, 30 comparisons PASS; timing excluded |
| Actual-down oracle 109784 | Seven exact mapping cases / 26,880 outputs PASS; adversarial case4 expected1.40625 → actual0.703125 for all3,840 rows (50% loss), raw tolerance FAIL |

`validation/performance-summary.json` and `docs/experiments/e133-adoption-performance.json` supersede the first-screen decision for performance validation. The first screen report and `screen-decision.json` remain immutable historical records, including their then-pending checks. Every measured archive is hash-bound, and all three outer batch payloads and normalized candidate/control three-file packages are identical across allocations. `experiment_index.measurement` and `check_comparable` pass.

The source recipe and performance control both select E128 source commit `cc82b9c`; the actual worktree started at `0f601fe`. Up/gate, GeGLU, public entries, the existing five 12-row down block-scale/reduction passes, `finish_down`, and the final epilogue retain E128 source bytes. Only the complete post-GeGLU down operand path changes:

1. Compact each cluster's 7,680 BF16 values; independently derive its amax and power-of-two scale.
2. Quantize sixteen 480-value chunks per cluster into three E5 residual terms using one common scale. Assemble the terms with explicit 480-byte Sub copies, compact, then broadcast each local 1,920-value quarter through ring32.
3. Read the unchanged public FP4 weight and FP8 block-scale matrices into their original compact DM layout. Direct FP4→E5 lookup feeds a single native down stream with a 3×1,920 E5 TRF operand.
4. Restore input scale in F32, write the existing H60×block120 F32 partial, and use the original block-scale, local-K, four-column and two-cluster reductions and BF16 boundary.

There is no extra packed-weight assembly. The full decoded down-weight DM write/read of 56.25 MiB each is removed. The F32 partial write/read of 14.0625 MiB each remains. Replicated activation payload grows from 1.875 to 2.8125 MiB, and the new amax/quantizer, term copies/redistribution, three-term dot and input-scale restoration are included in the whole measurement. These are logical byte counts, not measured bus traffic or reclaimable cycles. The static estimate is 153,072 versus E128 155,190; the NPU results above are separate evidence.

The first frozen implementation compiled but failed the descriptor gate: typed DMA into a padded term view read/wrote 1,440 bytes per 480-byte source. `down-e5-v1`, its build and `graph-verification-e133-down-e5-v1.json` preserve that failure. V2 changes only those three new down copies to explicit Sub streams. Its emitted index constants 0/1/2, offsets 0/480/960, 480-byte reads/writes, public matrix coverage and row/cluster/quarter mapping, scalar-scale replication, TRF/register lifetimes and native/reduction order pass `verify_graph.py`. No NPU ran the invalid V1 image; no unrelated inherited E128 source was altered.

Three E5 residual terms do not preserve every BF16 value. For example, with cluster amax 128, BF16 2^-24 maps to zero in all terms. `down-oracle/` calls the actual frozen `project_down_tile` and `finish_down` against original BF16 truth, separately reporting exactly representable mapping cases and an adversarial underflow case. The observed case4 output was1.40625 expected versus0.703125 actual on all3,840 rows. Its block scale256 is outside the Stage1 standard generator range2..15; this alone does not establish general numerical equivalence or settle reachable full-FFN impact. Synthetic raw-down inputs are not automatically proven reachable after GeGLU. A raw-helper discrepancy must remain visible and requires root's public full-FFN assessment; it must not be silently reclassified as either whole-FFN failure or general correctness.

Rechecks, without compilation or NPU submission:

```sh
python3 moa-dev/experiments/candidates/ffn-native-down-e5/verify_graph.py e133-down-e5-v2 --check
python3 moa-dev/experiments/candidates/ffn-native-down-e5/package.py e133-down-e5-v2 --check
python3 moa-dev/experiments/candidates/ffn-native-down-e5/validation_performance.py --check
python3 moa-dev/experiments/candidates/ffn-native-down-e5/validation_extra.py verify
python3 moa-dev/experiments/candidates/ffn-native-down-e5/provenance.py --check
```

`validation_verify.py` rechecks raw extra-input and oracle records as well as the existing validators. Its `valid` field describes evidence integrity; `adoption_ready` separately carries any unresolved adversarial review. `compiled-source-manifest.json`, `measured-build-e133-adoption.json`, and `source-build-binding.json` bind 46 compile inputs, the original exact and package records, all three embedded images, and byte-identical `kernels/` compatibility copies. They do not adopt code, assert live-source identity, test a full model, or submit an official competition entry.

`numerical-scope-review.md` and `numerical-scope-analysis.json` preserve the CPU construction: the first 1,920 full-FFN outputs are predicted to differ by 0.029296875, above tolerance 0.01404296875. Run `numerical_scope_analysis.py --check` to reproduce its source-bound scalar model. This is separate from the observed 50% raw-down loss and the nine passing seeded inputs. The measured same-scale variant is closed as **accuracy_limited_positive**; a follow-up may amplify only the third residual and restore its separate scale before summation. That coupled source/consumer repair is not implemented or validated in this record.
