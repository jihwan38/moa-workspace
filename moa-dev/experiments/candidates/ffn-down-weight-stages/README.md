# E150: staged weighted-down supply adopted

Adopted frozen `staged-v1` as `baseline-e150`, source commit `ad1ecb3d3868c7e0ca2e4d1428bbd66bf1111821`. Official `ac53ad0a` passed15 comparisons: FFN296854, score6.838931882, rank57/80 at2026-10-01 19:11 KST (rank unchanged). QKV/O images are unchanged; their timing variation is not attributed to the FFN source change. Actual start `024a0fd1d711d14c5c6355deb76f377828022d47`, branch `opt/ffn-down-weight-stages`. Source and independent performance control are E147; the inherited FFN is E142. E148/E149 O changes are not included.

| Allocation | E147 before / after FFN | E150 FFN | Paired change |
|---|---|---:|---:|
|141242|303848 /304036|297276|−2.193182%|
|141259|303744 /304075|298524|−1.772074%|
|141276|304657 /304240|297731|−2.206449%|

All three improve, with paired median **−2.1931816%**. The separate aggregate-median ratio is−2.043482%. The same uploaded candidate/control payloads, fixtures and runner are bound across allocations. All135 standard-stage comparisons pass; extra6 job141282 passes30 comparisons, giving nine distinct standard inputs. Repeated inputs and before/after controls are not independent candidate allocations. Unchanged QKV/O timing is not an optimization. [Performance](adoption/performance-summary.md).

Raw-down job141261 passes all nine original cases and34560 exact BF16 outputs, signed zero equivalent. Its complete production MLP prefix is unchanged and the diagnostic image is distinct from whole FFN. Whole regression job141273 passes3840 exact values, max error0, using the actual production image and original13input/reference checks. This is selected boundary coverage, not arbitrary BF16 or full-model correctness. [Boundary results](boundary-result-summary.md) · [Final gate](adoption/independent-final-review.json) · [Adoption](adoption-decision.json).

The complete path remains up/gate → GeGLU → canonical per-cluster weighted E5 quantization/broadcast → shared activation TRF and coefficients[s,s/16,s/256] → down. Only the local60-row FP4 ingress/native producer is replaced by two compact30-row allocations and two weighted native streams. Their direct F32 commits cover per-slice `[0,14400)` and `[14400,28800)` of the original full60 partial root. The original full blockscale load and five12-row scale/localK/K4 consumers, including rows24..35 across the half boundary, remain unchanged. No assembly copy or stage reduction is added.

Actual descriptor/register checks pass for29491200B original packed weights,960B source segments,28800B compact weight halves and disjoint partial writes. Weight DMA/native/lookup count each increases from one to two for down. Total weight bytes and F32 partial write/read14745600B each are unchanged. All46 compile inputs are bound; only `src/device/shared/mlp.rs` changes, and QKV/O images match E147. [Source/build binding](source-build-binding.json) · [Actual graph audit](audit-e150-staged-v1.md).

Selection followed trace141134, where up result gather finished just after long down-weight ingress and native followed activation TRF by one cycle. Those overlapping marker windows are not additive savings or proof of exclusive DMA contention. Rust source order gives no issue-priority guarantee. The new exact image's `readiness/` diagnosis is running separately to determine the changed completion path; its unfinished outputs are not required adoption evidence and no result is claimed yet.

Stop this row/stage tuning series. A reproducible gain below5% still qualifies for adoption after validation. Preserve C09's API precedent, E146's unreproduced staged-QKV gain, M04's older five-stage/materialized regression and C12's added-packing regression as different complete combinations. No padding retry follows from225×128B packed-half extent alone. [Frozen recipe](staged-v1/recipe.json) and [selection review](selection-review.md) retain preparation-time claims in their historical scope.
