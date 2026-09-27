# E128 — native FFN dot/scale/reduce fusion, adopted

**E128 adopted** as `baseline-e128` / `cc82b9ccff22795165faffd3ba9a68cc557da001`.
Branch `opt/ffn-native-fused-scale`, actual start `57e6a5b318b47e89912beda47cea19250f21f687`.
Source origin `baseline-e127` / `09cf22fcb54349cf5984640d815ade3c4478b3e8`; separate performance control `baseline/e127.json`.
[Design and prior-attempt audit](../../../../docs/research/e127-large-cost-shortlist-20260927.md) ·
[frozen source recipe](six-row-v1/recipe.json) · [final decisions](decision.json).

Up/gate each now executes five six-row native E5M2 streams. Each stream computes
block dots → Mul0(input scale) → Mul1(block scale) → F32 H reduction → final row
output. Its scales occupy compact VRF; the large block partial never enters DM or
partial VRF. The actual graph removes29,491,200B up/gate partial writes and the
same amount of reads. Native starts increase2→10. These are logical bytes, not
bus measurements or a latency prediction. Full FP8 weight materialization,
actual3term input preparation and M03 GeGLU/down remain.

Only `project_rows_with_scale_contraction` and the file header changed from E127;
the production source from `project_up_and_gate` to file end is byte-identical.
The F32 order `((dot*x_scale)*block_scale)` and first BF16 row-output boundary
are preserved. Source reload calls remain, but the compiler shares their TRF and
input-scale VRF; emitted lifetimes and VRF address non-overlap were checked.
[Graph/coverage proof](graph-verification.json) · [independent review](independent-review.md).

## Whole-kernel result

| Allocation | E127 before | E128 | E127 after | Change vs bracket |
|---|---:|---:|---:|---:|
| 108339 | 332115 | 329451 | 332020 | −0.787942% |
| 108342 | 331449 | 329159 | 331222 | −0.656887% |
| 108343 | 330548 | 329097 | 331329 | −0.556448% |

Paired median **−0.656887%**, aggregate331335.5→329159cycles. Each candidate
beats both same-allocation controls; all three use the identical standard
host/fixture/runner and outer batch bytes apart from Arena's job manifest.
[Reparsed uploads and aggregate](validation/performance-summary.json).
Static155190 vs154440 is **+0.485625%**, a separate compiler estimate.
There is no new internal NPU profile attributing the modest gain to a particular stall.

Correctness: candidate45 standard comparisons across three allocations, extra6
inputs/30comparisons in108344, and independent projection probe108341 with
8cases/245760BF16 outputs all PASS. There are9distinct standard inputs per kernel;
performance repeats reuse the original3. Extra/probe timings are excluded.
The probe uses actual quantization, distinct projection scales and row/block/cluster
boundaries, not a fixture-dependent device path. [Probe scope and repair history](validation/README.md) ·
[extra6](validation/extra-results.json). The diagnostic host's unsupported checksum
API and output-path setup failures are preserved; the corrected diagnostic device
image is identical. The production kernel compiled on its first attempt.

This is a local dataflow replacement. E73 already fused BF16 dot/scale/reduction
but repeated lookup per tile; E128 uses materialized native E5M2. C11 stored
partials in VRF and handed them to another Main stream; E128 keeps scales in VRF
and consumes partials immediately. Neither past attempt is relabelled as new.

## Adoption, tuning and larger research

- Adoption: adopt the frozen six-row-v1 after reproduced positive performance and
  changed-boundary correctness. The5% tuning budget does not veto small gains.
- Tuning: stop; no tile-size sweep. Removing56.25MiB logical round-trip work
  produced a modest measured gain, not a ranking breakthrough.
- Larger research: separately test projection-owned clusters to remove full FP8
  weight storage without E123 joint packed assembly. Public transfer and direct
  correctness are unresolved feasibility gates; this experiment does not prove them.

[Source/build binding](source-build-audit.json) · [adopted recipe](../../../../baseline/e128.json).
QKV/O images are unchanged E127. Inherited three-term FP8 input representation is
approximate; selected cases do not prove all-input/full-model equivalence.
No E128 official submission or new ranking result.

## Reproduction

From the workspace root, existing evidence can be checked without new NPU work:

```bash
python3 moa-dev/experiments/candidates/ffn-native-fused-scale/verify_adoption.py --check --check-live
python3 moa-dev/experiment_index.py --check --check-live
```

A new build needs a fresh label; do not overwrite frozen evidence:
`structural_build.py ffn-native-fused-scale six-row-v1 --label NEW --dump-graph`.
The current graph verifier intentionally binds only `e128-six-row-v1`; a new build
requires a separately reviewed binding. Original package/screen/probe tools and
logs remain available for reproducibility, not automatic resubmission.
