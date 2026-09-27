# E141 actual up/gate correctness diagnostic

The source recipe is the frozen E135 direct-six-row production file with exactly three up-term copy bounds repaired. The remaining whole-build inputs and contemporary performance control come from adopted E140. This diagnostic appends the original E135 wrapper to the entire actual E141 `mlp.rs`; it calls `prepare_ffn_fp8_split` → `project_up_and_gate` → canonical up/gate BF16 stores before GeGLU. No production arithmetic is replaced by a diagnostic implementation.

The five files in `probe-v1/` (host, wrapper, public entry, small crate entry, quantizer audit) are byte-identical to E135. Its eight finite inputs, independent F64 dot/BF16 reference, all FP4 code packing, distinct up/gate scales, row6/30/cluster and K16/64/3839 boundaries, cancellation and small-value checks remain unchanged. Original `E135_` console markers deliberately remain. The builder checks the historical24 input/oracle log rows and verifies the full source prefix against the whole build record, container source, executable image and eventual actual uploaded archive.

The immediate question is whether the corrected production prefix completes the previously timed-out standalone path and returns all245760 expected first-BF16 outputs. This separate image does not measure whole FFN performance, exercise GeGLU/down/RMS, or prove all-input equivalence. A failure is preserved with the stage reached; no source/layout/arithmetic retry is automatically selected. The original eight cases are not weakened or extended here.

Target `/tmp/moa-c09-validation-probe/target` is separate from the whole-build agent's m03 target. Build only after the exact whole source/build is frozen, then submit after its specific descriptor gate. No historical file is changed.

```sh
python3 moa-dev/experiments/candidates/ffn-independent-direct-copy-repair/probe.py build
python3 moa-dev/experiments/candidates/ffn-independent-direct-copy-repair/probe.py verify
python3 moa-dev/experiments/candidates/ffn-independent-direct-copy-repair/probe.py submit
python3 moa-dev/experiments/candidates/ffn-independent-direct-copy-repair/probe.py collect
python3 moa-dev/experiments/candidates/ffn-independent-direct-copy-repair/probe.py --check
```

Result status is recorded in `probe-build-v1.json` and `probe-results-v1.json` when those actions execute; their absence means not built/not measured. Adoption and tuning decisions belong to the complete whole-path record, not this diagnostic.

Job111574 completed and passed all8 cases /245760 exact BF16 outputs (signed-zero equivalent), with no nonfinite output. `probe.py --check` rechecks the complete production prefix, original historical fixture/checksum identity, diagnostic image in the executable and actual uploaded archive. `independent-probe-copy-review.json` separately verifies three exact480B copies and a live ring32 bitmap outside their destination ranges. Source/build/results are fixed by `verification-summary.json`.

This resolves the observed first-launch liveness and finite correctness questions for this repaired implementation. It does not prove the copy defect is the unique cause of every historical E135 failure. The contemporary whole run111571 passed45 comparisons but regressed3.151638% against E140; this validation record neither overrides that result nor requests performance repeats or broader accuracy tests. The candidate remains unadopted.
