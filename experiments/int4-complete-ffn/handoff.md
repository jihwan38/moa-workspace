# Frozen complete-FFN executable; actual gate pending

Functional source `70f4c15f`, build/evidence source `18888930`.
Fresh a005 device build passed56.00s; all411 recorded inputs remained unchanged.
Original Cargo.toml and ignored frozen Cargo.lock are unchanged.

Captured executable:

```
/round2/runs/a005-int4-complete-ffn-device-build/built/target/release/test_int4_complete_ffn
```

SHA256: `bd51ab5f4c0cf7e20918e022c023c7e42fcbad4fa11b30da382c33297088e61f`.
No arguments. Optional `FURIOSA_OPT_PROFILE=1` enables separately untimed traces.
Root alone submits any bounded70s actual gate; this agent has not run it.

Read-only embedded-image audit found19 images and18 exact prior matches:
the four actual-passed I4/E4 projections, two actual-passed fused quantize/pack
entries, q020's current complete production graph/four norm-tail entries/five
weight-preparation entries, and the two independently compiled BF16 bridge
resource images. The sole new independent P128 GeGLU reference wrapper has no
prior image and requires actual runtime validation. No prior image was missing
or different. Reproduce with `python3 experiments/int4-complete-ffn/audit_build.py`.

The original a003 compile failure is preserved. Canonical HBM→column-DM oracle
ingress used an unsupported2-byte DMA stride. The narrow repair changes only
the untimed reference's HBM wire order and independently checked CPU mapping;
its original GeGLU arithmetic, candidate, production graph and strict guards
are unchanged. batch/root mapping reviews and a004 hostcheck10.89s passed.

Expected correctness coverage (not yet an actual result):196,608,000 BF16
comparisons,6144 i64/F64 samples,4096 scale values and19,660,800 packed bytes,
two feedback updates and true row permutations. Timing includes all seven
candidate awaits versus the unchanged one-call production graph. Startup
encoding/upload/prepack and residual restores are separately disclosed.

Require both reported10% fullchain gates and final `performance_gate_passed=true`
before claiming a speed gate; a correctness-success exit alone is insufficient.
Step0 production drift uses matched input, step1 includes diverged feedback.
Layer0 sliding only; no padded/full-layer or model/task quality proof.
DOWN's245684>240000 static resource HOLD remains independent and unchanged.
