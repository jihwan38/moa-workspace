# Complete FFN stress reference

`generate_reference.py` actually executed the frozen official PyTorch
`gemma4.MLP`, `GloballyScaled` and `rms_norm` on CPU. It installs all three
dense BF16 weight matrices after folding the original FP4 values and per16
FP8 local scales, and retains pre-RMS, GeGLU, global scales, post-RMS, residual
addition and layer scaling. `upstream/` contains unmodified source from
`cc82b9ccff22795165faffd3ba9a68cc557da001`.

The deliberate input pattern is described in the E133
`numerical-scope-review.md`. All tensor components fit the standard generator's
value ranges, but no claim is made that its PRNG generates this correlated
pattern. This is additional numerical validation, not a modified official
input generator or acceptance threshold.

The actual result is3840 BF16 values **0.404296875**, each bits **0x3ecf**.
`expected-bf16.bin` is their little-endian7680-byte representation, SHA256
`d2930beb2df2274c5b316865d4563e4babe550c1ef930cf2bbcd02cca79f3a74`.
`reference.json` includes all13 original input byte hashes and the exact
`test_kernels.rs` PRNG checksum, plus actual hooked intermediate values.
GeGLU contains15358 copies2^-22 and anchors1024/0.5, as predicted.

The captured Docker CPU command, image ID, complete output and source hashes
are in `reference-build.json` and `reference-generation.log`. Reference
generation uses no NPU and changes no kernel. The host owner separately runs
the unchanged E128/E133/E134 public images against these exact input and
reference bytes; those outcomes must be recorded independently.

Read-only verification without Torch, compilation or device use:

```sh
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/full-ffn-stress/verify_reference.py --check
```

To regenerate in a scratch Docker directory, copy `generate_reference.py` and
`upstream/` together, then run the command/environment recorded in
`reference-build.json`. Preserve this frozen result when recording a rerun.
