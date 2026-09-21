# E55 with the original sliding-output kernel

Subsequent root-managed [Arena job71275](../../../../docs/experiments/e55-model-control-sliding-o.json) completed.
See the [combined model results](../../../../docs/experiments/e55-model-single-kernel-controls.md).
The build evidence below remains a preparation-time snapshot.

This diagnostic control restores `sliding_attention_output` while retaining the
other nine text kernel images from E55. It is not an adopted optimization.
The [build script](build.py) starts from the independently audited47-input E55
[four-prefix suite](../model-prefix-suite-e55/README.md), restores the original
sliding-output operation and its projection helper tail, and appends a dedicated
copy of the original residual-add loop. The generic E55 residual helper remains
unchanged, so full O does not change.

The three changed inputs are `src/ops.rs`,
`src/device/sliding/projection.rs`, and `src/device/shared/residual.rs`;
[source.patch](source.patch) records the exact changes. The probe host, runtime,
loader, dependency lock, and toolchain inputs are unchanged. Adding a private
helper does not change any public operation signature.

The fresh isolated build completed in56.324seconds. Its host SHA256 is
`ba5f7deec114d713de58e51b697b8e6fbb7010f1c7c4216c64aa6795dfaa569f`
(11,485,104bytes). The [ELF registry check](registry-summary.json) verified that
the target image is byte-identical to the original
`434f864af2f46a75c06cf2c5f2f42053e705596ad75d397809a73f50b488ead4`,
and the other nine images are byte-identical to E55. Every embedded image also
matches its compiler-emitted file. Actual Docker inputs match the47-file
[source manifest](source-manifest.json).

Run from the repository root:

```bash
python3 moa-dev/experiments/candidates/model-control-sliding-o-original-e55/build.py
```

The script builds only under this candidate's ignored artifact directory and
`/tmp/moa-model-control-sliding-o-original-e55` in Docker. It does not modify the
live crate, execute a model, or launch an NPU. Subsequent model runs and their
limits belong to the [control-pair workflow](../model-single-kernel-controls-e55/README.md).
