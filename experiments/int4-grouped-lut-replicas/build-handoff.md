Fresh standalone build `a004-int4-lut-replicas-device-build` passed in34.85 seconds from `0488a75a` (functional source `78054937`). Original dependency hashes match the plan, all421 source/config inputs are frozen, `--fresh-package --capture-artifact` completed with unchanged inputs, and a new empty `kernel-out-a004` compiled all four images. No Arena execution was performed in this lane.

Captured container path:
`/round2/runs/a004-int4-lut-replicas-device-build/built/target/release/test_int4_grouped_lut_replicas`

ELF SHA256: `1d292a33f7172a5d8c96c7aa411817473c68f98e6c5828c50e53efa6df17a3c6`,3,210,264 bytes.

Read-only registry audit finds exactly four empty-key entries: both original grouped-store kernels are byte-identical to submitted actual323173; both replica kernels are byte-identical to the frozen exact-resource images. All four match the new output directory. The audit never invokes objcopy or runs an ELF.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 experiments/int4-grouped-lut-replicas/audit_build.py > /tmp/replica-build-replay.json
cmp /tmp/replica-build-replay.json experiments/int4-grouped-lut-replicas/build-result.json
```

The fixed probe takes no tuning inputs. It covers five fixtures per H/L width: finite/distinct groups, exponent/zero/subnormal cases, exact half thresholds with constant mantissa, and two true permutations. Independent expected totals are49,152,000 codes,24,576,000 packed bytes and51,200 scales/offsets. Baseline and candidate have separate poisoned outputs; the new offset oracle checks destination-row replica selection while preserving the original quotient/code/scale oracle. Both complete graphs are timed in ABBA, ten awaited calls per block, then every mutable output and immutable input/table/base/shift owner is read again. Table/base host initialization/upload and CPU references/readback are excluded equally; the candidate's added device base-load and explicit offset write/read are included.

The fixed old and new LUTs intentionally have different extents and offset contents. Only packed codes/scales must equal each other; each offset array has its own independent expected ABI. Data-dependent gather behavior remains unmeasured. Neither compiler success nor a future lookup micro gain establishes a contention mechanism, complete-FFN improvement, model quality or an additive gain with another experiment. Root owns the only permitted actual submission.
