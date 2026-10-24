R2-BF01: B16 feedforward core
============================

`ops_batch::batch_ffn` consumes `[B,H]` rows after pre-FFN RMS normalization and
returns `[B,H]` before post-FFN RMS normalization, residual, and layer gating.
It uses the provided FP4 matrices, E4M3 block scales, and F32 global scales.
Up and gate expand FP4 to E4 once, then use two B8 contraction passes. Their
block-scaled BF16 weight construction, BF16 dot boundary, BF16 global-scale
boundary, GELU, and BF16 up-times-GELU result follow `device/shared/mlp.rs`.
Down partitions L into the baseline's eight chunks of1920 and reduces F32
partials before BF16 conversion. The inter-layer activation permutation uses a
four-row TU transpose because direct DMA cannot express two-byte SRAM strides.

`test_batch_ffn` compares all61440 results against sixteen calls into the
unmodified official serial `mlp::feedforward`, using deterministic independently
seeded FP4 weights/scales and BF16 inputs. It reports bit differences as well as
failures at tolerance `0.02 + 0.01*abs(reference)`. Timing labels distinguish a
cold batch launch+read, five warm batch launches with one final read, and sixteen
serial launches with per-row reads. These are kernel-level measurements, not
an E2E throughput estimate.

As of this commit: FFN compile passed and host test executable built; hardware
validation is pending root. Failed direct-DMA source/log is retained as v1.

```
docker exec -w /round2/batch-kernels moa2026-round2 env CARGO_BUILD_JOBS=4 \
  cargo furiosa-opt build --release --bin test_batch_ffn
```
