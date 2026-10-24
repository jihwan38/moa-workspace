B16 query projection feasibility (R2-BG01)
========================================

The initial kernel in `src/ops_batch.rs` reads the model's E4M3 query matrix once,
expands it to BF16 once in DM, and contracts against two B8 activation tiles.
Its BF16 dot rounding and channel-scale rounding match the baseline projection.
Output is row-major `[B, Qs]`. `test_batch_projection` independently checks every
output against exactly representable dyadic products, including both BF16
rounding boundaries. This is a projection test, not full-model validation.

Build, SDK 0.9.0:

```
docker exec -w /round2/batch-kernels moa2026-round2 env CARGO_BUILD_JOBS=4 \
  cargo furiosa-opt build --release --bin test_batch_projection
```

As of this commit: host and device compilation passed (`logs/build-v12.log` and
`logs/compile-v12.log`). Hardware correctness and timing are pending root's
coordinated execution. No speed claim follows from compilation.

Compiler findings preserved in versioned source snapshots and logs:

* v1/v2 used a padded BF16 fetch flit before F32 conversion; the extra time
  padding did not match collect. v3 removes the unnecessary fetch padding.
* v3's one-step transpose demanded 16 BF16 source rows, beyond the transpose
  engine's four-row input limit. v4 returns to direct output DMA permutation.
* v4-v11 repeatedly hit an internal duplicate-IR error **only because
  `cargo furiosa-opt compile --dump-summary DIR` was enabled**. v11 compiles
  without that flag; v12 restores full E4/channel-scale/B16 row-major behavior
  and also compiles without it. The summary directories are diagnostic artifacts
  from failed compilation, not successful-kernel timing measurements.
* v7-v9 explored B4 to isolate capacity: contraction lane output must have eight
  elements, and F32 collect packets must occupy a full 32-byte flit. That variant
  was abandoned after locating the diagnostic-option issue.

The TRF design is based on the SDK architecture documentation:
https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/index.html
https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/register-files.html

B8 × H3840 × BF16 fits full TRF capacity; B16 does not. Decode and weight HBM
transfer occur outside the B8 loop. Larger batches can reuse this placement by
adding B8 iterations, subject to DM capacity and actual hardware measurements.

Hardware result, root Arena317587:

* source4a7f256;65536 outputs checked, zero bit mismatches.
* launch+download1.060ms, a single cold kernel probe.
* raw log: `/round2/runs/r006-b16-query-oracle-submit/arena.log`.

The remaining raw Q/K projections now compile at actual model sizes. Full Q
uses the baseline32-byte E4 fetch packet to avoid a fetch-stride lowering error.
The output projections accept canonical head shapes, preserve per1024-column
BF16 dot rounding and the BF16 pairwise summation tree, and explicitly transpose
into row-major DM before output DMA. Both sliding and full O wrappers compile;
O hardware correctness is pending. A failed initial HBM-view clone lowering is
retained in `ops_batch_output_v15.rs`; each partial now creates its own metadata
view from the original HBM tensor, without calling non-primitive Clone on device.
