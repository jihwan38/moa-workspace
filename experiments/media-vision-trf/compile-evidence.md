# B8 TRF vision compile gate

Base: `a98ca2b`. The new source changes dense projection placement only; it is unused by serving. The input norm, K768 dot-to-BF16 boundaries, balanced BF16 addition tree, patch bias, position addition and final RMS are unchanged.

The fused kernel compiled with SDK 0.9.0 using `cargo furiosa-opt compile --exact ops_vision_trf::encode_patches --dump-schedule experiments/media-vision-trf/schedule-row-major.json`. The generated schedule ends at 268441 estimated units. It has one `dma.sto_d`, the final output; the `DramProvenance` tensors are tiled views of the two original weight arguments. This is allocation/compiler evidence, not an NPU speed result.

Three bounded compile repairs are preserved:

1. A cropped K768 view retained an outer token stride and the TRF fetch rejected an unread live axis. `tiled-trf-rejected.rs` and `compile.log` preserve this. The supported path explicitly materializes the compact B8×K768 DM tile.
2. `DmTensorView::to_dm` is not an SDK API. The compact destination uses `DmTensor::new` and `to_dm_view`; `compile-dense-tile.log` records the API failure.
3. Direct final `[output120,B8]` redistribution requires unaligned two-byte DMA strides. `strided-result-rejected.rs` and `compile-compact-tile.log` preserve the failure. The supported path first transposes to `[B8,output120]` with the TU, then gathers contiguous token rows.

The probe uses actual supplied vision weights and four independent paths: scalar sixteen patches, new staged B16, new fused B16, and current fused B16. Outputs begin as NaNs. All stage/final outputs, permuted rows, one/seven active fixture rows plus computed padded rows, and separate normalization outputs are compared bitwise. Independent F64 projection samples preserve all K768 and BF16 tree boundaries and use the pre-existing frozen F32 accumulation interval. The interval oracle is distinct from exact scalar parity.

Timing variants are 0=scalar, 1=new staged, 2=new fused, 3=current fused. The order is `0,1,2,3,3,2,1,0`; completed-kernel-only timings exclude uploads/allocations/download/scatter, while inclusive timings include position gathering, uploads and scratch/output allocation for every path and exclude downloads/scatter for every path. The scalar micro preallocates its three stage buffers before launch; this is equal frontend work, not exact serving allocation chronology. No HTTP or whole-model throughput claim follows from this micro.

Host typecheck passed. The full standalone device build and actual NPU bit/timing gates remain pending at this source freeze.
