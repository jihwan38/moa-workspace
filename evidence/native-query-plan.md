Control checkpoint08a4780 uses ops_qkv_p128_trf32::project_query_batch with
BF16 activations, original E4 weights, H960 F32 partial reduction, BF16 dot
boundary and per-output BF16 weight scales. Root actual wide profile318048
measured27.356ms sliding QKV across40 layers; candidate begins with Qs only.

Candidate ops_query_native_fp8 preserves original E4 weights and their layout.
It clones the proven per-row power2 quantizer from ops_native_fp8_trf64, then
redistributes actual rows (no seeded replication) into the existing H960
shards. A64-row FP8 TRF reuses each weight fetch. Four F32 partials reduce,
the sum multiplies input dequant, rounds BF16, then applies the original
BF16 channel scale and rounds BF16 again.

This deliberately changes input precision. Quantized layout/scale and F64
quantized-input CPU dots must pass independently; BF16-activation control
error is reported, not asserted zero. Timing includes quantization and layout
redistribution. No host dispatch or model acceptance follows kernel success.
Full-model and public accuracy gates are mandatory before adoption.

Initial e9fc3ec host compile rejected f8e4m3::to_bits (not an SDK API).
The diagnostic compares decoded F32 bits instead, a one-to-one representation
for the finite E4 domain here including signed zero. No kernel change.

Pre-NPU source audit incorrectly followed SDK documentation describing into_vec
as logical. Actual NPU318079/r310 then returned1024 physical elements for
[P128,1#8], failing the diagnostic length assertion before any quant/projection
comparison. Restore physical b*8 indexing and assert1024. Kernel source/math
is unchanged; r310 supplies no arithmetic or timing evidence.
