# Exact B8-TRF vision result: Arena 319170

Run `r795-vision-trf-complete-submit` succeeded. Source `085d734`, artifact manifest `2b47bbb`, ELF SHA256 `92bb30a739de461a47c112336c9bd5ddc5c966fc14339d716b9caddeae455778`. Full structured evidence is `r795-evidence.json`, including the source log SHA256.

All four cases passed. Across stage/current/fused/permutation/copy gates, all 51 comparisons comprising 11,501,568 BF16 values were bit-identical and finite. All 2048 independent F64 projection samples also matched the nearest BF16 tree result exactly; the separately frozen accumulation intervals had no failures. The test retains zero/constant/random/sharp rows, row permutation, and one/seven active rows with every padded fixture row computed and checked. The 18 indexed row-copy checks compare the entire destination and preserve sources.

| Sixteen patches | Scalar | Existing B16 fused | B8-TRF staged | B8-TRF fused |
|---|---:|---:|---:|---:|
| Completed kernels, ms | 7.0218–7.0219 | 1.5127–1.5166 | 0.3762–0.3812 | 0.3129–0.3148 |
| Position gather, uploads, allocations and kernels, ms | 12.6798–14.0662 | 2.3557–2.4641 | 1.3899–1.4701 | 1.1378–1.2236 |

Both scopes exclude output downloads and scatter. The order was scalar/new staged/new fused/existing fused then reverse, with five completed-kernel repetitions and three inclusive repetitions per observation. These are frontend microbenchmarks, not serving/HTTP measurements.

Actual cluster0 trace comparisons:

| Raw trace duration | Existing B16 fused | B8-TRF fused |
|---|---:|---:|
| Task | 3008539 | 596092 |
| DMA interval union | 2860298 | 448401 |
| TU execution interval union | 356480 | 328857 |
| TRF ingress interval union | 10379 | 60078 |

The large reduction is in observed DMA span duration, while TU duration changes modestly and the larger input TRF increases ingress duration. This supports the intended weight-delivery reuse mechanism; it does not establish physical HBM byte counts or bandwidth. Profiling spans can include waiting and overlap, and no conversion from raw trace units to chip frequency is assumed. The current mapping has 256 live output/token slice groups with one live TRF lane; the new mapping has 32 live output groups with eight live token lanes, so arithmetic utilization is not inferred from the lane count alone.

The compiler estimates were 1215300 units for existing fused and 268441 for new fused, both with only the final output HBM write. Actual device traces, above, remain the evidence for speed.

Independent read-only review by `e2e_evidence` found no concrete mapping/timing-control defect: real DM redistribution moves B%8 into TRF lanes; the V32→output/120 rename preserves32 live groups and padding; named output-axis bias broadcasts across B; output transpose/gather precedes unchanged per-token normalization. Hardware parity confirmed those mappings.

Minimal production source consists of `src/ops_vision_trf.rs`, `src/device/vision/projection_trf.rs`, and its `src/lib.rs` module registration from `085d734`. `encode_patches` has the same arguments as `ops_vision_b16::encode_patches`. The probe is `src/bin/test_media_vision_trf.rs` in the same source commit. This experiment changed no host dispatcher. Whole-model and serving adoption belong to the parent integration gate.
