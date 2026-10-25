# Fixed scale1 E4 sliding KV quality surrogate: read-only design

Reviewed production `r64-server` at `0bd19635`, fixed-cache kernel lineage `4bf039b2`, current micro `b16-fixed-e4-kv-micro` at `27471cbf`, and historical `kv-quality/src/ops_kv_e4.rs`. No implementation, build or device dispatch is part of this audit.

A minimal isolated numerical experiment is feasible while retaining all BF16 cache allocation/read APIs. Apply exactly BF16 -> fetch_cast F32 -> E4 conversion -> E4 table-lookup BF16 to sliding K and V after their original final BF16 head boundary. There is no per-head maximum, division, saved scale, reconstruction multiply or Q conversion. The current fixed append uses nearest-even E4 with finite saturation; exhaustive finite device coverage is a required micro result, not yet assumed here. Historical per-head E4 public-quality passes do not validate this different quantizer.

## Verified producer coverage

Each insertion is after K's final `rope_ds` / `apply_rope` and V's existing `value_ds` / `normalize_value`, before its first output/cache store. Q is retained unchanged.

| Active implementation | Actual caller/coverage | Insertion anchor |
| --- | --- | --- |
| `ops_cache_pools::sliding_project_qkv` | `host/runtime.rs:726`, canonical scalar | `ops_cache_pools.rs:47`, before cache scatter |
| `ops::sliding_project_qkv` | `host/runtime.rs:766`, legacy scalar | `ops.rs:73`, before cache scatter |
| `ops_head_batch::sliding_heads_b16` | prompt B16 `host/prefill.rs:478`; ordinary decode fallback `prefill_decode.rs:299` | `ops_head_batch.rs:432` |
| `ops_head_b16_blocked::sliding_heads_b16_stage` | direct blocked decode and `ops_b16_prefix::sliding` at line35 | `ops_head_b16_blocked.rs:68` |
| `ops_head_r64::sliding_heads_r64` | ordinary `Decode64::decode_sliding` | `ops_head_r64.rs:432` |
| `ops_head_r64_blocked::sliding_heads_r64` | current blocked `Decode64::decode_sliding` | `ops_head_r64_blocked.rs:52` |
| `ops_head_wide::sliding_heads_p128` | ordinary complete P128 `WidePrefill::sliding_after_qkv` | `ops_head_wide.rs:432` |
| `ops_head_wide_blocked::sliding_heads_p128` | blocked complete P128 without fused-head path | `ops_head_wide_blocked.rs:53` |
| `ops_head_wide_tile::sliding_heads_from_wide_tile` | partial Wide segments `prefill_wide.rs:858` | `ops_head_wide_tile.rs:53` |
| `ops_prefill_head_attn_commit::prepare_heads` | both current `sliding_fused_p128` and `sliding_fused_first_p128` | `ops_prefill_head_attn_commit.rs:630`, before headed scratch stores |

R32 is not a separate head producer: its exact hybrid uses the same R64 norm/QKV/heads and switches attention/FFN/O capacity afterward (`decode_r64.rs:385..428`). Group, same-owner Prompt512, pairs, three-owner512, Tail256 and Tail384 execute `GroupPrefill::advance_rows` over real PreparedWide states; `prefill_group.rs:987` calls each row's `sliding_after_qkv`. Tensor padding creates no extra cache owner. The ten sites cover these routes without modifying Group ownership or adding codecs to padded fake owners. The archived `ops_prefill` and scalar head-reference kernels are not active host producers in this call graph; preserve them as reference fixtures.

## Why the location is mandatory

Prompt B16 and partial Wide read both immutable old-ring K/V and current-chunk headed K/V, attend, then commit. Complete P128 does the same. Current fused P128 calls `prepare_heads -> attend/attend_first -> commit` in one graph. Rounding only a host commit would miss current-chunk attention and the start0 path, materially understating quantization error. Rounding headed scratch in fused `prepare_heads` covers both consumers and both fused entry points.

Scalar and independent B16/R32/R64 decode commit the current token before pooled attention. The same producer hook therefore covers their current-token and future-history reads. Leave those commit/mask/order rules unchanged. Do not compose multiple same-owner P128 stages into one graph: the archived four-stage wrap alias failure is unrelated and remains held.

## Minimal OFF experiment shape

Use one ordinary Rust source constant default false in a new fixed-roundtrip module, with a generic DM helper preserving the existing named layout. For the isolated enabled artifact, source-enable this constant for every producer from startup; no content, request, benchmark or width-dependent precision choice. No mutable per-call global mode, history conversion or mid-request precision switching is needed. Separate frozen OFF/ON artifacts with identical schedules can serialize all diagnostic rows and compare offline. This avoids changing the active kernel/host route solely to create the control.

The existing per-head `ops_kv_e4::roundtrip` is a useful placement/type precedent only. Reusing its `quantize()` would implement the wrong numerical contract. Its old host hooks predate blocked heads, composed B16 prefix and fused P128. Scalar uses a real named-axis DM move if necessary, as the archived scalar hook does; no unchecked physical relabel is introduced. A return in the same original DM layout limits downstream changes. No HBM cache datatype, byte offset, allocation, full-attention kernel or scheduler changes are required for this quality surrogate.

## Required gate

1. Confirm fixed E4 micro finite codec byte oracle and complete append/attention/O intended result first. Its source independently derives exact E4 magnitudes in units2^-9, nearest-even ties, signed zeros and saturation; inactive NaN rows are excluded from commits. It currently uses Rc64 synthetic pools/SLOTS, so a physical32 model fixture must derive its dimensions anew.
2. Prove each head layout produces original Q exactly and CPU-roundtripped original K/V exactly on shared inputs, with distinct per-row/head values, subnormals/zeros/near-ties, poisoned destinations and inactive tails. Fused P128 must expose/check headed scratch plus attention against the same rounded old/current K/V. This is the proof that a BF16-storage surrogate matches an eventual compressed consumer; table lookup representability is not a substitute for mapping tests.
3. Use fresh histories on both OFF/ON artifacts: prompt starts0/128/896/1024, same-owner512 and independent groups/pairs/tails, partial Wide17/31, B16 prompt segments, scalar continuation, independent decode widths2/8/14/16/17/31/32, reordered owners, reset0 and active P128 with R32 interleaves. Keep all arithmetic dispatch routes identical between controls, including partitioned LM. Split bounded jobs instead of exceeding70s.
4. Report all-vocabulary original-vs-candidate drift and top1; validate own-output greedy/stochastic behavior and finite logits. Candidate replay must be bitwise identical in logits and entire retained caches. Check sliding stored finite values belong to the E4-decoded set. Full-attention code is unchanged, but its cache values can differ indirectly after prior sliding layers change model activations; do not incorrectly require full-cache equality against the original model.
5. Six public accuracy suites, plus an actual C17 autoregressive history/width-regroup test, remain necessary. A one-token MMLU job alone does not exercise R32 decode. Quality-surrogate timing is not compressed-storage performance.

## Cost and eventual storage port

The surrogate adds two encoding/reconstruction paths per sliding producer in DM and retains existing HBM output stores. It needs no persistent HBM or whole-cache copies; exact DM liveness and timing remain unmeasured. No headroom claim follows from the compressed B16 resource estimate16.19%. The actual complete micro, then integrated serving, determines speed.

A sole compressed pool remains a broader ABI port: canonical `KvCache::SlidingCanonical` allocation/type and byte offsets, scalar append/read, B16 live-prefix append and composed/uncomposed readers, R64 live/full commits and full/512/768/mixed/R32 readers, P128/tiled prompt old-history readers and fused heads/attention/commit, diagnostic copy paths and pool identity all need consistent E4 handling. Full KV remains BF16. A temporary per-owner8MiB BF16 old-ring staging buffer could reduce prompt-reader implementation breadth, but adds decompression/copy cost and cannot be called a speed win. No duplicate full sliding pool is needed.

With physical32, BF16 sliding K/V is10GiB over40layers and full KV is2GiB. Fixed E4 sliding would be5GiB, so sole-storage savings are exactly5GiB with no scale metadata. This is capacity arithmetic, not measured allocator headroom. Fixed codec is idempotent on finite decoded E4 values, but double encoding should still be removed from a final compressed path for cost; the roundtrip quality hook should not silently remain in that port.

## Micro review scope

Current micro paired intended BF16 history/current K/V comes from the independent CPU codec; candidate is read before controls overwrite anything. Physical ring offsets use1byte E4 versus2byte BF16, with identical masks and Q. Whole-pool CPU scatters and immutable inputs are checked. ABBA awaits complete append+attention+supplied O and repeats an idempotent append at the same positions; setup/readbacks are outside both timers. Feedback supplies the same candidate attention to every arm, followed by a bijective row reversal. No concrete source control/timing defect was found. This is synthetic history with supplied O weights, not computed-model quality or an actual device PASS claim.
