# Sole fixed-E4 sliding storage, default off

Base: q022 source `9866221391bf463b6179b950ad26c63436de145c`. Root verified the frozen q022 image `c0985c25bca1a0fa80ceb6ab81bda96bee09dd9787eac4fcfb880209f210c2d7` passed all 124 public cases and the same fixed GSM32 with no new failed IDs relative to c081. This authorizes a storage implementation, not serving activation or a speed claim.

## Numerical and ownership contract

Keep all ten q022 head-codec hooks enabled. Q, full-attention arithmetic, every head/RoPE boundary and all current-chunk BF16 values remain unchanged. Replace only canonical sliding K/V owners with `HbmTensor<f8e4m3, Chip, m![Rc64,Ts,Ns,Ds]>`, where `Rc64` is the existing compatibility alias for physical32. Introduce a distinct `KvCache::SlidingFixedE4` variant; never reinterpret a BF16 allocation as E4. `Workspace::new_canonical64` remains BF16. An explicit diagnostic constructor selects E4 at allocation, before Model preparation; no mid-execution storage switch.

BF16 q022 head outputs are exactly decoded E4 values. A compressed commit recasts those values through the same BF16-to-F32-to-E4 sequence. A dedicated device gate must prove roundtrip idempotence, including signed zero and saturation. Readers use the original E4-to-BF16 LUT before the original dot/softmax/value arithmetic. No new query quantization, scaling, denominator extent, or reduction association is authorized.

Physical Rc32 and admission32 remain independent of query R64. Full-attention Fc256 allocation and full page ownership are unchanged. Prompt old-history reads complete before the corresponding live chunk commit; independent decode commits its current token first. Group/Prompt512/pair/three-owner/Tail256/Tail384 retain one awaited attention/commit launch per real PreparedWide state. No padded quarter is a cache owner; no multi-quarter shared mutable pool graph is introduced.

## Byte ABI

For E4, `token_bytes = Ns8 * Ds256 = 2048`, `ring_bytes = Ts1024 * token_bytes = 2,097,152`. Ring base is `slot * ring_bytes`; current-token append offset is `(slot * 1024 + position % 1024) * 2048`. The corresponding BF16 factors remain twice these values. Every metadata builder receives the storage byte width explicitly; RoPE/full-page/logit offsets retain their original BF16 widths. All existing mask/live-row policies remain unchanged.

One E4 pool is 67,108,864 bytes; 40 sliding layers times two pools is 5,368,709,120 bytes (5 GiB), versus original 10 GiB. Eight BF16 full-attention K/V layer pairs remain 2,147,483,648 bytes (2 GiB). Canonical allocation is therefore 7,516,192,768 bytes (7 GiB), exactly 5 GiB less. Verify public owning Buffer sizes at runtime. No whole-pool mirror is permitted. An initial scalar compatibility adapter may use a single shared 8 MiB BF16 K/V ring scratch, whose cost and allocation are reported separately; it cannot outlive an awaited scalar step.

## Interfaces and ownership split

| Owner | Files and responsibility |
| --- | --- |
| e2e_evidence | This tree. All `src/host/*` allocation/metadata/dispatch changes, scalar adapter/shared zero/copy utilities, storage identity, source/image binding, exact matched model and decoded-cache oracle. |
| batch_kernels | Separate `fixed-e4-kv-decode`. New `ops_fixed_e4_commit`, `ops_fixed_e4_b16`, `ops_fixed_e4_decode32`, and R64 reader variants. B16 live1..16 standalone commit, original attention, and complete attention/O; R32/R64 existing policy branches. No host edits. |
| round2_path | Separate `fixed-e4-kv-prompt`. New `ops_fixed_e4_prompt`, `ops_fixed_e4_wide`, `ops_fixed_e4_fused`, `ops_fixed_e4_commit_wide`. Partial B16 prompt, native P128/fused first+general and diagnostic fallback readers; no host edits. |

New device entries retain the corresponding original public argument order and all non-pool tensor types. Only pool dtype changes; all sliding offset tensors remain i32 with their current shapes but carry E4 byte offsets. BF16 headed K/V scratch is retained. B16 append offsets have exactly VALID entries; R64 partial live commits continue through B16 segments. P128 ring bases retain either the original scalar or duplicated query-cluster shape, as required by each entry. Full entries remain in their original modules.

## Required gates and provenance

1. Kernel codec/all-live-prefix/raw-pool gates: distinct rows, slot permutation, all32 physical slots, future/inactive NaNs, wrap/reset, unchanged neighbors and immutable inputs. Prompt gates include starts0/128/384/896/1024 and partial17/31/127, plus original diagnostic paths.
2. Matched whole-model runs use q022 BF16-rounded history from position0 as the control, identical token schedule, flags, output/LM route, and real physical ownership. Compare every BF16 logit and every decoded E4 cache value exactly; compare full BF16 caches exactly too. Include scalar/B16/R32/regroup, active P128 interleaves, and all11 packed owner layouts. No tolerance replaces storage parity.
3. Avoid simultaneous full models if memory requires it: sequential independently initialized passes with bounded host expected data or separately frozen paired reports containing complete deterministic hashes; do not compare R64/B16 LM routes against each other. Scratch/readback work is excluded from inference timing consistently.
4. Both storage variants use the same q022 precision helper and unchanged device caller images wherever not modified. Use a unique documented `FURIOSA_OPT_OUT_DIR`, fresh package capture, read-only registry comparison, and actual owning-buffer identity. Preserve all previous stale-OFF evidence; it is not a reference.
5. After structural parity, root decides public quality/performance dispatch. BF16 surrogate speed is not compressed performance; initial scalar fallback copies must remain included. Worker/default serving activation is a separate decision.
