# O projection: 256 slices, full reduction

Prepared 2026-09-21 from workspace commit `90f3c3fe16f4fe445f81e82d30b17b1b951e0082`.

`projection.rs` is the complete candidate; `projection.patch` changes only the output projection section. Q/K/V implementations and `apply_output_channel_scale` are preserved verbatim. The live source tree was not modified.

## What changed

The original O projection uses 32 live slice groups with 120 output rows per group. It runs four 1024-wide contractions, casts each partial to BF16, and sums those BF16 partials through a BF16 addition tree.

This candidate distributes all 3840 output rows over 256 live slices, with 15 rows per slice. The successful mapping is **interleaved**, `Slice=m![H % 256]`, `Time=m![H / 256]`. For slice `s` and temporal row `t`, the logical output row is `h = 256*t+s`. Every row and all 4096 reduction elements are retained. The weight mapping is `m![H / 256, Qs]`, while the replicated activation is placed in a single-lane BF16 TRF.

FP8 weights pass through BF16 table lookup directly into a full 4096-element contraction. Its result is cast to BF16 once, then committed with `m![1 # 4]`: one live BF16 scalar in an 8-byte physical slot. The three padding elements are not assumed to contain zero.

A regular `Broadcast1` gathers slices in `m![H / 256, H % 256]` order, which is dense H order. A transpose packs four live BF16 values into each 8-byte commit. The existing channel-scale implementation then runs unchanged using its original mapping. This avoids trying to DMA a 30-byte contiguous group of 15 BF16 results into a layout requiring aligned writes.

The initial requested contiguous mapping `Slice=H/15` with local `H%15#16` failed at weight DMA lowering. The compiler could not factor the logical H axis after padding each 15-row group to16. The failed source and exact diagnostic are retained under `failed-attempts/`; the successful candidate uses the interleaved arrangement above.

## Precision change requiring RNGD validation

**This changes reduction order and intermediate rounding.** Four 1024-element contractions with BF16 partial results and a BF16 pairwise sum are replaced by one 4096-element FP32 contraction with a final BF16 cast. Weight dtype, lookup decoding, input values, channel scales, output dtype, function signature, and downstream RMSNorm/residual operations remain unchanged.

No CPU oracle or RNGD job has run for this candidate. Compilation success does not establish numerical correctness. Apply only as an experimental candidate and run the full standard RNGD suite before accepting it. No values, positions, or test seeds are special-cased.

## Static results

| Variant | Schedule makespan | Speedup vs original |
|---|---:|---:|
| Original O projection | 195,823 cycles | 1.000× |
| Earlier FP8-fusion-only candidate | 162,813 cycles | 1.203× |
| This candidate | 100,581 cycles | 1.947× |

This is a 1.619× static speedup over fusion alone. It uses the **original ring256 input broadcast** for an independent comparison. That broadcast remains the longest instruction at 66,311 cycles. Weight DMA is 26,475 cycles, the fused projection computation is 9,863 cycles, and output gathering is 4,617 cycles. These durations may overlap; they must not be added to infer makespan or actual device latency.

Combining this candidate with a separate TDMA-broadcast change is promising but requires a new compilation, artifact hashes, and hardware measurement. These static numbers do not claim that combined performance.

## Reproduction and artifacts

The original crate's sources/manifests/toolchain were copied into Docker `/tmp/moa-output-256rows`, then only the scratch projection source was replaced. Exact compile command, run from the host workspace root:

```bash
./moa-dev/moa bash -c '
  set -o pipefail
  cd /tmp/moa-output-256rows
  export CARGO_TARGET_DIR=/tmp/moa-output-256rows/target
  export FURIOSA_OPT_OUT_DIR=/tmp/moa-output-256rows/target/furiosa-opt/kernel
  cargo furiosa-opt compile ops::sliding_attention_output --exact \
    --dump-schedule schedules/sliding_attention_output.json \
    2>&1 | tee logs/attempt-2.log
'
```

The command exited0, compiling one kernel and skipping14 nonmatching kernels with `cargo-furiosa-opt 0.8.1`, `furiosa-opt-std 0.8.1`, and `nightly-2026-05-01`. `evidence.json` records source/schedule hashes, context interval unions, longest instructions, baseline comparisons, and the failed first attempt. Raw schedules and compilation outputs remain under the isolated Docker scratch directory; no shared `/workspace/target` output was written.
