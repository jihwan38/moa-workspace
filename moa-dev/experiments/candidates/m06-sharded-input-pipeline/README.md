# M06 — consume pre-RMS shards and write packed input terms directly

Branch `opt/m06-sharded-input-pipeline`, following integrated M05 record `83d09dd`.
Source recipe chooses E74's crate plus frozen M05/M03 FFN ownership. E104 is the
independent performance control. This is a coupled pre-RMS/quantizer/storage
replacement, not a newly counted whole-FFN family.

## Motivation and exact prior overlap

M05's first amax begins earlier in the compiler schedule, but its complete input
broadcast ends later than M03. In the same Arena allocation M05 is1.08% slower
than M03. This does not identify measured per-engine stalls, but it rejects the
assumption that removing the initial copy makes the whole operand ready sooner.

Historical `ffn-sharded-input-rms-e53` already removed the pre-RMS whole-H gather.
Both old variants compiled but were closed on static cost; they fed replicated
BF16 up/gate and had no native quantizer consumer. Historical QKV/O direct-packed
quantizers also compiled, but their approximate parent kernels have separate
correctness limitations. M06 reuses their transport mechanisms with the current
three-term E5M2 preparation and M03 ownership; no prior validation is inherited.

## Full path and changed contracts

`ops::decoder_feedforward` keeps the canonical residual for the unchanged final
residual update. A typed DMA produces H480×8 RMS input shards, as before.
`normalize_sharded_inter_slice` returns BF16 normalized H480 shards directly.
The private `feedforward_sharded` now accepts those shards; its canonical
`feedforward` wrapper converts its input if used. Public competition signatures
and the final output path are unchanged.

```text
residual -> H480 shards -> inter-slice pre-RMS -> BF16 H480 shards
 -> local absolute maxima -> global max -> power-of-two restoration scale
 -> three E5M2 terms committed into disjoint final packed DM tiles
 -> prepared terms to both cluster leaders -> ring32 replicas -> native TRF
 -> unchanged materialized up/gate -> M03 local GeGLU and L-owned down
 -> F32 cluster-partial sum -> BF16/global scale -> final RMS/residual -> HBM
```

Before leader distribution, activation/scale work stays on cluster0. Physical
slices0..7 each own480 H coordinates. Each final term tile contains480 E5 bytes,
an8-byte multiple; three nonoverlapping tiles fill1440 bytes per active slice.
The low term reads the stored high term, and the third term reads the F32 sum of
the stored high/low terms. No uninitialized fourth term or padded slice is data.

M03 native up/gate, FP8 weight materialization, local GeGLU, L-owned down and
cluster-partial reduction are retained. Their weight bytes and register lifetimes
are not claimed eliminated. New preparation can change allocator and scheduling
interaction; total performance must include these effects.

Removed boundaries: pre-RMS gather to canonical H, BF16 repartition into
quantizer shards, scale DMA before quantization, and three DMA copies of the
quantized terms into their packed tensor. Added: an eight-slice global maximum
reduction and direct packed-view lifetime dependencies. v3 uses one explicit
Main scalar broadcast after canonical reduction to define scale ownership.

The original global amax/scale floor and three-term arithmetic are retained for
finite values. Local max then global max preserves that maximum. Moving the
positive floor before max also preserves it. The reused sharded RMS changes
ClipAdd to FpAdd and fuses EPS/Sqrt, so bit-exact equality is **not** asserted.
The normalized BF16 boundary remains; unchanged reference criteria must be run.

## Bounded lowering repairs

- v1 compiles Rust and emits a schedule but fails SFR lowering with
  `no entry found for key` at `sfr_impl.rs:868`. No executable/NPU result.
- v2 moves the scale floor into local maxima, leaving inter-slice Max without a
  subsequent VE operation. The same lowering error remains. This does not prove
  Max or the proposed architecture is unsupported.
- v3 replaces new replicated-Max/scale-selection composition with the canonical
  scalar-Max output used by old quantizers, then explicitly broadcasts scale via
  Main ring8. This removes the replica-selection reshape and retains the removed
  DMA boundaries. Its outcome is recorded separately.

A backend assertion is not itself proof of an SDK-wide limitation. Frozen sources,
logs, source plans and failed schedules remain separate from executable builds.
The first hardware gate is a complete unchanged benchmark with contemporary
E104 and M03. Regression/<5% gain closes without a parameter sweep; a promising
candidate requires independent repeats and expanded numerical validation before
adoption/submission. Live runtime remains E104 during the experiment.

## Final outcome — executable v4/v5, runtime timeout, on hold

v3 still fails the same SFR lookup. v4 moves only the low-term cast/add stream
from Sub to Main, keeping the direct packed views and arithmetic; full exact
compile and host build then pass. This establishes a composition-sensitive build
change, not the exact backend bug. QKV/O images remain byte-identical to E104.

Arena104566 / **E119**: candidate FFN times out before its first output comparison.
Its unchanged QKV/O pass12 comparisons. E104 before, original M03, E104 after
pass45 comparisons; their FFN medians are336352,331309,336282 respectively.
The candidate has no valid FFN cycle count or correctness result.

A final bounded runtime isolation v5 restores the proven separate hi/lo/third
buffers, Sub sum and three DMA packing writes. Sharded pre-RMS, global max and
Main scalar broadcast remain. This reduces the claimed optimization scope and
counts the restored copies; it is not called the direct-packed path.

Arena104588 / **E120**: v5 also times out before its first FFN output comparison.
Candidate QKV/O again pass12 comparisons. E104 before, original M03 and E104 after
pass45 comparisons, FFN335802,331423,336132. Thus direct packed aliasing alone
cannot be declared the failure cause; the shared new pre-RMS/quantizer path and
its lowering/lifetimes remain unresolved. Arena service failure is not supported
by these successful contemporary controls.

| Static only | M03 | M05 | M06 v4 | M06 v5 |
|---|---:|---:|---:|---:|
| Total schedule | 154440 | 156992 | 155735 | 157173 |
| DMA instructions | 37 | 36 | 32 | 35 |
| Third term preparation ends | 35822 | 39770 | 12977 | 12819 |
| Input broadcast ends | 56931 | 60233 | 49972 | 60065 |
| Packing DMA count | 3 | 3 | 0 | 3 |

Earlier static preparation is not hardware performance evidence for either
invalid candidate. The compiler intervals are not summed into a speedup.

Five frozen source variants are implementation/diagnostic attempts in one M06
combination. Three fail lowering, two produce complete images and fail NPU
execution. The prototype is **held unresolved**, without more parameter/context
sweeps. E104 remains adopted; no official submission or new ranking result.
A retry requires evidence of a specific failing boundary and supported repair.
`verify.py` checks source scope, frozen compile logs, images, actual uploaded
archives and all valid/invalid Arena stages. No candidate FFN comparisons passed.
