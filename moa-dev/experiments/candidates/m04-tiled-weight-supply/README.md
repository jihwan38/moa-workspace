# M04 — activation-first preparation and tiled FFN weight supply

Branch `opt/m04-tiled-weight-supply`, parent `db89a50`. Source components come
from E104 `7f891f9c48e670fe014b2ac5bd2c92bae3bac433`; comparison is independently
selected by `baseline/e104.json`. M03's producer-local GeGLU/down ownership
proposal remains unimplemented. M04 follows the newer whole-FFN NPU diagnosis.

Scope: coupled FFN operand-supply redesign, applied to all three projections.
Row/column ownership and arithmetic remain explicit reused components, not a
claim that every part of the FFN architecture is new.

## Evidence and first design

[Two immutable E104 profile runs](../e104-weight-path/README.md) found a median
42.2k interval between pre-RMS and activation amax. Main has no other operation
there, while bulk weight DMA overlaps the small activation copy. This is evidence
for changing the producer/transfer path, not proof of a specific HBM bank stall
or 42.2k removable cycles. FP4 loads and FP8 materializations are also substantial.

```text
residual HBM -> unchanged pre-RMS, cluster 0
  -> amax and three-term quantization on cluster 0
  -> packed terms to ring leaders in both clusters -> Switch broadcast -> TRF
  -> up/gate, each with five row tiles:
       packed HBM tile -> compact FP4 DM -> tile FP8 DM
       -> native dot / input scale -> tile F32 block partials
       -> block-scale VRF multiply / reduction -> final F32 rows
  -> unchanged gather / BF16 / GeGLU
  -> unchanged down activation distribution / TRF
  -> down, five row tiles:
       packed HBM tile -> compact FP4 DM -> tile FP8 DM
       -> BF16 lookup/dot -> tile F32 partials
       -> block-scale multiply / local and column reduction -> BF16 rows
  -> unchanged global scale / post-RMS / residual / layer scale / HBM
```

This removes the initial normalized-input cluster replication before amax and
the requirement to load/decode a whole projection before its first dot. It keeps
a real FP8 DM boundary per tile, separate input TRF reloads for up/gate, F32 block
partials, all public inputs, BF16 boundaries and accuracy thresholds. It does
not remove total weight payload, total FP8 materialization bytes, or the
up/gate-complete -> GeGLU -> down dependency. Main lookup and Main DPE cannot
execute simultaneously. No claim of enforced double buffering: actual scheduling
and allocation determine overlap and lifetime.

| Intermediate | Owner, placement and layout | Lifetime / consumer |
|---|---|---|
| Normalized x / amax | cluster0, slice0 DM, H3840 / padded F32 scalar | quantization; no initial cross-cluster x copy |
| Quantized terms | cluster0 eight H480 leaders, DM three E5M2 terms | packed leader DMA to both clusters, then ring32 Switch |
| Up/gate x | all512 slices, 3×3840 E5M2 DM/TRF; scale VRF32B | reused over five row tiles; separately loaded for gate |
| Up/gate tile | l=7680c+30s+6i+r; H stays contiguous | DMA -> Main lookup -> Main dot -> scale reduction |
| Tile packed / FP8 / partial | local6×3840 FP4=11520B / E5M2=23040B / F32 6×240=5760B | consumed before final row commit; compiler lifetimes verified separately |
| Up/gate scale | local6×240 E4M3 DM -> 5760B F32 VRF | one tile's block reduction |
| Down tile | h=1920c+60floor(s/8)+12i+r; l=1920(s%8)+u | unchanged eight-column ownership, compact local row tile |
| Down packed / FP8 / partial | local12×1920 FP4=11520B / E4M3=23040B / F32 12×120=5760B | DMA -> Main lookup -> Main BF16 lookup/dot -> reduction |
| Down scale / result | 1440B E4M3 DM -> 5760B F32 VRF; 12 BF16 rows per row group | local L then 8-column F32 reduction before BF16 |

Up/gate six-row packed segments are 11520B, a multiple of256B, with H contiguous.
Some segment starts remain128B aligned because a full slice's30 rows occupy
57600B. Down retains960B packed and120B scale row segments; tiling does not fix
their transaction efficiency. Tile output commits are24B, divisible by8.
Scale DMA segments are1440B per up/gate tile; no assertion that all transfers
are256B aligned. HBM reshape only relabels physical order.

Added work: fifteen packed-weight DMAs and fifteen scale DMAs instead of three
each; fifteen FP4 lookup streams instead of three; repeated down lookup setup;
more dot/task setup and potentially longer input-register lifetimes. Removed
work: one raw activation replication before amax, duplicate cluster quantization,
full-projection producer/consumer completion dependencies. Quantized broadcast
still transfers to both clusters. Overlapping intervals are never added as gains.

## Prior related attempts and limits

- E56 used BF16/fused decode and tiled HBM, failed numerical/device execution;
  root cause unresolved. M04 uses compact independently typed tiles and retained
  FP8 + F32 boundaries with native up/gate. It does not inherit E56 validation.
- E73 repeated FP8->BF16 lookup for all up/gate tiles and eliminated F32 partials;
  it regressed. M04 uses native up/gate, retains partials, and co-designs activation
  preparation and HBM tile supply. Down repeated lookup remains an explicit risk.
- C07 direct FP4-to-native DPE variants failed projection probes. M04 keeps the
  materialization boundary locally instead of repeating that direct fusion.
- C15 split H and inflated weight/scale transfer costs. M04 preserves contiguous
  full H for up/gate and changes row tile supply, not K ownership.
- C14 pipelines L stages through GeGLU/down with extra reductions. M04 does not
  claim that architecture or its early-down behavior.
- Installed SDK `engine/fetch_adapter.rs` implements table lookup only for
  `Tu::Main`; moving FP4 lookup to Sub is not a supported repair.

First gate: compile the full graph, verify tiled transfers and shortened
producer-consumer boundaries in the lowered schedule, then one whole-kernel
NPU screen with contemporary E104 before/after and unchanged three-kernel host.
Fix concrete source/layout errors with preserved snapshots. Stop a measured
regression or <5% FFN gain without tile sweeping; a promising result requires
independent repetition and expanded correctness before adoption/submission.

Primary references: [DMA engine](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html),
[memory performance](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html),
[schedule diagnosis](https://developer.furiosa.ai/furiosa-opt/book/scheduling/diagnosis.html).
Runtime measurements, compiler estimates and design hypotheses are kept separate.

## Result — E116 closed after a hardware regression

The first source compiled without a repair. The exact compile and packaging
recompile produced identical image and schedule hashes. Arena **103108** ran the
unchanged official three-kernel harness with E104 before and after the candidate.

| Stage | FFN median cycles | Three input runs |
|---|---:|---|
| E104 before |334906|338574 /333729 /334906|
| M04 row-tiles-v1 |381507|382844 /381507 /379951|
| E104 after |336369|338334 /335170 /336369|

The candidate is **13.67% slower** than the bracketing-control median335637.5.
All45 output comparisons pass; the candidate itself passes15, including the
three FFN inputs/11520 final elements. This is one allocation, not independent
repetition or expanded projection validation. QKV/O images, the host, fixtures,
public signatures and accuracy criteria are unchanged. No official submission.

[NPU report](../../../../docs/experiments/e116-m04-tiled-weight-supply.json),
[decision](decision.json), [source/build/package verification](verification.json),
[frozen raw log](evidence/arena-103108.log).

## What the lowered graph actually changed

[Recomputed compiler analysis](static-analysis.json):

| Compiler quantity | E104 | M04 |
|---|---:|---:|
| Whole schedule |157045|165192 (+5.19%)|
| Main interval sum |94944|101424|
| DMA interval sum |126267|155447|
| DMA instructions |35|74|
| FP8 producer streams |3|15|
| Dot streams |3|15|
| Selected FP8 buffers' peak live tensor size |56.25MiB|22.5MiB|
| Total logical FP8 writes |168.75MiB|168.75MiB|
| Last pre-RMS Main ends |7511|13942|
| Activation amax begins |30761|13942|

The FP8 peak sums the compiler's selected tensor sizes and lifetimes; it excludes
other tensors and physical-page reservation/fragmentation. It is not a measured
whole-device memory footprint.

The early amax dependency gap really disappears **in the compiler schedule**,
but pre-RMS itself starts/finishes later because the scheduler puts an initial
weight load/expansion before it. Subtracting only the old gap would overstate
even the static effect. The first down dot starts119782, before the second FP8
tile is ready126410 and well before the final tile152732: tile consumption is
expressed, not merely a renamed whole-matrix buffer.

Weight and scale loads each increase3->15. Source-less compiler-inserted
DmaLoads also increase6->22; their exact hardware cost is not separately profiled.
The retained short-transfer/lookup/task setup costs are therefore concrete
remaining liabilities. Context sums overlap and cannot be added into a predicted
hardware slowdown. The 13.67% hardware regression is measured for the complete
combination; individual intervals/stall causes were not measured for M04.

Source helper calls retain the separate projection TRF-load expressions used by
E104, but lowering shares one up/gate input TRF allocation. Source order does not
force per-projection hardware reloads or a particular DMA priority. The frozen
source comment describes the intended source composition, not an enforced device
reload requirement. Lookup is Main-only in the installed public API, so this
prototype does not claim a Main/Sub decode-dot pipeline.

Decision: **do not adopt; keep E104**. No tile-size sweep, repeats or extra
accuracy jobs for this regressing implementation. This does not reject the
producer-local input preparation independently, or all streaming architectures.
Retry requires a changed cost mechanism, such as eliminating repeated setup or
materialization while establishing correctness; simply reducing the tile size
does not meet that condition. M03 remains an unimplemented distinct ownership
combination. The ranking goal has not been achieved by this experiment.

Recheck existing evidence without rebuilding or submitting:

```bash
python3 moa-dev/experiments/candidates/m04-tiled-weight-supply/analyze.py
python3 moa-dev/experiments/candidates/m04-tiled-weight-supply/verify.py --check
python3 moa-dev/experiment_index.py --check --check-live
```
