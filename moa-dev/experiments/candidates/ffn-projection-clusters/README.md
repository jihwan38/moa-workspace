# E129 — Projection-owned clusters, complete direct FP4 FFN

This candidate starts from adopted E128 `cc82b9c` and compares the complete
`decoder_feedforward` against an explicitly packaged E128 control. Work starts
at `ff7b9df` on `opt/ffn-projection-clusters`. It is an up/gate ownership and
producer/consumer redesign inside a complete FFN, not a six-row parameter sweep.
Full compile failed at the first cluster destination view; no executable or NPU result. See [decision](decision.json) and [SDK lowering audit](sdk-lowering-audit.md). The design below is the frozen pre-build hypothesis, not achieved behavior.

Source: [cluster-direct-v1/mlp.rs](cluster-direct-v1/mlp.rs).
Contract and decisions: [recipe.json](cluster-direct-v1/recipe.json).

## Complete path and actual ownership

```text
unchanged pre-RMS → actual E5M2 three-term quantization and input broadcast
public up FP4 HBM   → compact DM on physical cluster0
public gate FP4 HBM → compact DM on physical cluster1
corresponding public FP8 block scales → same projection-owned DM positions
one shared xTRF / input-scale VRF preparation
  ten six-row streams, simultaneously covering both physical clusters:
  packed FP4 DM → table lookup E5M2 → native dot
  → Mul0(input restoration scale) → Mul1(block scale) → F32 H reduction
  → paired8B F32 stores into60-row output
60-row F32 →60-row BF16, four-BF16/8B commits
select actual source cluster0/1 → TDMA up/gate into the original L-half owners
unchanged M03 local GeGLU/global scales → local down → F32 cluster sum → epilogue
```

The internal `Projection` axis owns physical clusters:0 is all up rows,1 is all
gate rows, and `l=60*slice+row`. It is distinct from the existing `Dummy2` axis,
which remains L-half ownership for GeGLU/down. Both HBM sources retain compact
`[slice256,row60,H3840]` byte order. Public `cluster_tile` selects the destination
cluster; there is no padded Element view pretending one matrix is two matrices.
Each FP4 source has29,491,200 bytes, and each scale source has3,686,400 bytes.
Actual emitted descriptors must prove those bounds and the cluster masks.

Each slice retains115,200B compact FP4 and14,400B FP8 scales. Six-row scale VRF
is5,760B plus32B input restoration scale. The ten direct streams consume one
shared activation TRF rather than constructing two sequential direct-projection
producers. This is a specific different ownership/state lifetime from failed
E103, but its correctness is still an open empirical question before execution.

After F32 reduction,60 rows occupy240B per slice. A separate cast writes four
BF16 values at a time and produces120B/slice; six BF16 values are never committed
as a misaligned12B packet. For each projection, a source cluster view is selected,
its in-cluster rows are named `L/60,L%60`, and **actual TDMA** routes them to
`cluster=L/7680`, `slice=4*(L/120%64)`, `element=L%120`. Only after that transfer
is the result relabelled to the original Dummy2/TileL consumer contract.
There is no reinterpretation of the Projection axis as L-half ownership.

## Removed and added work

All byte figures are logical payload, not measured bus traffic or cycles.

| Up+gate work | E128 | Candidate |
|---|---:|---:|
| FP8 weight DM write/read |112.5/112.5 MiB|absent in intended graph|
| Packed FP4 HBM load / subsequent DM fetch |56.25/56.25 MiB|same|
| FP8 block-scale HBM load |7.03125 MiB|same|
| Native six-row streams |10|10|
| FP4 lookup stream setups |2|10|
| Sub six-row scale preparations |10|10|
| Activation TRF / restoration-scale VRF emitted preparation pairs |1 (source writes2)|verify graph; expected1|
| Block partial DM |already absent|intended to remain absent|
| F32 local120 gather payload |120 KiB|removed|
| BF16 output redistribution payload |none across clusters|60 KiB logical target data|

E123's extra56.25MiB joint packed write/read is avoided, but it did not exist in
E128 and is not added to the E128 saving. F32 final-row data remains120KiB total;
BF16 output data remains60KiB total. The new redistribution's packet splitting,
source reads, destination writes, routing and any compiler staging are included
in the complete comparison. Its60KiB payload is not a zero-cost transfer claim.

Added risks include eight more lookup setups, direct decode/native interaction,
both projection inputs and scale views becoming ready for each shared stream,
mutable cluster-write dependencies, longer TRF/VRF lifetimes, per-slice packed
weight/scale extent doubling, and cross-cluster output routing. E128 already emits one activation preparation pair after compiler reuse, despite two
source-level loads. This candidate claims no additional reduction of that emitted
work. Smaller logical traffic does not predict the cycle outcome.

## Prior paths and validation gate

E103 already attempted two compact direct projections with separate TRF/scale
reload and failed targeted correctness. V7b already attempted `begin_interleaved`
and failed an internal axis contract. Neither is renamed here: this candidate
loads different matrices into different physical clusters and uses one common
producer per row tile. E123 corrected joint direct passed basic whole FFN while
paying packed assembly. E128 validated native/scale/reduction fusion with
materialized weights. Those results do not prove this combined path correct.

The first compiled graph must show exact compact HBM source coverage, exclusive
cluster destination masks, no up/gate full FP8 or packed Main assembly, ten direct
fused native streams, aligned row writes and genuine output redistribution.
Concrete source/API errors are repaired with frozen evidence. This gate supports
the full FFN experiment; it is not a substitute for completing the architecture.

Next, run the unchanged standard whole FFN checks with contemporary E128 controls
and targeted actual-quantizer up/gate oracles. Distinct projection scales,
row5/6 and59/60, L7679/7680, K15/16/63/64/3839, zero, cancellation and tiny inputs
must be covered. Keep `((dot*x_scale)*block_scale)`, F32 H reduction, first BF16
and global GeGLU scale placement unchanged. The inherited three-term input
approximation is not claimed exact for arbitrary BF16 values.

The three decisions remain separate: adoption requires full-path evidence and
bounded validation of any positive result, including below5%; further tuning is
not a tile sweep; the larger research question is whether projection ownership
removes decoded-weight storage without expensive packed assembly. A failed
prototype or static timing alone does not reject the whole family.

## Actual result

[E129 build](build-e129-clusters-v1-repro.json) preserves the exact source and compiler log. The error at `mlp.rs:92` is `tiling a non-in-slice axis is not lowered yet`. The public API exists, but its installed target lowering does not implement this Cluster tile. Custom mapping and raw/join alternatives were inspected; no supported no-copy replacement was found. This is a blocked implementation path, not a measured slow architecture. Runtime E128 stays unchanged.

The separately named `ffn-joint-direct-fused` candidate continues the complete FP4-direct experiment with explicit packed assembly costs. It is not the successful completion of this projection-owned proposal.
