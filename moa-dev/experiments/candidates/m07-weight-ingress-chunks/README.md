# M07 — split weight ingress, retain full decode/contraction

**Closed after NPU screening: basic correctness passes, no qualifying gain.**
Arena104739 / E121: v3 FFN336059 cycles, E104335871/335305, original M03 330118.
That is+0.14% versus the bracketing E104 median and+1.80% versus M03. E104 stays
adopted; no additional tuning, independent repeat, expanded accuracy or official
submission. See [decision](decision.json), [verified evidence](verification.json),
and [E121](../../../../docs/experiments/e121-m07-weight-ingress-chunks.json).

Branch `opt/m07-weight-ingress-chunks`, integrated parent
`0c6dea50db58e75bb89c705fd7b937eb3254a638`. Actual crate origin is E74
`9e2058ec0947498d7cb693fbe98e7b5f0b5c9ea7`, with M03's frozen FFN as the
method source. E104 is the performance control; original M03 is the companion
that isolates ingress changes from its already observed ownership effect.

This is a **local transfer-dataflow replacement**, not a new whole-FFN family.

## Question, prior overlap and first gate

Can smaller HBM transfer tasks allow short activation/constant transfers to
complete earlier without repeating lookup/contraction setup? Existing E104
trace shows small activation replication delayed across a large weight load.
Those intervals overlap and are not an estimate of removable time. Whether
the compiler/hardware actually interleaves the new tasks remains unproven.

- M04 chunks both transfer and consumers: five FP4/FP8/dot streams per matrix,
  with changed input preparation; FFN regressed 13.67% on NPU.
- C09 splits O ingress and lookup/dot into two stages; static-only closure.
- E47 tiled up decode also splits decode/dot, using compact tile allocations.
- FP4-native diagnostic v7c puts up/gate into separate Projection-axis views,
  then directly decodes/contracts; it does not subdivide each matrix's ingress
  while keeping the verified materialized three-projection FFN consumers.
- M05 changes activation preparation; M06 changes pre-RMS/quantizer ownership.
  Neither change is inherited here.

M07 retains M03 arithmetic, ownership and activation preparation. The changed
cost mechanism is **three weight DMA tasks become fifteen**, while the three
FP8 materializations and three large contraction streams stay intact. Chunk
sizes reuse M04's five-way split; this is not a tile-size search. One full graph
compile/package and an unchanged benchmark screen with contemporary controls
are the first gate. Concrete source/API errors may be repaired with evidence.
Close regression or <5% target-kernel gain without parameter tuning; a promising
gain requires independent repeats and expanded accuracy before adoption.

## Complete path and ownership

```text
HBM residual/gamma -> unchanged pre-RMS -> BF16 canonical H
 -> unchanged three E5M2 terms/scale -> cluster leaders/ring32 -> native TRF
HBM FP4 up/gate -> 5 disjoint row writes into each original full FP4 DM buffer
 -> one full FP8 DM conversion each -> native dot -> block scale/reduction
 -> cluster-local L7680 compact/BF16 -> unchanged local GeGLU
 -> local compact/ring32 -> BF16 down TRF
HBM FP4 down -> 5 disjoint row writes into original full FP4 DM buffer
 -> one full FP8 DM conversion -> BF16 lookup/dot -> block scale/reduction
 -> F32 cluster H partials -> gather/sum/BF16/global scale -> RMS/residual/HBM
```

Up/gate: cluster `c`, slice `s`, local row `t` own `l=7680*c+30*s+t`.
Each transfer selects `t=6*i..6*i+5`; its HBM view preserves the full 30-row
pitch and writes the same interval of a full `[FfnRow30,H3840]` FP4 tensor.
No compact intermediate or DM-to-DM assembly is needed. HBM row segments
are1920 bytes and each slice's transfer is11520 bytes.

Down: `s=4*r+k`, `h=60*r+t`, `l=7680*c+1920*k+u`. Each transfer selects
`t=12*i..12*i+11` and writes the corresponding full `[H60,K1920]` FP4 view.
HBM row pitch is7680 bytes; each contiguous K segment is960 bytes, and each
slice's transfer is11520 bytes. All physical write offsets/lengths meet8-byte
alignment. Destination tiles are disjoint and cover the full matrix exactly.

Each matrix still transfers29491200 logical packed bytes; all three transfer
88473600 bytes. All three full FP8 matrices still write176947200 logical bytes.
Full FP4/FP8 allocations and producer-completion boundaries are retained.
No lower peak memory or transfer bandwidth improvement is assumed. Lifetimes,
actual task ordering and extra DMA setup must be read from the emitted schedule;
source call order is not a scheduling guarantee. Static timing is not NPU speed.

SDK0.8.1 `HbmTensorView::to_dm_view` writes into a typed DM view and validates
DMA layout. The installed implementation is `src/tensor/memory.rs:484`; tiled
view shape is preserved rather than reshaping an already tiled view.

Public ops, host, inputs, fixtures, tolerance, scales and floating-point paths
are unchanged. `compose.py` produces only the three ingress replacements on
the pinned M03 file. Layout coverage and source scope will be checked separately
from NPU numerical correctness. Live runtime stays E104 during the experiment.

## v1 emitted cost and bounded v2 repair

v1 compiles and packages, but its static makespan is352298 versus M03's154440.
The ten up/gate transfer commands each retain a full30-row descriptor:
`225 × 256 = 57600` bytes per slice, instead of the intended11520 bytes.
Down's five descriptors correctly contain12 rows. Thus source tile coverage
alone does not establish the emitted transfer volume. `inspection-v1.json`
records an identical-image/schedule recompilation with graph/LIR dumps; these
descriptor counts are not measured HBM bus transactions or hardware timings.

v2 makes each up/gate HBM destination a compact six-row owned tensor before
copying it into the matching interval of the full FP4 buffer. The specific repair
removes matching source/destination padded extents from the HBM copy. It adds
ten DM-to-DM transfers and temporary compact allocations, whose setup and lifetime
costs must be included. Down ingress remains v1's correctly limited twelve-row
path. Full decode/contraction/scale/activation paths remain M03's unchanged code.
This is one mechanism-based repair, not a new family or a parameter sweep.
The expected short descriptors and any extra copies must be checked in emitted
IR before interpreting an Arena screen. No speedup is claimed by this change.

v2's compact-to-full **DM assembly** fails verification: its emitted DMA still
reads57600 bytes per slice from a11520-byte compact source. The compiler rejects
`source_accessing_range_local.end <= input_tensor_range`. Enlarging the source
with uninitialized padding is not a repair. v3 replaces only that assembly copy
with a typed raw Sub fetch/collect/commit:64 FP4 nibbles per32-byte flit, no lookup,
cast or arithmetic. A historical FP8 tail-copy used this route after the same
DMA range failure; FP4 stream support still needs its own full compile/runtime
check. Ten Sub copies replace v2's ten assembly DMAs and may contend with
activation/TRF preparation. Decode/dot remain the original full-sized streams.

v3 passes complete compile and host build. An identical-image/schedule inspection
compile confirms fifteen5898240-byte weight descriptor volumes, totaling88473600
bytes as intended; v1 emitted324403200 descriptor bytes. The compiler result is
checked independently of source coordinate coverage. Main conversion and dot
streams remain three each. v1 is closed on its explicit redundant-transfer cost
without an NPU claim; v2 has no executable. v3 proceeds to Arena104739 with
unchanged E104 before/after and the original M03 package in the same allocation.

| Static schedule only | M03 | M07 v3 |
|---|---:|---:|
| pre-RMS last stream ends | 7511 | 28971 |
| amax begins | 30761 | 32764 |
| pre-RMS to amax gap | 23250 | 3793 |
| input broadcast ends | 56931 | 59184 |
| complete makespan | 154440 | 160799 |
| DMA commands | 37 | 49 |
| added raw Sub assembly streams | 0 | 10 |

The shorter gap does not mean earlier operand readiness: pre-RMS itself is
delayed and amax/broadcast finish later. The first four weight chunks precede
the residual input transfer in v3's schedule. Chunking permits smaller tasks
but does not force input-first arbitration. This is compiler scheduling evidence,
not measured NPU engine timing; the screen decides whole-kernel performance.

## Final hardware result and closure

The unchanged four benchmark processes all pass, totaling60 comparisons.
Candidate FFN passes three inputs/11520 output elements. QKV/O images are unchanged
and their timing variation is not an M07 optimization effect. The original M03
companion is its frozen previously built package, not a recompiled substitute.

| Arena104739 stage | FFN median cycles |
|---|---:|
| E104 before | 335871 |
| M07 v3 | 336059 |
| Original M03 | 330118 |
| E104 after | 335305 |

This one screen does not establish statistical significance for the small E104
difference. It does establish a valid basic execution with no gain meeting the
5% experiment-budget gate. The modified source never changes public inputs,
weights, tolerances or arithmetic; full contract validation is still required
for any future adoption. This closed prototype is not independently promoted.

The actual NPU contribution of Sub copies, transfer arbitration, memory lifetimes
or other engine stalls was not measured. Do not assign the observed difference
to one of them from compiler intervals alone. Shrinking DMA tasks is feasible,
but this implementation does not turn it into a faster complete FFN.
Reopening needs a concrete change to arbitration or assembly/materialization
cost, not another chunk-count sweep. It does not reject every streamed design.

There are three frozen source variants, five exact compiles (three candidate
attempts plus two identical-source IR inspections), two host builds and one Arena
job. v1's valid executable is retained but unmeasured; v2 has no executable; v3
is the only M07 variant measured on NPU. Compile failure/cost closure are recorded
separately from hardware performance. Uploaded files, embedded images, raw logs,
source composition, coordinate coverage and descriptor volume are independently
checked by `verify.py`.

```bash
python3 moa-dev/experiments/candidates/m07-weight-ingress-chunks/analyze.py
python3 moa-dev/experiments/candidates/m07-weight-ingress-chunks/verify.py --check
python3 moa-dev/experiment_index.py --check --check-live
```

For a new build label, use `structural_build.py m07-weight-ingress-chunks
stream-assembly-v3 --plan` first. Existing source/build/screen evidence is immutable;
never overwrite its labels or substitute newly compiled images into old results.
