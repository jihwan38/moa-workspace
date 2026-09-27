# E137 selection: activation-resident native O — 2026-09-27

Read-only review of adopted E134/C09 `src/device/sliding/projection.rs:160–245`, historical `output-fp8-sixteen-columns/projection.rs:283–335`, and installed SDK0.8.1. No compile/NPU/source edits. This is a specific precision/reduction repair of the native-O family, not a newly invented architecture or predicted speedup.

## Complete compatible path

`BF16 x[4096] -> 16 canonical K256 shards per duplicated cluster -> local absmax -> global K16 max -> common power-of-two s -> actual E4 [hi, lo16, third2^22, fourth2^27] using cumulative restored residual -> replicate each K shard across 16 output-row groups -> E4 TRF[Term4,K256]`.

Keep C09 two independent compact original-E4 weight DM stages `[row60,K256]` and existing row120 destination. Each stage uses weight as Main stream; no lookup:

```
fetch/collect Time[Row60,K/32], Packet[K%32]
contract_outer Time[Row60,K/64,Term4], Packet[K%64]
contract_packet<1>
contract_time<Row60,Term4>           # preserve Term; reduce only four K64 chunks
contract_lane<Row60,Term4,1#8>       # one useful native lane
vector narrow 1#4
Mul0(normalized coefficients[Term4,1#8])
intra_slice_reduce<Term, Row60, 1#4>(Add)
widen 1#8
inter_slice_reduce<existing OutputRows, Row60>(Add)
existing row-pair transpose / trim2 / indexed commit at row0 or60
```

`Row60,Term4` keeps Term innermost in VE time: inner-time1, unlike rejected free Row15 in the weight-resident O design. IntraTerm reducer uses `ReduceAccTree`, not E47's `vector_fp_zip(AddF)`. SDK `engine/vector/tensor/vector_tensor.rs:510–524` permits IntraFirst widened stream -> inter-slice reduction, and E134 down already uses this class of chain. Source expression appears supported; exact shape lowering and emitted native/TRF indexers remain unverified.

After existing canonical H480 F32 gather, use the existing scaling pass with **Mul0(common s), then Mul1(channel scale), then first BF16**. This adds no F32 partial materialization boundary. O RMS/residual/epilogue remain unchanged.

## Why use global s

Per-K256 s would require restoring column scale before the K16 sum. Ordinary FP Mul cannot be added after intraTerm reduction in the public VE stage order. Folding s into each term coefficient before Term sum is a different arithmetic order, and general BF16 overflow/underflow witnesses exist. Global s lets both reductions stay normalized and restores s afterward in the already-required channel-scale pass. It avoids both unsupported stage reversal and the per-column F32 spill. It does add a K16 amax reduction and scale distribution.

Both clusters can quantize the same HBM vector with the same s; a new cross-cluster max is unnecessary if all16 K shards are present in each. Quantization need not run on all512 row/column slices: do it on16 valid K owners/cluster, then replicate terms. Source `to_dm` mapping must retain K ownership; this is not an offline-prepack proposal.

## Full costs, not an estimated gain

| Item | Current C09/E134 | Proposed compatible native path |
|---|---|---|
| Weight HBM and packed E4 DM |15,728,640B total, two60row stages|Same; no weight assembly/transposition/materialization|
| Weight Main stream|E4 -> BF16 lookup, two stages|Original E4 direct stream; removes two table/decode stages|
| Activation resident payload across512 slices|512×256×2=262,144B BF16|512×4×256=524,288B E4; doubles resident payload/one TRF staging amount|
| Native arithmetic|One BF16 product per logical term|Four E4 products plus weighted F32 term reduction; native K64 vs BF16 K32 is not a cycle prediction|
| Activation preprocessing|HBM replication/TRF|absmax/global max/scale, four casts plus residual arithmetic, temporary VRFs/DM and replication of four terms|
| Coefficient storage|None|128B/slice VRF=65,536B aggregate plus creation/distribution|
| F32 projection destination/gather|3,840×4=15,360B logical|Same if both reductions fuse as above; no 512×120×8×4 padded F32 partial|
| Epilogue scale|channel-scale Mul|common input-scale Mul0 then channel-scale Mul1, existing BF16 boundary|

The current input may be read by multiple engines; the table gives resident logical payload, not bus-counter traffic. Term TimeBroadcast should reuse fetched weight packets (official Outer/Broadcast), but increases contraction work; it does not make four native products free. The weight-ingress dependency remains and may dominate. Do not infer saved cycles from prior marker sums or bytes.

## Precision and relation to E47/E136

The independent normalized-domain CPU audit covers19,328 BF16-significand F32 values up to255: factors[1,16,2^22,2^27] have cast maxima[255,128,256,256], maxabs reconstruction error2^-37, and exact reconstruction for |u|>=2^-29. For the public O scalar generator `BF16(-1+2*k/2^24)`, all21,887 admissible scale/value pairs were exact in the CPU model. Global s is compatible with that check. It is not a proof for arbitrary BF16, native accumulation, reduction reassociation, or final RMS tolerance.

E47 case7 and the independent .53125 cancellation fixture are exactly reconstructed by both four-term schemes in CPU; original-reference final expected values .609375/.59765625 are preserved there. E47 instead used two terms on native lanes, pair AddF, per-column scales, a full F32 partial DM pass, and an extra BF16 raw-dot boundary. The proposed sequence specifically changes those failed precision/cost mechanisms; merely increasing E47's term count would not describe it.

E136 QKV success would establish native E4 and quantizer pieces under its own tested layouts. It does not validate O's K16 reduction, global-scale placement, or final output RMS. Reuse the unchanged eight-case old O boundary host plus a global-outlier/mixed-K-scale cancellation case against the original BF16 reference, exact candidate image, and same public whole-O controls.

Verdict: a defensible one-prototype follow-up if a complete O comparison is authorized, contingent first on the fused intraTerm -> K16 inter-reduce exact graph. It avoids the weight-resident O Row15 refetch penalty and has a concrete distinction from E47. Neither global amax overhead nor four-term contraction cost is known to win; failure should be classified by the actual mechanism rather than copied from E136's different operand ownership.

Primary API sources: https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/outer.html ; https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/register-files.html . Numerical CPU artifacts are frozen under E136 `numerical/`; this note does not replace them.

## Final shape/ownership clarification for E137 selection

Use quantizer cluster mapping `m![2]` (same x duplicated), `QuantSlices=m![1 # 16, Qs / 256]`, and in-slice `Qs % 256`. Thus16 valid owners per cluster, each with256values; two clusters each see the same full4096vector. These are32 shard quantizers, not512 row replicas. Reduce local maxima to `Slice=m![1 # 256]` within each cluster; both maxima and resulting s are identical for the duplicated finite input, without a cross-cluster max. Keep this s semantically replicated rather than labelling it as a different value per `Dummy2` output-row owner.

Distribute `[Term4,Qs%256]` from those16 owners to `OutputColumns=m![H/120%16,Qs/256]`, so each output-row group gets the same K256 shard/terms. Check actual DMA packet coordinates `k=256*(slice%16)+u` and each native TRF `[Term4,K256]`; do not select unsupported runtime cluster_tile as a shortcut. Input sharded payload before quantization is2×4096×2=16,384B logical and compact four-term output before replication is2×4096×4=32,768B. Both numbers exclude descriptor padding/transfer replication and must be checked in the emitted graph.

Existing canonical final `Cluster=m![1 # 2]` is one valid cluster, not replicated full outputs. Existing H3840 gather keeps15,360B logical F32 result. Move the common scalar s from replicated clusters to this existing output mapping with a supported DMA view; verify cluster source ownership/constant equality. Stage graph must show final `intraTerm -> K16 interReduce`, no pre-K extra BF16 and no full padded partial DM; final epilogue must read common s then channel scale before first BF16. Preservation of that order and the global-aMax equality are source/descriptor/NPU gates, not assumptions to omit from the build recipe.

## Decision after E136 hardware result

E136 whole Arena110367 completed45 comparisons and regressed11.9973426%
(116321 versus E134 before102925/after104796, paired103860.5). Its full
production cancellation job110365 passed16384 outputs and8380416 untouched
cache values. Native four-term precision and weight-resident performance are
therefore separate findings. No extra performance repeats or parameter sweep
is selected for that frozen implementation.

E137 selects the complete activation-resident O path above, with explicit
source and control E1344f8fe7f. Preserve E134 QKV/FFN images. The actual branch
start must be the main commit after E135/E136 evidence is integrated and pushed.
Exact compile plus small descriptor/order checks precede early whole NPU
screen. A positive result, including below5%, freezes source for independent
controls, extra6 and original-reference O cancellation/mapping checks. A
source/shape bug receives a supported mechanism-based repair; actual regression
or an unresolved runtime failure receives its own scoped decision.

## Lane4 review before the first NPU screen

The original Time4/Lane1 form compiled as frozen `native-v1` (static33950);
it has no NPU result. A supported spatial form was identified before the screen:
TRF Lane=Term4, Element=K256, original weight stream without Term time repetition,
native lane folder interleaves the four terms into one F32 packet. Multiply its
four components by the normalized coefficients, then use packet
`ReduceAccTree` followed by K16 inter-slice sum. Public lower's
`reduce_in_packet_only_needs_one_slot` test directly supports packet4→scalar
reduction. This avoids E47's pair-AddF resource conflict while using native lanes
for the four terms; actual emitted mapping still needs checking.

The old coefficient layout `[Term4,1#8]` cannot be read as one contiguous four-way
operand (installed `engine/vector/operand.rs` requires broadcast or contiguous
packet cells). Build compact16B with two adjacent8B Sub transpose/commit pairs,
then replicate/load a padded32B VRF packet. Include that actual packing cost.
Preserve v1 and compile a distinct v2; choose the supported complete form before
the first NPU screen, not an in-place rewrite of measured evidence. Arithmetic
factors and precision cases remain unchanged. Native parallelism alone is not
a whole-O speedup claim.

## First hardware result and the bounded cost diagnostic

Frozen Lane4 v2 completed Arena110453 with45 comparisons PASS. O took56163
cycles against E13449037/49447 (paired49242), a14.0551% regression. Its actual
projection oracle110458 passed34560 BF16 values exactly, including a value
whose fourth term is nonzero; full public O/RMS/residual case7 in110465 passed
all3840 expected values0.609375. This establishes the tested precision and
mapping, not a performance gain.

The emitted graph exposes two new costs that a specific relocation can change:
the global-max instruction uses ratio256 despite only16 valid input shards, and
four-term replication moves512KiB before the next weight stage. Their static
durations2562 and5444 are not additive runtime savings. One frozen-image NPU
diagnostic is selected to compare actual input/global-max/quantization/term-DMA,
weight readiness, native and epilogue boundaries against E134, with plain
whole hosts in the same allocation to check profiling effects.

The conditional repair is consumer-local quantization: distribute the original
BF16 K256 input to all output rowgroups first, compute the same global K4096
scale separately in each identical rowgroup using K16 reduction, then produce
the four terms directly where their TRF consumes them. It removes packed-term
distribution, increases logical input HBM replication16KiB→256KiB and active
quantizer storage, and changes the reduction scope. It does not reduce term
count, assume input values, change tolerances or promise a gain. Actual profile
evidence and supported mappings must select the repair before implementation.

## Measured repair selection and first positive result

The frozen-v2 diagnostic Arena110511 passed57 comparisons and reproduced the
whole-kernel regression:55941 versus contemporary E13448579/48703 (+15.0079%).
The six run/cluster observations place median first-weight completion at15051,
quantization completion at23814.5, term distribution at23815.5–31242.5, TRF load
completion at31835.5, and first native start at31836.5. Thus native execution
followed activation readiness immediately even though its weight had arrived
much earlier. The first projection span fell from9541 to2479 cycles, but its
start moved from17112 to31836.5. These intervals include dependencies and queueing;
they are not pure service times or additive reclaimable savings.

This evidence selected frozen `native-v3`: original BF16 shards go to all512
consumer slices, each rowgroup computes the same full K4096 maximum through a
K16 reduction, and four terms are produced locally. The512KiB distribution DMA
is absent; local term storage/TRF load remain. Original input replication grows
from16KiB to256KiB and quantization runs in more slices. The exact static schedule
is25112 cycles versus v233532 and E13427432; only whole NPU establishes the gain.

Arena110558 passed45 comparisons: E13449585 → v342526 → E13449485, a14.1496%
decrease against paired49535. The implementation was frozen for two independent
repeats and extra6. Actual projection oracle110561 passed9 cases/34560 exact BF16
values with the unchanged fixture; whole O110569 passed3840 exact0.609375 outputs
including production RMS and residual. The final adoption/performance record,
when completed, is kept with the candidate rather than overwriting v2 evidence.

The general lesson is specific: faster native arithmetic can be retained when
its newly introduced producer/distribution delay has an implementable repair.
Fewer logical bytes alone did not select this change. Applying it to QKV is not
automatic: O owns K256 per slice, whereas QKV's full H3840 residual creates a
different VRF/ownership problem. Direct Sub Switch→TRF is also unresolved; the
closest E55 implementation failed its output slice bitmap check, so generic
frontend API availability is insufficient evidence for that next route.
