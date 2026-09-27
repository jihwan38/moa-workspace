# E131 — Joint direct FP4 + fused reduction + M03 consumers

## Actual result — both variants not adopted

| Frozen variant | E128 before / after | Candidate whole FFN | Change vs bracket | Separate oracle |
|---|---:|---:|---:|---|
| Two rows,15streams (job109430) |328536 /328473|378328|+15.1668%|job109373:8cases/245760outputs PASS|
| Four rows×7+tail2,8streams (job109408) |328813 /328144|348871|+6.2082%|job109411:case0 launch timeout(code4)|

Each whole job passed45 standard comparisons across its three original hosts.
They are two implementations with one allocation each, not repeated measurements
of one binary. Sources, embedded images, standard inputs and actual uploaded
archives were verified. V2's separate diagnostic timed out before producing a
comparison; do not label it an observed wrong answer, nor treat its validation as
complete. The v1 launch display name retained a copied E129 label; its source,
image, host and output log identify the actual E131 v1 diagnostic.

[Final decision](decision.json) rejects both variants and stops additional tile
exploration. Runtime E128 remains unchanged. No extra6, adoption repeats or
official submission were run for these regressions. The prepared repeat/extra
scripts are unexecuted. The measured result does not prove the exact hardware
stall cause or reject every direct-FP4 design. A compatible implementation of
projection-owned clusters remains separately blocked by E129's target lowering.

[Whole v2 result](e131-joint-four-row-screen-job-109408-results.json) ·
[Whole v1 result](e131-joint-two-row-screen-job-109430-results.json) ·
[V1 oracle](validation/probe-results-v1.json) · [V2 diagnostic](validation/probe-results-v2.json).

The sections below preserve the full design and distinguish intended work from
actual emitted graphs and measurements.

This is a complete FFN combination from E128 source`cc82b9c`, measured against a
separately packaged E128 control. Actual start is`5ff84a7` on
`opt/ffn-joint-direct-fused`. It combines E123's corrected packed joint assembly
with scale-resident native/F32 reduction and the M03 L-half consumer layout.
It does **not** implement projection-owned clusters. E129's full prototype and
its unsupported cluster-tile lowering remain separately preserved.

- [v2 source](joint-four-row-v2/mlp.rs) and [recipe](joint-four-row-v2/recipe.json)
- [Frozen v1 source](joint-two-row-v1/mlp.rs) and [recipe](joint-two-row-v1/recipe.json)

The selected v2 keeps the complete dataflow below and corrects v1's conservative
row grouping. `Projection2×Row4×240×F32=7680B` plus the 32B input scale fits the
8192B VRF. Seven four-row tiles and one two-row tail cover all 30 rows. Four rows
are transposed as two row pairs (`Time=[Projection,Row/2]`, `Packet=[Row%2#8]`),
so each commit remains 8B. Native, lookup and scale-preparation starts fall from
15 to 8; E128 has 10 native/scale starts and two lookup streams. Logical payload,
the two packed and two scale copies, projection-major output and M03 consumers
are unchanged. This is one concrete setup-cost correction, not another family
or a row-size sweep. The frozen v1 source/recipe remain available; claiming that
two rows were required by alignment or VRF would be incorrect. V2 requires its
own exact graph, oracle and full-FFN measurements.

The following original design details describe the frozen two-row v1; v2 changes
only the producer grouping and row-pair transpose described above.

## Complete execution graph

```text
unchanged pre-RMS → actual three-term E5M2 preparation/broadcast
up/gate public compact FP4 HBM → two compact DM tensors
  → two raw Main copies → joint FP4 DM[Projection2,Row30,H]
up/gate public FP8 block scales → two compact DM tensors
  → two raw Sub copies → joint scale DM[Projection2,Row30,H/16]
shared activation TRF and input restoration-scale VRF
  fifteen fixed joint tiles, each Projection2×Row2:
  joint scales → Sub F32 VRF[Projection2,Row2,block240]
  joint packed weight → Main FP4 lookup E5M2 → native three-term dot
  → Mul0(input scale) → Mul1(block scale) → F32 H reduction
  → two-F32/8B commit per projection → joint F32[Projection2,Row30]
projection source regions at0/120B → actual TDMA to M03 local120 F32
unchanged BF16 cast → GeGLU/global scales → local down → F32 cluster sum → epilogue
```

L-half physical ownership remains unchanged throughout. The main joint FP4
assembly is the exact corrected E123 `load_joint_compact` implementation: public
HBM loads have compact destinations; raw Main copies fill the two padded joint
slots. No padded destination HBM DMA may substitute for those copies. The scale
assembly similarly uses compact HBM loads and two exact raw Sub copies. Its
16-byte FP8 commits are aligned and include all required conversion/transfer work.

The fixed two-row tile serves two constraints: joint scale VRF is
`2×2×240×4=3840B`, and each projection produces two F32 values for an8B commit.
A six-row-per-projection scale VRF would exceed8KiB. The uncompiled three-row
paired-output proposal would expose4B deinterleave offsets to DMA; it was not
pursued. There is no row-size sweep. Output stays projection-major so every
source projection begins at an8B-aligned0/120B offset, and each source row region
is120B. Actual gather descriptors must confirm the live region and padding bounds.

Only the producer and its transfer to the original local consumer change.
`broadcast_geglu_scale` and all subsequent GeGLU/down/quantizer/epilogue source
are byte-identical to E128. `((dot*x_scale)*block_scale)` and the first BF16
conversion remain in their existing numerical order. The existing three-term
input approximation is not claimed exact for arbitrary BF16 values.

## Paid copies and removed storage

Logical payload is not measured traffic or cycles.

| Up+gate work | E128 | E131 intended graph |
|---|---:|---:|
| Full FP8 weight DM write/read |112.5/112.5 MiB|absent|
| Raw packed joint assembly write/read added |none|56.25/56.25 MiB|
| Raw FP8 scale joint assembly write/read added |none|7.03125/7.03125 MiB|
| Native starts |10|15|
| FP4 lookup stream setups |2|15|
| Sub scale-VRF preparations |10|15|
| Raw packed Main copy streams |none|2|
| Raw scale Sub copy streams |none|2|
| Activation TRF/input-scale preparation pairs emitted |1|expected1; inspect graph|
| Block partial DM |already absent|intended absent|

The original packed decode fetch becomes the assembly source fetch; the new
joint native fetch is the additional packed read. Similarly, the old scale
fetch role moves to assembly, and the joint scale-VRF fetch is the added scale
read. This avoids counting one source read twice. Net logical up/gate DM
write/read removed is49.21875MiB in each direction, excluding table traffic and
compiler staging. It is not a performance prediction.

Public HBM weight/scale bytes, input preparation, final F32 row data, local gather,
BF16 outputs and the entire down/epilogue remain. The two packed copies are
physical streams totaling56.25MiB, not one hypothetical free view. Scale copying,
13additional lookup setups, five extra native/scale tasks, joint-buffer lifetimes,
mutable-write dependencies and any scheduling waits are part of the complete
measurement. No E123 percentage is added to an E128 percentage.

## Related attempts and gates

E103's separate direct projections with reload failed targeted correctness.
E123's corrected joint direct producer passed basic whole FFN but retained F32
block partials and used its E104 consumer. E128 validated fused reduction only
with materialized E5M2 weights. The current combination has not inherited any
of those accuracy results. It keeps both projections in each direct producer,
explicitly pays the packed assembly, and connects to M03 consumers.

The full compile gate checks exact compact HBM ranges, two packed Main and two
scale Sub assembly copies,15fused direct streams, absence of up/gate FP8 and block
partial DM, and complete aligned projection-major row commits/gathers. Concrete
API/layout failures receive supported repairs with frozen attempts. Static timing
alone is not a hardware performance verdict.

Then run the complete FFN with contemporary E128 before/after controls and the
actual-quantizer two-projection oracle: distinct weights/scales, row/projection
commit boundaries, cluster L-half boundary, H block/native boundaries, all FP4
codes, zero, cancellation and tiny values. This is a full-path experiment, not
an isolated helper substitution presented as an architectural result.

Adoption, further tuning and broader research remain separate. Any positive
complete-kernel screen qualifies for bounded independent controls and expanded
correctness even under5%; no tile sweep is planned. Projection-owned clusters
remain a separate SDK-lowering question rather than a claimed success of E131.
