# Read-only P512 F4 lookup hoist audit

Inspected `p512-f4-partialscale/src/ops_ffn_f4_projection.rs` and the actual-gated
copy in `p512-f4-partialscale-micro`; both SHA256 are
`d23a8ad5cdefe6c503da5ebd230db9938666b1affdca7ba4ee57199395292452`.
Actual r1225–r1228 source9e4bec2a reported7.337–7.813% complete gain and was held
against its unchanged10% gate. This audit does not alter that decision.

`stage_upgate` loads the F4 tile into DM before its macro, but the main
`fetch_table_lookup::<f8e4m3>()` is inside `chunk!` (line46), instantiated eight
times (lines77–84). Each projection-owned cluster reads all512 input queries in
eight64-query groups. `stage_down` likewise holds F4 DM, but performs lookup
inside each of four token-cluster chunks (line252; calls288–291). The candidate
therefore does **not** hoist the conversion, even though its HBM weight load is
already hoisted. The actual report's12 added table loads per cluster agree with
8+4 lookup pipelines; it also reports12 extra TU and14 extra DMA operations.

Proposed distinct change: perform fixed-code F4→unscaled E4 lookup once to a
contiguous E4 DM tile before each stage's query chunks, then fetch E4 directly in
the unchanged contraction. Keep every group960 scale after its local F32 dot,
before the existing inter-slice sum, and retain all later BF16 boundaries. This
would reduce12 lookup-bearing pipelines to two conversion pipelines, without
another HBM weight copy, per16 scale reconstruction or a new numerical rule.
It is different from the archived exact original-NVFP4 reconstruction: no block
scale/row-dequant arithmetic runs over every reconstructed weight here.

Per-PE original packed tile240×960 is115,200B; expanded E4 is230,400B. During
conversion their combined nominal DM is345,600B. The compressed tile must die
after its last lookup use, before the query/result lifetimes. With the expanded
tile resident, a64×960 activation DM is61,440B and a240×64 F32 partial is61,440B;
BF16 result/transpose buffers are30,720B each. These sizes suggest a possible
sub512KiB lifetime schedule, but do not prove allocation or overlap: group-scale
and norm buffers, leader replicas, input-source placement, compiler retention
and all whole-graph lifetimes require a fresh resource gate. TRF is separately
61,440B. No geometry, H960 association or consumer ABI changes are needed.

The tradeoff is one explicit expanded-DM write/read and doubled resident weight
DM against repeated LUT/table setup and decode. Actual old evidence has roughly
85–90k table-load cycles plus a116–122k rise in TU span sums; those overlap and
cannot be treated as guaranteed savings. The previously measured full graph was
1061.16→980.10µs, so a new variant still needs its own inclusive resource/device
gate. No implementation, build or new NPU work was performed in this audit.
