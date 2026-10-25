# Same-cluster norm output store grouping

Base: current c075 `4bab6ea3199d0de99ebe9d17a06df208ee7c9e2f`.
No serving call, flag, math, weight, precision, or HBM ABI changes.

The production `GroupPrefill::advance_rows` projection-owned branch calls
`ops_group_ffn_projection::{sliding,full}_projection_owned`. These execute the
post-attention normalization/residual/pre-FFN/quantization chain, projection-owned
FFN core, and post-FFN normalization/residual/gate in one device launch. The actual
norm implementation is `ops_norm_blocked_chain`, using `ops_norm_quarters` math;
the older `ops_norm_shardquant` sequential-four module is not this branch.

All five original module hashes match the frozen all-owner model source. The
three norm module bodies also match the source used in `r924/319594`. The current
composed sliding resource `ffn-projection-clusters-model/experiments/ffn-projection-model/sliding-schedule-a2.json`
estimates1,178,145 total cycles, including71,908 for the qinput store and37,442
for the canonical input-scale store. Their sum is an idealized9.28% stage ceiling,
not a wall-time saving prediction. The old matched standalone norm Task in r924
lasts259,328 cycles and has overlapping96,104/184,045-cycle late DMA spans. The
resource versus full-bin DMA ordinal mapping is not established; do not assign
these individual actual spans to source operations by ordinal alone. The current
production composed graph has no matched operation-level trace yet.

The input store is currently emitted directly from cluster`P/64%2`,
slice`[P/2%32,H/480]`, local`[P/128,P%2,H480]` to `P512H960`. This yields eight
480-byte source row fragments per live slice. The authorized candidate adds a
real DM placement into slice`[H/960,P/16%4,1#16]`, local
`[P/128,P%16,H960]`, retaining the cluster. Each live destination then emits four
15,360-byte contiguous stripes. There are16 live destination slices per cluster,
with61,440 useful bytes each. The real16-source fan-in is included in the graph;
no reinterpret is used to fake movement and no cross-cluster assumption is made.

Input scales first undergo the existing supported scalar-packet TU transpose
to remove the eight-lane padding. A real DM gather then targets one live slice
per cluster (`1#256`), local`[P/128,P%64]`. The final store writes four256-byte
stripes per cluster instead of scattered scalar elements. This adds one TU and
one real DM move; logical payload is2,048 bytes globally. `1#256` is one live
destination, not256 valid replicas.

Other stores are lower-priority: joined UP/GATE already writes contiguous
64×240 BF16 tiles (30,720 bytes per leader); qhidden writes P4×L960 contiguous
blocks; DOWN writes the existing P64×H240 blocked output; persistent residual
matches its7,680-byte local physical block. Hidden scales are still small
fragments, but their four8,057-cycle resource spans are separate from this single
bounded mechanism and are not changed.

Resource gate: compile the whole production-equivalent sliding composed graph,
including both new movements and all old FFN/postnorm work. Reject spill or
obvious complete-graph regression before asking for hardware. Compare against
the unchanged control with the same SDK/target. If plausible, prepare a strict
candidate-first full-chain micro with all retained buffers, provided layer
weights, distinct/permuted/padded rows, input immutability, and complete ABBA.
Only root dispatches Arena; no host activation before full numerical/model gates.
