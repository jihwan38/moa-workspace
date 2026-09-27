# E134 weighted-v3 independent source/emitted-graph review

No blocking source or mapping error found. This review does not substitute for
NPU arithmetic checks. The companion JSON binds the exact source, graph and
schedule hashes.

`weighted-v3/mlp.rs:332–365` implements scaled lo×16, reconstructs hi+lo/16,
then scales the remaining residual×256. Lines388–412 produce coefficients
[s,s/16,s/256] using32B exact commits at term offsets0,32,64. V3 changes
engine placement of the sum and coefficient construction; its arithmetic
matches the frozen CPU bound. V1's SFR ICE alone did not establish an
unsupported native term-reduction family; v3 fully compiles.

In actual `evidence/e134-weighted-v3/graph.json`, RenegadeCommand#O249 keeps
native output `[H60, TileL/64%30, FfnTerm3, packet4#8]`. Narrowing preserves four
block lanes, Mul0 consumes the coefficient VRF, and `LReduce(+)[axes=FfnTerm]`
removes only the three-term time axis. The16B commit produces `[H60,TileL/16%120]`;
`commit_projection=0+[1800:4,4:1]` covers7200 consecutive F32 partials per slice,
matching the existing blockscale/local-K/cross-cluster suffix.

T261 holds `[FfnTerm3,1#8]` F32:96B per slice. DmaCommand#O240 transfers96B per
source slice and keeps separate cluster0/cluster1 sources across its eight PE
descriptors. The public SDK's `engine/vector/operand.rs:813–831` requires the
same cluster/slice and permits a scalar broadcast per packet; here the only
matching operand time axis is FfnTerm, so logical reads are offsets0,8,16 F32
and each is broadcast to four block lanes. The frozen graph does not expose
physical SFR read-index fields; this is a graph/SDK interpretation, not an
independent decode of the machine instruction.

The native instruction is schedule index637, static lifetime131280–140543.
Concurrent VRF allocation is consistent: blockscale bytes416..6176,
term-coefficients6176..6272, and post-RMS weight6272..8192 per slice. There is
no overlap. Intra-slice reducer verification preserves the packet and reduces
only FfnTerm (`furiosa-opt-lower/src/verify/engine/vector/tensor.rs:359–399`).
Its three terms are innermost time positions of each H/block group; the graph
shows the intended grouping, but does not expose the final accumulator-reset
register fields. The down boundary oracle and same-image full-FFN stress are
the value-level checks for coefficient selection, term sum and group resets.
