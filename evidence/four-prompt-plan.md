Measured baseline checkpoint: root e5b67d2 merged as21842da, including exact
P128 QKV TRF32, parallel O, local FP8 FFN, blocked R64 logits and raw host reads.
The merge only conflicted in lib.rs module registration; both sets were kept
once. The existing two-prompt phase factor and active allocation checks remain.
All prompt, singleton and native blocked-output entrypoints still reject active
prompt ring/page aliases. Legacy P128 calls pass include_ffn=true; paired calls
pass false and run their common FFN afterward. No Worker change is included.

Four-group candidate: four disjoint canonical request allocations, each exactly
128 prompt rows contained in one full-cache page. Retain four separate WidePrefill
objects and PreparedWide states. Per layer, each request performs its own QKV,
head/RoPE, causal attention and O/residual. Join only the four normalized FFN
inputs into P512; run quantize_hidden, project_up_scaled twice(up/gate),
geglu_quantize, project_down_scaled; split projected rows; apply each original
postFFN norm/residual/gate, then advance all four cursors. About41.3MB temporary
FFN scratch plus the four prompt states; no extra model weight copies.

Control: four uninterrupted forwards using the same optimized P128 kernels.
Gate row joins/splits separately on every raw BF16 bit and changed group order.
Then fullmodel final logits, following decode/cache reuse and interleaved same-R64
decode must match. Report preparation, layer-total and maximum-quantum times plus
FFN/join/split stage costs. Production scheduling stays inactive. Preliminary
P512 kernel318035 is exact but only modestly faster than four latest P128 FFNs;
wholemodel improvement is unmeasured and the agent's weight-cluster repair is
still a candidate.

Wide profiling now separates heads, attention, KV commit and optional tile copy
from existing norm/QKV/O/FFN stages. Optional marks avoid Instant::now entirely
when profiling is disabled in ordinary wide/paired serving. No kernel dispatch,
mask, commit order or arithmetic changes are made by profiling.
