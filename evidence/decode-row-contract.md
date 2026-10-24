Independent-request decode attention adapters

`ops_decode::sliding_attention_row` accepts batched Q `[B,Ns,Gs,Ds]`, K/V
`[B,Ns,Ds]`, unscaled `row_index: i32[1]`, the original KV byte offset `i32[1]`,
mutable request K/V rings `[Ts,Ns,Ds]`, the original serial mask `[Ts]` (0 or
infinity), and batched output `[B,Ns,Gs,Ds]`. It gathers exactly that row, commits
its KV to the supplied request cache, runs the original sliding attention helper,
and scatters only that output row. Other requests' caches are separate arguments
on separate launches; this wrapper never mixes their histories.

`ops_decode::full_commit_row` accepts batched Q `[B,Gf,Df]`, K/V `[B,Df]`, the same
row index and original KV byte offset, mutable current request K/V pages
`[Tf,Df]`, and scalar Q scratch `[Gf,Df]`. Call once per request/layer before
attending all its pages with existing `ops::full_attention_first_page/page`.
These original calls use one scalar max/sum `[Gf]` and numerator `[Gf,Df]`
scratch, which may be reused only after that request finishes all pages.

`ops_decode::full_divide_pack_row` accepts scalar numerator `[Gf,Df]`, scalar
running sum `[Gf]`, row index, and batched output `[B,Gf,Df]`. It performs the
original division and writes only the indexed output row. Call before reusing
scalar scratch for another request and before the shared batched O projection.

The wrappers use the SDK's position-indexed SPM gather/scatter, so callers provide
one unscaled row index despite different Q/K/V/output row widths. They leave
KV offsets in the existing byte convention. Position indices are replicated over
all four PEs; using PE extent one is rejected by SDK0.9. They introduce tiny
row-sized temporary HBM storage for the original sliding helper's interface,
while retaining per-request cache allocation. They do not claim to parallelize
attention across requests; they connect independent attention histories to the
joint batched QKV/O/FFN kernels without full-cache packing.

The `test_decode_rows` probe selects output rows 15,0,7,3,11 at positions
0,511,512,1023,1024 with distinct initialized caches. It requires exact complete
KV contents, exact Q extraction, exact inactive-row preservation, and exact
sliding output parity with original attention. Full division is compared with
the original numerator divided by its returned sum with BF16 output rounding.
