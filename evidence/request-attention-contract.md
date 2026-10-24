Independent-request pooled attention

Public pool axes are `ops_request_attention::{Rc=16,Fc=256}`. Sliding K/V storage
is `[Rc,Ts,Ns,Ds]`; full K/V storage is `[Fc,Tf,Df]`. Fc supports the current
16-request,8192-position allocator's256 physical full pages. Pool ownership must
follow `Blocks::sliding` and `Blocks::full`, and must reset when those are reused.

Full `full_attention_first_page_b16` and `full_attention_next_page_b16` arguments:
`device, q[B,Gf,Df], pool_k, pool_v, page_byte_offsets[B], visibility[B,Tf],
page_active[B,1 # 8], running_max[B,Gf,1 # 8], running_sum[B,Gf,1 # 8],
out[B,Gf,Df]`. Offsets are byte addresses relative to the relevant pool, equal to
physical_full_block * 512 * 512 * 2. All offsets must be legal, including inactive
rows. Visibility is exactly1/0. Page-active is1 only when that request visits this
page, replicated into the eight physical f32 padding slots. Inactive later pages
explicitly mask exponentials to zero and add no EPS, preserving previous state.
Call pages in order from zero through the longest active request. The separate
`full_attention_divide_b16(device,numerator,sum,out)` produces zero for inactive
padding instead of dividing0/0.

`copy_sliding_cache_to_pool(device,legacy_k,legacy_v,slot_byte_offset,pool_k,pool_v)`
and `copy_full_page_to_pool(device,legacy_k,legacy_v,page_byte_offset,pool_k,pool_v)`
copy complete legacy buffers into one selected pool slot/page. Their reverse
forms `copy_sliding_pool_to_cache` and `copy_full_pool_to_page` accept pools first,
then offset, then mutable legacy K/V. They provide an ownership-preserving bridge
without unsupported host subbuffer tensor construction.

`commit_sliding_pool_N` and `commit_full_pool_N`, N=1..16, accept batched current
K/V, an unpadded `i32[B=N]` byte-offset list, and mutable pools. Offsets address the
combined consecutive pool-position axes: `(slot*1024 + position%1024)*2048*2`
for sliding; `(physical_page*512 + position%512)*512*2` for full. Only the live
prefix is scattered. Invalid batch padding cannot overwrite another request.
Composite indexed axes are supported directly by SDK scatter; a mutable reshaped
view is not accepted by the compiler's scatter lowering.

The initial probe verifies full attention against repaired scalar attention for
16 independent histories, different lengths and permuted physical page indices.
It also verifies ragged-page state preservation,3-row updates, reverse migration,
finite-zero inactive outputs and16→3→16 batches. No whole-model speed claim follows
from this kernel probe. Actual full probe Arena317685 passed with all live outputs
bit-exact to scalar. Sliding Arena317713 also passed:39 live row comparisons
(159744 output values), future invariance,7-row updates and exact reverse migration.
Its current whole16-row warm launch takes about1.08ms; kernel correctness does not
imply a throughput improvement over scalar attention.

Sliding `sliding_attention_b16` arguments:
`device, q[B,Ns,Gs,Ds], pool_k[Rc,Ts,Ns,Ds], pool_v[Rc,Ts,Ns,Ds],
kv_byte_offsets:i32[B,Ts], visibility:f32[B,Ts], out[B,Ns,Gs,Ds]`.
Each offset equals `(physical_ring*1024 + physical_slot)*2048*2`; all offsets
must be legal, including inactive rows. Visibility uses exactly1/0. Before the
ring fills, slots up to the request position are visible; after position1023 all
1024 slots are visible. Commit current decode KV before attention. An inactive
row has an all-zero mask and produces finite-zero output.

The sliding image executes four B4 tiles, retaining the original32-key BF16
value partials and32-way interslice sum. Each K/V tile occupies64KiB per slice.
Four small index tiles are materialized in contiguous HBM because SDK gather
requires an owning HbmTensor rather than a tiled HBM view. No cache duplication
or request computation reuse happens inside attention.

Host ownership must survive batch shrink/growth and singleton fallback. After
pooled decode, either keep subsequent decode in pools or reverse-copy dirty
slots/pages before any legacy path. New request position0 resets ownership.

Faster equivalent sliding implementation: sourceb52ec49 in
`ops_request_attention_clusters::sliding_attention_b16` has exactly the same
arguments as the initial pooled sliding function. Host adoption only changes
that function path (or imports the module under a local alias). It splits B8
tiles across two clusters with four local requests each. Arena317736 passed all
39 scalar row comparisons and cache invariants, with warm534..547us versus
1079..1082us for the replicated-cluster control. A separate ring-gather candidate
changes index shape to[B] but remains unvalidated at this point.

Preferred sliding implementation after Arena317738:
`ops_request_attention_rings::sliding_attention_b16(device,q,pool_k,pool_v,
ring_byte_offsets:i32[B],visibility:f32[B,Ts],out)`. Sourcefaa33ce.
All tensor shapes except offsets match the preceding versions. Each offset is
`physical_ring*1024*2048*2`. The pool payload must retain contiguous
`[Ts,Ns,Ds]` storage per ring, as already declared. Padded rows use any legal
ring base and an all-zero visibility mask. Sparse current-token commit functions
and their offsets are unchanged. Host uploads16 ring bases instead of16384
per-token byte addresses. Actual39 live rows/159744 outputs remain bit-exact,
with future invariance, sparse updates, reverse migration and inactive zeros.
Warm16-row attention is181.8..183.1us (cluster-token-gather534..547us; original
replicated-cluster1079..1082us). These are isolated kernel timings.

Preferred full implementation after Arena317747: sourceafeac3b in
`ops_request_full_clusters::{full_attention_first_page_b16,
full_attention_next_page_b16,full_attention_divide_b16}`. All three argument
contracts are unchanged. Host changes only those function paths; full page
metadata, padded running statistics, commits and migration helpers stay intact.
Actual35 live rows/286720 outputs are bit-exact to scalar, including ragged
pages,3-row update, reverse migration and16->3->16 active counts. Paired warmed
firstpage: original59.94..61.27us, clusters47.43..48.51us. Nextpage cold overhead
is separate, and whole-model host ownership transitions remain an independent
validation. Freeze these kernels until whole-model profiling identifies a cost.
