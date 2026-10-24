B16 prompt attention integration

`src/ops_attention.rs` implements `full_attention_first_page_b16`,
`full_attention_next_page_b16`, and `full_attention_divide_b16`. Q/output are
`[B,Gf,Df]`; each shared K/V page is `[Tf,Df]`. Additive masks are `[B,Tf]`:
0 for visible keys, -1e30 otherwise. Running max/sum are physically padded
`[B,Gf,1 # 8]`, because reading one tightly packed f32 per slice violates SDK 0.9
DMA alignment. The padding is part of the interface and must be allocated.
Call divide exactly once after the last page, before the batched O projection.

For each valid row, pages must begin at page zero, be visited in increasing order,
and contain at least one visible key. Host chunking must cap at a 512-token page
boundary. This guarantees every valid row visits the same pages and prevents an
all-masked page from becoming a standalone softmax. Padded rows are not consumed.
The recurrence preserves the original BF16 rounding between old numerator rescale
and addition, and the original EPS addition per page.

`src/ops_sliding_batch.rs::sliding_attention_b16` accepts Q/output
`[B,Ns,Gs,Ds]`, old K/V `[Ts,Ns,Ds]`, and separate current K/V `[B,Ns,Ds]`.
The current key axis is renamed to local public `Bk=16` through a borrowed view;
callers retain ownership of their B-shaped tensors. Old masks are `[B,Ts]`,
current masks `[B,Bk]`, using the same 0/-1e30 convention. For query position p,
old slots must describe only keys before chunk_start and in `[max(0,p-1023),p]`.
Current keys are visible only through that query's own valid row. Commit current
KV to the 1024-token ring only after the kernel finishes. The kernel never writes
KV. This preserves old keys required by earlier rows when the chunk crosses a
ring wrap. Current-value accumulation adds one BF16 partial to the BF16 old-ring
numerator; this changes grouping relative to serial ring placement and requires
numerical validation.

Both kernels mask scores before taking a maximum. The original serial kernel
masks exponentials after taking its maximum; once future KV is present, that can
make BF16 outputs depend on future keys. Pre-max masking deliberately removes
that dependence; bit-identical results to serial code are not promised.

Shared work: old/page K/V preparation and query-key/value contractions now cover
all 16 queries in a launch. Added work: row-to-slice redistribution, per-row mask
storage, padded scalar state, and a separate current-chunk contraction for the
sliding ring. No throughput or end-to-end correctness claim follows from compile.

`src/bin/test_attention_b16.rs` initializes all old cache bytes deterministically,
then checks full attention at 0,496,512,1008,1024 and sliding additionally at
1020 and 1032, including partial batches. It compares against an independent CPU
implementation with the actual BF16 intermediate boundaries and against original
serial attention for each case's last valid row. It also changes only a future
chunk key and requires earlier-row output to remain exactly equal. The probe's
NRMSE < 0.025 and max absolute error < 0.035 bounds are diagnostic kernel gates,
not substitutes for official whole-model correctness. Root coordinates execution.
