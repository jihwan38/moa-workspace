True batch head normalization and RoPE

`ops_head_batch::{sliding_heads_b16,full_heads_b16}` have exactly the corresponding
`ops_prefill` argument lists. Replace only those two launch paths after the probe
passes. They accept raw, already channel-scaled BF16 projections, existing norm
weights, one byte offset per batch row into the original RoPE tables, and the same
Q/K/V output buffers. They neither read nor write request KV caches.

Each query head occupies one of the 256 slices: `[B,Ns,Gs]` for sliding and
`[B,Gf]` for full. Sliding KV uses `[B,Ns,1 # 2]`; full KV uses `[B,1 # 16]`.
RMS reduction and RoPE run over one head dimension per slice, without row or
query-head loops. A single replicated batch RoPE gather is redistributed to these
layouts; the SDK rejects directly broadcasting gathered positions over named
head axes. Full K and V reuse the same denominator computed from their shared
raw key projection.

Arithmetic retains the original boundaries: BF16 projection input; F32 square,
reduction, divide by head dimension, EPS addition and square root; sliding norm
uses divide then weight while full uses weight then divide; BF16 normalized
heads; F32 cosine product plus F32 swapped-half sine product; BF16 rotated heads.
The signed sine table continues to supply the negation of the first rotated half.
No global norm, softmax scale, or KV commit is folded into this operation.

The probe compares all Q/K/V values to original scalar helpers with distinct
nonzero random rows and weights, positions 0..32767, zero/tiny EPS cases, and the
runtime's F32-generated full partial-RoPE table. Its scalar oracle runs the
original functions, not a restatement of the optimized code. Warm host launch
times are diagnostic; comparing one batch launch with sixteen separate scalar
launches includes their launch overhead and does not itself measure the existing
ops_prefill row-loop kernel or whole-model performance.
