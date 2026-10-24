H960-partitioned P128 query (R2-BK01)
===================================

`ops_batch_kpart::project_query_wide` partitions the contracting H3840 dimension
across four H960 slices. Each slice owns64 output channels instead of16, so
weight bytes and multiply work are unchanged. Each input is replicated across
64 output groups instead of256, reducing activation DM replication by four.
All64 local token rows fit in122880 bytes of DM and are staged into TRF in B8
pieces without another HBM read. F32 partials reduce across four slices before
the original BF16 dot rounding and BF16 channel-scale boundary.

The F32 reduction order differs from the full-H control. The standalone probe
therefore has two gates:524288 exactly representable dyadic outputs checked
against an independent scalar oracle, and524288 seeded varied-exponent outputs
compared against the existing device kernel. The latter reports bit differences,
maximum absolute/relative error, and failures at `0.001+0.01*abs(reference)`.
Passing this kernel tolerance does not establish whole-model correctness.
Matched ABBA samples use five launches and one identical P128 result read.

Kernel and executable compile. Hardware correctness, numerical differences,
timing, and subsequent actual-model validation are pending root. The canonical
serving kernels have not changed.

Root Arena317698 query result: all524288 dyadic outputs exact. Varied-exponent
comparison had87 bit differences among524288 values, zero tolerance failures,
maximum absolute error0.25 and maximum relative error0.0125 (near-zero relative
errors use the documented denominator floor). Warm runtime fell from roughly
1.157–1.164ms to0.317–0.361ms. No whole-model correctness claim follows yet.

The candidate module now mirrors every P128 linear API: Q/K use the H960 split,
O reexports the previously validated wide output projections, and `batch_ffn`
uses H960-split up/gate while retaining cached-down. Up and gate share one
P64×H960 input cache. A48-row FP4 weight tile is decoded once and reused across
eight B8 chunks. GeGLU uses eight-token tiles because240×8 F32 values fit8KiB.
Reduced output rows redistribute into the existing L60 down-input layout.

Sixty E4 block scales per H960 shard violate eight-byte DMA tail alignment.
The implementation first expands those original E4 values exactly to BF16 in
aligned full-H scale rows, then redistributes120-byte scale shards. F32 VRF
fetch pads the60 live scale values to64; those padding lanes are never weights.
The failed unpadded and directly padded transfers are preserved in v4/v5/v7.

`test_kpart_ffn` compares491520 varied-exponent input results against the current
exact cached P128 FFN, reports bit differences/max errors, and fails beyond
`0.02+0.01*abs(reference)`. Both FFN and all Q/K wrappers compile; the FFN hardware
probe and subsequent whole-model comparison are pending root. Original wide
kernels remain available unchanged for that comparison.


Startup FP8 prepack plan afterf830b4a:original FP4×E4 blockscale expands to
the exact baseline BF16 value; calculate per-output-row amax, choose
weight_dequant_scale=2^floor(log2(max(amax,2^-112)))/128, and emit E4 q.
Reconstruct q×scale. Up/gate have30 rows/slice and five-row scale-dequant
tiles; down has8 rows/slice and two-row tiles. Globals are excluded. Verify
every emitted E4 byte and F32 scale against an independent CPU formula before
using the tensors. This is approximate weight quantization and needs full-model
accuracy validation; original weights remain the only source of model data.

The initial down placement padded a named240-slice H axis to256 and failed
DMA common-axis factorization. Final down startup placement uses64 output
groups per cluster,30 rows/group, and15 two-row passes. Each pass has aligned
8-byte F32 scale output and small DM buffers; source values are unchanged.
Up and down startup kernels and the every-value CPU probe now compile.

Arena317831 activation quantizers passed every H/L E4 output and per-row F32
scale against the independent CPU formula including RNGD F32 subnormal flush
to signed zero. Weight FP4×E4 products are normal F32 except signed zero.
Arena317828 packed L960 down candidate:97 bit differences among491520,zero
tolerance failures,max_abs0.0078125;3.068ms→2.396ms.
Arena317813 native original-FP4 projection:46 differences among1966080,
relativeL2=5.05e-6,zero tolerance failures;512 sampled F64 dots exact.
30-row and6-row native variants are bit-exact to one another, taking~0.92ms
and~1.17ms respectively. Final warm weighted-BF16 comparison~1.006ms; the
first1.634ms sample was an outlier. No full-model native4 adoption followed.
