# Keep the O epilogue on eight H480 slices

This candidate changes only the O output path relative to E24. The FP8
decomposition and projection arithmetic are unchanged. Exact compilation drops
from32,252 to29,088 static cycles. Arena E26 job70361 passes all15 official
comparisons; O measures52,579 cycles versus E24's59,027. These are single-job
screens; repeated and additional-input measurements remain separate evidence.

The projection's two-cluster result already contains the complete3840 rows:
`h=1920*cluster+60*row_group+local_row`, with32 groups per cluster. The new
`reshape` only renames the high-row cluster axis from Dummy2 to H/1920. It retains
the exact physical cluster, slice=`8*row_group`, local element and padding.
Typed DMA then directly gathers into eight480-value H partitions in one cluster.
The previous intermediate1920-value per-cluster gather and H120 layout are gone.

`apply_output_channel_scale`, `normalize_output_fused` and `add_output_residual`
all accept this same eight-slice layout. Channel multiplication retains its
original BF16 cast. RMS keeps the same480-value FP32 partial sums, H3840 divisor,
eight-way FP32 sum, EPS, square root, DivF and gamma multiplication. Its final
BF16 cast now occurs before any gather, in the same slices. The intervening FP32
DM write/read did not introduce a rounding boundary and is removed. Residual
addition consumes that BF16 result directly, retains the final BF16 cast, and
typed DMA writes the distributed result to the canonical HBM residual tensor.

Thus the BF16 boundaries remain projection→channel scale→RMS result→residual.
The changes remove gather/scatter and a redundant F32 store/read; they do not
cancel normalization, remove EPS or combine multiplication across BF16 casts.
The shared generic `normalize` function and other public kernel bodies remain
unchanged. `ops.rs` modifies only the local types in `sliding_attention_output`.

All3840 logical rows were enumerated through the cluster-axis rename and the
final eight-slice partition. The independent [review](review.md) checks callers,
padding and arithmetic order. The inherited FP8 input decomposition remains
approximate, with the counterexample documented in the parent candidate; this
layout change does not establish all-input equivalence for that quantization.

Sources, patch and hashes are frozen in this directory. The independent scratch
is `/tmp/moa-output-parallel-epilogue`, with its own target and compiler output.
`evidence.json` references the ignored schedule and the E26 hardware record.
