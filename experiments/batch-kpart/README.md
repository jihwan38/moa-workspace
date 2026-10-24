H960-partitioned P128 query (R2-BK01)

`ops_batch_kpart::project_query_batch` partitions the contracting H3840 dimension
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

Root Arena317698 query result: the independent H960 dyadic claim was withdrawn
after an oracle-selection bug (see corrected Arena317743 evidence below). Varied-exponent
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

Audit correction after Arena317733: the original query test's independent
dyadic oracle accidentally read the full-H control output. Its H960 random
differential comparison and timing were valid, but its reported H960 dyadic
independent proof is withdrawn. The test is fixed and `test_kpart_qkv` explicitly
checks both variants for Qs, Ps, Qf, and Pf against a seeded exact dyadic oracle,
then checks full varied-exponent tensors against each other and512 output
coordinates against independent F64 dot products. Distinct channel scales and
nonperiodic K values cover shard boundaries and all output/cluster layouts.

Arena317730 H960 FFN:491520 outputs,4640 bit differences,zero failures at the
documented tolerance,max_abs0.0078125,max_rel0.03662; warm4.66ms→3.116ms.
Arena317733 whole-model513 tokens:1753ms versus B16 3903ms and previous exact
P128 2661ms, but relativeL2=0.045957,max_abs1.875,argmax100→2946. This is not an
accepted model correctness result. The corrected all-projection oracle gates
bug investigation before any accuracy evaluation or adoption.

B16 follow-up plan (source9dde843): adapt H960 QKV/up-gate to one B8 row tile
per cluster, preserving parent's independently proven B8 down helper arithmetic
from native-ffn020aa5c. This removes three quarters of up/gate activation
replication versus split-cluster full-H, but introduces four-slice F32 reduction.
Compare all61,440 FFN outputs against exact baseline, report all numerical
differences, and measure matched ABBA launches. No serving adoption is implied.

Corrected all-QKV Arena317743:both full-H and H960 variants were exactly equal
to the independent dyadic oracle for all1,900,544 outputs acrossQs/Ps/Qf/Pf.
Full random comparisons had relativeL2 around2–3e-5, and512 sampled F64 scalar
dots per shape matched both variants exactly. No QKV layout/scaling defect was
observed. This replaces the withdrawn earlier H960 independent-query claim,
and does not erase the measured whole-model numerical drift.

Stage-profiling plan after Arena317730:expose post-GeGLU asHBM[L,P128], then
run unchanged cached-down from that tensor. Verify split output is bit-exact to
fused H960 FFN. Measure fused/upgate+GeGLU/down in ABC-CBA order with5 launches
and one stage result read. Stage measurements include additional intermediate
HBM write/read and differing download sizes, so they are upper-bound component
estimates, not an additive decomposition of fused runtime. This separates the
remaining down cost from up/gate scale-dequantization and contraction work.


Packed L960 down plan after83ae467:split down K15360 over16 slices rather
than8, cutting local activation cache245760B to122880B and output-group
activation replication32→16. FP4 weights stay packed; four60-row tiles expand
scale rows before splitting because60 E4 bytes violate DMA alignment. This
changes the16-way F32 reduction tree and doubles four-row dequant streams
30→60; compare491520 outputs against H960-up/L1920-down with tolerance and
matched ABBA. No loader change and no model storage growth.
