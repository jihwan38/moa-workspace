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

Scaled native matrix/fullFFN plan after016cb2f: per-token activation quantizers
copy verified0d5a253 formula as internal DM helpers (F32 subnormal FTZ follows
hardware). Native F32 dot multiplies token dequant scale then output-row weight
dequant scale before BF16; original global scale is afterward. Existing GeGLU
and BF16 boundaries remain. Hidden activations are transposed before DM
redistribution to avoid two-byte strided transfers; no HBM intermediates are
required. Independent scaled-matrix scalar checks precede honest full-FFN error
statistics and warm timing versus packed L960 control, then whole-model gate.

Startup prepack Arena317852 passed all117964800 emitted E4 values including
signed zeros and all row scales. Reconstructed-weight NRMSE≈0.02364.
Up160ms/down191ms included downloading each full60MB result, not pure startup
kernel time. FullFFN matrix checks sample1024 independent F64 dots for each
postscaled up/down mapping. FullFFN reports bit differences, strict old-tolerance
failures, relativeL2 and max error against packed L960; finite values and a broad
relativeL2<0.2 catch gross defects, not model quality. This approximate path is
explicitly UNVALIDATED for whole-model accuracy pending root's gate.
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

## Root partition-local FP8 quantization prototype (2026-10-24 UTC)

`ops_native_fp8_partition::batch_ffn` preserves the fused FP8 arithmetic but
quantizes in H960/L960 contraction partitions. Per-token amax reduces across
all H/L shards; scales use the same power-of-two floor and FTZ contract.
The baseline schedule spends 289,644 cycles each redistributing quantized
hidden and intermediate activations. The new layout replaces those with an
83,806-cycle HBM input load and a 99,932-cycle BF16 intermediate redistribution.
Compiled total: 1,459,380 -> 1,057,900 cycles. This is a compiler estimate,
not NPU timing. Whole-FFN bit equality and ABBA timing are required before adoption.
The probe includes zero and signed-subnormal BF16 rows in addition to random
finite rows. Baseline module and the frozen c007 server are unchanged.

Partition quantization NPU317887/r148: all491520 FFN bits equal, including
zero and signed-subnormal rows. Measured base1.421–1.424ms versus
partition1.414–1.423ms: no meaningful improvement, so not adopted.
Actual trace317901/r164 confirms Task2,457,703→2,432,305cycles. The BF16
intermediate repartition costs519,198cycles versus compiler99,932, replacing
the expected saving with a more expensive transfer. Operator spans can overlap;
their durations must not be summed into elapsed time.
Arena317869/r130 fused P128 FP8 result:1024 scaled-up F64 scalar samples
bit-exact; scaled-down had1 bit difference,max_abs3.05e-5,zero tolerance
failures. Full491520-value FFN comparison had477835 bit differences,zero
nonfinite values,relativeL2=0.0578963. This is real quantization error, not an
exact model path. Matched warm packed-L9602.407–2.413ms versus fusedFP8
1.397–1.410ms (three launches plus one identical result read, divided by3).
Root is integrating for actual-model accuracy validation; no official score
or model quality claim follows from these kernel results.

B16 native FP8 plan afterca4abc9:share the same prepacked model tensors with
P128; use one B8 tile per cluster, existing B8 GeGLU, and per-token activation
quantization with identical scale semantics. Preserve P128 source. Check1024
scaled-matrix F64 samples each and report full61440-value FFN error against
exact split-cluster packed control, followed by matched ABBA. Approximate
model accuracy remains a separate root gate.

Blocked-storage plan afterc5808b5: raw native FP8 schedule shows weight DMA
~116k cycles versus all eight contractions~60k. Row-major HBM causes240
strided960-byte pieces per slice. Reblock startup weights into
[L/240,H/960,L%240,H%960], and down analogously, so each live slice loads
230400 contiguous bytes. Startup uses one cluster and256 slices. Same FP8
values, scales, activation quantization and all arithmetic; only HBM layout
changes. Require all up/down weight values to match the independent CPU index
permutation and491520 final output bits equal to current fused FP8 before
matched ABBA timing. P128 original path stays available.

Arena317876 B16 native FP8:1024-sample scaled-matrix gates passed;61440 full
FFN outputs finite with relativeL2=0.05720 versus exact packed cluster. Runtime
0.882–0.889ms→0.521–0.529ms. Quantization error remains a model-level gate.
Fused P128 schedule now identifies two quantized activation redistributions
at289644 cycles each and GeGLU's three sets of tile copies at194760 cycles.
Root owns direct partition-local quantization; this worktree's immediate
blocked-weight variant changes only weight layout.

Default8-PE SDK topology rejects DM cluster extent1. Reblock startup therefore
splits output groups between two clusters,128 valid tiles each padded with an
anonymous axis to256 physical slices. Every weight is copied once; blocked
HBM order and live FFN behavior are unchanged. This avoids requiring a second
4-PE runtime topology only for startup.
