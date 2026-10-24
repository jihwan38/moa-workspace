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

Arena317750 B16 H960 FFN:803 bit differences among61440 outputs,zero tolerance
failures,max_abs0.0078125. Runtime0.957ms versus exact cluster control0.883ms.
This is a measured regression and is not adopted. `test_kpart16_qkv` applies
the corrected all-shape independent/differential gates to B16 and compares
ABBA timing against original replicated-B16 QKV (not the newer Qs-only split
cluster control). No speed claim can be made versus the newer Qs control from
that benchmark alone.

Predecoded H960 plan (control4140445): replace each original FP4×E4 product
with exactly the same BF16 stored in HBM; retain globals post-dot. Up/gate use
two120-row BF16 tiles, each230400B, beside122880B of cached input. Down uses
three40-row BF16 tiles, each153600B, beside245760B of cached input. This removes
all150 dequantization streams and shortens contraction tile loops, but expands
FFN weight HBM4.4495GiB→15.8203GiB. Candidate must be bit-exact to packed H960
for491520 varied-exponent results, and improve matched warm ABBA timing before
any full loader work. H960 whole-model numerical drift remains a separate gate.

B16 exact predecode plan afterc900306:share a B8-per-cluster full-H activation
cache across up/gate; use two30-row BF16 weight tiles per projection. Down
loads the full120×1920 BF16 shard after activation has moved to TRF. Reuse the
exact cluster GeGLU and eight-way down reduction, with no H960 reassociation.
The same predecoded model weights serve wide and decode paths. Require all
61440 random outputs bit-exact to packed cluster control, then matched ABBA.
This is an isolated kernel experiment; model storage/loader is unchanged.

Arena317756 stage profile:split/fused gate passed; fused3.01ms,upgate+GeGLU
1.584ms,down1.798ms including the documented extra stage IO. Arena317766
predecoded P128:all491520 outputs bit-exact,packed3.043ms→predecoded2.736ms
(about10%). Additional15.82GiB FFN weight storage is not adopted on this result.
The modest gain revises the hypothesis that per-launch weight expansion alone
dominates; down input repartition/contraction remains substantial.

Packed L960 down plan after83ae467:split down K15360 over16 slices rather
than8, cutting local activation cache245760B to122880B and output-group
activation replication32→16. FP4 weights stay packed; four60-row tiles expand
scale rows before splitting because60 E4 bytes violate DMA alignment. This
changes the16-way F32 reduction tree and doubles four-row dequant streams
30→60; compare491520 outputs against H960-up/L1920-down with tolerance and
matched ABBA. No loader change and no model storage growth.

Native FP4 plan after29eeda9: use caller's E4 activation and per-token F32
power-of-two scale with the provided FP4 weights and E4 blockscales. H960
shards limit six-row/B8 blockdot to60 blocks. Contract retains each16-element
block, applies its original scale in the vector stage, reduces H locally and
across4 slices, then applies xscale before the BF16 dot boundary. No BF16
weight requantization or expanded model storage. Arithmetic regrouping remains,
and single-term activation quantization needs a separate full-model accuracy
gate. First matrix probe reconstructs FP8×xscale as BF16 for the existing exact
weight kernel, also checks sampled F64 scalar dots, and measures warm ABBA.

Both native4 shapes compile. Because H960 has only60 blocks,30 scale rows fit
30×64×4=7680B VRF; the probe compares this8-iteration row tiling against the
initial40-iteration six-row shape. This reduces command/loop repetition while
preserving the same dot/blockscale/reduction computation. Native activation
input is generated directly and its exact E4×power-of-two reconstruction is
the BF16 reference, so this kernel gate excludes activation quantization error.

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

Direct-GeGLU plan afterfc3eb66: unchanged global-scale BF16 casts, Erf GeGLU
sequence and final BF16 cast. Read up/gate B8 tiles as padded DM views directly
in scale streams, and commit final GeGLU directly to the full P64 result view.
This removes three sets of eight strided DM copies measured at194760 cycles.
Variant `ops_native_fp8_blocked::batch_ffn_direct_geglu` preserves blocked-only
control; require491520 output bits equal before matched ABBA timing.

Arena317891/r152 blocked-only gate:all117964800 reblocked FP8 values match
the independent CPU permutation and all491520 final FFN values are bit-exact
to row-major FP8. Matched warm row-major1.3625–1.3635ms→blocked
1.2486–1.2526ms, about8%. Same quantization error as the existing FP8 path.
Direct-GeGLU variant compiles; its independent change is still awaiting NPU
bit-equivalence/timing. Compiler cycle estimates remain hypotheses: root's
partition-local quantization reduced estimated cycles without a measured gain.

B16 blocked-storage plan after7137db5: reuse exactly the same UpWeight and
DownWeight aliases/startup conversion as P128, changing only B16 matrix HBM
input layout. This permits one blocked FP8 model copy for both paths. Require
all61440 final output bits equal to row-major B16 FP8 and matched ABBA timing;
startup permutation already has its separate117964800-value proof.

Arena317898/r160 direct-GeGLU:all491520 final bits match blocked-only FP8;
blocked1.215–1.221ms→direct1.0868–1.0870ms,about10.6%. Arena317902/r165
B16 blocked:all61440 bits match;0.525–0.526ms→0.420–0.425ms,about20%.
Both paths share the same blocked model tensors. Root's physical cycle traces
confirm partition-local quantization had no meaningful measured gain; R64
therefore retains the validated original quantization/redistribution pipeline.

R64 plan afterbc64d0f:64 live queries,32 per cluster,four B8 contractions.
Reuse all original QKV tensors and the single blocked FP8 FFN model copy.
QKV copies H960 arithmetic; FFN copies proven blocked/direct-GeGLU arithmetic.
Compare all R64 FFN output bits with the first64 rows of P128, padding the
remaining independent P128 rows with zero. Distinct R64 query and Rc64 cache
slot axes prevent accidental query/cache ownership conflation.

R64 frozen source37ae696/352780a passed all three NPU gates. Arena317928/r178
FFN:245760 values bit-exact to P128 prefix; R64 0.7658–0.7706ms versus padded
P128 1.165–1.172ms. Each timing includes five launches plus its output read/5;
the P128 read is twice as large, so this is not matched output-transfer volume.
Arena317929/r179 O projections:245760 independent dyadic CPU outputs exact
for each shape, sliding0.471ms/full0.764ms (five launches plus one read/5).
Arena317934/r181 QKV:all four shapes exact against P128 for both dyadic and
varied-exponent inputs;950272 dyadic CPU outputs and512 F64 scalar samples
per shape exact. Full-query0.283–0.286ms/full-key0.169–0.174ms; remaining
timings are in the root run log. These are kernel gates, not R64 model accuracy.

R64 direct-O plan after352780a: preserve every1024-column BF16 partial dot,
the exact BF16 binary sum tree and output scale. Read each P16 sum operand
as a padded DM view directly into the existing vector stream, removing two
DM tile copies per add node. Existing entry points remain controls; require
all output bits equal on varied inputs and independent dyadic CPU oracles,
then matched-output ABBA timing. No weight or numerical-format changes.

Direct P16 operand fetch initially failed SDK lowering because its live padded
axis was left unread by the split fetch packet. Compilable direct variant uses
four B8 tiles per cluster rather than two P16 tiles; this adds two loop bodies
per sum node while removing both operands' DM copies. Arithmetic is unchanged;
actual timing will decide whether removed DMA exceeds extra control overhead.

P256 plan after9fde97c:128 independent token rows per cluster and16 B8
contractions reuse each blocked FP8 weight tile, instead of two P128 calls
loading it twice. Reuse the same model weights/scales and per-row quantization;
no approximate arithmetic change. Direct GeGLU removes tile copies. Larger
activation/result buffers may exceed512KiB per-slice DM live allocation, so
first compile before any host integration. Require all983040 output bits equal
to two P128 calls on the same256 inputs. ABBA batches compare one P256 call
with two P128 calls and matched total output bytes, reporting launch/read scope.

Original-E4 QKV storage plan after09078a1: isolate R64 Qs projection with
startup permutation [Qs/64,H/960,Qs%64,H%960]. This turns64 separate960-byte
HBM pieces per slice into one61440-byte tile, preserving every provided E4
weight, BF16 expansion, H960 dot/reduction and scale boundary. Require all
15728640 permuted bytes match CPU indices on two inputs, all262144 projection
outputs match canonical R64 bits, independent dyadic/F64 gates, and matched
ABBA timing. No model integration or other QKV shapes before measured benefit.

Arena r192 direct-O:245760 output bits exact for each shape on both dyadic
and varied-exponent inputs; both independent dyadic CPU oracles pass. Sliding
control0.415–0.443ms versus direct0.399–0.409ms; full control0.752–0.776ms
versus direct0.7126–0.7143ms, matched five-launch/read ABBA. Modest real gain.
A bounded direct-P16 follow-up retains two loop iterations: fetch the whole
16-wide padded live axis before collect splits it into8-wide vector packets,
addressing the original compiler error instead of adding two B8 iterations.
Compare control/direct8/direct16 bits and timings before any adoption.

Arena317947/r194 P256 FFN:all983040 output bits match two P128 direct-GeGLU
calls. Per256 rows, two-P1282.161–2.170ms versus P2561.715–1.716ms,about20.8%.
Scope is five batches plus matched total output bytes/5; control uses two
output reads and candidate one. Same blocked weight tensors and quantization.
This is an isolated FFN gain; no P256 host/model result follows from it.

Parallel-K1024 O plan after2f3aa4e: current O serializes4/8 partial matrices
on [H/120,1#8] slices. Map sliding partials to [H/120,Qs/1024,1#2], preserving
each1024-element full dot and BF16 boundary, then pairwise inter-slice adds
with BF16 casts at4→2→1. Each add combines the same two original BF16 values.
This exposes independent partial work across slices and avoids four separate
input/weight launch sequences. Original E4 weights and HBM layout unchanged.
First sliding-only gate checks every245760 output bit versus original on
dyadic/random inputs, CPU BF16-tree oracle, and matched ABBA timing. Full O
will follow only if pairwise mapping is correct and gives a real speed gain.

Arena317957/r201 blocked Qs:all15728640 reordered values pass on both dyadic
and varied-exponent weights;all262144 output bits per case,dyadic CPU outputs
and512 F64 samples exact. Row-major[0.210870,0.185773]ms versus blocked
[0.183383,0.179091]ms:matched final-warm gain about3.6%, so no expansion to
all QKV shapes or model-memory addition follows yet.

Parallel O initial source padding after Qs failed VRU's requirement that the
reduced axis be innermost. Compilable placement puts anonymous padding before
Qs:Parts=[H/120,1#2,Qs/1024],Pairs=[H/120,1#4,Qs/2048],then[H/120,1#8].
The two stages retain adjacent K pairs and the original BF16 tree; actual
NPU differential/CPU gates are still required to prove this mapping.

Arena r210 parallel-O first candidate launch failed:device request unanswered
for7s, no numerical/timing result. Source retained. Leading hypothesis is the
partial-axis VRU mapping after reduction, not proven. Separate gather variant
keeps all four K1024 contractions concurrent but transfers their BF16 results
as four contiguous7.5KiB planes into each output group, then applies the proven
direct-P16 addition tree. This avoids the questionable two partial-axis VRU
stages. Same final differential/CPU/timing probe; if it fails, isolate the dot.

Gather variant compiles after preserving nested-plane padding:leading1#4
normalizes H%120 to H%120#480, which must remain in the subsequent token
tile. The frozen probe first launches the concurrent partial dots alone and
checks all983040 dyadic BF16 partial values, then launches full gathered sums.
This distinguishes dot failure from gather/add failure if the device stalls.
Arena317962/r208 direct-P16 O passed all final bits/CPU oracles. Sliding
control0.423–0.442ms/direct160.398–0.404ms;full0.769–0.797/0.720–0.732ms.
Direct8 was0.411–0.447/0.731–0.785ms in the sameABC-CBA run.

Arena317977/r224 gathered parallel sliding O:983040 isolated BF16 partial
dots exactly match CPU;245760 final outputs match canonical bits on both
dyadic and varied-exponent inputs, and independent dyadic BF16-tree oracle
passes. Matched ABBA control[0.437694,0.423136]ms versus gathered parallel
[0.198380,0.194981]ms, about54% improvement. The concurrent-dot mechanism
is validated; the earlier partial-axis VRU implementation remains rejected.
Next full O uses eight gathered planes and the same BF16 tree8→4→2→1.

Arena317986/r229 full O:all1966080 isolated partial values match CPU and
245760 final output bits match on dyadic/random inputs. Control
[0.783718,0.746654]ms→parallel gather[0.222893,0.219208]ms,about71%.
Sliding repeat0.4247→0.1908ms. Next P128 mapping retains the same concurrent
K1024 partials and exact BF16 tree, with64 token rows per cluster and eight
B8 contractions. Original P128 serial O remains the differential control.

Temporal TRF reuse plan afterf40687c: SDK0.9.0 limits Lane to1/2/4/8,
but permits extra temporal axes in TRF Element and innermost stream broadcast
axes in contract_outer. Store all64 local FP8 token rows as Lane=P%8,
Element=[P/8%8,K960] (61440B) and broadcast each weightK64 packet over
P/8%8 inside the contraction, preserving each token's K-reduction order.
The64 live accumulator values fit1024-element capacity. This replaces eight
separate B8 TRF loads/contractions and repeated SRAM weight fetches with one
stream; does not change HBM weights or arithmetic/quantization. First exact
projection compile, then full FFN bit-equivalence and ABBA, with an up-only
variant to distinguish up/gate savings from down savings. No speed assumed.

Hardware/utilization audit (SDK0.9.0, actual source and official sources):
[Furiosa specifications](https://developer.furiosa.ai/latest/en/overview/rngd.html)
state512TFLOP/s FP8,256TFLOP/s BF16,1.0GHz and1.5TB/s HBM. The
[ISCA2024 paper, §IV-C](https://furiosa.ai/download/FuriosaAI-tensor-contraction-processor-isca24)
describes eight DPEs per slice,64 FP8 values per DPE input, temporal reuse
through the feed unit, and1024 accumulators. Current Lane8/K64 fills these
widths; there is no unsupported Lane64 shortcut. SDK lower verifier
`verify/engine/collect.rs:182` restricts Lane to1/2/4/8, while
`verify/engine/contraction/outer/stream_adapter.rs` permits an innermost
broadcast Time axis. TRF Lane8 × tokenGroups8 × K960 =61440B fits64KiB.
The new projection compiles with this exact temporal-broadcast arrangement.

From our model shapes, P128's three FFN matrices perform
2×128×3840×15360×3=45.2984832GFLOP. At0.947ms this is47.83TFLOP/s,
not30GFLOP/step. Our existing two token-owning clusters each read every
weight, totaling2×3×3840×15360=353894400 bytes per FFN. At1.5TB/s,
weight traffic alone has an optimistic0.23593ms floor and caps this mapping
at192TFLOP/s before activation traffic. This is a bound, not a measured rate.
Peak512TFLOP/s is therefore not the achievable roof of this storage mapping.
The observed0.947ms leaves room above the weight-only floor, but a10× FFN
gain would require changing the traffic mapping as well. Root's r222 actual
profile separates task spans from overlapping DMA/TU spans; summed span
durations must not be interpreted as serial wall time. TRF temporal reuse
addresses repeated SRAM fetch and command work; HBM and redistribution
remain independent limits. Current full-FFN probe includes root's opt-in
physical-cycle collector after untimed correctness and normal ABC-CBA timing.

Arena r237 P128 parallel O passed1966080 sliding and3932160 full isolated
CPU partial values,plus491520 final output bits per shape on dyadic/random
inputs and both BF16-tree oracles. Sliding serial0.7657–0.7848ms→parallel
0.3253–0.3286ms;full1.3759–1.3978ms→0.4164–0.4353ms. Matched
five launches plus one output read divided by5. Exact kernel substitution;
whole-model integration/latency remains root's separate gate.

Arena318008/r243 temporal TRF64:up/gate-only and all-matrix variants both
match all491520 control bits. ABC-CBA control1.1305–1.1498ms,up/gate-only
1.0401–1.0645ms,all-matrix1.0101–1.0148ms. This is a modest~11–12% full
FFN gain,not8×. Actual cluster0 task spans2007105→1757889 cycles;baseline
24 StoTrf operations disappear, and three combined matrix TU spans are
~58905–59097 cycles versus eight~15.7–15.9k spans per baseline matrix.
Large DMA spans remain (candidate maximum526652 cycles),so the successful
reuse change must be combined/tested with root's H-local/L-sharded quantization
before claiming their gains compose. Source keeps baseline quantization for
this isolated experiment. Root will integrate the matrix helpers separately.

BF16 QKV temporal-reuse plan afterfa4d51d: eachH960 shard holds32 token
rows in61440-byte TRF as Lane8 × temporal groups4. Fetch each original BF16
weight packet once, broadcast to four groups inside the contraction, retain
the original H960/F32 reduction and BF16 dot boundary before output scale.
R64 uses one32-row contraction per cluster;P128 uses two32-row contractions.
Original E4 weights and HBM layout unchanged. Verify all four shapes against
their prior batch kernels on dyadic/random inputs, independent dyadic/F64
oracles, then matched ABCBA/ABBA timing before adoption.

All four R64 and P128 QKV32 variants compile. P128's padded32-of64 token
view cannot fetch low token bits first,so it uses Lane=token/4 and temporal
group=token%4 to read in natural order. Immediately after the unchanged BF16
dot cast,a four-row TU transpose restores natural token order. Its output
packet is token%4 padded16 (not16 live values:that exceeded TE's four-row
limit). This avoids a full32×H960 BF16 DM copy. Source keeps failed compile
logs and separate R64/P128 exact-control/CPU-oracle binaries.


P512 staged FFN experiment (base50039c0 + axis2a969b9): compare four existing
P128 blocked/direct FFN calls with quantize_hidden, two project_up_scaled calls,
geglu_quantize, project_down_scaled. Each blocked weight is read once per cluster,
held across four64-token tiles using the exact gated TRF64 temporal packet reuse.
Each slice holds230400B weight +61440B input +30720B BF16 result (+transpose),
rather than all256 local token rows. GeGLU uses token-sharded1920-channel tiles;
its BF16 global-scale and final BF16 boundaries precede the unchanged row quantizer.
Weights and FP8 scales are shared unchanged. Matrix input HBM replication is about
377.5MB/512rows across allthree matrices; stage IO is additional. Weight transfer
falls1.416GB to354MB vsfourP128 calls, but actual speed is unproven. Gate all1966080
output bits vsfourP128, then compare matched ABBA total-output reads and full staged
launch cost. No model/score adoption before hardware result.

P512 buildv1: padded L1920-of15360 direct operand view cannot split its live
axis in Fetch. As in the proven directP16 O path, fetch the whole live L1920
packet, then Collect splits it into vector flits. Failure log retained.
