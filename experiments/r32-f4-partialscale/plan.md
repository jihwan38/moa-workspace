# Current R32 F4 partial-scale resource gate

Base: `07718f0739a16f439c3725e31cb61ba4bcd480d5` (physical sliding pool32).
Control: unchanged `ops_decode32_ffn::batch_ffn`, called from `Decode64::ffn`
only for capacity32. Its surrounding launches are unchanged
`ops_norm_r64::rmsnorm_r64` and `ops_norm_r64::post_feedforward_r64`.

The candidate copies this complete four-stage core and changes only immutable
weight storage from current E4 to static F4 with a F32 scale for every output
row/K960 group. This is approximate current-model weight quantization; it is
not the original provided NVFP4/per16 format. The same F4 storage and metadata
used by B16 can be borrowed: UpWeight[L/240,H/960,L%240,H%960],
DownWeight[H/240,L/960,H%240,L%960], with scales omitting the final K960 axis.
No second R32 weight layout is needed.

Direct F4 lookup produces E4 codes for the unchanged K64 contraction. Each
local F32 dot is multiplied by its group scale before the original H4 or L16
inter-slice Add. A real reduced-F32 DM boundary precedes the original
activation-scale, retained row-dequant scale and BF16 boundary. Global scales,
GeGLU rounding, row activation quantization and both blocked HBM scratch
boundaries remain unchanged. The canonical R64 input/output ABI still reads
and writes only its first32 rows; rows32..63 are untouched by the core.

Removed work: half of immutable weight DMA bytes. Added work: three scale DMAs,
lookup operations, local F32 multiplication and explicit F32 materialization.
Geometry remains L120/H960 for UP/GATE and H120/L960 for DOWN. No width sweep,
activation quantization change, startup implementation or host dispatch.

Gate: same-tree exact resource compilation of both complete cores and the two
unchanged surrounding norm kernels. Require no DramStack and >=10% estimated
complete FFN saving including unchanged norm costs before a runtime micro.
These separate-launch estimates are a bounded resource comparison, not a
new fused graph or measured latency; host submission overhead is unmeasured.
Actual byte/F64/retained-stage correctness, timing, model quality and serving
performance remain separate gates owned by root.
