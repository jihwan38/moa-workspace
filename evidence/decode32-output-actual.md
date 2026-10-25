# Exact32 O actual gate

Arena319434 / r892-decode32-output-submit,5.5489s, source b45a6f2 from a011,
ELF c6bc7582daabca0e406459e976238d7c846b5768ad1506e4bba4e04aa3d45aaf.
All three synthetic seeds and two provided-layer weight seeds pass for sliding
and full O. All1,228,800 final values and7,372,800 K1024 partials match current64
bits;1,727,488 independent F64-partial/BF16-tree values match exactly. The active
inputs include zero, tiny, cancellation, varied and permuted rows; upper32 NaN
inputs do not affect active outputs. All skipped output rows preserve their
poison bytes and all input bytes remain unchanged.

Completed-call ABBA (12 complete projections per group; downloads/CPU excluded):

|Projection|Current64 µs|First32 µs|
|---|---:|---:|
|Sliding|121.025,120.823|84.818,83.560|
|Full|173.674,172.399|123.697,119.414|

This is about30% lower kernel wall time, with unchanged weights/scales and
unchanged BF16 K1024 partial/add tree. Applying those differences to40 sliding
and8 full layers suggests~1.85ms per eligible model call, an estimate pending
actual whole-model timing, not a serving or scoreboard claim.

Actual trace, cluster0 (categories overlap, so do not sum them into Task):

|Metric|Sliding64→32|Full64→32|
|---|---:|---:|
|Task cycles|222374→154925|329098→226001|
|TU count / union cycles|18/113400→12/72558|34/149371→20/87125|
|DMA count / union cycles|6/126716→6/91302|6/222068→6/158656|
|Table-load cycles|5674→5676|5680→5677|

The trace supports removal of token-dependent contraction, partial-add and
movement work. DMA operation count and weight-table load remain; this is not a
claim that weight bandwidth halved.

Next gate is per-call `Decode32OutputDiagnostic`: only existing eligible17..32
hybrid32 calls select new O. The original hybrid32 diagnostic remains the exact
matched control; ordinary Automatic remains unchanged until the full-model gate.
All other stages, logits output policy, pool64 ownership and live KV commits are
unchanged. The model probe retains actual paired512/mixed histories, all logits,
greedy, full reserved caches, regroup/reset and active128 interleaves. Additional
33/64 live-row reset calls compare the original64 projection branch on both APIs.
