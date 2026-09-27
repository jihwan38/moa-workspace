# After E137: next research priority

E137 v3's three paired whole-O improvements (-14.1496%, -12.2978%, -12.2654%; median -12.297751%) plus standard/extra6, raw9 and whole-case7 checks support adoption of that frozen path. The transferable result is that operand ownership and preparation must put the activation in its consumer layout early. It is not evidence that native arithmetic, more slices, or removing any copy always wins. E136's complete weight-resident QKV regressed11.997%; E135 direct up/gate has no candidate arithmetic or valid FFN timing because whole and isolated projection launch both timed out.

## 1. First research priority: activation-resident, column-sharded native QKV

Investigate a complete replacement of full-H activation replication and BF16 lookup with small consumer-owned K shards. A concrete register-compatible starting geometry is **K4 × H960**, with two clusters ×64 output-row groups ×4 K columns. Q4096 has32 local output rows/group; K and V2048 each have16. This geometry follows H3840 divisibility and register constraints, not a4/8/16 sweep: H960 is15 native64-element chunks; the F32 reconstruction VRF is3840B plus32B scale, whereas fullH needs15360B. All four weighted E4 terms remain fixed.

Proposed whole path:

`existing preRMS BF16 H480 shards -> typed DM into H960 consumer shards -> local max + K4 max per output group -> identical global-H3840 scale -> four-term local quantization -> LaneTerm4/H960 TRF; original row-major Q/K/V E4 weights -> compact DM -> native term coefficient/Term sum -> K4 F32 sum -> canonical compact F32 Q/K/V -> common input scale then channel scale -> first BF16 -> existing head RMS/RoPE/cache`.

No direct Sub Switch->TRF is needed. The failed E55 direct-switch bitmap route is a constraint on that shortcut, not on this typed-DMA/local-quant path. Preserve the same full-H scale in every group; per-column scale would require an extra restoration boundary or changed arithmetic. A source proof must show each group gets exactly H0..3839 and each native weight owner gets the matching H960, then validate head/cache coordinates.

Compared with E136, logical four-term activation payload falls7.5MiB→1.875MiB and bulk weight-TRF staging30MiB disappears; original weight HBM30MiB remains. Compared with the adopted BF16 QKV, it adds actual four-term quantization, K4 reductions, and approximately32KiB compact F32 output write/read plus common-scale/channel-scale work. Local BF16 input payload is960×512×2=0.9375MiB. These are layout payloads, not bus measurements or predicted savings. More local output rows, source stride, Q/K/V TRF reuse/reloads and head completion ordering belong in the cost model.

**Prior overlap:** native activation-resident QKV and quantized terms already existed; this is their ownership/precision reconstruction, not a newly invented family. E47-style approximate terms do not supply numerical validation. E136 used output-row lanes and streamed large replicated activations; it did not test this Term-lane/K4 path. C10 used Main lookup->weightTRF. Bounded source search did not find this complete consumer-local H960/Term4/K4 combination; check the historical recipe index before selecting a build.

**Prerequisite / smallest discriminating step:** first write the exact coordinate, TRF/VRF and stage-order proof against the actual SDK. If it survives, a single exact raw-Q projection feasibility diagnostic should include real post-preRMS BF16 preparation, all four terms, full original Q weight rows and channel-scale-before-BF16—not prepacked inputs or an isolated MAC. It must verify emitted DMA ownership, Term->K4 reduction and canonical head conversion against cancellation/boundary inputs. This settles expressibility/mapping only. A performance decision needs one complete QKV screen including K/V and head tails, with contemporary E137 controls. No implementation is selected by this note yet; native K4/source shaping and changed accumulation still require those checks.

## 2. Conditional FFN direction: consumer-local weighted-down preparation

Keep the passing E134 native-down arithmetic, L-half cluster ownership and original weights, but ask whether post-GeGLU operands can reach their current K1920 consumers without canonical7680 gathers, packed leader copies and Main ring broadcast. In each cluster,64 Hrow groups ×4 K1920 columns could receive the same local7680 BF16 values through typed DM; each group's K4 Max then supplies the same existing per-cluster scale. Locally quantize the fixed E5 hi/lo16/third256 and load the existing Term3/H1920 TRF. Retain F32 block partials, block scales, K/cluster reductions and final epilogue initially.

This changes ownership at the complete down producer/consumer boundary; it is not another residual-factor experiment. K1920 F32 residual VRF is7680B plus32B scalar, below8192B only with a tight live set. Native term coefficients96B and block-scale VRFs must be scheduled with verified lifetimes; fitting nominal sizes does not establish allocation success. Input/quantization become active on512 slices: BF16 consumer payload1.875MiB and E5 local terms2.8125MiB remain. The proposal removes specific packed-copy/broadcast boundaries, not quantization, TRF traffic, weight work or the14.0625MiB F32 partial write and read. Additional local sums/VRFs and typed-DMA contention could lose.

**Prior overlap and prerequisite:** this is related to M03/E134 producer-local down and E137 locality, but is not M06's unresolved preRMS/upgate pipeline or E135's independent FP4 decode. It provides no explanation or repair for E135 timeout. The current successful FFN still uses `broadcast_down_quantized` -> full replicated DM -> Sub TRF (`weighted-v3/mlp.rs:264–275,428–432`). We do not yet have current native-down internal NPU evidence that this boundary delays a consumer; old BF16-down/E131 profiles cannot establish that.

**Smallest discriminating diagnostic:** one same-image whole-FFN trace that separates post-GeGLU gather/quantizer, leader+ring preparation, down TRF-ready, original weight-ready, first native down, and retained partial/reduction tail. Select the ownership repair only if readiness or Main/DM resource occupancy is actually limiting, and it remains favorable after counting enlarged local quantization. If activation is ready with slack while weights/other work determine native start, defer this implementation. Preserve the existing down oracle's tiny/cancellation and reachable full-FFN regression for any later change.

E135 should remain a separate unresolved liveness question until a specific blocked boundary/source hazard or supported repair is identified. The existing public profile wait path cannot be assumed to return partial spans on a timeout. Another padding/context/ordering retry without new evidence is not the next priority.

Source evidence: `axes.rs`; `qkv-native-weight-resident/decision.json`; `qkv-bf16-broadcast-trf-e55/README.md` and frozen compile log; E137 native-v3 source/independent graph review/profile; `ffn-native-down-weighted-residual/weighted-v3/mlp.rs`; `ffn-independent-direct-fused/runtime-analysis.md`. This assessment used read-only inspection, with no new build, NPU job or ranking promise.

## Follow-up review: the actual E137 comparison and experiment order

The refreshed [leaderboard snapshot](e137-next-priority-leaderboard-20260927.json)
places BEEF at46 with6.4427554× and the30th team at7.8574794×. Other scores fixed,
QKV and FFN each20% shorter would yield only7.4761554×. With O fixed and QKV20%
shorter, FFN must fall31.0911% to match that threshold. These are planning
scenarios, not predicted gains of either proposed implementation.

The current E137 QKV already keeps BF16 activation in TRF and streams weights
through BF16 lookup. It has **no weight-TRF staging**. The30MiB weight-TRF removal
and7.5→1.875MiB term payload comparison above are versus rejected E136 only.
Against E137, the proposed K4 path targets full-H BF16 replication (3.75MiB
logical resident DM) and lookup preparation, while adding0.9375MiB consumer BF16,
1.875MiB local four-term DM, actual quantization/global-max, K4 F32 reduction,
32KiB compact F32 output write/read and a separate restoration/channel-scale
consumer. Current channel scale is already fused into the BF16 dot stream;
separating it is an added cost, not a new fusion benefit. Quantities are logical
payloads, not measured bus traffic or cycle savings.

Recommended parallel work is one QKV feasibility prototype and one current-image
whole-FFN diagnostic. QKV first proves real post-RMS inputs, K4/Term4 ownership,
canonical head/cache output and unchanged arithmetic acceptance, then moves to
an early complete QKV comparison including K/V and all tails. A positive gain
freezes the source for adoption validation even below5%; a regression triggers
cost diagnosis only when a concrete repair can be selected.

The FFN diagnostic should retain up/gate readiness and GeGLU boundaries as well
as the down markers listed above. Native-down input preparation is not assumed
to explain the full FFN gap. If its delay is small or hidden behind weights,
do not implement local quantization merely because O benefited. Choose the next
large boundary from the whole trace. E135 direct-decode timeout still requires
specific liveness evidence before another repair; its missing candidate cycles
cannot be substituted for this cost measurement.

This follow-up changes the research recommendation only. No candidate source,
compile, Arena measurement, baseline or official submission was changed.
