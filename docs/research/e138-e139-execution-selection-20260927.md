# E138 QKV consumer-local native path and E139 current FFN diagnosis

Selected after E137 adoption and official330d5595, from main a7bdaeb. Source
recipe E137334a7059a82252a310436c6d7e02f8ba577e7f51 and performance control
baseline/e137.json are separate inputs. The candidate branches record their
actual start after this selection commit. The user authorized proceeding toward
a substantial complete-path gain; no ranking success is assumed.

E138 (`qkv-native-consumer-local`) replaces the QKV execution path with
consumer-owned H960 shards (K4), four weighted E4 residual terms held in TRF,
original E4 weights streamed into native arithmetic, fused Term/K4 F32 sums,
and canonical head outputs scaled before their first BF16 cast. Existing
pre-RMS BF16, head RMS/RoPE/cache correctness and all public interfaces remain
acceptance boundaries. Q/K/V should reuse prepared input terms where their
physical consumer layouts agree; exact lowering and lifetime must prove reuse.
O and FFN must keep E137 images. This is a coupled ownership/representation
replacement, not an independent new arithmetic family or a tile sweep.

Against actual E137, targets are full-H BF16 replication (3.75MiB logical DM),
BF16 lookup/table work and selected row/scale routing. Add consumer BF160.9375MiB,
four-term DM1.875MiB, real global-amax/quantization, K4 reduction, compact F32
Q/K/V output32KiB write/read and common-scale/channel restoration. E137 already
fuses channel scale into its dot. Removing30MiB weightTRF applies only versus
rejected E136 and must never be claimed as an E137 saving. Original QKV weight
HBM30MiB remains. Count all transfers, layouts, register lifetimes and head tails.

Before compiling, write the complete coordinate/precision contract and inspect
related QKV/native/column attempts, including C15 H960 FFN's strided-ingress
lesson. H960=15 native64-value chunks and a3840B F32 residual nominally fit;
this is not a lowering or hardware guarantee. First gate is exact complete QKV
(or a production-helper feasibility wrapper if needed), then actual descriptor,
Term/K4 reduction, initialized-scale and register checks. Source/API/shape bugs
receive a supported repair when their cause is known. Preserve failed variants.
Use original full-QKV cancellation/cache oracle and standard whole inputs;
raw-Q verification may isolate a changed boundary but cannot establish speed.

Early whole screen is contemporary E137 before -> frozen candidate -> E137
after. Any positive freezes source for three independent allocations, extra6
and changed-boundary validation before adoption. Negative results are scoped;
one bounded internal diagnostic is justified only if it selects a concrete
repair. No gain<5% rejection. Root owns separate adoption/tuning/research decisions.

E139 (`e137-ffn-readiness-profile`) is one current-image whole-FFN diagnostic,
not a performance candidate. Reuse E134 graphs/schedules only after exact E137
FFN source/image identity; preserve existing production task/code/constants and
change diagnostic profile-name bytes only. Include ordinary whole controls and
diagnostic info/trace to separate host/tracing effects. Inspect actual input,
pre-RMS, up/gate packed/decoded weights and native boundaries, GeGLU, down
global-max/quantization/leader/ring preparation, activation-TRF versus weight
readiness, native down, retained partial/scale/reduction and final store. Never
sum overlapping intervals into predicted savings or infer bus stalls from timing.

The question is whether a supported preparation/ownership change can move
completion: if down activation has slack, defer its local-quantization proposal;
if up/gate or retained partials dominate, assess that actual boundary instead.
E135 timeout remains a distinct unresolved liveness issue and gets no arbitrary
padding/context retry. Complete the bounded trace and record its scoped decision.

Each branch freezes source/build/upload/raw-result records and commits/pushes
completed work. Root merges evidence before a new series. Use separate warm
Cargo targets (E138 c09, E139 m03), maximum two concurrent Arena allocations,
and coordinate helper builds. No official submission of unvalidated candidates.
