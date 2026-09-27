# E140 weighted down preparation at its consumers

Source recipe: E137 `334a7059a82252a310436c6d7e02f8ba577e7f51`.
Performance control: `baseline/e137.json`. These are independent inputs.
The named branch records its actual main start after this selection commit.
E139 evidence has been integrated at `b9c02d6` before this follow-up series.

E139 job111124 measured the current FFN, including all137 operations, with
unchanged instruction/constants/IO bytes and before/after ordinary hosts.
GeGLU completion to down activation TRF readiness has median36680cycles;
native starts1cycle after TRF in every run/cluster. Down packed weight is already
ready. This selects an implementable preparation replacement, not a claim that
36680cycles are reclaimable. That interval is about11.7% of whole FFN; it cannot
alone satisfy the roughly31% FFN reduction needed if QKV were20% faster and O
were unchanged at the latest recorded30th-place score. Gate gather overlaps
down-weight ingress and is not a second additive37k saving.

Selected path: preserve `project_down_tile`'s compact per-cluster BF16 TileL7680
gather, then typed-DMA those values to the existing `DownColumns =
[H/60,TileL/1920]` consumers. Each(c,g,q) receives exactly its same-cluster
1920-element stripe. Compute local amax and K4 Max across q to reproduce the
original cluster7680 scale, then produce the same E134 E5 hi/lo16/third256
terms and coefficients[s,s/16,s/256] locally. Preserve direct FP4/native dot,
F32 term restoration/reduction, block partials, five block-scale reductions,
cluster merge and original BF16/RMS/residual boundaries. QKV/O stay E137.

This is a local preparation/ownership replacement of the already adopted native
down combination. It is not a new arithmetic family. Compatible methods are
E134 weighted residuals, current cluster L ownership and native block partials.
Conflicting methods removed are compact480-element quantization, packed-term
gather, leader DMA and term ring broadcast. Keep the cheap canonical gather
(observed roughly1469cycles), avoiding a direct LocalRows→consumer expression
that could produce8192 small240B source fragments. Canonical→consumer logical
stripes are3840B, but emitted descriptor splitting must be measured: logical
stripes are not a promise of512 physical DMA packets.

Added costs include1.875MiB BF16 consumer replication, quantization in all512
slices, K4 Max, local residual/scale preparation, local2.8125MiB E5 term storage
and retained TRF loads. Local F32 residual is7680B; scalar scale32B and native
term coefficients96B bring their nominal total to7808B. The later block-scale
VRF is5760B, not96B. Existing up/gate and post-RMS VRFs must be included in the
actual allocation/lifetime audit; nominal capacity alone proves nothing.
Typed term views must use exact stream commits, preserving the earlier fix for
padding-expanded DMA. No prepacking, changed public input, extra hidden host
work or relaxed tolerance is allowed.

First gates are exact complete FFN compile and emitted coordinate/bounds,
K4-max broadcast, scale/reduction order and register lifetime checks. A concrete
source/API error gets its supported repair with each failed attempt preserved.
Then run an E137 before→candidate→E137 after whole-kernel screen. Positive even
below5% freezes source for three independent allocations, extra6 and the
production-down/raw plus whole-FFN tiny-value regressions. Existing E134 checks
are templates, not inherited accuracy of this new mapping. A regression does
not close native-down broadly: one readiness diagnostic is justified only when
it selects a specific supported repair; arbitrary tile sweeps stop.

Adoption is pending whole performance and changed-boundary correctness. Further
parameter tuning is not selected. Major research remains complete QKV and FFN
paths: this bounded repair may recover a measurable gain but is not a30th-place
promise. Use a named `opt/down-native-consumer-local` branch and candidate
`down-native-consumer-local`; preserve all source/build/upload/result records.
