Control is current P128 BF16 H960 TRF32 Q/K/V with original E4 weights and
BF16 output-channel scales. Native Qs actual318096 showed the materialized
FP8 input path faster176–178us versus205us, while fused DM redistribution
was slower279us. Extend only the measured materialized path.

Quantize input[P128,H] once into E4[P128,H] plus F32[P128,1#8] per-row scales.
Launch independent native cores for sliding Qs and Ps K/V, or full Qf and Pf
(shared K/V model convention). Each core loads the same input without changing
weights, reduces four H960 F32 partials, multiplies input dequant, casts BF16,
then multiplies original BF16 output-channel scale and casts BF16.

Gate every projection against a bounded exact quantized-input F64 oracle,
including all128 rows and channel/slice boundaries, and report BF16-activation
precision error separately. Time complete sliding one-quantize+three-core
and full one-quantize+two-core sequences against their current BF16 counterparts
with all scratch allocated ahead. No host dispatch or model adoption here.
