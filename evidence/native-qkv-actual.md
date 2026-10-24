Actual NPU318102/r325-shared-native-qkv-submit, source079e5b3.
Binary0c212f247cd85a6e619b46eeaaad4fd9dcc2f92f3ddd9d3a431564122feac03c.

All491,520 quantized input values and128 per-row scales passed independent
CPU equality for each seed0/31/79. All five projection instances (Qs, PsK,
PsV with distinct random weights, Qf, Pf) passed their quantized-input F64
oracles. Seed0 checked all2,162,688 outputs; random seeds each checked9,984
distributed samples across every128 rows. Bounded dyadic domains make the
F32 partial sums exact, so these are strict arithmetic/layout tests.

BF16-activation comparison is deliberately approximate: random relativeL2
ranges.0243571–.0250288 across projections/seeds, with synthetic maxabs52–72.
Seed0 was bit-exact with the BF16 control. These do not establish full-model
accuracy, generated-text quality, public-suite pass rates or SLO performance.

Complete matchedABBA20 sequences include quantization and every projection:
- Sliding BF16 QKV648.091/650.903us → one quantize+native Q/K/V503.495/502.086us.
- Full BF16 QK491.824/493.566us → one quantize+native Q/sharedKV404.230/401.214us.
Scratch allocation/host input upload/logit reads are outside both timing loops.

Proceed only to an isolated full-model numerical candidate and public quality
gates. Preserve BF16 TRF32 control; do not adopt the slower fused-DM Qs path.
