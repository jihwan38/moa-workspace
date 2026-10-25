# B16 input-prefix launch composition

Base b078362. Model behavior and Worker policy stay unchanged. Current
Prefill::decode_sliding/full executes four host-awaited calls: ops_norm_batch::
rmsnorm_b16, ops_query_b16_native_fp8::quantize_input, ops_qkv_blocked_b16::
project_sliding_qkv/project_full_qk, and ops_head_b16_blocked head/RoPE.

Factor unchanged original device bodies into callable stage helpers, then invoke
those helpers in one device entry point. Retain every original HBM intermediate:
normalized BF16, FP8 input and padded F32 scale, blocked raw Q/K/V, headed Q/K/V.
No arithmetic, precision, KV, mask, request ownership, or output changes intended.
No DM boundary removal or hidden scratch aliasing. Both sliding and full paths.

Micro gate: supplied actual weights at multiple layers; candidate-first NaN
poisoned independent stage buffers; compare every normalized/quantized/raw/head
bit and live scale lane; row permutation and zero/tiny/magnitude cases; source
input immutability. Complete four calls versus one in alternating order excludes
input upload and downloads equally, includes all prefix computation. Inspect IR
for unexpected DramStack and legal HBM data dependencies.

Only after hardware micro proof should a Prepared/Prefill-scoped diagnostic
selector compare actual full-model all-vocabulary outputs and full KV bytes on
same histories, regroup/reset/active-prompt and scalar fallback. Default serving
off. Parent owns all Arena dispatch and production promotion.

Target directory is our inactive /round2/ffn-k256/target, preserving frozen prior
K256 executables. Build records freeze each new executable before another build.
