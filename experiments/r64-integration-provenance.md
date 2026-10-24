# R64 integration prerequisites

Dense source files imported unchanged from 37ae696: ops_batch_r64, ops_native_fp8_r64, ops_geglu_direct_r64. Shared dependencies imported from that same snapshot: ops_native_fp8_blocked, ops_geglu_direct, ops_batch_kpart, ops_batch_prepack8. Existing exact ops_batch_wide::down_cached visibility is widened for the shared kpart implementation; exact prompt dispatch is unchanged.

R64 attention source d8869ed, heads adee9da, IO d4443c6. Root has independently validated these kernels on NPU. NativeFp8Weights is copied from root checkpoint4769f96 and allocated once per MlpWeights at model load, corresponding to b6e14ca plus4769f96. It retains only the blocked copy and releases temporary row-major matrices immediately. R64 reads these shared fields; it allocates no second model-weight set.

Canonical model proof932f5a0 is frozen before this startup-weight change. Its reference remains the exact packed/BF16 branch. R64 matrix reductions and native FP8 activations change numerical behavior, so subsequent scheduler/ownership proofs must compare the same R64 arithmetic; original scalar comparisons are numerical diagnostics, not bit-exact scheduler gates.
