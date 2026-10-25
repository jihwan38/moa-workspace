# H480-local quantization, grouped sliding layers only

Actual full-chain microgate318844 (root r658, isolatedsourceb493991): all1,966,080FP8bytes,512F32scales andfour983040-byte residuals exact for each of three seeded/permuted inputs; independent CPU F64 residual also passes. Inclusive current blocked consumer274–296us versus candidate249–279us. Actual first-pair task555029→462023 raw units; this is not an HTTP speed claim.

Integration only selects the new quantizer when the existing grouped sliding blocked-O path is active. Full attention and all original-output/original-norm control paths retain prior helpers; their canonical new helper was not device-gated. PreparedGroup owns a per-execution shard_quant flag; normal constructors enable it and the matched diagnostic disables it before any layer runs. It adds no scratch ownership and preserves all four residual stores, KV commits and atomic cursor advancement. No request-content or benchmark condition.

Required: host checks; matched same-group all-logit parity with interleavedR64 and postdecode; independent four-owner gate; coherent public suites and long mixed HTTP evidence. Root also combines separately gated B16/R64 single-launch FFN and guardedprefix768, so its mixed-edge fullmodel gate must pass before submission.
