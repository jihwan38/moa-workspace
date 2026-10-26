# Frozen standalone diagnostic

Builda004 passed39.06s from449ab26f; functional source71cdf450 unchanged.
All418 source/config input hashes match before and after build. Original
Cargo.toml/lock preserved. Captured ELF SHA256 `4fb53b95157ed9a32afa91f0f78c7eea2d62fd8e08074d8e31c86ebc7f3085cd`, 4339808 bytes.

Captured path: `/home/jihwan38/moa-worktrees/round2/runs/a004-int4-grouped-activation-projection-device-build/built/target/release/test_int4_grouped_activation_projection`. Arguments: none. Root alone may execute one bounded
<=70s diagnostic; no Arena command was run by this agent.

All4 embedded images match their fresh a004 files. Both candidate projection
images also exactly match the separate resource compile images. The two new
P64 dense-E4 reference images compiled successfully but have no prior actual
correctness proof. Independent source review passed before build.

This gate compares typed/raw I4 and intended-code dense-E4 arithmetic; its
projection-only timings exclude activation quantization/packing/GeGLU. It is
not a current-production performance test, full FFN or model-quality gate.
The historical complete speed failure and old DOWN resource HOLD remain.
