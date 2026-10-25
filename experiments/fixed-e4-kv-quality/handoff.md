# Root-only dispatch handoff

ON worktree `/round2/fixed-e4-kv-quality-on`, source375bdcf4, captured prefix `/round2/runs/a005-fixed-e4-on-probes-build/built/target/release/`.
OFF worktree `/round2/fixed-e4-kv-quality-model`, sourceaa544f40 (runtime identical8c9016d7), captured prefix `/round2/runs/a006-fixed-e4-off-probes-build/built/target/release/`.
Do not use later shared-target outputs. Exact hashes are in build-pair.json and original run.json.

Head/scalar ON actual already pass. Next paired jobs must use the same executable basename/arguments on both flags:

- `compare_fixed_e4_kv model 512` (and1024 if needed): disjoint16-owner full logit/cache replay plus activeP128 and scalar/B16 paths.
- `compare_fixed_e4_kv r32 512` (or1024): 17-owner regenerated-history replay, scalar→17→17→8.
- `compare_fixed_e4_kv partial 896`: explicit tiled17/31, B16prompt16, scalar, native128 history and BF16-QKV128 reset. Also accepts0/128/1024.
- `compare_fixed_e4_packed 0 1`: quad varied positions with paired active-decoder interleaves, full caches/sentinels/reset.
- `compare_fixed_e4_packed CASE 0` for1..10: other packed owner shapes described in probe-contract.md. One case per bounded job.

Cross-program report reader: `python3 experiments/fixed-e4-kv-quality/compare_reports.py OFF_ARENA_LOG ON_ARENA_LOG --output FILE`. No all-logit original-vsrounded error norm is inferred. Same-program exactness is asserted separately by each executable.

No server artifact or public activation has been built. Six public suites and actual C17 autoregressive regroup are later quality gates; this handoff is numerical diagnostics only. Parent owns all Arena dispatch.
