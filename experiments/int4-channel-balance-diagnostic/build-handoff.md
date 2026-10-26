Fresh ELF build passed in 55.94 seconds from clean source `e6e45d96` (functional `52d821e2`). No diagnostic source or policy changed. Original dependency hashes and all 423 source/config inputs were verified against the frozen source and current files.

Captured artifact: `/home/jihwan38/moa-worktrees/round2/runs/a003-int4-channel-balance-device-build/built/target/release/test_int4_channel_balance`

ELF SHA256: `a71cf0d522cab1d731d3a95d84fabfa17ae4052bef965382f7250fe13781dbb5`; 6,584,096 bytes.

The new empty public `FURIOSA_OPT_OUT_DIR` was `/round2/tail256-model/target/device-out-int4-channel-balance-a003-20261026`; the build log records all 19 device compilations. The build used the exclusively owned tail256 target, `--fresh-package`, original locked dependencies and immediate `--capture-artifact`.

Read-only registry parsing verified exactly the same 19 symbols, keys, sizes and image SHA256s as the submitted artifact used by actual job 323213. The complete embedded section also matches. Each image also matches its fresh output-directory `.bin`. The prior 421 inputs remain unchanged; only the two new host diagnostic files were added. No `objcopy` or ELF execution was used.

Replay: `PYTHONDONTWRITEBYTECODE=1 python3 experiments/int4-channel-balance-diagnostic/audit_build.py`. Raw build records, source manifest and hashes are preserved here; external immutable ELF paths remain in the run records.

This establishes source/build/image identity only. Gamma compensation range and inverse checks, unquantized sampled products, intended-code parity and numerical drift still require the root-owned actual diagnostic. No Arena submission, model-quality result, serving activation or speed claim was made in this lane.
