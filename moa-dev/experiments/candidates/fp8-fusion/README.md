# FP8 lookup → contraction fusion candidate

Prepared 2026-09-21 from workspace commit `90f3c3fe16f4fe445f81e82d30b17b1b951e0082`.

`projection.rs` is a complete replacement candidate for `furiosa-opt-gemma4-12B/src/device/sliding/projection.rs`; `projection.patch` is the corresponding unified diff. Neither has been applied to the live source tree. `source-hashes.json` pins the original and candidate files.

The change removes the BF16 DM materialization and second fetch between FP8 table lookup and contraction in `project_query`, `project_one_kv_matrix` (K and V), and `output_partial` (O). It preserves HBM/DMA mappings, lookup type, contraction mappings/reduction order, output casts, scaling, and all function signatures. The original lookup-side fetch mapping is retained, including O's 32-element FP8 fetch followed by 16-element BF16 collect.

## Static compilation results

Both selected kernels compile with `cargo-furiosa-opt 0.8.1`, `furiosa-opt-std 0.8.1`, and `nightly-2026-05-01` without diagnostics requiring a fix.

| Kernel | Original schedule cycles | Candidate schedule cycles | Static speedup |
|---|---:|---:|---:|
| `ops::sliding_project_qkv` | 118,815 | 110,763 | 1.0727× |
| `ops::sliding_attention_output` | 195,823 | 162,813 | 1.2027× |

These are schedule makespans, not measured RNGD cycles. Numerical correctness has not been tested. No CPU oracle, NPU job, or competition submission was run for this candidate.

`evidence.json` contains schedule hashes, context interval unions, longest instructions, commands, and status. Raw schedules and compile logs remain in Docker at `/tmp/moa-fp8-fusion/{schedules,logs}` to avoid checking large generated files into Git.

## Commands used

The original crate's `src/`, `Cargo.toml`, `Cargo.lock`, and `rust-toolchain.toml` were copied into `/tmp/moa-fp8-fusion`, then only the scratch projection file was replaced. From the workspace root, compilation used:

```bash
./moa-dev/moa bash -c '
  set -o pipefail
  cd /tmp/moa-fp8-fusion
  export CARGO_TARGET_DIR=/tmp/moa-fp8-fusion/target
  export FURIOSA_OPT_OUT_DIR=/tmp/moa-fp8-fusion/target/furiosa-opt/kernel
  cargo furiosa-opt compile ops::sliding_project_qkv --exact \
    --dump-schedule schedules/sliding_project_qkv.json \
    2>&1 | tee logs/qkv-compile.log
'

./moa-dev/moa bash -c '
  set -o pipefail
  cd /tmp/moa-fp8-fusion
  export CARGO_TARGET_DIR=/tmp/moa-fp8-fusion/target
  export FURIOSA_OPT_OUT_DIR=/tmp/moa-fp8-fusion/target/furiosa-opt/kernel
  cargo furiosa-opt compile ops::sliding_attention_output --exact \
    --dump-schedule schedules/sliding_attention_output.json \
    2>&1 | tee logs/output-compile.log
'
```

Both exited 0 and reported one compiled kernel with 14 nonmatching kernels skipped. The baseline schedule comparison reads `/workspace/target/schedules/baseline/`; all compilation outputs go to the scratch paths above.

After applying the patch, run the full standard correctness suite on RNGD before accepting it. If combining with TDMA broadcast changes, record that combination separately: the static figures above use the original broadcast implementation.
