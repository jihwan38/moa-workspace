# QKV broadcast through a temporary HBM vector

`ops.rs` and `candidate.patch` preserve input RMSNorm and replace its ring256 broadcast with two typed transfers: normalized BF16 DM→temporary HBM `[H]`→replicated BF16 DM `[H]`. The temporary vector is7680 bytes (`H=3840`). Function signatures, input values, normalization, casts, projection math, scales, RoPE, and cache writes remain unchanged. This adds a value-preserving BF16 copy roundtrip rather than reordering arithmetic.

Exact compilation with compiler/std0.8.1 succeeded. Static QKV makespan is **95,433 cycles**, versus original118,815 (1.245×). HBM→replicated DM occupies18,400 cycles. No actual NPU latency or correctness result is claimed.

The rationale is measured scheduling behavior: the original ring occupies roughly62k main-context cycles, whereas directly broadcasting a single DM slice with TDMA contends with weight DMA and regresses. HBM can serve the replicated reads through a different memory route. This candidate uses only a temporary intermediate tensor, with no persistent state or dependence on test inputs.

No live source, NPU queue, or Git state was changed. `evidence.json` and `source-hashes.json` pin sources/schedules. No CPU oracle, RNGD job, or formal submission was run.

Scratch `/tmp/moa-hbm-relay-broadcast` uses isolated `CARGO_TARGET_DIR=/tmp/moa-hbm-relay-broadcast/target` and `FURIOSA_OPT_OUT_DIR=/tmp/moa-hbm-relay-broadcast/target/furiosa-opt/kernel`. Exact command:

```bash
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
  --dump-schedule schedules/sliding_project_qkv.json
```

This candidate is independent of FP8 fusion; a combination must be compiled and measured separately. The [DMA Engine contract](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html) and [0.8.1 memory tensor implementations](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/tensor/memory.rs) support typed value-preserving moves between HBM and DM. Accessed2026-09-21.
