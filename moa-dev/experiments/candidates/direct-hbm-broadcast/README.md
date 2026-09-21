# O input: direct HBM → replicated DM

`ops.rs` / `candidate.patch` replace the O entry point's HBM→single-slice DM→ring256 broadcast with HBM→replicated DM directly. The first destination is `DmTensor<bf16, Chip, Cluster, Replicated, m![Ns,Gs,Ds]>`, followed by the same physical-order reshape to `m![Qs]`. BF16 values, shape, downstream projection, normalization, residual, and signatures are unchanged. No arithmetic or rounding stage is added or removed.

The single-candidate exact compilation succeeded with compiler/std0.8.1 and nightly2026-05-01. Static O makespan is **134,996 cycles**, versus original195,823 and direct DM→DM broadcast172,560. The input DMA occupies19,589 cycles. This is a static result, not a device measurement.

`combined-evidence.json` separately compiles this ops candidate with `../output-256rows/projection.rs`: **78,618 static cycles**, versus original195,823 (2.491×). That combination carries the output-256rows candidate's changed reduction/rounding behavior and requires RNGD numerical validation. Its weight DMA remains26,475 cycles. Individual node durations overlap and must not be summed as measured latency.

No CPU oracle, RNGD execution, or submission ran here. Live files were not changed. The combination is reconstructed by using this `ops.rs` and the other candidate's complete `projection.rs`; no additional source modification is needed.

Scratch single: `/tmp/moa-direct-broadcast`; combined: `/tmp/moa-output-direct-combined`. Both use their own `target` as `CARGO_TARGET_DIR` and `target/furiosa-opt/kernel` as `FURIOSA_OPT_OUT_DIR`. In either scratch directory, the exact command was:

```bash
cargo furiosa-opt compile ops::sliding_attention_output --exact \
  --dump-schedule schedules/sliding_attention_output.json
```

Launch through `./moa-dev/moa bash -c` with an explicit scratch cwd and the two isolated environment variables, as recorded in the evidence. Source/schedule hashes are in `evidence.json`, `combined-evidence.json`, and `source-hashes.json`.

Semantics follow [0.8.1 HbmTensor::to_dm](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/tensor/memory.rs) and the [DMA Engine contract](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html): the destination mapping selects broadcast/distribution while preserving logical values. Accessed2026-09-21.
