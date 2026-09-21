# QKV RMSNorm: direct replication of the normalized shards

This is a compiled experiment with a **static regression**, retained as evidence.

The candidate appends a QKV-specific `normalize_replicated` helper to `shared/rmsnorm.rs` and calls it only from `sliding_project_qkv`. The original generic `normalize` is unchanged. Mean-square, epsilon, square root, normalization division, weight multiplication, and FP32 arithmetic retain the original order. Instead of gathering the eight normalized FP32 slices and casting globally to BF16, each shard is cast to BF16 and copied directly into replicated DM. Moving a per-element cast across pure copying is value-preserving at the API level; no hardware numerical test has run.

The typed shard layout is `m![1#32,H/480]` with480 BF16 elements per live slice, and the destination is the original `Replicated,m![H]`. The original ring256 broadcast is removed from this QKV path.

Exact compilation succeeded with compiler/std0.8.1. Static makespan is **121,157 cycles**, versus original118,815 (1.97% worse), though lower than the plain DM→DM replication129,141. The shard-replication DMA itself occupies45,152 cycles. This experiment is not recommended for adoption on static evidence.

`ops.rs`, `rmsnorm.rs`, and `candidate.patch` contain the complete candidate/change. `evidence.json` and `source-hashes.json` pin the sources and schedule. No live source, NPU queue, or Git state changed.

Scratch `/tmp/moa-rmsnorm-replicated` uses isolated `CARGO_TARGET_DIR=/tmp/moa-rmsnorm-replicated/target` and `FURIOSA_OPT_OUT_DIR=/tmp/moa-rmsnorm-replicated/target/furiosa-opt/kernel`. Exact command:

```bash
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
  --dump-schedule schedules/sliding_project_qkv.json
```

No CPU oracle, RNGD test, or competition submission was performed.
