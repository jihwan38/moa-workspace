# Typed TDMA broadcast at the two Stage 1 call sites

The candidate changes only two expressions in `src/ops.rs`: `sliding_project_qkv` replaces `layout::broadcast_hidden(device, &x)` with typed `x.to_dm(&mut device.tdma)`, and `sliding_attention_output` similarly replaces `layout::broadcast_sliding_heads`. `ops.rs` is the complete candidate, `ops.patch` the two-line patch. Live sources were not modified.

## Semantic review

Both replacements keep the original explicit destination type:

- QKV: `DmTensor<bf16, Chip, Cluster, Replicated, m![H]>`.
- O: `DmTensor<bf16, Chip, Cluster, Replicated, m![Qs]>`.

The original helpers produce exactly those types using main-context fetch, custom ring256 broadcast, collect, and commit. `Replicated=m![Dummy256]` names the destination copy on each slice; consumers subsequently reshape these identical copies into their row-partition mappings.

Release 0.8.1 `DmTensor::to_dm` keeps element type `D`, validates the source/destination DMA layout, and delegates to `self.inner.transpose(true)`. The `true` permits broadcasting into destination dimensions. It creates a distinct destination buffer rather than reinterpreting a single populated slice as256 copies. `Tensor::transpose` materializes the destination and performs the mapped copy. No arithmetic, cast, normalization, scale, reduction, or cache update moves across this boundary. Consequently the API-level value contract is the same for every live output element; this review is not a substitute for running the compiled candidate on RNGD.

The source and destination element mappings stay identical (`H` or `Qs`), and the per-slice BF16 spans are7680 and8192 bytes respectively, both divisible by the DMA8-byte alignment. `Cluster=m![1#2]` remains unchanged; this patch does not introduce second-cluster computation. Existing `decoder_feedforward` already uses the same typed TDMA replication pattern.

Primary evidence: [0.8.1 memory.rs, `DmTensor::to_dm`](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/tensor/memory.rs), [0.8.1 dma_layout.rs](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/tensor/memory/dma_layout.rs), [Tensor semantics](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/tensor-semantics.html), [DMA Engine](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html). Accessed2026-09-21.

## Static results: one improvement and one regression

| Kernel | Original cycles | TDMA candidate | Change |
|---|---:|---:|---:|
| `sliding_project_qkv` | 118,815 | 129,141 | 8.69% slower |
| `sliding_attention_output` | 195,823 | 172,560 | 11.88% fewer cycles |

Both exact compilations succeeded. The only warning is the intentionally retained unused `broadcast_sliding_heads` helper. The QKV broadcast DMA itself occupies53,136 cycles, but reducing one main-context operation does not guarantee an improved critical path: DMA now carries the broadcast alongside weight transfers. QKV therefore should not be adopted on static evidence alone. O improves by1.135× statically. These two changes may be evaluated or applied independently.

No CPU oracle, RNGD execution, or competition submission ran for this candidate. No numerical or hardware performance result is claimed.

## Reproduction

Scratch `/tmp/moa-tdma-broadcast` was built from the earlier pristine baseline snapshot, restoring the original projection implementation so neither FP8 fusion nor output-row redistribution affects this comparison. Sources/manifests are based on workspace commit `90f3c3fe16f4fe445f81e82d30b17b1b951e0082`.

For each kernel name above, run from the host workspace root (replace `KERNEL` with the exact name):

```bash
./moa-dev/moa bash -c '
  set -o pipefail
  cd /tmp/moa-tdma-broadcast
  export CARGO_TARGET_DIR=/tmp/moa-tdma-broadcast/target
  export FURIOSA_OPT_OUT_DIR=/tmp/moa-tdma-broadcast/target/furiosa-opt/kernel
  cargo furiosa-opt compile ops::KERNEL --exact \
    --dump-schedule schedules/KERNEL.json
'
```

`evidence.json` records hashes, exact kernel commands, context intervals, longest operations, and compile status. Raw schedules and logs remain under the scratch directory. No shared target directory, live source, NPU queue, or Git state was changed.
