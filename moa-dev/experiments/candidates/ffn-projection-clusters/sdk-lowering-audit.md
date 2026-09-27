# E129: public cluster API versus current lowering

E129 `cluster-direct-v1` is blocked by an explicit compiler lowering limitation. It has no successful exact image, descriptor PASS, or NPU result. The public `cluster_tile` API and CPU mapping implementation accept this intended operation, but that does not establish device lowering support.

The preserved [compile log](evidence/e129-clusters-v1/compile.log) fails at the first mutable destination cluster tile in [mlp.rs](cluster-direct-v1/mlp.rs), line92:

```text
visa: a resolved index step counts in-slice cells, but this tile targets Cluster;
tiling a non-in-slice axis is not lowered yet
```

[Build record](build-e129-clusters-v1-repro.json) preserves the exact command, exit1, and source manifest. Source SHA256 is `15f1b86df1757fe0f2b450fb3f349bee010ece01f5a8e0397915a0355bcd5e0d`; compile log SHA256 is `026cd0cd50e9bdd39e3aee1539c1c38b3a874a2294e28d3205d5733d73e64baa` (1413B). This is a lowering failure, not a source-bound failure, accuracy failure, or performance loss. The output's immutable cluster tiles were not reached; their similar lowering risk was not tested separately.

## Intended data path and API contract

The proposal assigns all up rows to cluster0 and all gate rows to cluster1. Each owns256slices×60rows and performs the same six-row native pipeline. Separate public row-major HBM inputs reshape only L into `[slice256,row60]`, then write the selected cluster of a common compact DM `[Projection2]/[slice256]/[row60,H]`. There is no packed DM assembly copy or full FP8 weight materialization in this proposed path.

Expected ingress per projection is29,491,200B FP4 weight and3,686,400B FP8 block scales. Per local slice these are115,200B and14,400B. The local Element shape is compact; the cluster selection is spatial. A successful lowering would have to prove each source's bounds, exact single coverage, and destination cluster ownership from its emitted descriptors.

Installed SDK0.8.1 was inspected at `/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.8.1/src` in `moa2026-dev`. Host mirror `/tmp/moa-opt-std-release/furiosa-opt-std-0.8.1/src` matched the installed SHA256 for `tensor/memory.rs` (`fec623ddf8fadd5cf31b32beb747243c2c463c48e4f73bb29bb2b6ef5d1aaff1`) and `tensor/view.rs` (`a6a93cc4487300b80fc19164f8206badd6ed79df46c9d33e1e9cb2afaf457e58`). Relevant implementation:

- `tensor/memory.rs:484`, `HbmTensorView::to_dm_view`: derives a transfer from source and destination mappings.
- `memory.rs:1274`, mutable `cluster_tile`: preserves physical Cluster::SIZE2, uses Bottom padding outside the selected slot, and delegates to indexed retile.
- `memory.rs:1102`, immutable `cluster_tile`: uses Top padding for a read view.
- `tensor/view.rs:173`: indexed retile retains base storage and records the selected live axis coordinate in offset.
- `constraints.rs:194`: preserving physical cluster size means the selected view remains two physical slots, not a tensor resized to Cluster::SIZE1.
- `storage/buf.rs:270–325,778+`: CPU transpose resolves partial-view base offset against the base mapping and packet positions against the new view mapping. This explains API/CPU semantics; it does not override the exact NPU lowering error.

The FP32 outputs would convert into60BF16 values per slice (120B), using four-BF16 8B commits. A full owned reshape to `[Projection]/[L/60]/[L%60]`, then selected source cluster DMA into `[L/7680]/[L/120%64,1#4]/[L%120]`, would restore existing M03 ownership: `l=7680*c+120*g+r`, live slice4*g. Each projection has30,720 logical output bytes. The actual transfer is required: merely renaming Projection to the L-half cluster axis would assign wrong values.

The intended output mapping takes adjacent source slices2g and2g+1 into one120-row destination slice; source slices0..127 feed target cluster0 and128..255 feed target cluster1. This coordinate proof describes the required graph verification; no emitted output descriptors exist for this failed build.

DMA type validation is not a substitute for source-bound analysis. `tensor/memory/dma_layout.rs:49–74` uses the reachable destination tail, and its destination-padding test at113–120 permits that tail to grow. This is why the older padded Element overread failure must still be checked independently. E129's observed failure occurs earlier, in cluster index lowering.

## Static alternatives inspected

No supported public repair preserving the same one-copy ingress was identified:

| Proposed route | Inspected contract and result |
|---|---|
| Static cluster1-only mapping / custom `M` | `furiosa-mapping-types-0.8.1/src/lib.rs:144+` contains Broadcast, Symbol, Stride, Modulo, Resize, Padding, Pair; no Offset, Slice, Concatenate, or arbitrary permutation. These constructors preserve a zero-origin live prefix and trailing padding. `dsl.rs:23` requires `M::to_value()->Mapping`; overriding only `map(index)` cannot supply an unsupported compiler mapping. |
| Fixed-address or raw DM handle | `DmTensor` inner/address are private, `from_parts` is pub(crate), `new()` delegates placement. No public `at_address`, raw constructor, or DM concat/join was found. |
| `cluster_swap` | Public deferred DM-source redistribution exists in `tensor/memory/redistribute.rs`; it is not an HBM-source or mutable-destination cluster selector. Using additional packed moves would change the cost model. |
| `begin_interleaved` + `fetch_cluster_lift` | Public methods exist (`context.rs:97`, `engine/fetch.rs:175`). A lifted axis requires appropriate values already present at source placements. No expression was established that combines the two separate HBM inputs into the desired distinct physical clusters without replication or assembly. This route is unverified, not a demonstrated impossibility. |

No private SDK patch, fabricated address alias, padded-shape sweep, or compilation of these unsupported alternatives was performed by this reviewer. Closing this frozen prototype does not reject projection ownership in general. A future compiler with non-in-slice indexed lowering, or a different complete data path, is a legitimate reopening condition.

## Scheduling consequence and evidence scope

`runtime.rs:35–45` exposes one Main, Sub and TDMA context; `context.rs:17–37` has Main/Sub, with no cluster-specific Main handle. Cluster mappings describe spatial ownership. Two calls to `main.begin` on different cluster views do not establish two independently scheduled instruction streams. The FFN proposal instead applies one compatible pipeline across both projections. QKV's unequal shapes/postprocessing would require a separate full-path design.

Official primary documents were fetched on2026-09-27: [DMA Engine](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html) for mapping-derived sequencers and alignment, and [Schedule](https://developer.furiosa.ai/furiosa-opt/book/scheduling/schedule.html) for same-context ordering and shared-resource constraints. No documentation bandwidth number is converted into a predicted NPU gain here.

The subsequent E131 alternative combines E123 joint packed assembly with fused native dot/scale/reduction. It must account for those additional packed copies, retain source-bound validation, and be measured as its own complete implementation. It is not a successful repair of E129's no-copy placement path.
