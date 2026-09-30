# E146 selection: staged Q/K weight supply after actual E144 readiness

Source/SDK review, 2026-10-01 KST, used to select E146 after D-E145. This research note does not itself establish E146 correctness or performance; candidate build and measurement records are separate. Performance conclusions below use the actual D145 frozen trace; source legality, historical lowering support, and new uncompiled compositions are separate claims.

## Fresh public ranking

`docs/experiments/leaderboard-e145-start-20261001.json` contains the exact raw public GET body, parsed response, both connection attempts, capture time and SHA. URL: https://micro2026-api.duckdns.org:7777/api/leaderboard . Captured 2026-09-30 15:05:32.773913 UTC (2026-10-01 00:05:32 KST), 27,061 bytes, SHA256 `fbc14b800c7a515548dfb85789a5d876e75ab5ef28dbb9b370f58f0e2b809a3d`. Initial default-TLS request reset; an explicit TLS1.2 HTTP1.1 GET succeeded. No authentication or private logs were used.

| Public position | Score | QKV / O / FFN cycles |
|---|---:|---|
| BEEF, 55 / 80 teams | 6.748109876962911 | 95918 / 42945 / 296581 |
| 30th | 8.068829902032553 | 75364 / 38860 / 244008 |
| 10th | 9.19578932242152 | 69798 / 35036 / 197414 |

BEEF best submission is `2a66e430`. Reaching the current 30th score requires +19.5717% score, equivalent to 16.3682% lower cycles if all three kernels improved uniformly. For 10th these are +36.2721% and 26.6174%. These arithmetic equivalents are neither forecasts nor a sum of kernel improvements.

## What the current execution establishes

Frozen E144 source: `moa-dev/experiments/candidates/qkv-fullh-weight-switch/fullh-v2/projection.rs`, SHA256 `3d2500295785592bebdc723fb62617398e2e001ba41c63b2a5c078cda5e00fed`. Exact graph/schedule: same candidate `evidence/e144-fullh-v2/{graph,schedule}.json`, SHA256 graph `5ad07991772f812457ea8a6e3b6121c64ba8d1efec18924bafe623ced7cec75e`, schedule `62ae976d64e582da0a8b8e71f3fe11d777bb09402741a86901c6a9cc99bb13f7`.

`prepare_qkv_terms` produces one shared activation TRF from consumer-local H960 shards. K4 max retains Group64; `s=2^floor(log2(max(abs(x),2^-119)))/128`. Four RNE E4 terms use factors `[1,16,2^22,2^27]`. Two cumulative reconstructions preserve signed residuals and are materialized to F32 DM, then loaded into VRF. Four Main constant streams produce normalized coefficients `[1,1/16,2^-22,2^-27]`; a Sub packet pack and typed DMA distribute them. Native term sum precedes K4 sum, then common s and channel scale precede the first BF16 boundary. No proposal below changes these numerical contracts.

Actual D145 source: `moa-dev/experiments/candidates/e144-qkv-readiness/profile-analysis.json`, `summaries.candidate`. Six cluster observations show Q and K native begin exactly two cycles after their own weight-ready markers; V begins one cycle after activation TRF. Median task-relative endpoints: TRF 43856.5, V native 43857.5–48932.5, Q weight 58219.5 / native 58221.5–67132.5, K weight 82513.5 / native 82515.5–87589.5. Last native to task end is a paired median 11128 cycles.

Scale-ready to quant-scale-ready is a paired median 9914 cycles; this is a producer-to-consumer interval, not a measured pure bus stall. In particular the scalar DMA begins a paired median **5976 cycles before V weight ends**, so the actual trace does not support a sequential “scalar DMA starts after V weight” claim inferred from the static schedule. Marker envelopes include waiting. We cannot add their durations into expected savings.

## First complete-path priority: Q/K full-H two-stage weight supply

The measured late Q/K predecessor makes row-stage overlap more directly relevant to completion than only accelerating V's activation preparation. Keep V, global quantization, shared TRF, coefficient order, K4 ownership and head boundaries fixed initially. Split Q32 into two 16-row pieces and K16 into two 8-row pieces, each retaining E144's full-H row DMA and low-two-bit row ownership. Switch/native consumes the first piece while the next compact piece is supplied. Each piece writes a disjoint row range of the same complete raw F32 buffer; the original canonical-head gather occurs once after both writes. No new decoded-weight materialization, intermediate result assembly pass, or split-K reduction is required by this design.

This is a coupled supply/compute pipeline, not a claim that a smaller tile is intrinsically faster. Source recipes must preserve original HBM pitch and offset: Q row `2048*c+32*g+r`, K row `1024*c+16*g+r`, full H3840. For compact half-row inputs the source half offset must not be expressed by DMA into a destination padded to the full row count; prior E133/E140 overruns make actual source extent/stride validation mandatory. Explicit half-shaped loads and exact disjoint output commits are safer contracts to audit. The switch's physical output row ordering must be derived anew for the half shape, not guessed from the API name.

Added costs: Q/K each go from one to two weight loads and from one to two Main Switch/native tasks; Switch setup/bitmap operations may grow; mutable output writes add dependencies; smaller transactions may be less efficient. Total useful weight bytes and raw output bytes remain unchanged, but that does not prove unchanged physical traffic or setup. The compiler may serialize half loads and Main tasks or conservatively keep full allocations live. The essential gate is actual first-piece weight readiness, next-piece DMA/Main overlap, exact disjoint raw coverage and unchanged full boundary. The first whole screen decides the net result. C09's two-stage principle is a relevant prior, not an already-tested E144 composition; C12/other static overlap results do not transfer a numerical speedup.

## Second, conditional path: one K4 max/pow2 producer and scalar Sub broadcast→VRF

Current separate K4 max and exponent-mask/scale Main streams can be composed in the public InterFirst pipeline: `vector_inter_slice_reduce(Max)` → `vector_intra_slice_tag(Zero)` → reinterpret/BitAnd(0x7f800000) → F32 → narrow → Mul0(1/128) → widen/final. Local max and its floor stay unchanged. Installed `furiosa-opt-std-0.8.1/src/engine/vector/tensor/vector_tensor.rs:649–677` explicitly allows inter-slice output to enter the intra-slice tag pipeline. This removes one leader DM handoff/Main setup in source; this exact new whole composition is not yet compiled.

The stronger part removes scalar-distribution TDMA competition without inventing a free constant array. Keep existing four coefficient builders and packet pack initially. For scale and packed coefficients separately, use Sub fetch → `CustomBroadcast { ring_size:4 }` → Collect → VRF. Input Slice is `[QkvGroup64,1#4]`; output Slice should be **anonymous** `[QkvGroup64,4]`. Then use the public unsafe `VrfTensor::reshape` to name these already-real copies `[QkvGroup64,H/960]`, and consistently relabel cluster m![2]/Dummy2 only where identical physical copies are proved. Preserve the separate common-scale DM route to canonical head epilogues.

Concrete source geometry: scale fetch/collect Time1, Packet1#8 (32B F32); coefficient source has compact Term4 (16B), fetch Time1/Term4, collect Time1/Term4#8 (32B VRF). Expected output slice `4*g+q` reads input slice `4*g`, same cluster, same element; all four q receive the same scalar or ordered four coefficient cells. Test actual 256 bitmap entries, initialized source q0, packet extent, output masks and live VRF intervals. Useful replicated DM payload removed is 16KiB scale +8KiB coefficients across two clusters. VRF writes remain 32B×512 each; Switch, bitmap load, Sub occupancy and coefficient-generation tasks remain/add costs. This cannot be called free or assigned a latency from byte count.

**Support is stronger than the failed E55 route, but not yet exact proof.** `engine/switch.rs:57–86` is context-generic and leaves Packet unchanged; `engine/collect.rs:89–99` supports Sub→VRF. `tensor/memory.rs:1488–1527` explicitly documents VRF reshape for naming already-broadcast Chip/Cluster/Slice copies, with identical per-level size and no element movement. Naming Element broadcast as data would be invalid. The direct named-output shortcut should not be assumed to pass.

Historical evidence: `c04-down-transfer/four-ring16-sub/mlp.rs:383–401` compiled CustomBroadcast on Sub, wrote DM with anonymous replica axes, and then relabelled. E80 passed basic NPU accuracy; three repeated FFN measurements regressed (+8.03%), proving support for that path, not its performance here. `c08-fused-qkv/c03-columns8/projection.rs:30–36` uses Sub InterTranspose→VRF but has no custom bitmap. E55 `qkv-bf16-broadcast-trf-e55/README.md:25–38` failed MIR named-output bitmap enumeration at `.to_trf()`; it lacked a public TRF relabel workaround. VRF's documented reshape is a concrete difference. Exact Sub CustomBroadcast4→VRF→named reshape has not been found in a prior measured kernel. Native `config_switch`/bitmap implementation is behind FFI in `furiosa-opt-lower-0.8.1/src/lowering/mod.rs:167–194`; exposed generic Rust API and mapping unit tests do not establish new target lowering.

Choose this only if the whole cost/dependency case remains after weight staging or if a combined design is explicitly justified. Advancing V readiness alone does not demonstrate earlier late Q/K or task completion. Do not add it automatically to the first weight-stage experiment and lose attribution.

## Not immediate candidates

Main F32 sum→direct VRF is a real API (`collect.rs:103–115`, `CanApplyToVrf` includes VectorFinal), with exact-compiled C11 two-row precedent and no C11 NPU validation. It is not an in-place alias guarantee. E144's actual residual registers are 3840B/slice at address4352: T118 lifetime16319–18023, T126 18023–20008, T132 20008–21272. Existing DM boundaries let these reuse one allocation. At head-owning slices early Q scale2048B (T89), V scale1024B (T71), quant scale32B (T112) and coefficients32B (T121) coexist. Naively producing the next residual VRF while reading the old one requires at least `3840*2+2048+1024+32+32=10816B`, above 8192B. The older E138 9824B figure is not the current E144 bound. Splitting a producer into 480 values does not cure this if both whole allocations stay live. Any revived design must change the actual allocation/lifetimes or downstream scale representation as a coherent whole path.

No arbitrary VRF immediate array constructor was found. `DmTensorViewMut::memset` accepts a constant whole-allocation scalar fill; ranged/tiled fill is rejected. Four whole fills plus copies/packing are possible but not a free replacement for four coefficient values. Prior bitwise review already rejected naive mantissa splitting and nonexistent variable-shift/RNE shortcuts; no term-count or precision relaxation is proposed.

## Evidence status

All named repository paths above were inspected against `/home/jihwan38/moa-workspace`; D145 evidence is merged in main before the E146 branch starts. Installed SDK read from container `moa2026-dev`, `/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`, version0.8.1. No generic paper claims or GPU timing estimates were transferred to RNGD. The primary useful result is a ranked complete-path question, a specific legal scalar alternative, and an explicit VRF blocker, not a promised speedup.

## Implementation evidence and separate decisions

E146 `opt/qkv-qk-weight-stages` starts at `6ec626988820cb3fa7ba18da5dc417d42cb9b8b4`. The source recipe selects E144 `3586c417e43f4ea3afc7df7c428d381b7dcbcc5e`, and independently declares `baseline/e144.json` as the contemporary performance control. Frozen attempts are under `moa-dev/experiments/candidates/qkv-qk-weight-stages`.

v1 used direct disjoint native writes into full raw Q/K outputs and a constant-bool K/V helper; scheduling failed at a zero-size Scratchpad request. v2 added explicit valid-region initialization (24KiB logical fill, two Sub starts), but the scheduler still failed. v3 introduced compact native results and bounded Sub assembly (24KiB logical read/write each and four Sub starts), but also failed. These are compile failures, not NPU regressions.

Independent vISA inspection found constant-true/false Branches and tensor PhiFunctions still present for K/V despite constant callers. v4 derives from v1 and separates unconditional key/value functions: no v2 fill and no v3 assembly are retained. Exact compilation succeeds with static schedule49218. This establishes a successful source representation, not a speedup or proof of the sole internal compiler failure cause. Actual descriptor/register gates precede first NPU screening.

Adoption: pending actual gate, contemporary complete-kernel performance and changed-boundary validation. Any reproducible positive gain can proceed to bounded adoption validation. Tuning: no arbitrary row-size series; only specific documented producer/consumer repairs. Major research: change late Q/K whole-weight completion boundaries; scale-only work remains conditional because V advances alone may not advance final completion.
