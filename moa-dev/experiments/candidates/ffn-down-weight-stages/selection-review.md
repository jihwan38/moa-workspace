# E147/E142 FFN down ingress: one unselected two-stage design

Read-only, 2026-10-01. No candidate source, compile or NPU run. Source and evidence hashes plus a logical-coordinate enumeration are in `moa-ffn-down-staged-ingress-review.json`; the neighboring Python script reproduces the source-bound CPU coverage calculation. This is a scheduling hypothesis, not a projected saving.

## What the new trace actually shows

E148 readiness job141134 profiles the current E142 FFN image. Six observations share one allocation. Endpoint medians are:

| Landmark | Begin | End |
|---|---:|---:|
| down FP4 HBM ingress |146923|213604|
| up F32 gather |178692|214282|
| GeGLU output |215152|217157|
| down canonical BF16 gather |217158|218627|
| down FP4 table ready |216705|223916|
| down activation TRF |253148|254921|
| down weighted native |254922|275930|
| down blockscale HBM ingress |245249|265340|

The same-observation GeGLU-end→down-TRF-end median is36829, and native starts exactly one cycle after TRF in all six observations. Down weight is ready about40.6k before native. Therefore earlier first weight readiness alone does not remove the observed native wait. The useful question is whether dividing the long down-weight transfer lets the already-ready up gather and then GeGLU/activation preparation complete earlier. Their overlapping envelopes and adjacent completion are evidence of a scheduling opportunity, not proof of exclusive bus service or hardware-bank contention. In particular, independently summarized endpoints cannot be subtracted as a claimed cycle saving.

## Minimal complete path

Current `src/device/shared/mlp.rs::project_down_tile` (429 onward) has canonical per-cluster L7680 input → unchanged weighted E5 quantization/broadcast → one shared `[Term3,K1920]` TRF and96B coefficient VRF → full local60×1920 FP4 HBM→DM → one direct lookup/native → F32 `[H%60,TileL/16%120]` partials → original five12-row blockscale/localK/K4 reductions → cluster gather/add/BF16.

Proposed single change: local60 weight rows become compact30+30 ingress and native streams. Allocate the original F32 `[H%60,TileL/16%120]` partial tensor once, and have each weighted native commit directly to its30-row half. Retain the existing full blockscale DMA and all five12-row consumers verbatim; retain quantizer, TRF,96B weighted coefficient VRF, K4 ownership and the entire finish/down epilogue. This avoids any new assembly copy or new reduction across stages. It also deliberately makes no promise that the first12-row consumer starts before the second native: mutable whole-allocation dependencies may wait for both.

The analogous public pattern exists in C09: HBM view row tile → fresh compact DM via `.to_dm()` → stage helper → `.commit_view()` into a full F32 allocation. For current down, expose HBM as `[H/60,H%60,Dummy2,TileL]`, tile `H%60=30#60` at0/30, and use compact destination `DmTensor<f4e2m1,Chip,UpGateCluster,DownColumns,m![H%60=30,TileL%1920]>`. The current native chain is unchanged except row extent60→30; its output destination is `partials.view_mut().tile::<m![H%60],30,m![H%60=30#{!}60,TileL/16%120]>(0 or30)`.

SDK0.8.1 `tensor/memory.rs:610` implements `HbmTensorView::to_dm` with destination layout/alignment checking. `engine/commit.rs:33` implements `commit_view` with slice preservation and `config_commit` verification. These contracts and C09/E146 target precedents justify a concrete attempt; they are not proof that this new full graph compiles or schedules as intended. Do not use typed DMA into padded mutable partial views.

## Exact geometry and complete costs

Physical slice `s=4g+q`, cluster `c`: `h=60g+30t+r`, `l=7680c+1920q+k`, with `g<64,q<4,t<2,r<30,k<1920`. Packed source byte start is `(h*15360+l)/2`. Each source row segment stays960B with original7680B row pitch; staging does not repair its transaction efficiency. Each stage reads14,745,600B; the two unions cover exactly29,491,200B with no holes or duplicates. The CPU enumeration covers all30,720 row segments, not emitted DMA packets.

The current exact E142 graph `DmaCommand#O244` (graph operator197) emits `H_1=60:7680` source row stride and `H_1=60:960` destination row stride, with960B terminal segments. A new graph must retain all other owner strides/offsets and change only the row extent/start; there is no assumed256B transaction or bandwidth value.

Each native stage writes900 ×16B =14,400B F32 per slice directly at offsets0/14,400 into the original28,800B full partial. Full F32 partial write/read remains14,745,600B (14.0625MiB) each direction across512 slices. Total weight payload, activation replication, native arithmetic, scale bytes and blockscale work are unchanged. Added costs are one weight DMA, one Main native/lookup setup, one additional table preparation unless compiler shares it, and mutable-output IndexWrite/synchronization. Neither peak packed DM nor shorter register lifetime is guaranteed: the compiler may retain both weight stages and the full partial allocation simultaneously. Existing coefficient96B plus blockscale5760B plus post-RMS gamma1920B already totals7776B/slice when concurrently live, before any other VRF payload; verify actual lifetimes, not only nominal capacities.

No public source-order hint has been found. The compiler can issue both weight transfers consecutively before the up gather and reproduce the old completion barrier. Merely placing the second `.to_dm()` after the first native in Rust does not prove an execution dependency. The discriminating check is the emitted whole schedule and a bounded actual readiness trace: did up gather finish between the weight stages; did GeGLU/quant/TRF become earlier; did the extra table/native setup erase the change? Whole timing decides adoption.

## Prior attempts and exclusions

- C09 two-stage BF16 O was adopted after hardware repetition. Its output-row split/direct F32 commit is the API precedent, not an imported gain.
- E146 Q/K two-stage full-H→Switch native is the closest recent dependency test: all exact/oracle cases pass, but paired changes −1.495%, +0.711%, +0.407% did not reproduce an improvement. Extra stage tasks can cancel readiness benefits.
- M04 `m04-tiled-weight-supply/row-tiles-v1` changed all three projections to five stages, preserved FP8 materialization, used the old BF16 down/K8 path, and regressed13.67%. Current weighted E5 down/direct FP4-native and a two-piece down-only intervention are materially different. M04 nevertheless warns that960B down-weight and120B scale segments do not become more efficient merely by tiling.
- C12 split Main dot/Sub reduction and added padded partial/packing work; it regressed2.46% and did not achieve intended overlap. This proposal retains the existing reduction contexts and adds no partial copy.
- C14 was L-stage producer→GeGLU→down pipeline with different cluster/K ownership and extra stage sums. It is not this output-row ingress split.

First gate must check source identity outside the down weight/native producer, actual original HBM bounds and disjoint partial offsets, all shared TRF/VRF lifetimes and bitwise-preserved per-row weighted term/scale/reduction order. Keep the original down-boundary oracle, whole controls and input/reference contracts. No speculative fill, buffer padding or stage-count sweep is justified if scheduling does not expose the intended opportunity.

### Source preparation follow-up

The deterministic proposed source is now prepared under `/tmp/moa-ffn-down-two-stage-preparation/` (`prepare.py`, `mlp.rs`, `helper.rs`, `source.diff`, `source-contract.json`). `prepare.py --check` passes: output39763B, SHA256 `813aa7068376a41833723748ba31bf43008972bb5ff9b4e1e7cbc3d185a23530`. Only the packed-weight/native partial producer in `project_down_tile` plus one unconditional helper change; all other source, full scale load and five12-row consumers are byte-preserved. No compilation has been attempted.

Each compact packed half is28800B/slice =225×128B, not a whole number of256B units. The emitted descriptor's source extent, destination extent and128B phase require inspection. An odd number of128B units is not evidence of a timeout, and no padding is proposed. The F32 partial half is14400B, with exact full-root ranges0..14400 and14400..28800; the unchanged12-row consumer covering local rows24..35 correctly reads both halves.

The shared mutable partial root adds an explicit IndexWrite dependency between native0/native1, so the second weight's consumer deadline may differ from the original one-task graph. It does not force up gather between weight transfers. If the static schedule issues both transfers consecutively before up gather, expected readiness benefit is weak; that is still not a complete hardware failure verdict. A standard contemporary45-comparison early whole screen remains necessary after actual descriptor/lifetime gates. Do not tune stage counts merely because the first proposed ordering is unhelpful.
