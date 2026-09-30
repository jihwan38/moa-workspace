# D-E143: E142 versus frozen E138 v2 QKV readiness

One bounded Arena allocation **136455** completed with **81/81 comparisons passing**. E142 standard QKV was104551/104192cycles before/after; frozen E138 v2 was108642, **+4.091634%** versus the104371.5 bracket. No production source was changed or adopted. Actual start`6c3a3f51188a53efa68f5ada5f88b6d2bd0123ce`, branch`diag/e142-e138v2-qkv-readiness`.

The question is resolved at this scope: first V waits for activation readiness, while later Q and K still follow their respective weight arrivals. In **every one of the six same-run/cluster samples**, V native starts1cycle after shared TRF completion; Q and K start2cycles after their own weight-DMA end. TRF was already ready a median21629cycles before Q native and47147 before K native. This supports investigating a different weight producer/route, not attributing all remaining loss to quantization.

| Frozen E138 v2 landmark | Begin median | End median |
| --- | ---: | ---: |
| pre-RMS final apply |7738|8982.5|
| BF16 seed DMA |8990|13145.5|
| input ring32 |13147.5|17362.5|
| K4 global amax |18221.5|18632.5|
| scale production |18633.5|19040.5|
| quantization-scale DMA |21832.5|32620.5|
| quantization-scale VRF |32622.5|32967.5|
| fourth/final term |44644.5|46219.5|
| coefficient VRF |34984.5|35301.5|
| activation TRF |46220.5|47515.5|
| V weight / native |13504.5 /47516.5|31975.5 /49748.5|
| Q weight / native |34129.5 /69196|69194 /72411|
| K weight / native |75960.5 /94573.5|94571.5 /96824.5|
| Q output store |98610.5|99468.5|
| V cache scatter, cluster0 |100117|101705|
| K cache scatter, cluster0 |106687|108275|

Times are relative to each cluster's Task start. Endpoints above are **separate medians**, so subtracting them is not generally the median elapsed interval. Same-run/cluster differences give pre-RMS-end→TRF-ready38655.5cycles, seed-ready→ring-ready4213cycles and ring-ready→quant-done29017cycles. These windows include scheduling/dependency effects and are not reclaimable-time estimates. The quant-scale DMA overlaps the long V-weight marker, so neither its10788cycle span nor the weight span is a measured isolated bus-service duration. Coefficient generation is also overlapped; its four spans must not simply be summed into a quantization speedup. Term/K4 reductions are inside each native marker and cannot be timed separately here.

Control E142 is declared independently by`baseline/e142.json`, source`b3fe0ab4be2a5ece6382ae5745112722200f1b66`; frozen candidate is E138`native-v2` built from E137 source. The candidate standard host therefore retains the older E134 FFN, while the control host has E142 FFN. **Only QKV is compared.** O images are identical; full-host runs independently validate each original bundle. [Source/image binding](source-image-binding.json) checks all46source files per side, original fixture/runner and all three embedded images. The changed sources between these two declared bundles are`projection.rs`, `ops.rs` and`mlp.rs`; this is not presented as an isolated whole-model comparison.

E142 QKV168960B SHA`5bdab7eafec5ac0aca352101deee334180b550c87db23e33c70d008892edba9e` and E138v2 QKV205824B SHA`1c4a10c86461d35eb3fb10dd2e32b1cd2a7216069c832bcbb91a428006c574f6` are frozen. Each diagnostic host reproduces its original three device images exactly before only existing QKV profile-name bytes are replaced. Instruction/task payloads, constants, addresses, inputs/outputs and marker counts remain unchanged. All76/92executable markers correspond to the actual schedules. Candidate cos88/sin293/Vscatter432/Kscatter503 are absent on cluster1 because their actual graph descriptors target only cluster0. [Independent image review](independent-image-review.json) and [source/readiness mapping](independent-source-readiness-map.json) are separate from the packer's checks. Historical control`graph.json` holds binary compiler IR, not JSON/text; see [artifact format note](artifact-format-and-scope.json).

Standard host info, diagnostic info and diagnostic trace QKV medians were104551/104906/105675 for E142 and108642/107166/108283 for E138v2. Host/instrumentation modes therefore differ measurably; internal trace timings are not adoption-performance samples. Three inputs and two clusters in one allocation do not constitute six independent repetitions.

[Decision](decision.json): no adoption from this diagnostic; no source-order, row/term or input-routing sweep; select a complete weight-ingress/ownership feasibility review using the observed later Q/K readiness. Root selected canonical full-H3840 weight ingress→Main Switch→same K4 native for the next supported-path review. Its added full-H staging/switch/storage and changed scheduling must be included and measured before any gain claim. No new implementation or additional NPU diagnostic is included here.

Reproduction from the workspace root:

```sh
python3 moa-dev/experiments/candidates/e142-e138v2-qkv-readiness/analyze.py e143-qkv-v2-readiness-v1 136455 --check
python3 moa-dev/experiments/candidates/e142-e138v2-qkv-readiness/verify.py e143-qkv-v2-readiness-v1 136455 --check
```

[Raw log](evidence/arena-136455.log), [uploaded archive binding/result](e143-qkv-v2-readiness-v1-job-136455-results.json), [compact summary](readiness-summary.json), and [full source-bound analysis](profile-analysis.json) preserve the evidence. Build/plan records and original source snapshots remain frozen; no official submission occurred.
