# Clean capacity32 integration export

Base b078362: current root with exact B16 FFN epilogue enabled. Export retains that code. Only merge conflict was additive src/lib.rs exports; both B16 epilogue and three capacity32 modules were kept. All three kernel source files are byte-identical to NPU-tested a006.

Automatic independent canonical singleton batches17..32 now select the exact first32 sliding-attention/FFN/LM hot stages. This is a hybrid32 path: norm, QKV, heads, O and full attention still use64-row kernels. Query work is reduced; physical Rc64/Fc256 storage and its ownership do not change. Added private FFN scratch is1,597,696 bytes; no new KV or weights. Existing partial live-prefix commits and Blocked64 output/greedy contracts are retained. All inactive outputs remain outside every live output/cache scatter. Counts1..16 follow current ordinary scalar/B16 dispatch;33..64 keep existing64 hot stages.

Explicit forward_outputs_capacity64 remains a matched originalR64 control. forward_outputs_capacity32 exposes the isolated per-call route and falls back to full64 outside17..32; neither changes persistent state. Disable activation by removing Automatic from the new matches! condition in Model::forward_sequences_inner, without deleting the diagnostics or validated kernels.

Meaningful gates are in decode32-model-actual.md:83,099,648 full-vocabulary logits and18,085,838,848 cache bytes exact, device greedy exact, computed histories, page/wrap boundaries, reorder/reset, active128 prompt and all48 layer interleaves. Warm complete model improves~19–20%, approximately16ms. The reference deliberately forcesR64 for7/16, so those timings do not describe the faster ordinaryB16 path or new c067 epilogue. Kernel micro results/actual trace and original build/run provenance remain documented separately.

Root should build the combined ordinary server and run publicC17/width-transition quality before an official submission. This export changes no Worker policy, admission count, API, weight precision, KV encoding, or sampling behavior. No Arena jobs were dispatched by the implementing subagent.
