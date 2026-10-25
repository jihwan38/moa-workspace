# Exact B16 single-launch FFN integration

Extends the passed R64 integration cc8a088 / Arena318826. B16 device source
ccf19ff passed Arena318834 with every scratch/final bit exact;0.246–0.249ms
versus the four-stage control0.318–0.320ms. Existing Prefill HBM scratch is
reused, so this integration allocates no additional tensors. Only ffn_native's
four FFN launches change; pre/post norms and all prompt/R64 paths are retained.
Model::set_b16_staged_ffn exposes the four-stage same-width diagnostic control.

The paired actual-model probe rebuilds17 independent512-token histories for
each pass with bijective physical slot/page remaps. Widths1,2,8,10,16,17 exercise
scalar fallback at1, B16 at2–16 and R64 at17, followed by position0 reuse and31-token
prompt interleave. All vocabulary logits must match bitwise and remain finite.
CPU sample is checked on actual matched logits for greedy and seeded stochastic
(temperature0.8, top_k37, top_p0.9) configurations with banned token IDs. This
checks CPU sampling parity; it does not claim an HTTP/device-greedy integration
gate. Root's existing device-greedy tests cover that independent code path.

Root owns NPU execution and production adoption. Model and stage times are
pure diagnostics and do not establish official score or SLO compliance.
