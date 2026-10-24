# Two independent P128 prompts with one P256 FFN

Base: root native/canonical a9616bc merged without changing its numerical choices. P256 FFN source09078a1 is hardware bit-exact to two P128 native direct-GeGLU calls, with2.16ms→1.716ms measured by root. QKV/O remain two P128 operations; no unsupported P256 linear kernels are assumed.

First dependency: exact pack/unpack of two [P128,H] hidden tensors into one [P256,H] tensor. Device-only gather/scatter uses the original whole allocation and two explicit byte offsets0 and128*H*2. Hidden values distribute over120 H/32 slices, avoiding a full matrix in one slice. The proof uses all16-bit BF16 bit patterns, swapped halves, all983,040 values, and times the additional transfers. Adoption requires exact copies plus measured overhead below the saved FFN work.

Planned host phase ordering per layer: each request independently computes attention/post-attention, both normalize for FFN, pack, one P256 FFN using shared model weights, unpack, each request independently applies post-FFN norm/residual/gate. Distinct rings/full pages remain owned for the entire group; only matrices share weight reads. Default dispatch remains unchanged until paired full-model and interleaving gates pass.

Initial device build46e9f0e rejected120 physical slices against the fixed256-slice device config. The repair pads the120 live H/32 shards to256 physical slices; payload addresses and half offsets remain unchanged.

Second compiler stage rejected padded120-way DMA partitioning because its lowerer rounded to128 physical partitions. The revised transfer uses32 live H/120 partitions with an explicit eight-way padded physical axis, matching established DMA layouts. No arithmetic or addressing changes.

Actual I/O proof317989/r232 passed all983,040 BF16 payloads with swapped halves. Warm pack121.548us plus unpack122.729us costs244.277us/layer. Against the measured450us FFN saving, the optimistic net is about206us/layer (~9.9ms per48-layer pair), before host overhead. The newer standalone P128 H-local FFN (.947ms each) would erase this gain if adopted alone; this pair diagnostic deliberately retains the prior P128 baseline for an internally consistent comparison. Production pair dispatch remains disabled.

The model exposes explicit start_prefill_pair/advance_prefill_pair APIs, with two persistent WidePrefill states and an allocation guard covering both sets of KV blocks. Existing single-request advance_layer still executes its full FFN. The paired probe compares against two original P128 native forwards at independent page/ring boundaries; optional same-R64 decode interleaves every layer. Sixteen host tests pass after factoring.
