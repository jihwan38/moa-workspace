# Two independent P128 prompts with one P256 FFN

Base: root native/canonical a9616bc merged without changing its numerical choices. P256 FFN source09078a1 is hardware bit-exact to two P128 native direct-GeGLU calls, with2.16ms→1.716ms measured by root. QKV/O remain two P128 operations; no unsupported P256 linear kernels are assumed.

First dependency: exact pack/unpack of two [P128,H] hidden tensors into one [P256,H] tensor. Device-only gather/scatter uses the original whole allocation and two explicit byte offsets0 and128*H*2. Hidden values distribute over120 H/32 slices, avoiding a full matrix in one slice. The proof uses all16-bit BF16 bit patterns, swapped halves, all983,040 values, and times the additional transfers. Adoption requires exact copies plus measured overhead below the saved FFN work.

Planned host phase ordering per layer: each request independently computes attention/post-attention, both normalize for FFN, pack, one P256 FFN using shared model weights, unpack, each request independently applies post-FFN norm/residual/gate. Distinct rings/full pages remain owned for the entire group; only matrices share weight reads. Default dispatch remains unchanged until paired full-model and interleaving gates pass.
