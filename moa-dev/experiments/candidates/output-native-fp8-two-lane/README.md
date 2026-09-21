# Native FP8 O projection with column-local input scales

This independent candidate starts from the eight-column, two-cluster O projection
and the fused DivF normalization helper. Exact compilation on std/compiler0.8.1
passes: **32,252 static cycles**, versus32,560 for the same O path with BF16
activation contraction. This0.95% static reduction does not establish an NPU gain.
The combined E21 package ran as Arena job70300. O passed all3 official input
comparisons at median59,016 cycles (61,852/58,442/59,016). Its printed max/mean
errors matched the BF16 O path in E20. The overall job failed because QKV V
outputs failed; no valid combined score is claimed. The QKV portion changes
independently. O needs a fully passing combined package and additional inputs.

`project_output` reads the original4096 BF16 inputs into eight512-value column
shards per row group. Each shard computes its own runtime maximum absolute value
and power-of-two scale. Row groups and both clusters see identical values for a
given column, so they compute identical scales without inter-slice communication.
The two finite FP8 components are `hi=FP8(x/s)` and
`lo=FP8(16*(x/s-hi))`. The source values and all weights are retained; input
quantization is approximate for sufficiently small values relative to a shard's
maximum. The scale floor and decomposition are the same independently derived
algorithm described in [the QKV arithmetic analysis](../qkv-fp8-split/README.md).

Each TRF holds lanes`[hi,lo]`, each with512 elements. The original FP8 weights
stream once into two native FP8 contraction lanes. Sequential lane folding makes
Time`[H%60,Dummy2]`; pair unzip, low-part multiplication by1/16, zip addition,
and runtime scale multiplication produce an FP32 column partial. Applying the
scale **after combining** hi/lo preserves their cancellation before rescaling.
Each column's own scale is applied before summing the eight different columns.

The backend cannot share AddF between pair zip and the inter-slice reduction in
one pipeline. Attempt1 passed Rust type checking but failed TensorUnit lowering
with `no available alu for op GlobalReduce(GlobalReduceAddF)`. Attempt2 stores
the60 column partials in FP32 DM, then starts a fresh inter-slice sum. That storage
is1,920 bytes per physical slice including padded packets and adds no BF16
rounding boundary. After the sum, the original BF16 projection cast, channel
scale cast, normalization and residual rounding boundaries remain in place.
The row gather, channel scale and residual implementations are unchanged.

`analyze.py` probes four synthetic distributions with64 projection rows. Ordinary
Gaussian, zero and column-dependent dynamic ranges had no BF16 output mismatch
in this CPU probe. Two isolated large outliers produced1/64 mismatches. All
per-column reconstruction errors met the recorded `s*2^-14` absolute bound in
these probes. NumPy FP32 dot/sum order is not RNGD's order, and these projections
are not the official final RMSNorm/residual oracle. Cancellation can amplify
relative error; this is not an all-input exact transformation. No test fixture
values, output caches or seed-dependent branches are used.

Frozen sources and hashes are in `evidence.json`; `candidate.patch` changes only
the O projection and appends its helper, relative to the canonical two-lane QKV
source. `ops.rs` selects the separate fused-DivF O norm in `rmsnorm.rs`. The
scratch `/tmp/moa-output-native-fp8` uses its own Cargo target and compiler output.
Attempt logs and CPU probe output are retained. The observed O result is limited to the3 official input configurations and one
job. It cannot be inferred from the unrelated two-lane QKV candidate: E20 and
E21 had V-only accuracy failures despite Q/K passing. See the independent
review for an outlier/cancellation counterexample to all-input equivalence.
