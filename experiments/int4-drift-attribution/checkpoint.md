Functional source `5da37fcd`, based on `1a835e18`, passed plain host check in a002 in
10.98 seconds (exit0). a001 at `61b198f3` also passed in11.12 seconds. Logs,
commands and413 input hashes are retained here. All411 original a005 complete-
FFN build input hashes remain identical; the only added inputs are the new
binary and host metrics module. Final `60a26576` adds only two explicit scalar-
reconstruction metric labels and passed a003 host check in0.65 seconds. No new
device function or dependency exists.

The old complete probe, device functions and independent u128 quotient oracle
remain byte-identical. `audit_source.py` verifies the common helper bodies and
records19 prior actual images plus two existing standalone quantizers requiring
a new captured binding. `audit_build.py` is prepared for the later authorized
artifact; no device artifact exists at this checkpoint.

round2_path independently reviewed5da37fcd: factor scale selection, required EE
negative controls, fixed production hidden, original midpoint tail, genuine row
permutation, raw-call-inclusive seven timers and immediate scratch immutability
before timing recomputation passed. Root separately reviewed the source. These
are source reviews, not runtime equivalence or numerical results. batch_kernels
independently passed the native prepack/code/scale/mapping metric review at
5da37fcd; the agreed scalar reconstruction label was added in60a26576.

Reconstruction statistics use scalar F32 code-times-scale products, then F64
error accumulation. In particular this is a declared scalar reconstruction
metric; the actual projection scales local dot sums and can round differently.
EE must exactly reproduce production before interpreting factors; mixed E4
arms do not reuse the integer-only F64 bound. Interaction statistics use F64
II-IE-EI+EE directly. None of these values are model-quality or speed claims.

The phase timing follow-up uses four restored executions per arm/order, two
samples in each ABBA block. Seven completed-await spans and their enclosing
total are reported with clock gaps; no intermediate download or validation is
inside the timed path. Readbacks and scalar statistics remain outside.

No calibration, balancing, clipping, model integration, official submission,
Arena or final/main change was performed.
