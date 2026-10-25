# Joint model gate on current exact output projections

Base is root `91e5dd7`, including the activated, independently validated sliding
and full weight-stationary output projections. This export adds the exact
projection-owned FFN source from `d7a3480`, with the startup-copy proof in
`r1028/320092` and all11 isolated model/cache gates in `r1031`–`r1041`.
Those isolated gates compare 68,943,872 vocabulary values and 27,179,089,920
raw KV bytes exactly; reported available HBM remains at least 3,520,347,904B.

The merge preserves `weight_stationary_output=true` in every production and
packed-tail constructor and retains its propagation and both output branches.
`projection_owned_ffn` remains false in every constructor. New joined-weight
storage remains explicit and startup-only. No Worker or serving default changes
are included. Kernel and startup-join bytes match the independently tested
source; only the model diagnostic adds an explicit stationary-output true
selection in both comparison arms.

`compare_projection_ffn CASE INTERLEAVE` now compares current exact stationary O
plus original composed FFN against the same O plus the new projection-owned
FFN. All eleven owner-route, history, reset, all-vocabulary/cache/sentinel and
post-decode checks remain unchanged. Case0 includes active B16/R32 interleaves.
The probe measures actual allocator availability after joining and after both
model routes, because the additional output kernels' program allocations were
not present in the earlier memory reservation run.

Host checks and all111 existing library tests are required after this merge.
Root coordinates the joint all11 hardware queue. Activation is a separate step
only after those gates pass; no source-level or isolated-gain inference replaces
the actual composed-model check.
