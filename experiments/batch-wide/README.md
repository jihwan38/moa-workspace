P128 linear execution (R2-BW01)
==============================

`ops_batch_wide` uses `axes::P128` independently of B16 attention/decode storage.
`Cluster = P128/64` gives each physical cluster64 different token rows. Each
weight shard is shared by eight B8 contraction tiles within its cluster.
Raw Q/K projections load and expand the original E4 weights once. The FP4 FFN
expands a12-row up/gate or24-row down weight tile to BF16 once, then reuses it
across all64 local rows. Inputs are staged eight rows at a time. The FFN's
GeGLU uses16-row VRF tiles so its F32 tensor fits8KiB. All baseline rounding
boundaries and the down projection's eight L1920 F32 partials are preserved.

The explicit output transpose must fetch token blocks outside channel rows:
`Time=[P128/16%4, channel_rows]`, `Packet=P128%16`. Transpose then emits time
`[P128/16%4, channel_rows/4, P128%16]`, packet `channel_rows%4#16`; commit permutes
that into row-major DM. Putting the token block inside channel time caused the
preserved v1 transpose rejection. DM views expose `to_dm_view`, not `to_dm`;
v4's source/log preserve that failed initial attempt.

`test_wide_projection` checks524288 outputs against the independent dyadic
query oracle. `test_wide_ffn` checks491520 outputs against eight validated B16
FFN calls. Its ABBA timing groups consist of either eight B16 calls or one P128
call, repeated three times, then identical P128 and B16 result reads. All
weights and scales are shared between controls and candidate. These are
kernel-level tests, not full-model correctness or benchmark throughput claims.

At this commit, P128 query and FFN compile and both hardware probe executables
build. Hardware correctness and speed are pending root's coordinated Arena
runs. O wrappers and their independent validation follow separately.

Root Arena317636/317637 results: query checked524288 values exactly and measured
1.189ms warm. FFN checked491520 values exactly, but P128 took12.78–12.83ms versus
10.38–10.41ms for eight B16 calls. P128 FFN is therefore a correctness control,
not an adopted performance improvement. The primary remaining mechanism is
repeated activation replication/repartition for each weight-row tile. Component
probes and cached/sharded input placement are the next tests.

Both canonical O wrappers now compile. P16 add-partial tiles are staged into
contiguous DM before F32 VRF fetch because a direct padded DM-view fetch fails
SDK lowering (v9/v11 failures preserved). `test_wide_output` independently checks
both491520-element outputs against the baseline1024-column BF16 rounding tree.
Its compiled hardware probe is pending root.
