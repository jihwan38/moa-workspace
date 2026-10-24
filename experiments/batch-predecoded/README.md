R2-BP01 exact predecoded BF16 FFN
================================

`ops_batch_predecoded::batch_ffn` accepts the exact BF16 tensor produced by the
baseline's FP4 lookup and E4 block-scale multiplication. No global factor is
folded into those weights: dot rounding, global scaling, GELU, and the down
projection's eight L1920 partials remain unchanged. The original FP4 matrices
and block scales determine every predecoded value.

The device loads15 up/gate rows or30 down rows per pass; that tile remains in
DM while both B8 token chunks contract against it. This trades additional HBM
bytes and repeated TRF preparation for removal of repeated per-request weight
lookup and F32 block scaling. The initial B16 prototype still stores allB16
inputs in DM. Future larger batches must stage inputs in B8 pieces and tile
GeGLU's VRF use; simply changing B is insufficient.

`test_batch_predecoded` generates original FP4 and E4 matrices, predecodes the
same values on the host one118MB projection at a time, and checks all61440
outputs against the validated packed-weight row-major FFN. ABBA samples use the
same five-launch plus one-final-read scope for both variants. As of this commit,
compile and executable build passed; hardware equality and speed are pending.

Read-only budget audit from the team's evidence agent, checked against loader
shape assertions: full BF16 model approximately20.02059GiB, including1.875GiB
embedding,2.24121GiB attention weights,15.82031GiB FFN weights, and other scales,
norms, vision/audio weights. KV is0.4375GiB/request at8192 context;32 requests
would therefore need approximately34.02059GiB plus allocator and scratch space.
The original FFN FP4+E4 allocations occupy4.44946GiB and must be released during
actual integration. Host RAM is16GiB, so decoding the whole model into host RAM
at once is invalid. Hardware HBM capacity has not yet been verified by this
experiment. No loader or serving model uses the BF16 weights yet.

Root Arena317619 checked all61440 values bit-exact. Warm time improved only from
approximately1.34ms to1.21ms. That gain does not justify the additional model
storage for this experiment, so the serving loader is unchanged. The next
measured mechanism is wide packed-weight reuse, recorded under `batch-wide`.
