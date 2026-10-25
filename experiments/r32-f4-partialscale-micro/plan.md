# R32 MSE F4 complete-chain actual micro

Base98a6d7e5, current physical32 production control07718f07. Resource gate
already passed20.93% complete estimated saving and no DramStack. Disk at start
22GiB free; reuse only tail256-model target, package debug0/incremental off.

Freeze encoder8e7ce99e (MSE25 dyadic choices) and current candidate1a4a57a3.
Use actual current nativeE4 UP/GATE/DOWN from layers0/5/6/47, with retained row
and global scales. Intended reference is mechanically copied candidate with
F4 replaced by dense unscaled E4 codes and LUT removed; per960 scale stays
before each local F32 partial sum. Original currentE4 graph is separate.

Each complete chain is exactly the current three calls: original R64 pre-RMS,
original/candidate/intended four-stage R32 FFN, original R64 post-RMS/residual/
gate. No new composition/fencing change. Candidate first, independent poisoned
scratch, all eight retained outputs read before scratch reuse; all intended
bytes must match, and original norm/qinput/xscale must match. Original-E4 drift
is reported, never treated as intended-reference exactness.

Use32/17/31/32 meaningful rows, zeros in unused live32 positions, true per-case
permutation within meaningful rows, two matched candidate-feedback iterations.
R64 rows32..63 use distinct NaN input payloads and finite projected sentinels;
FFN core must preserve projected tail bytes exactly, and full original postnorm
computes the same inactive residual tail on all arms. The test does not falsely
claim the existing R64 norm kernels skip these rows.

Independent CPU decoders calculate sampled F64 H960/L960 local partials and
final DOWN BF16/global boundaries using the exact frozen B16 tolerances. Probe
exports partials via separate diagnostic kernels, excluding them from timing.
Check all weight/code/scale/norm-weight immutability and raw row permutations.

Predeclared speed gate: >=10% lower mean COMPLETE three-call FFN wall time
under completed ABBA8 per arm, both row orders. Reset/upload/encoding/readbacks
and CPU oracles are outside timers equally; all normal device calls remain.
Report gate failure after all numerical/timing output. Profiles are untimed.
No host/default integration, Arena dispatch or model-quality claim.
