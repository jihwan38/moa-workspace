# R32 MSE F4 actual gate: PASS

Root jobs321351/321352 (r1258/r1259), source6f6c0de5, completed16.815/16.359s.
Frozen ELF SHA35a203a04eb4c4a0de3e5eceecbaeeda6bde615d4d1f189ae666d7ebfb3de1a5.
The reproducible collector preserves complete logs and per-case timings, stages,
F64 results, immutable assertions and engine spans in actual-r1258-r1259.json.

All49,156,096 retained bytes equal the intended dense fixed-code reference with
identical fitted group960 partial scales. All1,773,035,632 immutable bytes and
11,796,480 separately checked projected-tail bytes pass. Supplied layers0/5/6/47
cover meaningful32/17/31/32 rows, true within-live permutation, zero padded rows,
two matched feedback steps, distinct NaN R64 input tails and finite output-tail
sentinels. The unchanged full64 norm kernels still process inactive rows; the
R32 core preserves projected rows32..63 exactly, while final inactive residual
and norm tails match the original current path separately.

All1,536 independent F64 local partial samples pass the frozen B16 tolerance
(1e-5 times absolute product sum plus1e-5, max observed absolute0.0078125).
All256 sampled final BF16 outputs equal the F64-derived boundary reference.
No tolerance or speed criterion changed after execution.

Complete three-launch ABBA timing (pre-RMS → four-stage core → post-RMS,
residual and gate),8 completed calls per arm, both orders:

| Layers | Current mean us | F4 mean us | Lower time |
|---|---:|---:|---:|
|0/5|382.6461|325.6399|14.8979%|
|6/47|380.5292|324.0611|14.8394%|
|All four|381.5876|324.8505|14.8687%|

This passes the predeclared >=10% complete FFN gate. Static weight fit/upload,
resets, readbacks and oracles are excluded equally. Lane0 traces contain three
Tasks per chain: summed Task614277–627478→507970–516247 raw cycles; DMA union
542677–556001→406127–430173, TU sum108132–110520→131352–133284. Reduced movement
wins despite extra local scaling/LUT work. These are profiler cycles with
possible engine overlap, not a clock-derived wall-time estimate.

Original-current-E4 drift remains approximate: projected relativeL2 is
0.10622–0.20969; residual0.09330–0.19268; hidden0.14631–0.24024 across both
feedbacks/permutations. Norm/qinput/xscale remain exactly equal to the current
E4 baseline. All detailed code/scale drift stays in JSON. This validates the
intended new weight computation and its speed, not model/task quality or C17
SLO/top3. No host/default integration occurred. Static F4/group metadata ABI
is shared with B16; no R32-specific weight copy is required by these kernels.
