# BF16 block partials followed by scale contraction

Independently implemented from the mathematical observation that sum_b scale[b] * dot16[b] is a second dot product. The technique was identified during the attributed [public implementation analysis](../../../../docs/research/public-solutions-20260921.md); no external source was copied.

Up/gate full32-row FP32 block partials are rounded to BF16 and packed four at a time. Local FP8 scales are converted through the supported BF16 LUT and placed in TRF with time axes row,block. A second BF16 contraction reduces only the240-block H axis, preserving each row, before the existing final BF16 cast. Down, GeGLU and epilogue remain the E16 implementations. This adds intermediate BF16 rounding and is not numerically equivalent to the FP32 blockscale path.

Both exact FFN compiles passed in the same scratch with identical ops/residual: baseline 167296, candidate 207851 cycles. The candidate regresses and is not adopted or NPU-tested. Smaller local dot work did not offset conversion/lifetime/scheduling costs. The long schedule interval before activation replication and serialized matrix staging is observed; a unique causal allocation mechanism was not proven.

Source, patches, parent ops/residual snapshots and hashes are preserved. Raw schedules are ignored under moa-dev/experiments/artifacts/ffn-scale-contraction. Independent target: /tmp/moa-ffn-scale-contraction-target. The scratch source was restored to the baseline only for the fair comparison; mlp.rs in this folder remains the candidate.
