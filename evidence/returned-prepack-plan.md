Parent startup profile318023 measured36.402s in load_weights before Workspace.
At controlc3543f7, NativeFp8Weights::from_packed allocates six large FP8 buffers
per layer by HostTensorzero/PDMA: three temporary row-major weights and three
retained blocked weights. All six are overwritten completely by prepack/reblock.
This is16,986,931,200 redundant zero bytes across48 layers, plus small scales.

Candidate ops_prepack_alloc preserves every original arithmetic/rounding/layout
operation, but returns newly allocated q/scales and blocked weights from the
kernels. Up prepack writes all rows; down's15 passes cover all30 rows per shard.
Reblock returns its complete DM write to HBM. No uninitialized region is read.
Original control APIs and loader stay untouched until actualNPU proof.

Gate: two distinct packed/scaled seeds including signedzero rows; compare every
row-major FP8 byte, every F32 dequant bit, and every blocked FP8 byte against the
unchanged original kernels. Keep the prior returned allocation live across the
next seed and reread it to verify ownership. Time original Hostzero+prepack+
reblock versus returned variants; then integrate the loader and screen model
logits/public quality with unchanged blocked representation.
