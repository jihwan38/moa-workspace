# Exact H480-local ingress quantization

Baseline fcf2d7f is the direct blocked post-attention/residual/pre-FFN fused consumer. Actual Arena318838 traces show complete blocked task532155 raw units, 36DMAs with long tail transfers, but summed TU execution74021. This does not support per-element division as the leading cost. Existing pre-normalized BF16 H480 shards are redistributed to whole H3840 rows only to compute amax and quantization.

Candidate keeps the existing H480 RMS sum tree, both BF16 normalization boundaries and residual storage. Compute each row's amax locally over H480, then Max across its eight H shards; amax is order-independent for the same finite inputs. Broadcast only the scalar scale, quantize each original shard with the existing power-of-two scale/DivF/cast, and write it into the unchanged P512H960 output ABI. Removing the full BF16 redistribution may add fragmented H480 writes and scalar broadcast, so actual inclusive timings decide.

Required gates: candidate-first poison output; all512 FP8 rows, FP32 scales and residual bits versus current direct blocked consumer; CPU F64 residual; zero/tiny/signed/random inputs and quarter permutations. No serving integration before matched whole-model output gate. No assumed speedup or claim that this removes all traced DMA.
