# Canonical R64 host execution gate

Control arithmetic: the same native R64 QKV/FP8 FFN/attention/norm/LM kernels, with all selected requests in one call. Candidate replays identical request histories in different physical slots/pages and groups of1/7/16/17/31/64. The probe resets each logical request at position0 between passes, stores the first pass's logits on host, and compares every emitted logit exactly. It tests ownership and grouping, not equivalence to the original scalar numerical path or official accuracy.

The host keeps one Rc64 sliding pool and one Fc256 full pool per layer. R64 scratch is separate from the resumable B16/P128 prompt checkpoint. Every direct64 KV commit has64 live rows; smaller calls extract B16 K/V tiles and scatter only their valid prefix. Prompt commits retain chronological causal ordering. R64 FFN references MlpWeights.native_fp8, whose temporary row-major prepack is dropped after one blocked copy is stored.

Automatic canonical dispatch selects R64 for17..64 independent singleton inputs, existing B16 for2..16, and scalar for1. Explicit forward_sequences_r64 forces the same numerical R64 route for partial-batch diagnostic comparisons. Default Worker::new remains the legacy control; Worker::new_canonical64 allocates canonical storage and enables FIFO admission with epoch-owned complete prompt+generation page reservations. No partial request waits for a missing page after execution begins.

Evidence before this gate: canonical B16/P128 whole-model narrow/wide boundary probes317951/317952 passed exact, all R64 component kernels passed independent NPU gates,16 host tests passed. Full R64 model performance and official accuracy are still unmeasured.
