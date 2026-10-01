# E152: gate weight supply and initial activation readiness

Actual start `ffce3481842b8f71a635a2dc2ba773518856ec05`, branch `opt/ffn-gate-weight-stages`. Source E150/ad1ecb3 and independently declared performance control baseline/e150.json. This runs in parallel with unadopted E151; E151 source is absent and joint gains cannot be added.

Current E150 trace141326 shows the ready pre-RMS activation gather overlapping the initial large gate-weight ingress and ending afterward. Split gate30 into compact15+15 original FP4 loads/native streams, directly writing offsets0/14400 into the original28800B full30partial. Original five6row consumers, up helper, all quantization/GeGLU/E150down arithmetic remain. Cross-half rows12..17, actual native fetch/commit, sharedTRF/VRF lifetimes require exact graph verification.

This adds one DMA/native/lookup and removes no logical weight or partial bytes. There is no packed or decoded assembly. Earlier activation completion is a hypothesis, not a source-order guarantee. Preserve M07/M04/E56/E142 contrary evidence in selection-review.md and staged-v1/recipe.json. One complete prototype, no stage-count sweep; early contemporary whole comparison and bounded adoption of any reproduced net gain.

## Result: not adopted

Arena141424 passes45 original comparisons but whole FFN is327706 versus E150297644/297885 (bracket297764.5): **+10.05543% regression**. Actual static ordering had placed initial input gather between the gate DMAs, but that observation did not establish a complete-path gain. No exclusive physical cause is assigned. Stop further stage/row tuning.

Original raw-upgate eight-case245760-output and fullFFN3840-output packages were built and their reference/source/image bindings checked, but **neither ran on NPU**. Three standard inputs are the finite observed correctness scope. Prepared boundary evidence remains preserved. E151 is a separate independently validated candidate, not part of this source. [Decision](decision.json).
