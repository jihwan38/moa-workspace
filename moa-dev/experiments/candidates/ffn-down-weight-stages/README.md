# E150: staged weighted-down supply without partial assembly

Actual start `024a0fd1d711d14c5c6355deb76f377828022d47`, branch `opt/ffn-down-weight-stages`. Explicit source E147fb45202/E142 FFN; independent control baseline/e147.json. E148/E149 O candidates are not inherited.

Current-image trace141134 observed up result gather overlapping long down-weight ingress and finishing just after it. GeGLU-end→down-TRF-end median36829cycles, native atTRF-end+1 in all6 observations. The selected repair splits only down60rows into30+30 compact originalFP4 HBM allocations, two unchanged weightedE5 lookup/native streams and direct commits into disjoint halves of the existing F32full60partial allocation. Original canonical quantizer, activation TRF, coefficients[s,s/16,s/256], fullscales and five12row consumers remain. No partial assembly copy or new stage sum.

The intended benefit is earlier small up-gather/GeGLU progress between weight transfers, not earlier weight readiness alone. Rust source order gives no priority guarantee. Include the extra DMA/native/lookup and full partial lifetime; actual source bounds/half offsets/128B phase/cross-half24..35 consumer and8KiB VRF limits must pass before whole NPU. Same weight29,491,200B and partial14,745,600B per direction; logical byte saving0. If both DMAs stay consecutive, expected overlap benefit is weak, not an NPU verdict. No padding or row-count sweep.

Adoption pending gate/whole; any net positive receives bounded repeats/extra6/unchanged rawdown and fullFFN regression cases. Further tuning and major research are separate decisions in staged-v1/recipe.json. Selection-time source preparation/review is preserved in selection-review.md; its prior unselected/zero-compile statements describe preparation, not future results.
