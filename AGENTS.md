# Ownership and scope

This isolated tree is owned by `/root/e2e_evidence` for the exact-rational BF16
A4 quantizer foundation. Base: `507b9555fe600bbf447e58279d418e2d29868015`.

Only the new quantizer, independent probe, and evidence may change. The model,
official candidate, Cargo dependencies, and existing device kernels stay intact.
Use `/round2/prefill-head-attn-commit/target` exclusively and a new empty
`FURIOSA_OPT_OUT_DIR`. Root alone dispatches Arena. Initial authorization ends at
source/type checks and exact resource compilation; no device ELF build or model
activation before the resource report.
