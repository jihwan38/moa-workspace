# Ownership and scope

This isolated tree is owned by `/root/e2e_evidence` for the exact 20-bit
reciprocal-table BF16 A4 foundation. Base: `a384b467947b73f98fbc4378f0acb9891430dbe8`.

Only the new LUT quantizer, independent probe, and evidence may change. The model,
official candidate, Cargo dependencies, and existing device kernels stay intact.
Use `/round2/prefill-head-attn-commit/target` exclusively and a new empty
`FURIOSA_OPT_OUT_DIR`. Root alone dispatches Arena. Initial authorization ends at
source/type checks and exact resource compilation; no device ELF build or model
activation before the resource report.
