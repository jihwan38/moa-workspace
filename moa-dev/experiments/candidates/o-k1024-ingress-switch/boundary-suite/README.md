**Current status: both packages built and host-verified; NPU validation not run after the first whole regression.** See [result summary](result-summary.md) and [bound status](status.json). The preparation text below is historical.

# E149 original output boundary suite

Prepared from original E137 raw9 and E47 fullcase7 wrappers, fixtures, references and tolerances without source edits. Exact selected E149 production projection must be bound by the builder after its emitted graph gate; no standalone reimplementation. Raw checks34560 exact BF16 outputs (signed zero equivalent), whole3840 at public atol.05/rtol.01. Separate raw image is not a performance image. Whole host must contain the exact production image. Builds/NPU results are not inherited from E148 and do not exist yet.
