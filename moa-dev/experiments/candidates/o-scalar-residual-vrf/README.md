# E148: O scalar and cumulative F32 supply directly to VRF

Actual start `f9451626d508197901deee9bafbb03aa8edf2d34`, branch `opt/o-scalar-residual-vrf`. Source E147 `fb45202` and independent performance control `baseline/e147.json`.

The frozen direct-v1 combines one-Main finite-BF16 global scale, Sub ring16 scalar/coefficient supply directly to VRF, and two unchanged F32 cumulative reconstruction streams ending directly in VRF. Four E4 terms, Mul/Add order, native Term/K16 reduction, two60-row weight stages, canonical gather, common/channel scale and all BF16/RMS/residual boundaries remain. QKV prefix and FFN are unchanged.

Removed: two Main scale boundaries, replicated scalar/coefficient DM, cumulative F32 logical DM writes/reads1048576B each, two Sub reload tasks. Added: pointwise Mul131072 values, ring16 bitmap/control setup and Main/Sub/VE coupling during direct VRF writes. Two K256 residuals use2048B/slice before all other allocations; actual lifetime checks are mandatory. No speedup is inferred from these byte counts.

Adoption pending exact graph gate and whole performance. Any reproduced positive qualifies for bounded adoption checks. No row/term sweep is selected. A regression must be analyzed as a complete cost trade; it does not universally reject direct VRF. Source/API mistakes may receive concrete supported repairs. This is a local coupled replacement, not a new full architecture. No NPU result exists at preparation.
