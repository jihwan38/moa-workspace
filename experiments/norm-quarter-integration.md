# Quarter-batched grouped sliding normalization

Isolatedsourcec86ea4c, microgate318898: exact all512 FP8 rows/F32scales/four residual arrays for three seeded quarter permutations, CPU F64 passes. Current independent-quarters local-quant chain262–283us (one303us outlier), candidate220–252us. No serving speed claim. Independent e2e_evidence source audit verified quarter/row factor relabel, unchanged H480 sum tree, and7680-byte residual VRF halves.

Only the existing blocked sliding and shard-quant path selects this helper. Full attention and all original-output/norm paths retain their current code. New PreparedGroup quarter_norm flag controls one prepared operation; original_quarter_norm diagnostics select the passed independent-quarter chain before execution begins, with no persistent mode or new HBM ownership. All four residuals are stored before yield and output ABI is unchanged.

Before measured serving:67hosttests, same-group all53logit checkpoints with R64 interleaves and postdecode, and independent four-owner gate. Keep c056 source/binary and all public evidence as the control.
