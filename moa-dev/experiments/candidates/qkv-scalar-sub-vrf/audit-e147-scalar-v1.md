# E147 scalar-v1 independent gate

`python3 audit.py e147-scalar-v1 --check` reproduces the frozen [record](audit-e147-scalar-v1.json). This is an emitted-code/source gate; NPU accuracy and measured performance are separate requirements.

The 46 compile inputs match E144 except `projection.rs`. Four-term quantization, coefficient constants/pair packing, all three full-H weight/native consumers, head/cache path, and O suffix retain their original semantics. The unchanged operation signatures include actual reader, Switch, contraction, scale order, output masks and descriptor details. Q/K/V HBM descriptors cover all 30 MiB once with 256-byte packets.

Actual Main O12 performs exponent mask → multiply by 2^-7 → H960 Max → clip at 2^-126 → K4 Max, with only q=0 valid in each group. The separate [CPU proof](numerical/scale-proof.json) exhausts all 65,280 finite BF16 bit patterns; monotonicity extends scalar equality to a finite H960/K4 maximum. It does not claim behavior for NaN/Inf or substitute for NPU validation.

Sub O14 and O70 both use the same emitted bitmap T34: all 256 destination slices select `4 * (slice // 4)` within their cluster. Scale valid fetch indices are 0..7 (32 bytes). Coefficients use indices 0..3 (16 valid bytes) and pad four terms to the 32-byte VRF packet. The coefficient sequencer, printed `fetch_size: 8`, and output padding equal E144's compact coefficient reader. This establishes the exposed valid-element contract, not a hidden bus-transaction width. The two VRF renames preserve addresses and packet shapes and name actual broadcast copies.

All three native consumers share one TRF and the renamed coefficient VRF for their full scheduled lifetimes. The four quantizer consumers fit the scale VRF lifetime. All scheduled live VRF regions fit 8 KiB per slice and do not overlap. The added shared bitmap load is 16 KiB; consumer scale/coefficient DM replication removes 24 KiB of writes and the subsequent 24 KiB of reads. Two former Sub loads become two Sub broadcasts, and the common scale-to-head DMA remains. The new per-input multiply runs over 491,520 values, so neither fewer operators nor removed bytes alone establish a speedup.
