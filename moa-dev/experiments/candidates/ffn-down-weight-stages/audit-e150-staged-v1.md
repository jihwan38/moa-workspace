E150 exact emitted-graph gate passed. Reproduce with `python3 audit.py e150-staged-v1 --check`; `geometry.py --check` separately checks source and logical coordinates.

The46 compiled inputs differ from E147 only in `shared/mlp.rs`. Quantization, up/gate, the down shared TRF and96B weighted coefficient VRF, five12-row blockscale/reduction consumers, and finish/epilogue are byte-preserved. Down native arithmetic differs only in row extent60→30 and direct partial commit destination.

Actual down ingress operators O246/O249 retain960B source segments and7680B HBM row pitch. Their HBM indexed views start at0/30. Each half writes28,800B per slice, and their combined30,720 segments cover29,491,200B exactly. The128B phase of a225×128B half is recorded; it is not classified as a runtime hazard.

Native O258/O265 read the compact halves, preserve term-dependent F32 restoration before term reduction, and each commit900×16B per slice. Their destinations are disjoint `[0,14400)` and `[14400,28800)` ranges of one original full partial. The actual IndexWrite chain includes both producers before all five12-row consumers. The consumer covering rows24..35 crosses the stage boundary correctly, without an assembly copy. Total partial write/read stays14,745,600B each direction.

Both down native tasks share the same live activation TRF and coefficient VRF. Every emitted register consumer lies inside its allocation lifetime; all whole-FFN VRF allocations fit8KiB per slice without overlapping live addresses. The three repaired480B upstream term copies and live ring32 bitmap remain bounded and separate in this new allocation.

This is a source/layout/resource gate. It does not establish a whole-kernel gain, force the up gather to execute between the two weight transfers, or inherit NPU accuracy from the earlier single-native path. Additional lookup/native setup and mutable-partial joins remain part of the complete measured candidate.
