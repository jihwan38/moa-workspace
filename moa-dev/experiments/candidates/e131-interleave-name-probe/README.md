# E131 interleave axis-name diagnostic

Frozen `fp4-native-boundary/v7b` source is reproduced from its recorded nine-file build input. Only complete `Projection` tokens in `src/probe.rs` become `interleave`; `source.diff` records every edit. The source recipe remains historical v7b, while the adopted performance reference is E128.

The first gate is exact compilation of `probe::fused`. Success establishes only target lowering of this fixed probe. Two-source bounds/order still require emitted-descriptor review, and accuracy/performance require later authorized NPU tests. No host build, NPU run, SDK modification, prepacking, numerical change or layout sweep is performed.

Reproduction: `python3 build.py prepare` creates a new immutable snapshot; `python3 build.py compile` compiles it once. Existing evidence is never overwritten.

Exact compile **failed before contraction lowering**: `Ident must start with an uppercase ASCII letter` while evaluating `AxisName::NAME`. The installed `Ident` has a private field; its public constructor, `TryFrom`, and deserializers all enforce uppercase. The macro allowing lowercase Rust type names does not override this mapping-identifier contract.

The previous v7b `Projection_1` / `interleave_1` mismatch remains unresolved. This result closes the lowercase-name workaround through the current public API; it does not prove the full architecture impossible. No graph/image, host package or NPU result was produced. No second source variant was attempted. Source transformation, API excerpts/digests and decision limits are in `analysis.json`.
