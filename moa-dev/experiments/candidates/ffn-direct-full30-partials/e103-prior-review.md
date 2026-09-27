# E141 alternative: two full-row direct native streams, then F32 block scaling

This is an expressible, bounded composition worth selecting **only if the frozen E140/E141 profile shows the added lookup setup/readiness work on the limiting path**. It exchanges eight lookup setups for F32 partial materialization and two additional Main tasks; it does not establish a gain from bytes or static durations. No compile or NPU was run for this note.

## Exact prior and unresolved failure

The real E103 full/probe source is `moa-dev/experiments/candidates/c07-native-scale-dpe/evidence/e19939d8ad38c19ab811fdd3cd0ada9e16c9ae0fb934ff896ecee313287f3967.rs` (28,253B), bound by `build-c07-up-e5-reload-v3-repro.json` and `build-projection-probe-reload-v1.json`. The generated `up-e5-f32-reload/mlp.rs` is not the authority for the historical build. The frozen full build compiled and standard performance screens ran; its partial implementation is not a demonstrated API/lowering impossibility.

`projection-probe-reload-v1-results.json` and frozen log `evidence/7e3f9bf31ef125ea66016ddb54473129fc65dd8872c9c9a2ff0af460511d2477.log` preserve actual bad outputs: case0 up and gate each15,360/15,360 mismatches; case1 up5,441 and gate5,440 mismatches; down cases0/1 had0 mismatches; case2 then timed out withcode4. These are genuine failures of that image. The old direct-family correctness cannot be inherited from its standard timing.

That exact source also contains the three unsafe padded term-view DMAs at lines549–554. No final graph/vISA for this frozen full/probe image was found; only the full schedule and exact binary remain in ignored artifacts. Thus neither the old copy victim nor the sole cause of its wrong outputs is established. Do not assert that E140's finding explains E103 retrospectively. The note above the old helper incorrectly says BF16 block rounding: the actual partial tensor and commit are F32 and contain no BF16 cast.

## Minimal complete source composition

Start from frozen E141 `direct-six-row-copy-v1/mlp.rs`, preserving its exact480B input copies, independent compact FP4/scale ingress, per-projection TRF/scale reload, M03 same-cluster F32 row gather/GeGLU, E134 weighted down and public epilogue. Change **only `project_rows_with_scale_contraction`** to the following source path already expressed by the E103 helper (lines30–77):

1. One Main stream per projection fetches full `[FfnRow30,H3840]` packed FP4 and performs `fetch_table_lookup::<f8e5m2>()` once. Keep E141 `contract_outer<Row,H/64,FfnTerm>` → `contract_packet<H/16%4>` → `contract_time<Row,H/64>` → `contract_lane<...,H/16%4#8>` unchanged. It produces each block16 F32 dot after the same native three-term sum.
2. Apply input `x_scale` in F32 before storing `DmTensor<f32, ..., m![FfnRow,H/16]>`. Commit four F32 values per16B packet. This is28,800B/slice, `[row,block]` order with240 blocks per row,7200 F32 values. **No BF16/FP8 cast and no new weight/joint copy.**
3. Keep five six-row scale passes per projection. Each Sub reads FP8 scale `[6,240]` and loads5760B F32 VRF. Main reads matching partial rows via `fetch/collect<[Row6,H/16/8],[H/16%8]>`, narrows to four lanes, applies block scale in F32, reduces H, and commits six F32 rows as three8B pairs into the original output view. The old helper uses standalone `FpMulAlu::Mul0` here (E141 fused form used Mul1); preserve the semantic order and validate the actual emitted path rather than assuming physical-unit equivalence.
4. Leave all consumers after the helper unchanged. Do not import E103's old cross-cluster60-row gather, old down implementation, or old unsafe input copies.

The contraction/TRF geometry and six-row scale/output shapes have compiled independently before. Their composition with the current whole source needs a new exact graph and allocation check. No generic API claim substitutes for that check.

## Complete cost difference against E141

| Item | E141 | Proposed full30 partial path |
|---|---:|---:|
| Up/gate native+lookup Main streams |10|2|
| Up/gate lookup preparations |10|2|
| Extra Main blockscale/reduce streams |0|10|
| Total up/gate Main streams |10|12|
| Scale Sub/VRF tiles |10 ×5760B/slice|unchanged|
| Up/gate F32 block partial writes |0|29,491,200B =28.125MiB|
| Corresponding partial reads |0|29,491,200B =28.125MiB|
| FP4 HBM and scale ingress |unchanged|unchanged|
| Full decoded E5 weight DM |absent|absent|
| Up input quantization/broadcast/TRF and three bounded copies |retained|retained|

The partial is14.0625MiB per projection per direction. Both added directions together are56.25MiB of logical traffic, not measured bus traffic. Producer completion before scale passes is another dependency; potential overlap is compiler/runtime dependent. Table setup reduction alone does not prove that this trade is favorable.

## Smallest meaningful gates and numerical guard

Before NPU: bind the exact source/46-file recipe to current QKV/O and explicit contemporary control; prove only this helper changes from E141. Enumerate all six public weight/scale descriptors, full30 packed reads, 1800×16B partial commits/slice, five disjoint6×240 partial/scale windows, all8B output pairs, and original bounded480B prefix copies/bitmap. Verify two direct native/table producers, actual TRF/x-scale VRF consumers, ten5760B scale VRFs and whole allocation lifetimes. Partial DM store/reload must preserve F32 and `(native_dot * x_scale) * block_scale` before H reduction.

Run one whole comparison and the unchanged E141 eight-case production up/gate oracle on the new actual helper. That oracle includes the original dense/row/block-code cases which exposed E103, cancellation/tiny cases and H15/16,63/64,3839 plus row6/30/cluster boundaries. Its 245,760 exact outputs are a finite regression gate, not an arbitrary-BF16 theorem. Retain current down numerical/full-FFN regression checks as source composition requires. If wrong, use the actual graph and first mismatching row/block before a concrete repair; do not sweep row sizes. A positive whole result, even below5%, calls for bounded validation; no demonstrated gain means no adoption, and further tuning needs a separately measured mechanism.
