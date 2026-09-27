# E137 adoption update

Root adopted frozen native-v3 as `baseline-e137` / `334a7059a82252a310436c6d7e02f8ba577e7f51`.
[Adoption decision](adoption-decision.json), [source audit](source-build-audit.json),
[baseline recipe](../../../../baseline/e137.json). Three independent paired whole-O
reductions are14.1496%,12.2978%,12.2654%; median12.2978%. Standard9 inputs,
raw9 cases/34560 outputs and original full-O case7/3840 outputs pass. QKV/FFN
images preserve E134. Stop parameter tuning. Official grading is recorded separately.
The following snapshots preserve their status at creation; they are not later adoption decisions.

# E137 — activation-resident native O

Frozen `native-v3` reduces complete O cycles in all three contemporary E134 comparisons: **paired median −12.297751%**, aggregate medians **42,957 vs 49,122** (−12.550385%). All required finite correctness and source/package checks passed. `decision-v3.json` recommends root adoption review; this branch changes neither live kernels nor the baseline, and makes no official ranking claim.

| Allocation | E134 before | E137 v3 | E134 after | Paired change |
|---|---:|---:|---:|---:|
| 110558 | 49,585 | 42,526 | 49,485 | −14.149591% |
| 110573 | 49,272 | 42,957 | 48,689 | −12.297751% |
| 110584 | 49,126 | 43,097 | 49,118 | −12.265380% |

Each allocation runs the original three-input benchmark before/candidate/after. All three candidate and outer batch payloads are byte-identical; 135 comparisons passed (45 candidate comparisons). Extra6 is correctness-only, not performance evidence. QKV/FFN images are identical to E134, so their incidental timing changes are not optimization gains.

Source and control are explicitly E134 `4f8fe7f`; actual branch start is `5c94b51`. The only changed compile file is `src/device/sliding/projection.rs`. QKV functions and the residual helper remain byte-identical, all 45 other compile inputs are unchanged, and public interfaces/tolerances are preserved.

## Complete dataflow

`sliding_attention_output` calls the changed `project_output`. BF16 input is replicated into native consumer layout `(cluster c, rowgroup g, K shard q, element u)`, where destination slice `16g+q` reads `x[256q+u]`. Each of 32 `(c,g)` groups receives the same 4096 values. Per-group K16 max and power-of-two normalization produce the same common scale. Four cumulative F32 residual passes encode E4 terms with factors `[1,16,2^22,2^27]`; the packed `[Term4,K256]` tensor is already consumer-local and transfers directly into TRF.

Original E4 weights stream from the two C09 row60 DM stages. Native Lane4 owns the four activation terms. The output packet receives normalized coefficients, packet ReduceAccTree sums Term, and K16 inter-slice F32 reduction finishes each row. Two-row8B commits and canonical F32 gather retain the existing row contract. The existing final channel pass restores common activation scale, applies channel weight scale, then casts to BF16 before unchanged RMS/residual/output code. Coefficients are genuinely packed from four32B scalar slots into compact16B using two8B pair commits, replicated and loaded into32B VRF; no padded reinterpretation is used.

Independent emitted-descriptor review verifies all input coordinates, unchanged15MiB weight coverage, per-group K16 max/scale, scalar valid masks, canonical eight H480 scale shards, native Term→K16 order and register lifetimes. `native-v3/recipe.json` and `consumer-local-quant-review.md` preserve the full layout and cost contract.

## Why v3 differs from the slower prototype

`native-v1` preserves complete Time4/Lane1 source (exact compile33,950 static cycles, no NPU). `native-v2` uses Lane4 and the supported packet term reduction (33,532 static cycles). Its first whole job110453 passed45 comparisons but regressed **+14.0551%**. Separate v2 oracle110458 and public whole-O case7 job110465 passed. The original `decision.json`, source and evidence remain unchanged.

One frozen v2 profile, job110511, observed W0 ready at15,051 while native0 waited until31,836.5. Packed-term DMA occupied23,815.5–31,242.5; activation TRF ended one cycle before native0, and W1 began36 cycles after that DMA. These same-cluster observations selected consumer-local quantization as a concrete repair. They do not prove a particular physical resource conflict, and their overlapping durations are not summed into predicted savings.

V3 removes the512KiB packed-term redistribution and replaces sparse ratio256 max with K16 max per rowgroup. It **adds** BF16 HBM input16→256KiB, logical max/four-quantizer BF16 DM reads80→1280KiB, and grows each of two F32 cumulative-sum allocations32→512KiB with its read/write work. Local512KiB term materialization, quantization, coefficient preparation and TRF loads remain in the measured kernel. V3 exact static estimate is25,112 vs E13427,432; the table above is actual whole-NPU evidence. No v3 internal timing attribution is claimed.

## Validation and provenance

- Nine distinct standard inputs: original three in each performance allocation, plus unchanged generator runs3..8 in job110583 (30 extra comparisons, exact same three device images).
- Actual production `project_output`/quantizer and channel-scale boundary: job110561, nine original BF16/F64 oracle cases, **34,560 exact BF16 outputs** (signed zeros equivalent), before RMS. Cases include K255/256/4095, row59/60/119, cluster boundaries, cancellation, scale-before-BF16, and a nonzero fourth residual term.
- Actual public O image, legacy E47 case7 with original reference: job110569, **3,840 outputs exactly0.609375**, passing unchanged tolerance. Fixture bytes are identical to the v2 test.
- `source-build-binding.json`, `compiled-source-manifest.json` and `measured-build-e137-adoption.json` bind all46 compile inputs, exact/standard images, original runner/fixture, build commands and package digests. `validation/final-verification.json` binds the performance and accuracy evidence for root adoption.

The four-term CPU audit has a finite normalized error bound; it is not a general BF16 exact encoding or arbitrary projected/post-RMS guarantee. Three-term relaxation remains rejected by the preserved CPU counterexample. These nine standard inputs, nine selected projection cases and one whole-O construction do not establish full-model correctness.

Recheck without compile or Arena:

```sh
python3 moa-dev/experiments/candidates/o-native-activation-resident/validation_verify.py --check
python3 moa-dev/experiments/candidates/o-native-activation-resident/profile/verify.py e137-cost-profile-v1 110511 --check
```

`validation/`, `profile/`, `numerical/`, all prior source variants, failed diagnostic-tool/host-only preparation records and raw logs are retained. Parameter tuning stops; root owns final live source/tag/baseline adoption and official submission.
