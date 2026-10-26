Complete fixed-H64 FFN numerical gate: PASS ONLY; no promotion.

Actual323462 completed in32.994208248s. Strict replay checks all91 records, all32 drift rows, four predeclared per-order/step decisions and all16 original-grouped anchors fromactual323322. Both row orders pass; aggregate last-bit differences are allowed only for accumulation order.

| Step | State | Original grouped L2 vsq020 | Rotated L2 vsq020 | Rotated/original |
| --- | --- | ---: | ---: | ---: |
| 0 | joined | 0.396333742 | 0.173826230 | 0.438585 |
| 0 | down | 0.482091560 | 0.219266780 | 0.454824 |
| 0 | residual | 0.423880819 | 0.192335810 | 0.453750 |
| 1 | joined | 0.318219852 | 0.162658496 | 0.511151 |
| 1 | down | 0.452227876 | 0.229145418 | 0.506703 |
| 1 | residual | 0.374812605 | 0.186208608 | 0.496805 |

Fixed gate remains residualL2<=0.75×original and joined/DOWN no worse, with positive finite reference denominators. Hidden drift also improves, but is reported separately from the gate. Candidate DOWN consumes its own GeGLU hidden after CPU rotation; it does not reuse production hidden. Bothsteps share original q020 incoming feedback, so this is not an autonomous approximate multilayer history.

Correctness:471,859,200 intended BF16 values;12,288 i64/F64 samples;334,561,280 permutation owner bytes;31,457,280 same-input bytes;2,127,323,280 direct immutable bytes;928,661,504 replay/unchanged-owner bytes. Source-derived quantizer checks additionally cover39,321,600 packed bytes and81,920 scale plus81,920 offset values. All176,947,200 rotated weight values have independent row-transform checks before184,320 group encodings. Counts represent repeated comparisons, not distinct allocation size.

All432 source/config inputs match build and submission; original429 remain unchanged. Captured/submitted ELF SHA96ce0228ce20f0afda25206d2bc4ec1e181eb45d9a77e48da759be4d430ce34d (5,686,792B). All19 images equal actual323213, actual323322 and fresh compiler output, including the q020 control. Log SHA dcacd90ecb1ccec88cb864987b524ac3f4f93d05d4a4d5e08aaaee7f2c059045. Root and round2 independently verified raw metrics/counts/provenance.

Replay verifies all13 field values, but residual, rotated_prefix and rotated_hidden are reallocated; same-allocation reuse is not claimed for those3. Activation metrics are measured in each arm’s own BF16 coordinates, and the two hidden inputs differ. CPU D2H/rotation/H2D and diagnostic setup are included in the execution; no serving timing was measured. Layer0 synthetic inputs plus provided weights do not establish task accuracy or full-model quality. Official q020 is unchanged.

Replay: `python3 experiments/int4-hadamard64-complete-diagnostic/analyze_actual.py --verify-local-artifacts`. Raw files are under `actual-323462/`; full metrics and counts are in `actual-result.json`.
