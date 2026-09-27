# E136 — native E4 weight-resident QKV

Measured complete QKV prototype: +11.9973% slower, not adopted. Source and performance control are E134 `4f8fe7f`; actual branch start is `9de90ea`. O and weighted native-down FFN must remain byte-identical images.

The full graph, removed/added work, placements, numerical boundaries and separate adoption/tuning/research decisions are frozen in `native-v1/recipe.json` before compilation. Current source uses four residual factors `[1,16,2^22,2^27]`; these are derived CPU bounds, not a factor sweep. Common activation scale is applied after the F32 term sum, and channel scale before the first projection BF16 cast. No free host preprocessing or changed public tensor input is used.

First gate: exact complete-kernel compilation plus emitted source ranges, plain E4 Sub-to-TRF, normalized term coefficients, canonical row-lane outputs and unchanged downstream source/image checks. Capacity checks alone do not establish TRF sequencer support. Then one original-host E134→candidate→E134 allocation, with all conversion/transfer costs included. Positive results enter bounded repeat/extra-input/boundary validation even below5%; root owns adoption.

`boundary-suite/` is reserved for root's full public-image oracle. No historical native-E4 validation is inherited.

Frozen result: the first exact full kernel compiled without source repair. `graph-verification-e136-native-v1.json` binds the independent exhaustive Q8/KV4 weight-DMA row mapping and native arithmetic order review, plus emitted TRF/VRF capacity/lifetime checks. The original standard host contains the exact compiled QKV image; O and FFN images are unchanged E134.

Arena110367 passed all45 comparisons. Whole QKV medians were E134102,925 → E136116,321 → E134104,796 cycles: **+11.9973%** against the103,860.5 bracketing median. The compiler estimate was58,200 vs51,293; these are separate measurements. No internal NPU trace was collected, so neither TRF loading nor quantization/broadcast is established as the individual hardware cause. All of those added tasks are included in the measured whole kernel.

Root's independent `boundary-suite/result-110365.json` ran the same public QKV image with two original cancellation diagnostics. All16,384 Q/K/V outputs met the unchanged tolerance; all8,380,416 untouched-cache elements passed and all38 input hashes matched regenerated CPU truth. This includes the old tiny-value failure mechanism, but is not an arbitrary-input or end-to-end numerical proof. Some Q/K values are within tolerance rather than bitwise equal; both V cases were exact. These diagnostic inputs are outside the official generator support and are not performance inputs.

`decision.json` records three separate decisions: do not adopt this slower prototype, stop this weight-resident parameter series, and retain the validated finite four-term native arithmetic as evidence for a separately selected activation-resident weight-stream design. No repeats, extra6, live changes or official submission followed the measured loss. Current public behavior remains E134. The original recipe's `not_run` fields are immutable pre-build plans, superseded by the measured decision.

Recheck without compiling or submitting:

```sh
python3 moa-dev/experiments/candidates/qkv-native-weight-resident/verify_graph.py e136-native-v1 --check
python3 moa-dev/experiments/candidates/qkv-native-weight-resident/screen.py check e136-native-screen
python3 moa-dev/experiments/candidates/qkv-native-weight-resident/results.py e136-native-screen 110367 --check
```

The three normalized stage reports are `docs/experiments/e136-qkv-native-weight-resident{,-e134-before,-e134-after}.json`. Diagnostic source, builder, original input generator/truth, package hashes, upload archive and raw log remain in `boundary-suite/` and its referenced ignored artifact directory.
