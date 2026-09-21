# Experiment ledger

All candidate measurements below use the unchanged official three-kernel, three-input harness.
An Arena result is a development measurement; official ranking requires `moa-submitter` evaluation.
Baseline checkpoint: workspace commit `2890176`, official kernel source `8559916`.

| Variant | Arena jobs | Accuracy | QKV cycles | O cycles | FFN cycles | Decision |
|---|---|---|---:|---:|---:|---|
| Original | 69783, 69913, 69931 | Pass | 244051 | 404222 | 4264297 | Baseline; medians across jobs |
| E5 direct-layout, two-row commits | 69992 | FFN fail | 244661 | 402884 | 1249479 | Rejected; speed is not a valid optimization result |
| E4 two-row only | 70000 | FFN fail | 244172 | 404030 | 3241077 | Rejected; isolate common two-row path |
| E2 FP8 lookup/contraction fusion | 70009 | Pass | 227840 | 334457 | 4265795 | Retain for subsequent combinations; single-job screen |
| E3 256-slice output + QKV fusion | 70014 | Pass | 228450 | 195210 | 4263461 | Retain; output 2.071× baseline, single-job screen |
| E3 direct HBM replication + E3 rows + E2 QKV | 70022 | Pass | 227338 | 143976 | 4267115 | Retain; single-job screen |
| E5 four-row direct-layout + preceding projections | 70029 | Pass | 227578 | 154254 | 1139366 | Retain; independent package repeats in progress |
| E5 same package repeated | 70029, 70038, 70046 | Pass | 227916 | 146508 | 1139366 | Medians across 3 jobs; FFN range 0.063%, O 6.600% |
| E1 QKV HBM relay + preceding optimizations | 70039 | Pass | 188573 | 145998 | 1140479 | Improves QKV; hierarchical copy candidate next |

The failed E4 and E5 experiments have identical reported FFN max/mean errors and tolerance pass rates
for all three inputs. Their common two-row contraction/reduction/commit/copy path is the current
suspect; the additional column distribution is not isolated as the failure cause. This is a
working diagnosis, not a proof. The four-row candidate restored the original output path while
retaining direct column placement and passed all official input comparisons. The precise mechanism
of the two-row failure has not been established. O cycles differed by about 7% between the two
last jobs even though its source was unchanged; the identical package repetitions also show a 6.600% range for O, versus 0.225%
for QKV and 0.063% for FFN. Small O improvements require repeated comparison.

`*-provenance.json` files record sources and artifact hashes. `*.json` measurement records retain
job metadata, the actual submitted package hashes, validity and, for passing runs, cycle aggregates.
Raw logs, binary snapshots and large schedules are under Git-ignored `moa-dev/logs` and
`moa-dev/experiments/artifacts`. Failed candidates and compile-time comparisons are preserved under
`moa-dev/experiments/candidates`; static schedule predictions do not establish correctness or speedup.

The `capture.py` helper snapshots current source and an existing binary independently. The root
experiment sequence builds successfully, captures immediately, and submits before the next edit.
The helper itself does not claim to prove that a binary was built from the current source.
