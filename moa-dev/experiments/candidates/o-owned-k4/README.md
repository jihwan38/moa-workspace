# E132: K4 O ownership with compact F32 consumers

The frozen implementation is **not adopted**. Complete O regressed in both contemporary allocations, while all original-input comparisons passed. A subsequent source-order repair emitted the exact same machine image and was closed without another NPU run. E128 remains the performance control; this record does not reject every K4 layout or input-distribution method.

Source/control is E128 `cc82b9ccff22795165faffd3ba9a68cc557da001`; actual branch start is `8ee561558faedaf68e42c080a515f28e3b82ddb5`. Only the candidate projection module changed. QKV/FFN images, fixture, official runner, O channel scale, first BF16 conversion, RMS and residual retain their original boundaries. The standard host embeds the exact compiled candidate O and byte-identical E128 QKV/FFN images.

## Complete path and earlier attempts

The O path changes from 16 K slices ×120 rows, stages60/60, to four K slices ×30 rows, stages16/14. Coordinates are `h=1920*c+30*g+r`, `k=1024*q+u`, `slice=4*g+q`, with `c<2,g<64,r<30,q<4,u<1024`. Each stage keeps all512 slices live.

BF16 input → K1024 replicated DM/TRF → separately loaded compact FP8 stage weights → Main lookup to BF16 and dot → F32 four-slice sum → two-F32/8B commits into contiguous30 rows → canonical F32 H480 gather → unchanged F32 channel scale → first BF16 → RMS/residual/store.

This reopens [the historical static-only K4 attempt](../output-column-sweep/README.md) with a specific consumer correction: its BF16 scalar-padding compaction cost2,185 static cycles and preceded channel scale. E132 writes compact F32 pairs and scales before BF16. It also retains [C09](../c09-output-first-npu/README.md)'s two independently loaded stages. [E53 K8](../output-bf16-eight-columns-e53/README.md) established only a static regression, and warned that a longer logical K stripe does not create a larger hardware burst. [M08](../m08-output-pair4-ingress/README.md)'s fragmented8B gather is avoided: E132 emits128 contiguous120B packets. This is a coupled reopening of an existing ownership family, not a new family name or K/stage sweep.

| Cost | E128 O | E132 v2 |
|---|---:|---:|
| FP8 weight HBM payload | 15MiB | 15MiB (8+7MiB stages) |
| Lookup/table preparations | 2 | 2 |
| Logical replicated activation DM | 256KiB | 1MiB |
| BF16 TRF per slice | 512B | 2,048B |
| Valid F32 gather | 15KiB,32×480B | 15KiB,128×120B |
| K reduction ratio | 16 | 4 |

Logical bytes are not measured HBM traffic. Actual weight descriptor packets remain256B. Less reduction and fewer local rows do not remove the full weight/dot workload. The added input replication and changed issue dependencies must be included in whole O timing.

## Frozen attempts and correctness scope

- `two-stage-v1`: exact compile failed at `cannot factorize common axis H` for the16-of30 HBM tile. This was a source-axis expression problem, not an observed hardware failure.
- `two-stage-v2`: independent `OutputRow=30` separates local row from H-group ownership before slicing. Consuming reshape restores canonical H before gather. Exact compile passed at28,595 static cycles versus27,432 for C09. [The descriptor gate](graph-verification-e132-k4-v2.json) checks both stages' disjoint full15MiB coverage, all512 slices, compact destination bounds, full BF16 input/TRF coverage, F32 K4 sum, commit offsets0/64B, all3,840 canonical gather rows, channel scales and scale-before-BF16. Static costs are not calibrated hardware predictions.
- `two-stage-v3`: after the trace identified a changed issue order, one diagnostic moved first-weight source construction before input TRF construction. [The source diff](v3-source-order-only.diff) changes no shape or arithmetic. The emitted order remained input DMA → input TRF → first-weight DMA. All49 instruction metadata entries except source descriptions were unchanged. The exact81,920B image was byte-identical to v2: `a5d96722468d74048fc2a90642634f15457712554a8a6b38a2a16012258f629e`. [The issue-order gate](v3-issue-order-gate.json) therefore stopped without host rebuild or NPU rerun. Source construction order is not a demonstrated scheduler control.

The two allocations passed45 and57 comparisons respectively, totaling102 original-host/diagnostic comparisons. These repeatedly use the original three inputs; they are not102 distinct inputs or expanded boundary validation. The prepared eight-case pre-RMS oracle under `validation/` was **not executed** after negative whole screens. Changed F32 reduction grouping does not inherit C09's earlier oracle result.

## Whole NPU measurements

| Arena job | E128 before | Candidate O | E128 after | Candidate vs bracket |
|---|---:|---:|---:|---:|
| 109654: original hosts | 48,583 | 55,052 | 49,216 | +12.5819% |
| 109683: original hosts within diagnostic allocation | 48,998 | 57,450 | 50,249 | +15.7718% |

Both actual uploaded archives, source manifests, embedded images and raw logs are bound to their records. The second allocation also ran control/candidate info/trace diagnostic hosts. Those diagnostic timings are kept separate from the original-host whole comparisons.

## Internal profile and remaining uncertainty

The [profile analysis](profile-analysis.json) validates all26 native markers per side against exact schedules and original image metadata. The diagnostic host changes only host tracing; image patches change profile-name bytes only, preserving commands/constants/slots/IO/marker positions. E128's unchanged O image and projection source are explicitly bound to the preserved C09 graph; its historical full crate is not substituted for E128.

Medians below summarize three runs ×two cluster clocks per side. Every subtraction within an observation uses its own Task clock. Marker intervals include waiting; overlapping medians must not be added into an attributed latency saving.

| Observed interval/boundary | E128 | E132 v2 |
|---|---:|---:|
| Input DMA interval | 1,637 | 4,833 |
| Input TRF ready from Task | 8,221 | 8,547.5 |
| First weight marker begins from Task | 2,440 | 7,735.5 |
| First weight begin → last weight end | 29,973 | 30,820 |
| First dot begins from Task | 17,147.5 | 22,811 |
| Last dot ends from Task | 39,173 | 44,838 |
| Last dot end → gather begin | 2 | 2 |
| Complete F32 gather interval | 1,200 | 1,347 |
| Scale begin → Task end | 6,921.5 | 7,003.5 |
| Task cycles | 48,398 | 54,409 |

The observed regression is concentrated before the first dot, rather than in fragmented gather. The first weight command follows input DMA almost immediately in the control (begin+8 cycles in all observations), while the candidate issues input TRF before first weight. Candidate W0 begins33 cycles after TRF begins, with a527-cycle median gap after input DMA ends. [All per-run/cluster windows](profile-input-weight-timeline.json) are retained. This identifies an issue/dependency question; it does not prove physical HBM saturation, bank conflicts, or that the entire delay is removable. Both dots still begin2 cycles after their latest observed weight-ready boundary. W1/dot0 overlap remains present.

The v3 byte-identical result specifically rejects source-let reordering as a repair for this build. No supported public API mechanism has yet been demonstrated to enforce earlier W0 issue. A future reopening needs a concrete changed producer/dependency/ownership mechanism with its full conversion, copy and consumer costs, rather than another K/stage sweep.

[The decision](decision.json) separates: no adoption of v2; stop this parameter series and ineffective v3 repair; preserve the unresolved input/weight issue question without declaring the whole ownership family impossible. Runtime and official submission were not changed.

## Recheck

All commands below are read-only and run no compilation or Arena jobs:

```sh
python3 moa-dev/experiments/candidates/o-owned-k4/verify.py --check
python3 moa-dev/experiments/candidates/o-owned-k4/results.py e132-o-k4-screen 109654 --check
python3 moa-dev/experiments/candidates/o-owned-k4/verify_graph.py e132-k4-v2 --check
python3 moa-dev/experiments/candidates/o-owned-k4/verify_graph.py e132-k4-v3 --check
python3 moa-dev/experiments/candidates/o-owned-k4/analyze_profile.py moa-dev/experiments/candidates/o-owned-k4/e132-o-k4-profile-job-109683-results.json --check
```

The candidate-local builder copies isolate `/tmp/moa-m03-validation-probe/target`; global builders and live runtime were untouched. `package.py`/`screen.py` handle standard hosts. `profile_package.py`/`profile_screen.py` handle the bounded diagnostic; its explicit build label is `e132-profile-v1`. Legacy `*-ffn.image-profile.json` filenames are retained from the reused profiler, but their verified target and26-operation contents are O.
