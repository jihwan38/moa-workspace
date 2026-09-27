# E137 frozen-path cost recovery diagnostic

One allocation, **110511**, compared actual E134 and frozen E137 `native-v2`. All **57 output comparisons passed**. Original standard hosts measured O at48,579 /55,941 /48,703 cycles; candidate regressed **15.0079%** against the48,641 bracketing midpoint. This reproduces the earlier14.0551% regression; diagnostic trace timings are not promotion samples.

`package_v2.py` binds E134's actual source manifest and all three images. Its O image and production projection source match the original C09 schedule exactly. The candidate binds `build-e137-native-v2-repro.json` and its passed graph gate. Diagnostic hosts preserve all device sources and three exact images; only existing O profile-name bytes are changed. Tasks, constants, inputs/outputs, slots, profile markers and marker counts remain unchanged. All26 control and46 candidate executable operations map in sequence to the original schedule and marker kinds.

`package-profile-v1.json` preserves the initial tool failure: the old classifier called the coefficient `fetch<1,OutputTerm>→collect<1,OutputTerm#8>→to_vrf` a direct StoVrf. Its actual padded-lane operation is TuExec. `package.py` and that failed record are immutable. `package_v2.py` adds the narrow classification for this exact collect pattern; the final package is `package-profile-v2.json`. No production source or image repair occurred during profiling.

The seven sequential stages were standard control_before, diagnostic control_info/control_trace, diagnostic candidate_info/candidate_trace, standard candidate_whole, standard control_after. Every stage used the three original inputs. These are one allocation and three inputs, not independent performance repetitions. The standard hosts retain their original info-level task timing; diagnostics record existing markers at info and trace levels.

| Stage / elapsed landmark | E134 | E137v2 |
|---|---:|---:|
| Standard O median |48,579 before /48,703 after |55,941 |
| Diagnostic info median |48,421 |55,969 |
| Diagnostic trace median |48,963 |57,322 |
| Input HBM begin→end |2,457→4,458 |98.5→1,190 |
| W0 HBM begin→end |2,465→17,110 |1,906.5→15,051 |
| Global amax begin→end |not present |2,406.5→7,219.5 |
| Four-term quantization completion |not present |23,814.5 |
| Term replication DMA begin→end |not present |23,815.5→31,242.5 |
| Activation TRF begin→end |7,786→8,248 |31,244.5→31,835.5 |
| First dot begin→end |17,112→26,760 |31,836.5→34,311.5 |
| W1 HBM begin→end |19,462.5→32,581 |31,278.5→44,800 |
| Second dot begin→end |32,583→39,303 |44,802→47,246.5 |
| Final output DMA begin→end |47,833→48,694 |56,396→57,258 |

Landmarks are medians of the six run/cluster observations, each measured relative to its own cluster Task start. Median endpoints do not necessarily subtract to the median duration. They are not six independent allocations. Full samples and original schedule bodies/input/output tensor IDs are preserved in `profile-analysis.json`.

The candidate's per-observation median gaps are: quantization completion→term DMA start1 cycle; term DMA end→W1 start36 cycles; TRF ready→first native start1 cycle; W1 ready→second native start2 cycles. W0 completes16,955 cycles before first native starts. In E134, the first dot starts2 cycles after W0 completion and its activation TRF is already ready. The readiness bottleneck has moved to activation preparation in E137. The native dot itself is shorter on these traces, but the complete path is slower.

This supports one concrete repair experiment: prepare the same four activation terms directly in their final consumer slice layout, with every16-slice row group reducing the same complete K4096 amax. That removes the512KiB term replication operation and changes the max reduction from256 slices to16. It adds duplicated input HBM transfer (logical16KiB→256KiB) and more active quantizer storage/work. Same four terms, common global scale, channel-scale-before-BF16 and public inputs must remain unchanged.

The near-adjacency of term DMA and W1 is observed; it is not a direct tensor dependency proof or an identified physical stall fraction. Both are TDMA operations, while TRF consumes the replicated terms directly. No overlapping durations are summed. In particular, the measured7,427-cycle replication interval and4,813-cycle max interval are not an additive speedup prediction; the replacement changes other transfers, work and scheduling.

Decisions: **adoption** of frozen v2 is rejected for complete-path regression despite finite precision checks passing; **parameter tuning** remains stopped; **major research** proceeds only to the selected consumer-local quantization repair, followed by full-path controls and the same precision checks. The previous native-O family is not declared exhausted.

Recheck the exact upload, source/images, name-only patch and all marker correspondence:

```sh
python3 moa-dev/experiments/candidates/o-native-activation-resident/profile/verify.py e137-cost-profile-v1 110511 --check
python3 moa-dev/experiments/candidates/o-native-activation-resident/profile/analyze.py e137-cost-profile-v1 110511 --check
```
