# E128 / frozen E131 v2 whole-FFN profile — completed

**No kernel adoption or official submission.** One bounded allocation109609 completed all57 comparisons. Original whole hosts: E128328539 → E131349566 → E128329639; bracket329089 gives **+6.222329% regression**. This reproduces the earlier v2 whole regression; the standalone probe timeout is a separate diagnostic result, not a whole arithmetic failure.

[Compact data](analysis-summary.json) · [Complete per-run/cluster/source mapping](profile-analysis.json) · [Independent uploaded-byte verification](verification.json) · [Raw log](evidence/arena-109609.log).

Actual start8ee5615. Source/performance controlE128cc82b9c; candidate is unchanged E131joint-four-row-v2 exact image119790f1… . Same original fixture and correctness criteria. The profile host emits names/times and selects FFN; the device image edits touch fixed-width profile-name bytes only. All task code, constants, IO/slots/stacks, marker endpoints/counts and image lengths are unchanged. Every120/130 operation matches the complete schedule-kind/native-kind sequence and all operations are present in every run/cluster. The actual uploaded four host binaries were independently rechecked against the original standard hosts and reconstructed labelled diagnostic hosts.

Seven stages were E128 whole before, E128 FFN info/trace, E131 FFN info/trace, E131 whole, E128 whole after. Diagnostic info medians327069/347568 and trace327558/349374 are separate from whole performance. One allocation and three original inputs; two cluster clocks and seven processes are not independent repetitions.

## Actual readiness and completion

Values below are medians of three inputs for each cluster, cycles relative to that cluster’s Task start. Current down uses BF16 lookup/DPE; it is not an E5 native-down implementation.

| Landmark | E128 cluster0 | E128 cluster1 | E131 cluster0 | E131 cluster1 |
|---|---:|---:|---:|---:|
| Input HBM begin | 2,792 | 2,164 | 104,855 | 104,130 |
| Pre-RMS ready | 16,357 | 15,723 | 115,898 | 114,463 |
| Three-term broadcast end | 120,726 | 120,355 | 182,689 | 180,310 |
| First up/gate native begin | 123,943 | 123,570 | 186,261 | 183,882 |
| Last up/gate native end | 192,945 | 191,267 | 242,858 | 240,445 |
| GeGLU ready | 231,301 | 226,553 | 249,226 | 246,604 |
| Down activation TRF ready | 260,791 | 256,035 | 282,706 | 280,099 |
| First down BF16 dot begin | 260,792 | 256,036 | 282,709 | 280,102 |
| Final output store end | 326,692 | 324,482 | 349,367 | 346,567 |

## What the profile establishes

1. **Input preparation is scheduled behind full weight/scale ingress in E131.** In all six candidate samples, `up_scale_HBM.end → input_HBM.begin` is exactly8cycles. The two compact packed weight loads and up scale transfer precede the input read. Pooled landmark medians: input HBM begin2478→104492.5; pre-RMS ready16040→115391; first native123889.5→185874.5. The source already calls input normalization/quantization before `project_up_and_gate` (`src/ops.rs:228`, frozen candidate `mlp.rs:416`), so a source statement reorder alone is not a demonstrated fix. This is an observed DMA/scheduling order, not a mathematical dependency of the input on the weights.

2. **Early delay is partly recovered later.** Last-native completion is49581cycles later in pooled medians, but GeGLU readiness is only17792cycles later. In E128, all six up-gather endpoints equal `down_packed_HBM.end +678cycles`; its marker duration is32900–36222cycles. In E131 it is1110–1114cycles. The E128 marker includes waiting behind the large down ingress; treating the entire interval as copy service time or an additive saving is incorrect. Earlier input/native readiness alone need not move final completion by the same amount.

3. **Additional lookup preparation and broadcast overlap, without proved resource causality.** E131’s eight native-table preparations complete during the input/broadcast supply interval. In all six samples, the last table endpoint is2cycles before term-broadcast completion. Broadcast marker duration medians are23415→38579. This alignment is evidence to inspect, not proof that a table directly depends on the broadcast or a measured bandwidth/stall fraction. Original data dependencies do not connect that table output to the broadcast.

4. **The unchanged down tail is entered later.** Pooled down BF16-dot start260566.5→282359.5 (+21793); final output-store end326469.5→348965.5 (+22496). Dot interval durations remain about40753/40775cycles. There is no evidence here that changed down arithmetic causes the regression; it receives later inputs. This does not prove all downstream intervals are identical.

The two candidate packed copies each have a3915cycle median interval. Adding those intervals as the promised recovery would miss the much larger changed readiness order and overlaps. Conversely, subtracting the~102k input-start delay from the whole kernel would overstate recoverable time. The observed whole regression is20477cycles in this allocation; a new complete candidate must remove enough completion-limiting work, not just a large sum of spans.

## Native-down scope and conditional suffix window

For the E128 critical cluster in each run, the elapsed window from GeGLU readiness to Task end is96405/96257/96217cycles (median96257); first down BF16-dot begin to Task end is66914/66766/66726 (median66766). Relative to trace Task327558 these are about29.4% and20.4%. These are **conditional same-prefix/same-schedule suffix windows**, not operation-duration sums, guaranteed gains, or universal upper bounds for redesigned kernels. A new down path can change earlier scheduling and pays its own quantization, broadcast and scale costs. Native-down still needs its complete current-input/layout contract and numerical validation.

## Decisions and reproducibility

Adoption: diagnostic only, E128 unchanged. Further tuning: this bounded profile is complete; no extra allocation, row sweep or kernel-source change. Major research: investigate an implementable removal of joint storage/readiness barriers or the separately specified native-down representation; the profile does not reject either family or establish a speedup. Shared Cargo target was released immediately after the two host builds.

Only this candidate directory was edited. Small sources/schedules/graphs/metadata/logs are frozen under `evidence/profile-v1`; large crate/host/archive artifacts are local ignored `artifacts/`.

```sh
python3 moa-dev/experiments/candidates/e131-cost-recovery-profile/package.py profile-v1
python3 moa-dev/experiments/candidates/e131-cost-recovery-profile/screen.py prepare e131-cost-recovery-v1 --build profile-v1
python3 moa-dev/experiments/candidates/e131-cost-recovery-profile/screen.py submit e131-cost-recovery-v1
python3 moa-dev/experiments/candidates/e131-cost-recovery-profile/screen.py collect e131-cost-recovery-v1 --job 109609
python3 moa-dev/experiments/candidates/e131-cost-recovery-profile/analyze.py e131-cost-recovery-v1 109609 --check
python3 moa-dev/experiments/candidates/e131-cost-recovery-profile/verify.py e131-cost-recovery-v1 109609 --check
```

Build/submit/collect commands refuse duplicate output records; rerun only the final two read-only checks on this frozen evidence.
