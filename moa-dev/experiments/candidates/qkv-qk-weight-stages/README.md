# E146: Q/K weight completion split into two native stages

Status: **not adopted**. Three same-package contemporary allocations all pass45 output comparisons, but paired QKV changes are −1.494967%, +0.710594%, +0.406876%; median **+0.406876%**. The first screen's gain did not reproduce. Raw9 has73,728 exact BF16 values (signed zero equivalent); full2 has16,384 tolerance-pass outputs and8,380,416 untouched cache values. Extra6 host was built but not run after the performance reversal. Baseline remains E144.

The independent branch `opt/qkv-qk-weight-stages` and every frozen repair are preserved. Aggregate median/control ratios are not substituted for the paired result. No official submission was made. Further row/stage tuning stops; this is one unsuccessful measured combination, not a global rejection of overlap. Next selected research removes scalar intermediate/setup and TDMA distribution from the original E144 path.

Actual start `6ec626988820cb3fa7ba18da5dc417d42cb9b8b4`, branch `opt/qkv-qk-weight-stages`. Source recipe selects adopted E1443586c41 independently from performance control baseline/e144.json. Prior D-E145 job136924 shows Q/K native at own weight-end+2 in every run/cluster; V waits for TRF. This is a local completion-boundary replacement, not a new native/precision architecture or a tile sweep.

Original E4 HBM -> two compact full-H row DM allocations -> each Main fetch/Switch/native -> disjoint halves of original raw F32 allocation -> original single canonical gather/common scale/channel scale/first BF16/head/cache. Q halves16of32rows/group, K8of16; V remains one16-row stream. Source slice4g+b still owns rowlow2=b, each half selects localr range; two-stage offsets recover all original rows. The original four-term activation quantizer/sharedTRF/coefficientVRF and native arithmetic are unchanged. No new full-output assembly or weight transpose buffer.

Weight HBM and DM logical30MiB each are unchanged. Adds two weight DMAs and two native/Switch setups (3->5), sliced output commits and changed liveness/scheduling; setup and smaller-DMA throughput can erase overlap. Exact gate must verify all source/destination extents, stage0/1 disjoint output coverage, actual Switch packet coordinates, five native shared-register lifetimes and whether the intended second ingress can overlap first consumption. Static overlap is feasibility, not performance. One contemporary whole E144-before/candidate/E144-after NPU screen decides net gain. Positive freezes implementation and completes bounded adoption validation regardless of5%; negative is not a global ban on staged designs.

## Frozen source representation attempts

| Variant | Change from v1 | Outcome | Added useful work |
|---|---|---|---|
| qk-half-v1 | Direct native half commits; constant-bool K/V helper | Scheduler Scratchpad allocation failure | DMA/native3→5 |
| qk-half-v2 | Explicit initialization of full raw Q/K | Same failure class |24KiB fill, two Sub starts |
| qk-half-v3 | Fresh compact native output then bounded Sub assembly | Same failure class |24KiB read/write each, four Sub starts |
| qk-half-v4 | Unconditional separate K/V functions, based on v1 | Exact compile PASS, static49218 |No v2 fills/v3 copies retained |

`branch-phi-audit.py/json/md` bind the v1 source, diagnostic vISA and exact compiler log. Constant true/false callers still generated tensor-return Branch/Phi nodes. Removing these source conditions allows v4 to compile. The scheduler's renumbered failing tensors are not directly mapped, so this is not proof of its sole internal cause. No failed build was run on NPU. `failure-qk-half-v*.json`, recipes, source snapshots and build records preserve the original attempts.

Source inspection confirms the complete quantizer prefix, V helper and O suffix are unchanged from E144. K/V dispatch is now unconditional. v4 raw Q/K output writes remain direct disjoint native partial commits followed by one original canonical gather. Static49218 versus E14450510 is an uncalibrated compiler estimate, not an observed2.56percent gain.

## Measured result and decisions

| Arena allocation | E144 before / after | Candidate QKV | Paired change |
|---|---:|---:|---:|
| 137020 | 97,971 / 97,351 | 96,201 | -1.494967% |
| 137025 | 98,260 / 95,803 | 97,721 | +0.710594% |
| 137030 | 95,869 / 95,590 | 96,119 | +0.406876% |

[Performance verification](adoption/performance-summary.md) binds the same candidate/control stage bytes, original fixtures and runner across all three actual uploads. Paired median is **+0.406876%**. The separate aggregate-median ratio -0.855908% combines different allocations and is not evidence of reproduced improvement; no observation was excluded. The unchanged O/FFN images' timing variation is not an optimization.

All135 standard comparisons pass, including45 candidate comparisons over three distinct inputs repeated in three allocations. [Boundary validation](boundary-suite/result-summary.md) separately passes nine raw cases/73,728 exact BF16 elements (signed zero equivalent) and two whole-QKV cases/16,384 outputs at unchanged tolerances, with8,380,416 untouched cache values preserved. Raw exactness does not imply bitwise whole-QKV equality or arbitrary-input/model correctness. The extra6 host was built with all three identical images but [was not run](adoption/extra-not-executed.json); do not count those six inputs as validated.

Adoption: root rejects this frozen prototype because the net gain did not reproduce, retaining E144. Gain below5% did not block validation: the initial small positive received both independent repeats and changed-boundary checks.

Further tuning: stop this stage-parameter series and additional E146 NPU work. Preserve v1–v3 compile failures, v4 implementation, first positive and both subsequent losses separately. Closing this prototype does not exhaust all staged dataflows.

Major research: return to current E144's scalar/VRF supply boundary as the next supported question. No span-sum savings, exclusive bus-stall cause or ranking gain is inferred from this result. Source/input/image provenance is in [source-build-binding.json](source-build-binding.json); it describes evidence, not live adoption.
