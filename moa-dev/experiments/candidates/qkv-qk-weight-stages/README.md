# E146: Q/K weight completion split into two native stages

Status: `qk-half-v4` exact compile and unchanged standard-host build PASS; actual descriptor/register review and first contemporary NPU comparison pending. No improvement or adoption claimed. Baseline remains E144.

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

Adoption awaits actual gate, whole screen and bounded positive validation. Further tuning remains stopped except specific supported contract/cost repairs. Major research tests Q/K completion dependencies, not precision relaxation.
