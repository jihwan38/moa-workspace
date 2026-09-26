# D-E104-QO — unchanged QKV/O image internal NPU profile

Branch `diag/e104-qkv-output-profile`, record parent `0a4417c`.
Source and comparison are independently explicit E104 / `7f891f9`.
This is a bounded diagnostic, not an optimization family or official submission.

Question: does whole QKV/O completion wait mainly for weight supply/lookup-dot,
activation preparation, or head/epilogue/cache transfers? The answer selects the
next full projection design. C03 retained typed head gather; C08 combined weight
readiness, and M01/M02 added partial movement. None proves the current QKV/O
tail is the dominant opportunity. Previous E104 whole-FFN profiling is the tooling
source, not QKV/O cost evidence.

Compile exact QKV and O schedules/IR/vISA from unchanged E104. Reuse the frozen
generic diagnostic host from D-E104-WEIGHT: verify its hash, embedded images and
host source. Label only existing QKV/O profile-name bytes at unchanged length.
Keep all task code, constants, marker IDs, binding metadata and numeric checks.
Run E104 control → QKV info/trace → O info/trace → E104 control in one allocation.
Three original inputs are kept; no new output boundaries or free preconversion.
One repeat is permitted to check profile observations if the first is valid.

First stop: validate marker-to-schedule correspondence and passing output checks.
If correspondence fails, do not invent source attribution from duration. If the
profile overhead changes execution, retain that limitation and compare only
trace observations with explicit controls. Overlapping task durations and waits
are never summed into speedups or stall fractions.

No live kernel source is changed. Candidate architectures follow the measured
completion path; a small tail does not justify a head-fusion implementation.

## NPU result and design decision

Arena **104914 / 104933** passed **108 output comparisons**. Source, embedded
images, metadata-only name patches and the actual uploaded payloads are checked
by `verify.py`. `analyze.py --check` verifies all 76 QKV / 22 O engine markers
against the exact compiler operation sequence. QKV cluster 1 consistently omits
the two RoPE HBM gathers and two cache scatters; its 72 observed engines are
checked as that exact set, not filled in from cluster 0.

[All measured windows and controls](profile-analysis.json) and
[complete source correspondence](marker-source-map.json) preserve every input,
cluster and allocation. The following are medians across 12 run/cluster
observations; event differences use one cluster's clock.

| Measured boundary | Device cycles | Consequence |
|---|---:|---|
| Q input TRF ready → first Q dot | 20,118 | Removing the input broadcast does not directly remove its duration from completion. |
| Q weight ready → first Q dot | 7,497 | Constant-table DMA completion and Sub table preparation still precede the dot. |
| Q table preparation interval | 5,670 | Q dot starts one cycle after its last prerequisite in all 12 observations. |
| Q Switch end → K dot | 2 | K weight is already ready; the Main sequence also matters. |
| Q RoPE end → V dot | 2 | V weight is 221–2,105 cycles earlier than Q RoPE completion. Weight-only tuning cannot remove this remaining Main dependency. |
| O input TRF ready → dot | 20,955 | Input staging is ready well before weight supply. |
| O weight DMA interval | 26,736 | Dot begins two cycles after weight completion in all 12 observations. This does not prove bandwidth saturation. |
| O dot + 16-slice reduction interval | 12,014 | Keep the complete dot/reduction cost when comparing a new supply path. |
| O dot end → Task end | 9,389.5 | Includes gather, scale, RMS, residual, output and waits; deleting this entire tail still would not explain the roughly 41% gap to the audited tenth-place O result. |

**Prioritize weight delivery/decode/consumer layout together, including Main
scheduling and table setup. Do not start a broadcast-only or head-tail-only
sweep from these numbers.** QKV's table wait and input slack now have runtime
evidence, but [the existing SDK audit](../../../../docs/research/fp8-lookup-table-sharing-audit.md)
found no prepared-table handle. Shared-TRF and interleaved-KV are already tested
source families. O row staging is also not new: [C09](../c09-output-pipeline/README.md)
already emitted supply/dot overlap with extra loads/setup, and was closed on a
small static increase without NPU measurement. A retry must explicitly address
those added costs or justify a bounded NPU test of that uncertainty. This
diagnostic alone does not establish a faster replacement architecture.

Intervals can contain waiting and overlap. They are not summed into runtime
shares or predicted savings. Compiler makespans are 51,293 / 27,239, with no
assumed fixed conversion to device cycles. QKV control-after in 104933 includes
a 123,912-cycle sample, so no stable profiler-overhead or speedup claim is made.

```bash
python3 moa-dev/experiments/candidates/e104-qkv-output-profile/verify.py
python3 moa-dev/experiments/candidates/e104-qkv-output-profile/analyze.py --check
```
