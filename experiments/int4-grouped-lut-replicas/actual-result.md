Job 323310 passed all exact-output checks. Timing effects are mixed across fixtures; this result does not select a runtime policy or promote the change to a complete FFN/model. H cases 0 and 3 regressed, while H case 2 improved by 22.33%; all five L observations improved by 2.31–8.64% in this single ABBA run.

[Terminal status](actual-323310/arena-status.json): SUCCEEDED, exit 0, 1.554078326 seconds. Built/submitted source `0488a75a`, functional source `78054937`. All 421 source/config inputs equal the captured build; the ELF SHA256 is `1d292a33f7172a5d8c96c7aa411817473c68f98e6c5828c50e53efa6df17a3c6` (3,210,264 bytes). Both baseline images match actual 323173, both candidate images match their frozen resources, and all four match the new empty build output directory. [Build bindings](build-result.json), [raw manifest](actual-323310/manifest.json), [raw log](actual-323310/arena.log).

The fixed five cases per width cover 49,152,000 rational codes, 24,576,000 packed bytes, 51,200 F32 scale values and 51,200 relative byte offsets. These totals count the full fixture coordinates once, not every baseline/replay comparison. Candidate-first and post-timing reads checked separate poisoned baseline/candidate outputs against the unchanged independent quotient/scale/packing oracle and separate old/new offset oracles. The source-bound assertions also check both true permutations, output reuse and every immutable input, LUT, replica-base and shift byte. The frozen binary reached all 10 EXACT markers and the final PASS marker. No tensor dumps were retained; these are verified assertions from the source-bound executable.

Each number below is the mean of two 10-call completed blocks. Block order is original→replicated→replicated→original. Positive gain means lower candidate time; individual-call distributions and statistical confidence were not collected.

| Width | Case | Original µs | Replicated µs | Mean gain |
|---|---:|---:|---:|---:|
| H3840 |0|36.41860|37.49075|−2.944%|
| H3840 |1|40.92470|37.89875|+7.394%|
| H3840 |2|60.80360|47.22635|+22.330%|
| H3840 |3|36.09470|36.94265|−2.349%|
| H3840 |4|41.27675|38.03525|+7.853%|
| L15360 |0|80.84410|78.97940|+2.307%|
| L15360 |1|102.64035|93.76830|+8.644%|
| L15360 |2|181.70140|169.80255|+6.549%|
| L15360 |3|81.02745|79.11200|+2.364%|
| L15360 |4|101.89725|93.44570|+8.294%|

Case 2 has `A=14*2^shift`, normalized mantissa 224/index 96: all original offsets are 3,072, while the candidate uses 16 adjacent records at 49,152+32×(p%16). Its complete-call savings are 13.57725 µs H and 11.89885 µs L. This supports an input-dependent effect of the changed graph/address layout, but it does not identify cache/bank contention as the cause. The candidate also adds a real replica HBM→DM→VRF load and integer address operation. No NPU_SPAN or Arena profile records were emitted; two PROFILE markers alone cannot identify the critical operation.

Cases 3/4 are true row/group/channel permutations of cases 0/1. Their mantissas, codes and scales follow the source data; the replica component follows the destination row. New offset contents intentionally differ from the old ABI. Every code and scale bit remains equal to the unchanged original kernel.

Complete-call timing includes max reduction, replica-base load, index generation, explicit 8/32 KiB offset write+reread, full-record gather, quantization, packing, real H16 staging where applicable, and grouped-scale stores. CPU oracle, uploads and readback are excluded from both arms. Retained LUT/base storage is 65,536+512 bytes versus the original 4,096-byte LUT, adding 61,952 bytes. There is no GEMM, complete FFN, actual model activation distribution, task-quality or serving-throughput measurement. These gains must not be summed with the separate seven-to-three graph result, averaged into a workload prediction, or used to introduce fixture-specific dispatch. Prior retained-axis NO-GO, slower complete-FFN evidence and original timing records remain intact.

Read-only replay:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 experiments/int4-grouped-lut-replicas/analyze_actual.py > /tmp/replica-actual-replay.json
cmp /tmp/replica-actual-replay.json experiments/int4-grouped-lut-replicas/actual-result.json
```

The analyzer verifies raw hashes, terminal status, 421 source inputs, ELF/four image bindings, all 53 lines in strict order (40 timing blocks, 10 exact records, 2 profile markers and 1 final PASS), counts and every per-case ABBA calculation. It retains the two block means, conservative extrema and all regressions in [actual-result.json](actual-result.json). Log SHA256: `e689c6f1ba277a6b0008860b4796be8af89b2ba10d01cf9c3d1e00df460e113b`. Independent read-only review by `/root/round2_path` corroborated source/ELF bindings, raw counts and every reported gain; no new build or Arena execution occurred during analysis.
