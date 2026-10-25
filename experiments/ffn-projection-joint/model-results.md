# All11 joint model routes pass

All **68,943,872 vocabulary values** and **27,179,089,920 raw KV bytes compared** match exactly. Frozen source `acbc934`; all submitted ELFs and recorded source hashes verified.

| Case | Job | Reset control ms | Reset projection-owned ms |
| --- | --- | ---: | ---: |
| 0 | 320202 | 125.322 | 122.566 |
| 1 | 320203 | 126.360 | 123.127 |
| 2 | 320208 | 128.461 | 125.559 |
| 3 | 320205 | 112.355 | 109.280 |
| 4 | 320210 | 111.208 | 108.169 |
| 5 | 320213 | 120.330 | 116.389 |
| 6 | 320206 | 119.135 | 117.190 |
| 7 | 320215 | 120.628 | 118.918 |
| 8 | 320217 | 127.409 | 124.386 |
| 9 | 320218 | 129.642 | 126.892 |
| 10 | 320220 | 126.918 | 123.667 |

Case0 candidate-only independent B16/R32 interleaves are excluded from total_ms.
Stage-profile aggregates include interleaved decoder calls, so their sum is not the prompt timing.
Round0 is control-first; reset round1 is candidate-first. These are paired diagnostics, not ABBA or HTTP throughput.
Untouched sentinel and full physical cache scans include repeated values; byte totals count comparisons, not unique cache capacity.
Both arms explicitly enable current exact sliding/full stationary O and retain composed norm stages and global preload; projection-FFN default activation is still off. Startup joins are excluded from per-request timings and measured memory includes actual joined storage.

Coverage: four independent128, single512, pair256+256, both Tail256 owner shapes, one/two/three-owner Tail384 and all doubled-owner positions in ThreeOwner512. Every route independently refills actual prefixes, executes48 layers, reverses owner reservations on reset, and compares all-vocabulary output, whole allocated raw KV, sentinels and teacher-forced postdecodes. Tail reset padding is NaN. Case0 includes six matched B16/R32 decoder calls at complete layer boundaries with disjoint reservations.
