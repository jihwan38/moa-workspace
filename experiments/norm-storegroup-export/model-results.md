# All11 grouped norm-store model routes pass

All **68,943,872 vocabulary values** and **27,179,089,920 raw KV bytes compared** match exactly. Frozen source `cfb3cca2`; all submitted ELFs and recorded source hashes verified.

| Case | Job | Reset control ms | Reset grouped-store ms |
| --- | --- | ---: | ---: |
| 0 | 320923 | 121.013 | 116.331 |
| 1 | 320925 | 123.228 | 119.186 |
| 2 | 320930 | 124.505 | 119.935 |
| 3 | 320927 | 107.742 | 103.394 |
| 4 | 320931 | 106.689 | 102.150 |
| 5 | 320932 | 114.467 | 109.884 |
| 6 | 320929 | 115.330 | 110.822 |
| 7 | 320933 | 114.404 | 109.835 |
| 8 | 320934 | 123.483 | 118.725 |
| 9 | 320935 | 123.778 | 119.150 |
| 10 | 320936 | 124.792 | 120.473 |

Case0 candidate-only independent B16/R32 interleaves are excluded from total_ms.
Stage-profile aggregates include interleaved decoder calls, so their sum is not the prompt timing.
Round0 is control-first; reset round1 is candidate-first. These are paired diagnostics, not ABBA or HTTP throughput.
Untouched sentinel and full physical cache scans include repeated values; byte totals count comparisons, not unique cache capacity.
Both arms explicitly enable projection-owned FFN and stationary sliding/full O plus stationary sliding QKV; only norm-store placement differs; grouped norm-store activation is still off. Startup joins are excluded from per-request timings and measured memory includes actual joined storage.

Coverage: four independent128, single512, pair256+256, both Tail256 owner shapes, one/two/three-owner Tail384 and all doubled-owner positions in ThreeOwner512. Every route independently refills actual prefixes, executes48 layers, reverses owner reservations on reset, and compares all-vocabulary output, whole allocated raw KV, sentinels and teacher-forced postdecodes. Tail reset padding is NaN. Case0 includes six matched B16/R32 decoder calls at complete layer boundaries with disjoint reservations.
