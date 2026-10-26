Job323322 completed with all intended-code correctness checks passing. The fixed weight-MSE policy is **NO_PROMOTION**: weight SSE decreased, but matched final residual drift increased.

Frozen ELF `64dd3b267d6bfe13003172d57a26414d1daa76a7f0b26ed13a9cb3824e7caefc`;427 inputs and all19 prior/fresh device images match. The strict reader consumes all108 records and verifies every raw-file hash, submitted/captured ELF and source binding. Replay: `python3 experiments/int4-weight-mse-diagnostic/analyze_actual.py`. Raw records are in `actual-323322/`; no device profile payload was returned.

| Projection | Original weight L2 | MSE weight L2 | Original max error | MSE max error |
|---|---:|---:|---:|---:|
| up_proj | 0.203050318 | 0.153262002 | 192.000244 | 1080.000000 |
| gate_proj | 0.224122948 | 0.170875404 | 178.285767 | 1188.000000 |
| down_proj | 0.216504585 | 0.161993954 | 192.000244 | 1559.999939 |

All3×61,440 groups strictly improve exact-real pre-global weight SSE, measured on the exact2^-38 grid. L2 above derives from exact-real SSE; max error uses the separate RN32 scalar reconstruction metric. Scalar RN32 L2 also improves here, but this does not guarantee dot/FFN improvement.

| Matched step | Stage | Original vs q020 L2 | MSE vs q020 L2 |
|---|---|---:|---:|
| 0 | joined | 0.396333742 | 0.439559141 |
| 0 | hidden | 0.630499371 | 0.611818383 |
| 0 | down | 0.482091560 | 0.482275268 |
| 0 | residual | 0.423880819 | 0.466080828 |
| 1 | joined | 0.318219852 | 0.386066125 |
| 1 | hidden | 0.556667232 | 0.547987741 |
| 1 | down | 0.452227876 | 0.457512049 |
| 1 | residual | 0.374812605 | 0.422980636 |

Both steps use the same original-production incoming residual for original, fitted and q020 arms. The permuted row order confirms the same result. Hidden L2 slightly improves; joined, DOWN and final residual L2 worsen. Own-hidden quantization metrics use different hidden tensors and cannot replace this matched comparison.

Correctness totals:393,216,000 intended BF16 comparisons;12,288 sampled quantized outputs;157,368,320 heterogeneous permutation elements;35,454,976 same-input bytes;1,198,661,776 immutable-byte comparisons;771,375,104 reuse-byte comparisons. These totals include repeated checks, not unique memory footprint.

The14.639ms format-table setup,5,938.457ms original decode/rational encode, and5,461.555ms fitted decode/encode/full audit are diagnostic CPU timings, not isolated encoder or48-layer startup measurements. Job duration26.859419s includes all references/readbacks. There is no complete-chain timing gate, full-model/task-quality result or serving activation. Official q020, final repository and prior performance HOLDs remain untouched.

Batch independently audited the actual counts, source/image binding and numerical interpretation before this report.
