Job 323173 succeeded in 1.222040653 seconds. All 10 cases passed the frozen intended-code oracle: 49,152,000 codes, 24,576,000 packed bytes, 51,200 group scales and 51,200 relative offsets. Initial/post-timing checks, reused outputs, two true permutations per width and complete input/table/shift immutability are source-bound assertions reached before each EXACT marker. There are three base fixtures and two permutations.

| Fixture | H3840 two 10-call means (µs) | L15360 two 10-call means (µs) |
|---|---:|---:|
| 0 | 36.268, 35.805 | 81.886, 88.854 |
| 1 | 40.857, 40.237 | 103.114, 102.371 |
| 2 | 60.248, 60.086 | 182.177, 182.056 |
| 3 | 37.011, 35.677 | 81.302, 81.211 |
| 4 | 40.371, 39.884 | 102.413, 102.385 |

Each observation averages 10 completed calls, including max/quantization, the explicit offset HBM write+reread, full-record LUT gather, packing, H16 staging where applicable and grouped-scale stores. CPU expected values, uploads and readbacks are excluded. There is no production/control arm, GEMM or complete FFN in this run.

Case 2 uses A=14×2^shift, so every normalized LUT index is 96 and relative offset is 3072. Its greater latency correlates with that fixture; the run emitted no NPU_SPAN records or Arena profile records, so it does not establish gather contention or another hardware cause. Other fixtures and actual model activations can have different distributions.

Replay: `PYTHONDONTWRITEBYTECODE=1 python3 experiments/int4-grouped-activation-quant/analyze_actual.py`. It verifies raw snapshots/hashes, terminal status, ordered marker coverage, 418 Git-bound source/config inputs, original dependencies, captured/submitted ELF and two embedded resource images. The captured ELF remains external in the recorded runs directory; it is never executed by the analyzer.

Build source `ff51c812`; submission source `0dbea7d9`; functional source `f6ad9c83`. Captured ELF SHA256 `b71c05511a3217c8d6a3758360996203232df88d6bd5dfb6ae3d429edd37bfb3`. Original grouped resource H60292/L74636 and metadata-repair H29069/L67271 remain separate compiler evidence. No full-FFN performance, model quality or serving adoption claim.
