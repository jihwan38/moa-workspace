# Activated grouped norm stores on c078

The default-OFF export is d01ed090 + 73ff6d4f on ac2a8895. The two device modules and host implementation were byte-identical to the all11 frozen model source cfb3cca2 before activation. The three real prepared constructors now select grouped stores only when the existing joined projection storage exists, matching projection-owned FFN eligibility. Internal GroupExecution stays false until overwritten by a real prepared state. Explicit false diagnostics and legacy normalization/composition fallbacks remain available. No Worker policy, startup allocation, weight, arithmetic or cache change.

Independent read-only review by round2_path: PASS. Only three constructor defaults change; wrapper/tail propagation, fallback clearing and execution guards remain coherent. O/QKV and Worker are unchanged.

All11 actual model routes pass: 68,943,872 all-vocabulary BF16 values and 27,179,089,920 compared raw KV bytes, including sentinels, rolling histories, reset NaN padding, teacher-forced decode and B16/R32 interleaves. Submitted executable and all source hashes were verified. Comparison totals include repeated scans. Minimum reported free allocator memory is 3,520,479,488 bytes.

Reset rounds are candidate-first paired diagnostics, not ABBA or HTTP throughput. Interleaves and downloads are excluded from prompt totals, whereas aggregate stage counters include candidate interleaves.
Across the 11 reset comparisons, mean complete-prompt reduction is 4.504249 ms (range 4.041476–4.758601 ms). Four-owner128 improves 121.012729→116.331070 ms; single512 improves 123.227577→119.186101 ms. Root must measure serving latency/throughput after freezing the built server.
