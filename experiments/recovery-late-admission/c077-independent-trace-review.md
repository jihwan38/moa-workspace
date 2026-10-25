# Independent c077 trace accounting check

Read raw r1134/r1137(C15) and r1112/r1138(C16) arena logs plus SchedulerTrace::work/report and Worker completion timers. Confirmed the first idle cumulative SCHED_TOTAL follows exactly15/16 warm requests, and totals never reset. The peer report5a79a276 at prefill-prefix512/experiments/c077-lm-runtime correctly subtracts warm work and keeps prompt dispatch versus phase decompositions disjoint.

C15 measured calls are504 width15 in both arms; prompt chunks218→220 and layers10464→10560. Decode costs reconstruct−316.89ms, complete prompt+689.76ms; their sum+372.87ms (rounded bound±137.34ms) covers actual+358.36ms window. Extra96 layers account for about233ms; preparation increases~375.88ms, chiefly higher per-preparation time. No chunk trace identifies the precise arrival-dependent packing cause. Faster decode therefore does not imply better wholeTPS:158.958→157.787.

C16 measured calls510/rows8064 and prompt233chunks/11184layers are equal. Both have498width16 calls; only12partial calls differ. Decode−737.88ms and prompt−348.96ms sum−1086.84ms (rounded bound±145.08) covers actual−1043.51ms. TPS162.093→165.510 while p90TTFT remains near3775ms receipt-budget releases. These are aggregate completed dispatch timers including model, read/selection, Generation and sends, not isolated LM timing or hardware clock measurements. Source scope is exact LM movement with recovery disabled; no all-concurrency throughput claim follows.

Use peer5a79a276 for sanitized inputs, source hashes, reader and full accounting; this review adds independent raw-log/caller confirmation rather than duplicating its large dataset.
