# c081 physical32 runtime capacity verified

Root job 321376 / `r1263-c081-c40-queue-submit` completed in 15.448 s. This used frozen source `3d497087` and captured ELF `b1aa162208a29f2688de7926ec76eea5de419fa2ce0d04b553cc25a83fcf1705`. The startup assertion and observed scheduling independently establish the intended capacity.

The actual owning-buffer check reports physical rings 32, admission 32, query batch 64, full pages 256, context 8192; each sliding K/V owner and each full K/V owner is 134,217,728 bytes. Forty sliding layers and eight full layers total **12,884,901,888 bytes (12 GiB)** of KV. This reads real public Buffer extents, rather than inferring allocation from source constants.

All 40 requests passed the harness text/SSE/usage/completion protocol checks. Each independently used 256 prompt tokens and generated 64 tokens. All client dispatches occurred at 0.508–14.043 ms, before the first response completed at 5,690.096 ms: the workload genuinely exceeded the 32 resident slots.

The parser checks every `SCHED_CHUNK` row, pending snapshot, cohort member/passthrough and `SCHED_REQ` completion in server-log order. All 40 worker IDs have contiguous prompt histories totaling 256 tokens each; 20 chunks cover 10,240 tokens and 960 layer quanta. No ring outside 0–31 appears, no owner changes ring, and no new owner uses a ring before its old owner completes. Maximum observed simultaneous ownership is 32. IDs 1–32 all complete before any of IDs 33–40 is first observed. The eight actual reuses are:

| Ring | Previous worker ID | Completion log line | Next worker ID | First observed line |
|---|---|---|---|---|
| 24 | 25 | 93 | 40 | 103 |
| 25 | 26 | 94 | 39 | 103 |
| 26 | 27 | 95 | 38 | 103 |
| 27 | 28 | 96 | 37 | 103 |
| 28 | 29 | 97 | 36 | 103 |
| 29 | 30 | 98 | 35 | 103 |
| 30 | 31 | 99 | 34 | 103 |
| 31 | 32 | 100 | 33 | 103 |

Two completed cohorts have 32 and 8 fixed members and both release when ready. The first cumulative decode histogram is `d32=63`; the final adds `d8=63`. Thus 2,016 + 504 = 2,520 decode rows, plus 40 prompt first outputs, equals all **2,560 reported generated tokens**. This reconciles complete output delivery with actual work counts.

The eight later HTTP responses have TTFT 6,164.49–6,173.30 ms. Overall TTFT p90 is 6,169.40 ms, so this C40 case **fails the 4 s latency target**. Its 300.24 tokens/s is a bounded single-length capacity diagnostic, not a ranked mixed-workload result or a performance claim. Their Worker `q≈2.1 ms` excludes time waiting before mpsc dequeue and must not be used to deny the externally visible waiting.

The API does not carry a correlation ID between HTTP `request_index` and Worker IDs. This proves bounded ownership, complete overflow handling and no observed premature reuse; exact client-index FIFO order is not independently established. FIFO at mpsc/pending admission remains supported by the actual source and existing CPU tests. Event observations also do not invent absolute admission timestamps.

## Scope of the earlier artifact failure

The c080 server demonstrably admitted rings 32–39 and ran width 40; it was not the recorded physical32 serving program. It remains quarantined. The cached library timestamps, relative dep-info and absent a005 library rebuild support stale shared-target reuse, but no controlled compiler-cache reproduction establishes that as the unique cause. The forced-fresh, original-dependency c081 artifact now produces the intended real allocation and capacity, so the observed contradiction is resolved at the server-artifact boundary. This does not retroactively validate c080's serving results.

The separately frozen 32/64 model probes, their exact cross-program digests and measured 10 GiB allocation difference remain separate evidence. c081's fresh public quality/performance gates are root-owned and are not claimed by this capacity report.

Reproduce the checks without NPU access:

```sh
python3 experiments/physical-pool32-export/analyze_queue40.py \
  /round2/runs/r1263-c081-c40-queue-submit \
  --output /tmp/c081-capacity.json
```
