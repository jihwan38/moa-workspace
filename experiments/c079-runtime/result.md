# c079: normal C16 improves; overlap diagnostic remains flat

All comparisons use collected completed runs, the existing join-aware reader and a verified warm checkpoint after exactly16 completed warm requests. Client performance/scoring/entrypoint scripts match within each pair. c078 and c079 differ only in the grouped FFN norm-store modules, prepared-state plumbing and its diagnostic; Worker, recovery policy, Generation and API handler hashes are unchanged.

| Workload | c078 | c079 | Observation |
| --- | ---: | ---: | --- |
| C16,128 measured requests: generated tokens/s | 163.514 | 168.850 | +3.26% |
| Same: TTFT p90 / TPOT p90 | 3774.17 /43.16ms | 3771.25 /40.23ms | Both meet these p90 bounds |
| Overlap C16,96 measured, one128-output request: tokens/s | 156.518 | 156.386 | −0.084%, effectively flat single-run result |
| Same overlap: TTFT p90 /TPOT p90 | 3807.94 /56.04ms | 3828.40 /56.99ms | No overlap improvement claim |
| c079 longer C16,144 measured | — | 169.513 | TTFT3767.60ms, TPOT40.07ms |

## Warm-subtracted attribution

Normal C16 (r1152→r1173) has the same8,064 decode rows. Decode calls fall512→506 and full16 calls rise496→502. Of253ms lower decode dispatch cost,238ms is explained by the changed call/width counts at observed control mean costs; only15ms is a within-width difference. There is no decode-kernel change in this source delta.

Prompt work falls234→233 chunks (11,232→11,184 layer calls). Layer time falls1,235ms: about117ms is the removed48 calls, while1,118ms is lower mean layer time (2.431→2.331ms). Preparation falls133ms and finalization is essentially unchanged. The whole prompt-dispatch difference is−1,370ms. Combined decode+prompt accounting gives−1,623ms versus−1,583ms measured window; the40ms residual is inside the145ms bound from rounded scheduler means. This supports a real prompt-layer benefit plus a smaller packing benefit; it does not attribute the entire HTTP gain to the norm kernel alone.

Both normal runs have nine16-member cohorts including warmup, no late joins and two deadline releases. Unfinished prompt work at those releases falls from1,024/1,280 tokens to512/512. That is consistent with fewer partial decode rounds. c079 has0 individual requests over4s versus7 in c078, despite both p90 values passing.

The overlap diagnostic (r1150→r1181) holds total work constant:466 decode calls,6,112 rows,173 prompt chunks,8,304 layer calls. Layer time improves664ms (2.420→2.340ms/call), but preparation rises400ms, finalization21ms and decode265ms. Changed width counts alone would save53ms; observed within-width cost rises318ms. Neither preparation nor decode arithmetic changed, and the trace does not identify the causal source of their variation/context mix. Prompt-dispatch improvement235.7ms is offset by decode265.1ms, predicting+29.5ms against the actual+33.4ms window, within the115.8ms rounding bound.

Both overlap runs have seven cohorts and the same deadline leftovers1,536/2,560/2,048/768 tokens. c078 joins six members in five events, c079 seven in seven events; the fifth/sixth cohorts are12→16 and14→16 versus12→16 and13→16. The reader finds no repeat participation, changing existing membership, deadline extension or open snapshot. Old decoder completions occur during active joined cohorts. Thus this is not a broken recovery lifecycle; faster prompt layers simply do not guarantee a better aggregate overlap schedule. The unique128-output stream's maximum recorded content gap rises174.18→183.25ms. The150ms target is not a hard per-gap guarantee.

## Long-run completeness and quality

r1180/Arena320993 completes all144 measured requests plus16 warm requests, with160 distinct Worker completions,9,216 measured output tokens, no missing/duplicate/error result and a final144/144 summary. All finish by the requested length. Its measured decode workload is9,072 rows in568 calls,566 of them width16; prompt work is261 chunks/12,528 layers. No individual measured request exceeds4s. Arena duration67.101s includes startup/warmup; the scored measured window is54.367s. Standard C16 logs do not contain per-content-event offsets, so exact individual gap claims are limited to the overlap diagnostic.

The same c079 source1aad0fe6 and ELF SHA256 `c72b572f8d6ca8532508bf1b81eeb0ebebdfdb05e974d6f624d5c209bec6aa2d` were used for both performance runs, the long run and all six public quality jobs:

- r1174 IFEval20/20; r1175 GSM8K34/34 atC17; r1176 MMLU34/34 atC17.
- r1177 VQAv212/12; r1178 FLEURS-EN12/12; r1179 FLEURS-KO12/12.

Every individual quality record passes:124/124, six successful terminal statuses and six zero-failure summaries. There is no alternate quality binary or API/Worker branch in this comparison. These are public gates, not an official private score or proof for untested inputs.

## C17 is not justified by the existing width cliff

Current `Model::forward_sequences_inner` sends17+ independent singletons into Decode64; `Decode64::run` selects its capacity32 kernels for17..32. The latest change accelerates grouped prompt normalization, not that route. Historical warm17-row O32 model gates r899 measured64.144–64.371ms before selection; the later r1077 same-route17 step measured64.785ms. Current c079 width16 dispatch averages39.44ms in the long run.

A simple steady-cohort screen illustrates the mismatch: c079's169.51tokens/s corresponds to6.041s per16×64 outputs. Subtracting63 width16 decode rounds leaves an inferred3.556s of nondecode work. To match that throughput with17 requests, the decode step must be≤45.4ms even if prompt cost does not rise, or≤41.9ms if nondecode work scales17/16. The existing≈64ms route is far outside that bound; even an optimistic64ms step excluding selection gives about139tokens/s under the scaled scenario. This is not a new C17 prediction, but it is sufficient to reject another expensive C17 throughput run until that width discontinuity changes.

`analyze.py` regenerates `result.json`, verifies all summaries, matches client/source hashes and records rounding bounds. It imports the previously reviewed reader by recorded SHA256. No source edit, build, NPU job or serving change was made for this audit.
