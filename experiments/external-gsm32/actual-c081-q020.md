# Fixed public GSM8K TRAIN32: c081 versus q020

All four preregistered jobs completed. The original c081 scored31/32; the q020
MSE-F4 candidate scored32/32. Both completed32/32 HTTP requests, with32 `stop`
finishes and no length truncations or unscored requests. This is one fixed small
public TRAIN sample, not an official score or evidence of a general improvement.

| Frozen candidate | Shard1 | Shard2 | Completion | Correct | Completion tokens |
|---|---:|---:|---:|---:|---:|
| c081 physical32, original E4 FFN | 16/16 | 15/16 | 32/32 | 31/32 | 4990 |
| q020 MSE F4 B16, parallel startup | 16/16 | 16/16 | 32/32 | 32/32 | 4996 |

The single discordance is source row4066 (`gsm8k.train.004066`): the unchanged
organizer scorer extracts6 from c081 and15 from q020; the pinned dataset answer
is15. The c081 error remains in the original log/records and paired report.
There was no resampling, filtering, correction, retry, or answer-dependent
candidate/source edit. All31 other extracted answers agree. All32 full response
texts differ, as expected to be possible after approximate FFN arithmetic.

`analyze_results.py` rechecks the preregistered indices, source/shard manifest,
frozen server hashes, all four client/scorer helper hashes, request-set hashes,
unique IDs and request indices. It independently reruns the original scorer on
every saved response and reconciles completion/accuracy summaries and token
usage. Both candidates received identical request-set hashes per shard and
reported the same prompt usage for each ID (3784 total). This establishes frozen
client/request identity plus observed input-length agreement; it is not a
separate network packet capture.

| Run | Arena job | Candidate/shard | Arena seconds |
|---|---:|---|---:|
| r1308 | 321567 | c081/1 | 23.340665 |
| r1309 | 321568 | q020/1 | 40.881909 |
| r1310 | 321574 | c081/2 | 23.797466 |
| r1311 | 321577 | q020/2 | 45.105428 |

Each duration includes startup. The quality runs use different generated
completion lengths and are not a throughput comparison. Each request kept the
organizer GSM prefix, temperature0 and max768. Ground truth stayed in the
harness scorer and was never included in server request bodies.

The complete outputs remain in each original `runs/.../arena-records.json` and
`arena.log`; `actual-c081-q020.json` records their SHA256 values together with
candidate/source binding, per-ID numeric answers, output SHA256 and usage. No
full solution texts are duplicated in this report. Reproduce the read-only
analysis on the host with:

```sh
python3 experiments/external-gsm32/analyze_results.py
```

The original preregistration `0ddf46b5`, dataset revision
`3101c7d5072418e28b9008a6636bde82a006892c`, seed20261026 and all32 selected rows
remain unchanged. No private/test data or response-selected examples were used.
