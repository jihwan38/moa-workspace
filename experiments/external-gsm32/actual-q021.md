# Same fixed GSM8K TRAIN32: q021 extension

The two q021 jobs completed32/32 requests and scored32/32 with the unchanged
organizer scorer. The previously frozen c08131/32 and q02032/32 results remain
unchanged; their response files and original paired report were not overwritten.

| Candidate | Correct/completed | Prompt tokens | Completion tokens | Finish reasons |
|---|---:|---:|---:|---|
| c081 |31/32|3784|4990|32 stop|
| q020 |32/32|3784|4996|32 stop|
| q021 |32/32|3784|5019|32 stop|

q020 and q021 agree on all32 extracted numeric answers;23 full response texts
are byte-identical and9 differ. Both differ from c081's scored answer only at
the already recorded row4066: c081 returns6, both approximate candidates return
the dataset answer15. No response was filtered, replaced, or used to resample.
This fixed small TRAIN diagnostic supports no claim of general accuracy
improvement, equivalence, private accuracy, or official score.

Root dispatched `r1326`/job321620 and `r1327`/job321621 for q021's two unchanged
shards. Their Arena durations were40.755s and44.288s (full precision is in JSON).
Durations include startup and outputs have different lengths; they are not
throughput measurements.

The frozen q021 server SHA is
`7d89bb85e53fd9cd1cfcc44ebeaed4226aafe5f9cdb2063ac270ddcf4a272072`.
The exact shard request hashes, prefix/scorer/helper hashes, selected rows,
temperature0 and max768 match the prior two candidates. All96 actual responses
were independently rescored, each ID occurred once per candidate, and observed
prompt usage agrees per ID across all three candidates. No truncation or
unscored request occurred.

`actual-c081-q020-q021.json` contains all source/binary/log bindings, raw-record
SHA256 values, per-ID extracted answers/output hashes/token usage, all three
paired outcome tables, and the SHA of the preserved original two-candidate
report. Complete solution text remains only in the original run logs/records.

```sh
python3 experiments/external-gsm32/analyze_results.py --include-q021
```

This command reads existing results and writes a distinct three-candidate
report. Without the flag, the original two-candidate analysis remains available.
Neither form launches a model, sends HTTP, dispatches Arena, or alters selection.
