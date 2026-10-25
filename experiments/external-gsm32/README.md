# External GSM8K TRAIN32 diagnostic

Preregistered before responses in `0ddf46b5`; selection/harness frozen in
`a9ed7f36`. Source is the MIT-licensed primary
[OpenAI grade-school-math repository](https://github.com/openai/grade-school-math),
revision `3101c7d5072418e28b9008a6636bde82a006892c`.
Only [train.jsonl](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl)
was read:7,473 rows, SHA256
`17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`.

Selection is precisely `random.Random(20261026).sample(range(7473),32)`, split
in sampled order into two16-example shards. Indices, original question/answer
records and complete harness cases are retained. There was no response-based
filtering/resampling. Public training accuracy is an additional diagnostic,
not an official score or a guarantee about private/test accuracy.

Each shard sends16 different requests exactly once at C16, with no warmup.
The organizer GSM user prefix, max_completion_tokens768, temperature0.0 and
scoring.py are preserved. Ground truth stays in the client-side `check` field;
only the original request fields, prefix and sampled question enter HTTP bodies.
Frozen server bytes and all serving sources remain unchanged.

Prepared folders under `/round2/evaluations/`:

- `c081-physical32-clean-external-gsm-train32-s1` and `-s2`
- `q020-b16-f4-mse-parallel-external-gsm-train32-s1` and `-s2`

Server hashes are the original c081 `b1aa1622...c1705` and q020
`ba8cc68d...3f817`. Both candidates have identical request-set hashes:

| Shard | Canonical request SHA256 |
|---|---|
| 1 | `3c3a6c75d8d9eb7bf549b263f03ef7b418330f6c15239e65f5cf131f8fed697c` |
| 2 | `361e948160eddc57c097fe6f91a5bd9cb7600353e90f1d538127abead894d42d` |

`prepared-evaluations.json` and each folder's `harness.json` retain full server,
scorer, prefix, helper and entrypoint hashes. `selection-preview.json` verifies
each case appears once. `validate.py` checks all request/scorer/binary identities
without model or HTTP execution. Synthetic protocol mocks verify16 wrong answers
still yield16 completed requests and exit0; one HTTP failure yields15 scored,
one unscored, exit1, and all16 result records. Wrong answers never fail early.

The summary retains organizer `passed/failed` and adds `completed/incomplete`,
`accuracy_scored/accuracy_passed/accuracy_failed/accuracy_unscored`, and
`accuracy_denominator_complete`. Full answer text, finish reason and usage are
logged per result. Exit status describes completion; inspect accuracy fields
separately. A hard timeout can prevent the final summary: preserve partial
records, do not turn them into a completed32-example denominator.

Root dispatch only. The exact per-folder command is:

```sh
docker exec --workdir /round2/evaluations/FOLDER moa2026-round2 \
  bash scripts/stage2/accuracy_test.sh --test gsm8k --concurrency 16 \
  --no-build --no-wait
```

The copied launcher fixes Arena timeout to70 seconds. `dispatch-plan.json` lists
all four independent jobs with `stop_on_failure=false`; run later shards/models
even if a preceding job is wrong or incomplete. No job has been dispatched by
this agent.

For a later frozen q021 candidate, reuse the same selections:

```sh
python3 experiments/external-gsm32/build_evaluations.py \
  --verify-source --candidate EXACT_FROZEN_Q021_NAME
python3 experiments/external-gsm32/validate.py
```

The generator verifies the pinned raw SHA and selected records, copies the
frozen binary, verifies scorer identity, and executes only `--prepare`.
Existing evaluation directories are never overwritten. No serving build or
dependency change is required.
