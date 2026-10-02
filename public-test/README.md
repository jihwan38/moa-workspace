# Stage 2 Public Test Dataset

A publicly available test dataset containing six text, image, and audio cases.

| Dataset | Cases |
|---|---:|
| MMLU | 1 |
| GSM8K | 1 |
| IFEval | 1 |
| VQAv2 | 1 |
| FLEURS — English | 1 |
| FLEURS — Korean | 1 |
| **Total** | **6** |

## Contents

- `cases.jsonl`: one test case per line.
- `performance.jsonl`: the image and audio requests of `performance_test.sh`. They use the
  `vqav2` and `fleurs_en` media, with the instruction repeated to pad the prompt to 512
  (image) and 768 (audio) tokens, as in grading.
- `corpus/`: images and audio used by the cases.
- `ATTRIBUTION.md` and `LICENSES/`: credits and licenses.

Each line of `cases.jsonl` is a JSON object with these fields:

| Field | Meaning |
|---|---|
| `id`, `suite`, `modality` | Case name, its suite, and `text`, `image`, or `audio` |
| `request` | The request body sent to the server, including the prompt and `max_completion_tokens` |
| `check` | How the response is scored (`type`) and the expected answer (`expected`) |
| `provenance` | Source dataset, revision, and row |

In `request`, `$image` and `$audio` are file paths relative to this directory. The test
scripts replace them with the base64-encoded file before sending.

## How the cases differ from the source datasets

- **Selection.** Graded cases are ones the reference implementation answers correctly on
  every repeated run with the server's default sampling. The check is a regression gate,
  not a quality score: every case is expected to pass, and a failure means the optimization
  changed the model's output.
- **Prompts.** Each suite adds a fixed instruction and a fixed `max_completion_tokens`, as
  shown in `cases.jsonl`. Private cases use the same format.
- **Scoring.** Stricter than the official metrics, so results are not comparable with
  published numbers. A multiple-choice reply must be the letter itself. IFEval uses the
  instruction types implemented in `scripts/stage2/scoring.py`. VQAv2 needs agreement with
  at least 3 annotators. WER and CER are checked per case, not over the corpus.
