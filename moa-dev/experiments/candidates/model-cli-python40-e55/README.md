# Unmodified E55 CLI: Python prompt, at most 40 new tokens

This package reuses the existing `gemma4` ELF without source edits or compilation.
It exercises the real `generate → decode_stream.step → on_delta → printer`
text path that the fixed-logits diagnostics bypassed. Preparation verifies the
historical [46 compiler inputs and 14-image build audit](../../../../docs/experiments/e55-model-host-build.json),
all preserved source bytes, all 14 images against their compiler outputs, and
the ten text images against the frozen E55 prefix-suite host.

The CLI is 12,378,864 bytes, SHA-256
`7e801ed65016b5d0faa71da3fd3f3f9d012db393adc47863ad8e4665532678eb`.
This is the exact ELF used in [71173](../../../../docs/experiments/e55-arena-model-smoke.json),
which timed out before its model-loaded log. That earlier cause remains unknown.
The unchanged inspector is 880,872 bytes, SHA-256
`1fb587ee436c5eb27099fe6177b4b04911b9b6ea09699de4a781c41bb932cf11`.

Run the read-only provenance checks and create a fresh three-file package with:

```sh
python3 moa-dev/experiments/candidates/model-cli-python40-e55/prepare_package.py
```

[prepare_package.py](prepare_package.py) refuses to overwrite an existing package
or [manifest](package-manifest.json). It copies only `gemma4`,
`moa-model-asset-inspector`, and [remote_entrypoint.sh](remote_entrypoint.sh) into
`moa-dev/experiments/artifacts/model-cli-python40-e55/package`.
Python optimization mode is rejected because the audited ELF parser uses assertions.
It checks preserved sources against the old compiler manifest; it does not claim
to rebuild or establish a new compiler-input capture.

The runner hashes the two files it will execute, performs full asset identity
inspection, and directly runs this command with stderr merged into stdout:

```sh
GEMMA4_MAX_NEW_TOKENS=40 FURIOSA_OPT_PROFILE=warn \
  ./gemma4 'Complete the Python expression: len([1, 2, 3])'
```

`RNGD_MODEL_DIR` remains inherited from Arena. The CLI only recognizes `--image`
as an argument flag: the new-token limit must be the environment variable above.
No image argument or new model path is supplied.

Markers use the prefix `MODEL_CLI_PYTHON40`: `BEGIN`,
`ASSETS_BEFORE_BEGIN/END`, `CLI_BEGIN`, `CLI_EXIT=<code>`,
`ASSETS_AFTER_BEGIN/END`, `STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT`, and
the EXIT trap's `EXIT=<code>`. The post-run inspector uses `--header-only`,
including when the CLI returns nonzero. The original CLI error code takes
priority over a later inspector error; structural completion is printed only
when both return zero. An external kill/timeout may prevent the EXIT trap from
running, so missing completion remains an incomplete run.

The CLI region contains a model-loaded stderr line, then **the first exact
prompt line as an echo**, followed by generated stdout and the final
`ttft: ... | decode: ... | finish: Stop|Length` stderr line. A collector must
separate this initial echo from generated text and preserve later repetitions
as generated content. Both streams use the same destination descriptor; no
pipeline filters, text rewriting, or added model-side markers are used.
The CLI ignores the printer thread's join result, so exit zero alone does not
establish complete output: retain raw logs and check the final summary and any
panic/error output as well.

This CLI uses its default stochastic sampler: temperature 1, top-k 64, top-p
0.95, with `rand::rng()`. It has no CLI option to select the earlier greedy
policy. Therefore token IDs, text, or the 13 non-EOS tokens observed in
[71362](../../../../docs/experiments/e55-model-continuation-python40.json) need
not match. `Stop` implies one of EOS IDs 1, 106, 50 on this no-stop-string,
non-cancelling path, but the CLI does not report which ID was selected.

The job limit is 70 seconds. Recent E55 load/prefix/continuation observations
suggest headroom for one short prompt, but no completion-time guarantee follows.
The full weight hash before this run can warm the file cache; the 71173 runner
did not perform that hash. A successful retry therefore does not explain or
resolve the earlier timeout, or establish improved cold-load performance.
Displayed TTFT is based on the first callback and the CLI's throughput summary
is not an official or controlled benchmark. Completion establishes one text
generation path; it does not supply a correctness oracle, whole-model accuracy
verdict, long-context/multimodal coverage, or a competition score.
The four additional registry images are vision/audio entries; this text-only
request does not execute or validate those paths.

Preparation and wrapper checks execute no model or NPU. Remote execution and
its result record belong to the root agent; this package record remains the
pre-execution snapshot.

Subsequent [Arena job71395](../../../../docs/experiments/e55-model-cli-python40.md)
completed in12.284 seconds and printed the result3, with11 completion tokens and
Stop/exit0. The [collector checks](collector-tests.json) and
[wrapper failure-propagation checks](runner-tests.json) are separate from the
pre-execution package snapshot. This run leaves accuracy/official verdicts null.
