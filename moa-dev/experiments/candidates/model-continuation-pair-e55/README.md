# Original/E55 continuation pair

This package compares full next-token distributions along the original model's
greedy token history. The [host implementation](../model-continuation-e55/probe_model_continuation.rs)
uses four fixed prompts, at most eight checkpoints per prompt, a fresh Workspace
per prompt, and the same Workspace/Decode across that prompt's continuation.
The original EOS choice ends a case. Candidate EOS or a different candidate
choice is recorded while replay continues along the original history.

Actual [job71329 results](../../../../docs/experiments/e55-model-continuation.md):
42.448 seconds,32 checkpoints,8388608 finite logits,32 matching legal greedy
choices, and null accuracy verdicts. [Collector tests](collector-tests.json)
include the actual positive and six mutations of its log; all63 tests pass.
The build preparation records below remain separate from this subsequent run.

[prepare_package.py](prepare_package.py) requires47 verified compiler inputs per
host: the unchanged46 model/build inputs and one new diagnostic host. It parses
the actual original/E55 ELF registries and requires all ten text images to match
the corresponding frozen prefix-suite host. It copies two hosts, the frozen
asset inspector, and [remote_entrypoint.sh](remote_entrypoint.sh); it refuses to
overwrite an existing package or manifest. Preparation itself executes no NPU.

The runner performs these actions inside one70-second Arena job:

1. Hash the three executed ELF files and fingerprint the full model/metadata.
2. Run original `--emit reference`. The host writes each complete checkpoint
   array and publishes `reference/manifest.json` only after all cases finish.
3. Hash the manifest and the fixed checkpoint filenames that actually exist.
4. Run E55 `--compare reference`. It validates and loads all reference arrays
   before Device creation, then compares every checkpoint at the same history.
5. Rehash the reference files, check weight stat/header and metadata hashes,
   then emit structural completion and exit status.

The [collector](../../collect_model_continuation.py) validates terminal success,
the actual uploaded archive, ordered complete events, original-token replay,
EOS/count rules, reported finite error statistics, legal sampler summaries,
before/after reference hashes, and asset identity. Reference arrays remain
remote; this is validation of the audited host's reports, not an independent
recomputation from downloaded logits. File hashes after execution do not replace
a second full model-weight hash.

All numerical and token-choice differences remain observations with null
accuracy thresholds/verdicts. This diagnostic does not score generated answers,
run E55 on its own divergent continuation, or cover long context/multimodal
inputs. It uses `temperature=1, top_k=1, top_p=1` through the existing sampler,
not the CLI's default stochastic settings. See the
[execution design and source trace](../../../../docs/research/model-continuation-diagnostic-design.md)
for token, cache, tie, and timing boundaries.

Run preparation from the repository root after both builds and CPU checks:

```bash
python3 moa-dev/experiments/candidates/model-continuation-pair-e55/prepare_package.py
```
