# First real-model original/E55 comparison

[Arena job71207](e55-model-fixed-prefix-pair.json) completed the original and
E55 text paths sequentially in the same job. Both tokenize `Hi` with the actual
chat template into the same14 IDs, run every input token through the decoder,
and read all262,144 logits at final position13. There is no sampling or generated
token feedback. [Host implementation and provenance](../../moa-dev/experiments/candidates/model-fixed-prefix-e55/README.md).

**The comparison completed; model accuracy has not been accepted.** Both arrays
are finite, but they differ. Stage2's official acceptance threshold and workload
remain unspecified, so the diagnostic reports `accuracy_pass: null`.

| Observation | Result |
|---|---:|
| Compared logits | 262,144 |
| Bit-identical logits | 46,239 (17.64%) |
| Maximum absolute difference | 1.5 |
| Mean absolute difference | 0.1944870011 |
| RMSE | 0.2510387867 |
| Absolute difference above0.5 / above1.0 | 9,097 / 50 |
| Maximum-difference token ID | 705; original−11.0625, E55−12.5625 |
| Raw top1 token | 9259 in both |
| Raw top1 logit | original24.25, E55 24.375 |
| Raw top1–top2 margin | original12.4375, E55 12.0625 |

The large top1 margin for this one greeting makes its agreement weak evidence
for prompts with close alternatives. These are raw logits, including tokens that
the normal sampler may suppress. A matching top1 is not a QA score, a generated
sequence comparison, or a general accuracy guarantee. The next accuracy work
must cover varied fixed prefixes and locate where differences accumulate;
Stage1 fixture tolerance is not automatically an end-to-end logits threshold.

| Diagnostic timing (seconds) | Original | E55 |
|---|---:|---:|
| Device initialization plus model load | 3.825773 | 3.698346 |
| Workspace plus Decode allocation | 0.611408 | 0.667281 |
| 14-token prefix execution plus logits read | 1.814920 | 0.375348 |
| Total before reference-file write / comparison output | 7.126181 | 5.621031 |

The prefix/read ratio in this run is4.8353. It is a **single uncontrolled
observation**, not a repeated benchmark or official speedup. The two processes
ran original-first, filesystem/device cache state was not controlled, and
neither sampling nor generated-token throughput was measured. Model load also
varied substantially from [the earlier E55-only diagnostic](e55-arena-model-progress.json),
which took16.058s through model loading. The first [CLI timeout](e55-arena-model-smoke.json)
remains unexplained; the successful jobs do not retroactively establish its cause.

The downloaded upload archive exactly matches the two frozen hosts and runner.
Each has47 verified build inputs and ten text images. The measured Stage1 images
match their respective original/E55 hosts. The original reference JSON stayed in
the job directory; the runner recorded its SHA256
`6c45746671ee88099c390c45ec26aed72ef63f93093029aae7113289d029dec2` before E55
read it. Raw logs, terminal status, and archive hashes are retained in the linked
result and ignored artifact directory.

Both loaders resolve the same model environment in one job and the package
writes no model assets. This job did not fingerprint the weight file before each
load; later asset fingerprints are separate observations. It did not test image
or audio inputs, position512 full-attention page transitions, the1024-position
sliding ring wrap, multiple requests sharing one Workspace, or long contexts.
The live E55 runtime remains unchanged, and this result is not an official
submission, score, or ranking.
