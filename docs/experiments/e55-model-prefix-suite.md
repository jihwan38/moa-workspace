# Four real-model fixed-prefix comparisons

[Arena job71239](e55-model-prefix-suite.json) completed in34.074s with four
fresh Workspaces per model,82 total input tokens, and1,048,576 compared logits.
All outputs were finite and all four raw top1 IDs matched. **This is a diagnostic
comparison, not an accuracy acceptance test or official performance result.**

| Prompt | Tokens | Max abs difference | Mean abs difference | RMSE | Original / E55 prefix+read seconds |
|---|---:|---:|---:|---:|---:|
| `Hi` | 14 | 1.5 | 0.194487 | 0.251039 | 1.815601 / 0.375337 |
| `What is 2 + 2?` | 21 | 0.8125 | 0.131520 | 0.170123 | 2.719336 / 0.559233 |
| `대한민국의 수도는?` | 19 | 0.625 | 0.078391 | 0.111337 | 2.460720 / 0.506477 |
| `Complete the Python expression: len([1, 2, 3])` | 28 | 0.583984375 | 0.107761 | 0.129746 | 3.625034 / 0.744902 |

Each comparison covers all262,144 final-position logits. The raw top1 IDs are
respectively9259,236778,146220,818 in both variants. E55 top1 margins are
12.0625,10.125,15.5,6.5. These are still large-margin examples; they do not test
near-tie decisions. No tokens were sampled or decoded into an answer, and this
is not a four-question answer-accuracy score.

The observed original/E55 prefix-plus-read ratios are4.837–4.866. The models
were executed sequentially, original first, without repeat/order/cache controls.
These timings exclude each case's Workspace allocation and the initial model
load; they do include final logits download. They do not measure generated-token
throughput or the unpublished official Stage2 workload. Source arithmetic can
also differ legitimately from the original implementation; a difference alone
does not identify which implementation is closer to an independent oracle.

Unlike the [first single-prefix comparison](e55-model-fixed-prefix-pair.md), this
job also fingerprints assets inside the same job. Before either model loads,
the standalone inspector hashes the entire9,304,966,064-byte weight file and
three metadata files. The weight SHA is
`7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b` and all four
file identities match [job71224](e55-arena-model-identity.json). After both models,
the inspector checks weight size/mtime/header again and rehashes the metadata.
All observed values are unchanged. This is not a second full weight hash or
proof against an external mutation that evades the observed stat checks.
The optional JSON chat template is absent in both snapshots; the actual Jinja
template is present and fully hashed.

[The suite implementation](../../moa-dev/experiments/candidates/model-prefix-suite-e55/README.md)
validates the exact four cases and token IDs before creating the Device, limits
their total to128 tokens, and creates/drops a fresh Workspace/Decode for each.
There is no reuse of a previous case's KV cache. The existing46 inputs per
variant are unchanged, the same new host is added, and each of its ten text
images matches the corresponding earlier original/E55 model host. The actual
four-file upload, three executed ELF hashes, entrypoint, and70-second job limit
were checked against the frozen package.

The [collector](../../moa-dev/experiments/collect_model_prefix_suite.py) requires
terminal success, both completion markers, all12 model events in order, all four
comparisons, null accuracy verdicts, finite/range-consistent reported statistics,
asset identity checks, and the exact upload hashes. It does not independently
recompute logits from raw output arrays; those arrays remain in the job's
reference file. [Mutation checks](../../moa-dev/experiments/test_collect_model_prefix_suite.py)
accept the actual evidence and reject20 changes, including missing completion,
missing/reordered cases, false accuracy verdicts, corrupt statistics, partial or
changed asset identity, modified ELF bytes, and a modified job configuration.

Next, [one-kernel numerical controls](../research/e55-model-numerical-differences.md)
can attribute the observed differences without changing the runtime or inputs.
Long context, page/ring transitions, multiple requests sharing a Workspace,
multimodal inputs, full generated sequences, and an independent accuracy oracle
remain outside this result. No official submission, score, or rank was obtained.
