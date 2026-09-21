# Real-model single-kernel controls around E55

The [combined machine-readable record](e55-model-single-kernel-controls.json)
links all four terminal jobs and their detailed provenance.

These controls restore one whole kernel implementation at a time while keeping
the other nine text images byte-identical to E55. The four-prefix host, runtime,
loader, input prompts, and tokenizer settings are unchanged. They investigate
numerical differences; they do not replace the accepted E55 implementation.

All four completed comparisons use the same four prompts and82 input tokens per
model as [job71239](e55-model-prefix-suite.md). Each pair compares all262,144
finite final-position logits per case, with fresh Workspace/Decode per case.
The model weight and metadata hashes match the previous asset identity; each
job checks them before execution and checks weight stat/header and metadata
hashes again afterward. No second full weight hash is claimed.

| Restored kernel | Job | Comparison reference | Result |
|---|---:|---|---|
| QKV | [71290](e55-model-control-qkv.json) | Original | Four finite comparisons; raw top1 unchanged; RMSE increases in all four |
| Sliding O | [71275](e55-model-control-sliding-o.json) | Original | Four finite comparisons; raw top1 unchanged; error direction varies by prompt |
| Full O | [71281](e55-model-control-full-o.json) | E55 | All1,048,576 logits bit-identical |
| FFN | [71282](e55-model-control-ffn.json) | Original | Four finite comparisons; raw top1 unchanged; numerical differences remain |

All four jobs succeeded with exit0. They reuse four distinct prompts, producing
16 complete vector comparisons rather than16 independent prompt cases.

For QKV, sliding O, and FFN, the newly emitted original-reference SHA256 is exactly
job71239's `81da45073cc45fff93f9940658ae7589de2d11803544cd5944980749588ce2bb`.
This connects the compared arrays to the prior E55 statistics without assuming
that matching top1 IDs imply equal full vectors. The reference hash is reported
by the remote runner; the generated JSON itself is not downloaded.

| Prompt | E55 RMSE vs original | Original QKV only | Original O only | Original FFN only |
|---|---:|---:|---:|---:|
| `Hi` | 0.251039 | 0.256669 | 0.286202 | 0.250870 |
| `What is 2 + 2?` | 0.170123 | 0.184378 | 0.190651 | 0.254436 |
| `대한민국의 수도는?` | 0.111337 | 0.242541 | 0.174904 | 0.111842 |
| Python `len([1, 2, 3])` prompt | 0.129746 | 0.148154 | 0.102162 | 0.151316 |

Restoring QKV increases RMSE in all four cases. Restoring O increases RMSE in three cases and reduces it in one. Restoring FFN
changes the differences but does not remove them. Effects propagate through
later layers and KV state, so these error statistics are not additive kernel
contributions. Original output is a controlled comparison, not an independently
validated mathematical or task-accuracy oracle. A smaller difference from it
would not by itself justify reverting an optimization.

The full-O control is a direct negative control against E55. Only the generic
residual-add loop changes from expansion back to the original loop; its sole
current caller is `full_attention_output`. All four compared vectors are
bit-identical, consistent with the [source audit](../research/e55-model-numerical-differences.md)
finding no explicit arithmetic change. This tests the four observed execution
paths; it is not a proof for every input or for other combinations of kernels.

The [source/build audit](../../moa-dev/experiments/candidates/model-single-kernel-controls-e55/source-audit.json)
checks47 actual local and Docker compiler inputs for every control. The unchanged
probe SHA is `07cfaf4646da18a78e03bb386632ac455ccf545d98b858385bded281c3104da4`.
QKV changes four source files, O and FFN three each, full O one. Private original
residual helpers keep the O/FFN controls from also changing full O. The
[package workflow](../../moa-dev/experiments/candidates/model-single-kernel-controls-e55/README.md)
requires the target image to equal original and the other nine to equal E55;
the collector repeats those checks against the actual downloaded upload archive.
Build preparation snapshots and subsequent NPU reports are separate artifacts.

All accuracy thresholds and verdicts remain null. No generated answers,
independent accuracy oracle, near-tie coverage guarantee, long-context/page/ring
boundary, shared-Workspace multi-request, or multimodal result is established.
Sequential per-case timings are recorded in each JSON but are not repeated,
order-controlled benchmarks or official scores. E55 remains the accepted
checkpoint. The [next continuation diagnostic](../research/model-continuation-diagnostic-design.md)
will replay original-generated tokens and compare up to32 distributions and
actual legal greedy choices. That design is not implemented or executed yet;
its added logits-head and JSON cost has no70-second completion guarantee.
Official submissions remain unavailable according to the
[maintenance notice](https://micro2026-moa.github.io/leaderboard.html); no rank
has been obtained.
