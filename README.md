# Gemma-4 Kernel Optimization Competition on Furiosa RNGD

This repository is the competition skeleton code for optimizing Gemma-4-12B-it on Furiosa
RNGD. Stage 1 focuses on three decoder-layer kernels; Stage 2 will focus on end-to-end
(E2E) performance. Competitors modify the allowed implementation while preserving the
public interface and the model's numerical behavior.

The two-stage format is confirmed. Only the specific submission and scoring values marked
**TBD** remain to be announced.

## What you are optimizing

The three Stage 1 kernels are declared in `src/ops.rs`:

| Kernel | Operation |
|---|---|
| `ops::sliding_project_qkv` | RMSNorm, Q/K/V projection, Q/K RMSNorm, RoPE, and K/V ring-cache writes |
| `ops::sliding_attention_output` | Head broadcast, O projection, post-attention RMSNorm, and residual add |
| `ops::decoder_feedforward` | RMSNorm, GeGLU MLP, post-FF RMSNorm, residual add, and layer gate |

The kernels use the model's existing quantized weights and tensor layouts. Each kernel is
measured across 3 independently-seeded input configurations for the Stage 1 test (see
"Grading criteria" below), not a single fixed invocation.

Stage 2 covers the complete E2E model-serving path except for the public API endpoint in
`src/api/`. This includes model execution, host orchestration, tokenization, image and
audio preprocessing, and runtime integration.

## Competition format

The competition has two stages. The scope of each stage is confirmed below; the remaining
unknown submission and scoring values are listed explicitly.

### Stage 1 — Kernel optimization

Stage 1 is for optimizing the three kernels listed above. A dedicated grading server will
run the provided `src/bin/test_kernels.rs` against each submission. This test is the source
of truth for Stage 1 correctness and kernel performance.

The Stage 1 test checks:

1. **Buildability:** the allowed code compiles with the competition toolchain.
2. **Correctness:** each kernel satisfies the published tolerances on every one of 3
   independently-seeded input configurations, not just one -- a kernel that special-cases
   a fixed input fails as soon as a different configuration exposes it.
3. **Performance:** the test reports each kernel's median RNGD cycle count across those 3
   runs, so one unusually fast or slow run doesn't move the number that's graded.

**Stage 1 submission deadline: September 30, 2026, 11:59 PM AoE.**

### Stage 2 — End-to-end optimization

Stage 2 is for optimizing the complete Gemma-4-12B-it E2E path, excluding the public API
endpoint in `src/api/`. The harness supplies iteration scheduling and a bounded KV pool;
participants can implement batch calculation through `host::model::Model::forward` and its
device kernels. See [Serving](SERVING.md#batched-serving) for the contract.

The Stage 2 evaluation sends text, image, and audio requests to the server built from your
code and checks two things:

1. **Correctness:** every response meets the threshold of its suite, listed under
   "Grading criteria" below.
2. **Performance:** the server's throughput on each input type. Where its TTFT p90 exceeds
   the limit, the submission earns no speedup.

See [Submit for grading](#submit-for-grading) for how a submission is scored.

Stage 2 rules:

1. **No input-specific optimization.** Optimizations must be general. They must not target
   particular prompts, media, or datasets.
2. **No test-specific behavior.** Performance requests set `ignore_eos`, which makes the
   server generate exactly `max_completion_tokens` tokens. Keep that behavior, and do not use
   the flag, or any other way of telling a performance request from an accuracy request,
   to change how a request is computed.

Stage 2 values to be finalized:

- **TBD:** the TTFT p90 limit;
- **TBD:** submission deadline and tie-break rule.

## Grading criteria

The competition uses one grading policy across both stages. Correctness is a hard gate: a
submission that fails a required correctness check receives no performance credit, even if
it is faster.

For the Stage 1 kernel evaluation, `src/bin/test_kernels.rs` defines the following tolerances:

| Kernel | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| `sliding_project_qkv` | `0.04` | `1e-2` |
| `sliding_attention_output` | `0.05` | `1e-2` |
| `decoder_feedforward` | `0.01` | `1e-2` |

For the Stage 2 end-to-end evaluation, every case must meet the threshold of its suite:

| Suite | Metric | Pass | Graded cases |
|---|---|---:|---:|
| `mmlu` | exact answer letter | `1.0` | 4 |
| `gsm8k` | numeric final answer | `1.0` | 2 |
| `ifeval` | every instruction satisfied | `1.0` | 1 |
| `vqav2` | VQA accuracy | `1.0` | 5 |
| `fleurs_en` | word error rate | `≤ 0.2` | 2 |
| `fleurs_ko` | character error rate | `≤ 0.2` | 2 |

Prompts and scoring are adapted from the source datasets; see
[public-test/README.md](public-test/README.md#how-the-cases-differ-from-the-source-datasets).

The grading server will measure performance using the official evaluation. The Stage 1 test
reports each kernel's median RNGD cycle count across 3 independently-seeded runs. The Stage 2
test reports the server's throughput on each input type; where its TTFT p90 exceeds the
limit, the submission is scored a speedup of 1. The limit is TBD.

The following scoring values are still TBD:

| Field | Planned rule |
|---|---|
| Stage 2 TTFT p90 limit | **TBD** |
| Reproducibility and code review | **TBD** |

Schedule makespan is a useful development metric, but it is not a substitute for official
grading results.

## Stage 1 rules: skeleton contract

The following rules apply to Stage 1 kernel submissions. Stage 2 follows the E2E scope
described above; its remaining submission rules are listed in the Stage 2 section.

The Stage 1 skeleton is intentionally fixed so that submissions remain comparable.

1. **Keep device function names and signatures unchanged.** The name, parameters, types,
   and return type of every `#[device]` function are part of the evaluator contract.
2. **Only permitted implementation changes are graded.** Changes to `src/device/` and the
   function bodies in `src/ops.rs` are included in the evaluation. Everything else is
   ignored, including `src/ops_vision.rs`, `src/ops_audio.rs`, `src/axes.rs`, `src/host/`,
   `src/api/`, `src/bin/`, and `src/lib.rs`.
3. **Keep kernel module paths stable.** `src/ops.rs`, `src/ops_vision.rs`, and
   `src/ops_audio.rs` must remain at the crate root because compiled kernel names include
   `module_path!()`.

Shared code may affect more than one path. In particular, `device/shared/rmsnorm.rs` and
`device/shared/mlp.rs` are also used by full attention, vision, and audio code. Check the
impact of a shared change before measuring a result.

## Evaluation commands

### Prepare the public fixture

Run this once when `ref/fixtures.safetensors` is missing:

```sh
python3 scripts/generate_references.py
```

The fixture contains expected outputs and checksums. Inputs are generated deterministically
from the shared PRNG on both the Python and Rust sides.

### Optimize a kernel

Follow [OPTIMIZATION.md](OPTIMIZATION.md) for the schedule-dump, bottleneck-analysis,
kernel-tuning, and makespan-comparison workflow.

### Run the public RNGD check

```sh
./scripts/rngd_test.sh
```

This builds the test binary, submits it through the Arena server, and reports accuracy
and real RNGD cycle counts. Use `--no-build` to reuse the latest binary or `--no-wait` to
submit without waiting for the result.

For a locally available RNGD setup, the repository also provides:

```sh
./scripts/local_test.sh
```

### Test Stage 2 on the public cases

```sh
./scripts/stage2/accuracy_test.sh --test mmlu
./scripts/stage2/performance_test.sh --test 256 --concurrency 4
```

These build the model server, submit it through the Arena server, and test it with a public
input. `accuracy_test.sh` runs a case from `public-test/`, which holds one example per suite,
and reports whether the responses pass. `performance_test.sh` reports TTFT, latency, TPOT,
and throughput.

| Script | `--test` |
|---|---|
| `accuracy_test.sh` | `mmlu`, `gsm8k`, `ifeval`, `vqav2`, `fleurs_en`, `fleurs_ko` |
| `performance_test.sh` | `256`, `1024`, `2048` (text prompt tokens), `image`, `audio` |

Performance requests set `ignore_eos`, as in grading, so every response has 64 tokens.
`--concurrency N` (default 1) sends the input `2 × N` times with N requests in flight, so
that later requests are admitted while others are still running, as in grading. Unlike
grading, there is no warmup: the first requests include the server's cold start. Use
`--no-build` to reuse the latest binary or `--no-wait` to submit without waiting for the
result.

### Submit for grading

```sh
cargo binstall moa-submitter-cli   # for Stage 2, run again to update to the Stage 2 version

moa-submitter login
moa-submitter submit         # run from the repository root; auto-detects furiosa-opt-std version
moa-submitter submit --concurrency 4   # concurrent requests, default 1
moa-submitter status         # the state of every submission you have made
moa-submitter status <id>    # the results and score of one submission
moa-submitter log <id>       # the complete log, stage by stage
```

For Stage 1, `submit` uploads `src/ops.rs` and everything under `src/device/`. For Stage 2,
it uploads all the code except `src/api/`. Keep the original directory structure. Run it
from the repository root, or pass `--source path/to/furiosa-opt-gemma4-12B`.

**Stage 1.** A submission is scored as the geometric mean of its speedup over the baseline
across the three kernels.

**Stage 2.** A submission is built into a server and evaluated with AI-BMT in two Arena
jobs. Both use the `--concurrency` you chose, and each is limited to 5 minutes including
server startup.

- **Accuracy:** every one of the 16 private cases must pass, and the job must finish within
  the 5 minutes. With a concurrency of 16 or more, the 16 cases are sent
  `⌊concurrency / 16⌋ + 1` times in a row, so that new requests are admitted while others
  are still running, and every case must pass every time.
- **Performance:** the server is measured on five input types, in this order: text prompts
  of 256, 1024, and 2048 tokens, an image prompt, and an audio prompt. For each type it receives `concurrency` warmup requests, then `3 × concurrency` measured requests with `concurrency`
  of them in flight at a time: a new one starts as soon as one finishes. Every request sets
  `ignore_eos` and `max_completion_tokens: 64`, so each response has 64 tokens. The speedup on a
  type is the server's generated tokens/s divided by the baseline's, and the score is the
  geometric mean of the five speedups. If the server's TTFT p90 over the measured requests
  exceeds the limit on a type, or it does not finish that type within the 5 minutes, its
  speedup there counts as 1.

A higher concurrency leaves more room to raise throughput, but it also makes the TTFT limit
harder to meet.

## Toolchain

The supported development environment is x86_64 Ubuntu 22.04 or newer with GLIBC 2.34 or
newer:

```sh
sudo apt install build-essential libclang-dev
sudo apt install gcc-aarch64-linux-gnu

rustup toolchain install nightly-2026-05-01
cargo +nightly-2026-05-01 install cargo-binstall
cargo +nightly-2026-05-01 binstall cargo-furiosa-opt@0.8.1
cargo install furiosa-schedule-viewer
```

### Furiosa Arena CLI

Configure the Furiosa Arena CLI once before using `scripts/rngd_test.sh`:

```sh
cargo binstall furiosa-arena-cli

furiosa-arena login
```

The cli commands used for troubleshooting are:

| Command | Purpose |
|---|---|
| `furiosa-arena submit <file>` | Submit a script or binary |
| `furiosa-arena status <id>` | Check job state |
| `furiosa-arena logs <id> --follow` | Stream job output |
| `furiosa-arena list` | List your jobs |
| `furiosa-arena cancel <id>` | Cancel a queued or running job |

### Baseline source code per `furiosa-opt-std` version

The baseline is published for each supported `furiosa-opt-std` version. Each version's
baseline code is available under the matching git tag:

| `furiosa-opt-std` | Tag | Checkout |
|---|---|---|
| 0.8.1 | `v0.8.1` | `git checkout v0.8.1` |
| 0.6.0 | `v0.6.0` | `git checkout v0.6.0` |

`main` tracks the latest supported version.

## Source code sharing and competition fairness

Publicly sharing competition source code before the official deadline is strictly
prohibited. If publicly shared code is copied or plagiarized by another participant, both
the code provider and the user will be immediately disqualified from receiving any awards.
If you currently have any competition code publicly accessible, please set it to private or
remove it immediately. Thank you for your active cooperation in maintaining a fair
competition environment.

## References

- [Programming Tensor Contraction Processors](https://developer.furiosa.ai/furiosa-opt/book)
  — mappings, movement, computation, scheduling, and tuning.
- [`furiosa_opt_std` API docs](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/)
  — tensor types, mapping expressions, and engine modules.
- [furiosa-arena-cli docs](https://github.com/kreatinj/furiosa-arena-cli#installation)
- [moa-submitter-cli](https://github.com/micro2026-moa/moa-submitter-cli) — the submission CLI.
- [OPTIMIZATION.md](OPTIMIZATION.md) — the Stage 1 kernel optimization workflow.
- [ARCHITECTURE.md](ARCHITECTURE.md) — repository layout and host/RNGD split.
- [SERVING.md](SERVING.md) — running the model as an HTTP server.
