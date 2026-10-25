# BEEF — MOA2026 Stage2 final submission

## 1. Submission to evaluate

    Submission code:

The code is intentionally blank: use BEEF's highest-scoring valid submission at
the deadline, as permitted by the organizer. The verified official baseline is
`00549d43`. **The q020 candidate described below is pending an official result.**
Its local measurements do not replace the baseline's official score.

| Official checkpoint | Concurrency | Private accuracy | TTFT p90 | TPOT p90 | Generated tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00549d43 | 10 | 16/16 | 3734.43ms | 48.67ms | 95.6183 |

The baseline is preserved in [submissions/00549d43](submissions/00549d43).
The q020 source snapshot and build manifest are in
[candidates/q020](candidates/q020). No q020 private-accuracy result or official
score is claimed here.

## 2. Workshop attendance

    Able to attend: Undecided
    Attending member(s): Not yet determined

## 3. Embedded data and startup transformations

The serving source embeds no precomputed q020 weight blobs, calibration corpus,
benchmark answers, or cached request outputs. It derives its additional weights
at each startup solely from the organizer-provided model tensors.

`src/host/native_fp8_weights.rs` first derives blocked E4 FP8 FFN weights and row
scales from the provided FP4 matrices and block scales. Then
`src/host/b16_f4_weights.rs` prepares the UP, GATE and DOWN projections of all
48 layers. For each group of 960 E4 weights,
`src/host/f4_group960_mse.rs` selects among 25 fixed dyadic scales using an integer
histogram and squared-error calculation, and emits packed E2M1 FP4 values plus
the selected scale. Its lookup tables are generated from numeric-format
constants; they contain no calibration examples. Original model tensors and
existing row/global scale factors remain available.

The encoder uses at most four standard-library scoped threads, bounded by
available CPU parallelism and complete 240-row blocks. Threads run the same
serial encoder on disjoint contiguous blocks; ordered joins preserve every
packed byte and scale. This parallelization is bit-identical to the serial MSE
encoder. **The F4 weight approximation itself is not claimed bit-exact to the
previous E4 model.** Downloads, encoding and uploads remain sequential between
projections, and only a complete immutable 48-layer weight set is published.

RoPE tables are generated from model parameters by `runtime::rope_data`.
Diagnostic fixtures and external evaluation answers are not used to select
serving behavior or quantization parameters.

## 4. Execution and request ownership

The additional F4 weights apply only to supported B16 decode calls containing
2–16 independent canonical requests, each supplying one token. Scalar decode,
R32/R64 decode, prompt processing, mixed prompt/decode calls and legacy storage
retain their existing FFN routes. All 48 model layers execute. Token embeddings
and the final B16 vocabulary projection use the original BF16 embedding table;
q020 does not include the later FP8 language-model head.

Sliding and full KV storage remain BF16. There are 32 physical sliding rings,
256 full-attention pages and an 8192-token context limit; the R64 query ABI remains
64 rows. Startup checks the actual owning-buffer extents. Admission follows
available physical capacity, with excess requests queued rather than discarded.
Each request retains independent KV ownership and actual token computation.

Recovery cohorts may hold genuinely computed first events until their members
are ready or the oldest original receipt reaches its 3750ms budget. Fresh eligible
admissions can join an active cohort, but its deadline can only stay unchanged
or shrink. Existing decoders and previously participating unfinished prefills
remain eligible for service. Held ready members do not perform extra decode
steps; all holding time remains part of actual TTFT. Prefill yields between
complete layers, so an atomic operation can overrun a scheduling deadline.

## 5. Reproduction and recorded local evidence

Use the organizer's SDK0.9 baseline environment and provided model/tokenizer files:

```sh
cd candidates/q020
q020_build_root="$(mktemp -d "${TMPDIR:-/tmp}/q020-build.XXXXXX")"
export CARGO_TARGET_DIR="$q020_build_root/cargo"
export FURIOSA_OPT_OUT_DIR="$q020_build_root/device-kernels"
export CARGO_INCREMENTAL=0
mkdir -p "$CARGO_TARGET_DIR" "$FURIOSA_OPT_OUT_DIR"
cargo check --locked --bin server
cargo furiosa-opt build --release --bin server
```

The frozen source revision is `d52f0c08504d4827948abd8536b2d0011645fcce`.
`rust-toolchain.toml` pins `nightly-2026-05-01`; `Cargo.toml` retains the original
dependencies. Use the snapshot's frozen `Cargo.lock`, whose SHA256 is
`a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e`.
That lockfile was Git-ignored in the development checkout, so a bare source
revision alone is insufficient to recover the recorded build inputs.

The captured local server SHA256 is
`ba8cc68d6610a05634319bdb5741017ba4fa79bb4902cc11c77f33d2b163f817`.
The [candidate manifest](candidates/q020/candidate.json) records its build inputs.
The recipe uses fresh package and device-kernel output directories to avoid stale
shared build artifacts. The organizer replaces `src/api/` during source grading;
q020 retains the existing API integration, and its official source-only result is
still pending.

The planned source-only C16 submission, from `candidates/q020`, is:

```sh
moa-submitter submit --source . --concurrency 16
```

This command is documented as a future action; this draft does not assert that
it has run or that q020 has passed official evaluation.

Recorded q020 local evidence consists of:

- Six public fixtures/suites: 124 successful scored responses, including repeated
  fixtures; this is not 124 independent unseen problems.
- A preregistered sample of 32 distinct public GSM8K TRAIN questions: 32/32 correct,
  with fixed requests and scorer and no response-based resampling. This small
  additional diagnostic does not establish private accuracy.
- One C16 performance run with 16 genuine warmup requests followed by 96 measured
  requests: **174.04 generated tokens/s**, TTFT p90 **3765.21ms**, and estimated
  TPOT p90 **38.33ms**. This is a local warm96 result, not an official score or a
  guarantee for the official workload length.

The [HTTP validation record](candidates/q020/evidence/q020-http-validation.json)
binds the public/performance results to the frozen source and ELF. The
[GSM32 report](candidates/q020/evidence/external-gsm32-c081-q020.json) preserves the
fixed sample and paired outcomes. The
[candidate note](candidates/q020/evidence/q020-candidate-note.json) records source
and transformation details, while [preparation.json](candidates/q020/preparation.json)
maps historical log paths to the bundled, hashed evidence files.

Official q020 submission identification, private accuracy and performance will
be recorded only after an actual organizer result is available.
