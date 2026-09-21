# E56: sixteen-row up-weight staging

**Rejected after Arena job72023 failed FFN correctness.** Exact compilation predicted
162,209 → 159,236 cycles (−1.83%), but the first FFN input had only30.96% of
outputs within tolerance (max absolute error0.63281). A subsequent device launch
exceeded its time limit and the process exited101. QKV/O passed all12 comparisons.
There is no valid FFN cycle measurement or speedup for this candidate.
[Arena result](../../../../docs/experiments/e56-up16-screen.json).

The immutable E55 package was rerun as job72024 alongside the candidate. It passed
all15 comparisons at104,432/51,283/380,911 median cycles. This is one control run,
not a replacement for the earlier three-job E55 aggregate.
[Control result](../../../../docs/experiments/e55-up16-control.json).

## Implementation and hypothesis

Base: `baseline-e55`, commit873f64ed94229c88cd18d2731bdcca7eea0d6365. The live
crate remains E55. The isolated candidate changes only `src/device/shared/mlp.rs`:

`decoder_feedforward → feedforward_sharded → project_up_and_gate` loads up weights
in two16-row chunks. Each chunk follows FP4 lookup→FP8→F32 cast→BF16 storage,
then two unchanged eight-row dot/FP32 local-scale/reduction/BF16-output pipelines.
Gate, GeGLU, down and the output epilogue remain byte-identical to the base.

The earlier [eight-row staging](../ffn-up-tiled-fused-decode-e47/README.md) increased
DMA and lookup setup costs. Sixteen rows halves the number of staged chunks while
retaining the existing dot size. Logical immutable tensors do not imply a fixed
physical buffer count or force prefetch ordering.

For cluster`c`, live slice`s`, chunk offset`t∈{0,16}`, and local row`r<16`, the
global weight row is `7680*c + 32*s + t + r`. Scale/output indices retain the full
32-row offset; dot reads use offsets0/8 inside the compact16-row BF16 buffer.
[verify.py](verify.py) checks all15,360 rows, independent column factors and the
unchanged arithmetic suffix. This source-level check passed but did **not**
establish hardware correctness, as the actual failure demonstrates.

The first compile rejected a tile index covering32 rows on a compact16-row buffer.
The second uses the restricted axis `L % 32 = 16` and compiles successfully.
[Failed attempt](../../artifacts/ffn-up-sixteen-row-e55/compile-attempt1.log),
[successful attempt](../../artifacts/ffn-up-sixteen-row-e55/compile-attempt2.log).

## Static result and evidence

| Compiler schedule metric | E55 | Earlier8-row candidate | This16-row candidate |
|---|---:|---:|---:|
| Complete makespan |162,209|162,644|159,236|
| Main interval union |125,274|114,280|113,754|
| DMA interval union |135,607|138,928|136,156|
| Main/DMA overlap |104,921|97,581|97,254|

Up decode work fell from29,326 to17,806 cycles; up-weight DMA grew from26,475 to
27,024. These overlapping costs cannot be summed into a runtime improvement.
The first down dot waits until141,428 despite BF16 down weights being ready at
139,924, because its scale DMA finishes at140,827. See
[schedule summary](schedule-summary.json) and [analyze.py](analyze.py).

[Build evidence](build-evidence.json) records the46 actual compiler inputs, log
hashes and the uploaded host/package fingerprints. QKV/O images match E55; the
FFN image in the host matches the separately compiled exact image.
No numerical root cause has been isolated. The current failure does not establish
that fused conversion, tiling, or any individual allocator choice caused it.

## Reproduction

Run `python3 prepare.py` from this directory to reconstruct the isolated crate
from the E55 tag and pinned Cargo.lock; then run `python3 verify.py`. The builder
requires the live crate's inputs to still match that tag. Docker compilation used:

```bash
cd /tmp/moa-ffn-up-sixteen-row-e55
export CARGO_HOME=/home/moa/.cargo
export CARGO_NET_OFFLINE=true
export CARGO_TARGET_DIR=/tmp/moa-ffn-up-sixteen-row-e55/target
export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-up-sixteen-row-e55/target/furiosa-opt/kernel
cargo furiosa-opt compile ops::decoder_feedforward --exact \
  --dump-schedule candidate.json --dump-graph candidate-graph.json
cargo furiosa-opt build --release --bin test_kernels
```

The Arena package used the unchanged E55 fixtures and runner, with a70-second job
limit. Raw schedules, binaries, build logs and the frozen crate are in
`moa-dev/experiments/artifacts/ffn-up-sixteen-row-e55/`; these large artifacts are
ignored by Git. Candidate source, patch, manifests and results are tracked.
