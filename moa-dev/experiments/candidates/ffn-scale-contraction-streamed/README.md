# Stream FP8 weights into block dots, then contract BF16 block scales

This variant removes the full unscaled BF16 weight matrices from the earlier
[scale-contraction experiment](../ffn-scale-contraction/README.md). It preserves
FP4→FP8 DM decoding. The Main fetch unit converts that FP8 stream to BF16 through
the supported LUT and immediately contracts it with the original BF16 activation.
There is no FP8 activation quantization in this candidate.

For each of32 rows per slice,240 independent16-element FP32 block dots are
rounded to BF16 and packed four at a time. A second BF16 contraction multiplies
them by the exact BF16 widening of their FP8 block scales, reduces only the block
axis, and applies the original BF16 row-output cast. GeGLU, down and the E16
parallel residual/layer-scalar path are unchanged. The extra BF16 block rounding
means this is approximate, even though weight widening itself is exact.

The exact FFN schedule is171,727 cycles. The same E16-tail core without the new
rounding is167,296, and the earlier fully materialized scale-contraction candidate
was207,851. Removing the large BF16 matrix buffer substantially improves that
prototype, but the static schedule still regresses2.65% against the verified core.
Attempt1 fetched only16 FP8 values before lookup and failed fetch stride lowering;
attempt2 fetches a contiguous H3840 row before LUT and then collects16 BF16 values.
The compile logs and source patch are retained.

Arena job70327 (E23) passed all3 FFN comparisons, with median384,728 cycles
(396,175/384,728/382,583). This is about2.0% below the E18 repeated median392,608,
but it is only one measurement and additional inputs have not been checked.
Maximum FFN errors were0.00586/0.00977/0.01562, each within the official combined
absolute/relative tolerance for every element. The combined package failed its
unrelated QKV V outputs, so no valid combined score is claimed.

`mlp.rs` is the frozen candidate; `ops.rs`/`residual.rs` are the exact E23 full-build
callers. The earlier exact scratch used identical FFN caller/residual bodies but
different QKV code. Source and schedule hashes, comparisons and hardware status
are in `evidence.json`. Scratch compilation is isolated at
`/tmp/moa-ffn-scale-streamed`. This source was independently derived from the
block-dot identity and the previously documented primary-source research;
no external competition implementation was copied.
