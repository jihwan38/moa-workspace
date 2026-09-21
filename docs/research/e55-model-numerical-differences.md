# Original/E55 model arithmetic and next controls

The [four-prefix model comparison](../experiments/e55-model-prefix-suite.md)
observes finite logits with the same raw top1 but nonzero differences. This
source audit identifies arithmetic changes that can contribute; it does not
attribute the measured maximum1.5 to any one change. The original implementation
is a controlled comparison point, not an independently validated exact oracle.

The frozen original and E55 inputs in the
[suite source audit](../../moa-dev/experiments/candidates/model-prefix-suite-e55/README.md)
differ in seven Rust files. All ten public text-operation signatures are unchanged.
Let `B` mean a BF16 rounding boundary.

| Kernel | Observed source-level arithmetic change |
|---|---|
| Sliding QKV | Original `B(scale × B(dot3840))` becomes `B(scale × F32dot3840)`. E55 moves the first BF16 cast after channel scale. Input activation remains BF16 and FP8 weight lookup preserves its values in BF16. |
| Sliding O | Original four1024-column dot results are rounded to BF16, combined with a BF16 add tree, then channel-scaled and rounded again. E55 combines sixteen256-column F32 partials in F32, scales, then performs the first BF16 cast. Both intermediate rounding and reduction grouping differ. |
| FFN | Original forms `B(FP4 × local_scale)` weights before a long dot. E55 computes16-element dots with unscaled BF16 FP4 values, applies each local scale in F32, then reduces blocks. Up/gate have240 blocks; down has120 per column followed by the existing eight-column F32 reduction. |
| Full O | No explicit arithmetic change found. The shared residual's480-element×8 loop is expanded. The public full-O body, full projection/attention, generic RMS, and elementwise F32 add→BF16 sequence are unchanged. |

Current E55 Q/O do **not** use the earlier lossy FP8 activation hi/lo paths.
References to rejected E47/E53 activation approximations must not be applied to
this checkpoint. The current implementation is in
[sliding/projection.rs](../../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs),
[shared/mlp.rs](../../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs),
[shared/rmsnorm.rs](../../furiosa-opt-gemma4-12B/src/device/shared/rmsnorm.rs), and
[shared/residual.rs](../../furiosa-opt-gemma4-12B/src/device/shared/residual.rs).

GeGLU, generic shared `normalize`, `scale_by_layer_gate`, and the RoPE arithmetic
suffix are byte-identical between these frozen sources. FFN global-scale
positions and the BF16 boundaries after projection, residual addition, and layer
scaling remain. O/FFN post-RMS still uses eight480-element partial reductions,
but replaces an inter-slice reduction with gather plus an intra-slice reduction;
F32 summation order is another candidate difference. Q input RMS retains its
previous inter-slice grouping.

The earlier exhaustive check of4,064 finite FP4×FP8 products established that
those products are exactly representable in BF16. Thus the FFN transformation
does not by itself add finite weight quantization loss. It still changes F32
distribution and summation order and does not prove bit identity; extreme
unscaled intermediate overflow remains a separate limitation. Each model token
executes40 sliding layers and48 FFNs. Changes can propagate through later layers
and KV caches, so separate final-logit effects need not add linearly.

The next bounded experiments are four separate E55 controls, each restoring one
kernel kind to the original. Restore its original operation body and necessary
private helpers while preserving public signatures and host/loader inputs.
For sliding O and FFN, use dedicated original residual helpers so full O is not
changed unintentionally. For full O alone, restoring the generic
`shared::residual::add` loop is a particularly clear control: the prior
[shared-caller audit](../experiments/e47-shared-caller-audit.md) obtained the exact
original full-O image with that change.

Before interpreting any control as a single-kernel comparison, require its target
image to match the original and its other nine text images to match E55. Reuse
the same fixed-prefix suite and asset identity checks. No such control has been
built or run in this audit, and no new optimization has been adopted from it.
