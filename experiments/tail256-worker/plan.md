# Tail256 Worker integration

Base: root `a052254` (batched media and exact head-attention fusion enabled).
Model source: `7490200`, ported as `0f1be62`. The source-only merge preserves the
root's default `PreparedWide::fused_head_attention` initialization for both real
states. Group execution loops over real states only; four-owner validation is unchanged.

Model evidence before scheduler changes: Arena319230 case6 (single256 at1792,
48 R64 interleaves) and319233 case8 (two128 at1920, NaN inactive tensor quarters)
passed all logits, full caches, untouched sentinel and reversed reset. Complete
model time was189.14→135.10ms and190.84→136.06ms respectively. These gates used
the original Tail256 snapshot; the composition with root's enabled head fusion
still requires the rebuilt model oracle. Other boundary cases remain parent-queued.

Selection order remains cyclic oldest-owner order: eligible first-four real128
quad, then legal full512 two-owner pair, then oldest singleton512. When those do
not apply, two aligned complete128 ranges may use Tail256; otherwise an aligned
oldest request may consume an advertised256 range. A short or unaligned oldest
request keeps its original turn. No younger-owner search, new wait, extra
reservation or arrival-dependent timeout is introduced.

A Tail256 checkpoint retains one or two actual ScheduledInput ranges across all
48 layer yields. Only completed whole layers yield to independent decode. The
existing admission reservation, startup-cohort membership/receipt deadline,
terminal-event handling and atomic complete_work validation remain unchanged.
Both the pending credit prediction and measured recurring cost classify Tail256
as width4, because existing dense kernels execute512 tensor rows. Preparation,
finalization and reads remain separately measured and do not train layer cost.

Required host tests cover priority, disabled fallback, short/unaligned boundaries,
cyclic order, exact range retention through session-vector compaction, cancelled
owners, wrong output shape before mutation, final-only output mapping, finite
cohort receipt deadline and conservative cost/decode responsiveness. Hardware
composition and extended HTTP gates remain parent-coordinated before adoption.
