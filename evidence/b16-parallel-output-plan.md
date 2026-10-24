Control checkpoint4b31cf1 uses ops_batch::{project_sliding_output_batch,
project_full_output_batch}, each computing4/8 K1024 partials sequentially with
BF16 rounding after each dot and each node of a pairwise addition tree.
Actual root width2 profile318052 measured slidingO9.166ms andfullO3.670ms across
40/8 layers, making output projection a concrete remaining cost.

Candidate ports the actual-proven R64 parallel gathered output mechanism to B16.
One active cluster holds16 rows and executes two B8 dot tiles, while H output
shards and4/8 K1024 partials occupy separate slices concurrently. Gather partial
planes and retain exactly the original BF16 tree and channel-scale boundary.
No native quantization, reassociated dot, or host dispatch change is included.

Gate both final outputs and each K1024 partial against dyadic CPU arithmetic,
then nonrepeating randomized inputs against unchanged B16 controls and sampled
CPU F64-dot/BF16-tree reference. Both domains use bounded dyadic values, so each
K1024 partial is exactly representable in F32 before its BF16 rounding. Include all16 distinct rows, signed/nonunit scales,
zero/cancellation cases and two random seeds. MatchedABBA timings determine
adoption; successful compile or kernel speed alone is not a model result.

Initial16cd993 device compile rejected the inherited one-iteration16-row sum
loop: a full-width no-op tile must have literal index0, not16*loop_variable even
when the loop range is0..1. The repair removes those three one-iteration loops
and uses constantzero for full-width views. Dot tile loops remain two B8 rows;
no arithmetic, mapping or rounding changes are made.
