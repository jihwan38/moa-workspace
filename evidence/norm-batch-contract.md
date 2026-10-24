B16 hidden RMSNorm and residual epilogues

`ops_norm_batch::{rmsnorm_b16,post_attention_b16,post_feedforward_b16}` have the
same arguments as the corresponding `ops_prefill` functions. Inputs and outputs
remain `[B,H]`; norm weights are `[H]`; the feedforward layer gate is `[1 # 8]`.
The host replaces only these launch function paths after numerical validation.

Layout `[B,1 # 2,H/480]` places all sixteen rows and their eight hidden fragments
on the 256 slices. Each slice owns 480 hidden values. The code retains the
original sum tree: square and sum each 480-vector in F32, divide each partial by
3840, reduce the eight slice partials, add EPS, then square root. It retains the
original divide-then-weight normalization. This avoids changing rounding by
reducing an entire 3840-vector on one slice.

Normalization commits BF16 before residual addition; the sum commits BF16 before
multiplication by the layer gate; that product commits BF16 again. All three
stages operate in the same distributed layout and store directly to batched HBM,
without gathering a full hidden vector merely to split it again on the host.

The probe compares 368640 output values exactly against original shared scalar
helpers. Two independent random configurations include all-zero and tiny rows,
nonuniform weights and residuals, and positive/negative gates. The warm RMSNorm
launch timing is diagnostic; root measures the complete-model change.
