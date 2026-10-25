# B16 F4 with fitted dyadic scales: complete micro

Base `ae7b102f` is the measured ceil-max fixed-F4 candidate/intended reference
from r1204/r1205. All device kernels remain byte-identical. Only the CPU static
encoder is replaced with `f4_group960_mse` from8e7ce99e (25 deterministic
candidates, exact histogram SSE, baseline retained on ties). Packed F4 is
expanded through the explicit16-code fixed E2M1→E4 table for the intended dense
reference; fitted group scales stay separate and multiply local F32 partials.

Original source/results are untouched in the previous worktree. This new test
retains all provided layers0/5/6/47, two feedback steps, true row permutations,
eight candidate-first retained-state comparisons, independent F64 partial/final
samples, original-E4 approximation reports, immutable weights/scales and
complete-epilogue ABBA. CPU fitting/upload is outside device timers as before.

This gate specifically checks actual non-power-of-two dyadic scale arithmetic
in the unchanged runtime graph. CPU no-worse weight SSE does not imply
no-worse activations, residuals, selections or task quality. No host/model
activation or Arena dispatch by this agent.
