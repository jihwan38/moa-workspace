# Exact R64 FFN host integration

Base 30038f5. Kernel source 5111dc5 from batch-kernels passed Arena318794:
all scratch/final bits exact, inclusive0.399–0.402ms versus original0.481–0.484ms.
This worktree is isolated; root coordinates hardware and production adoption.

Decode64 owns quantized input245760B, input scales256B, BF16 hidden1966080B,
quantized hidden983040B, hidden scales256B:3195392B total. Allocated once at
model construction, reused only after the awaited prior invocation completes.
Original weights are reused. No request computations are shared. The existing
RMSNorm and post-FFN norm/residual/gate launches are byte-for-byte unchanged.
The original R64 kernel remains an explicit diagnostic control behind
Model::set_r64_staged_ffn; serving default in this candidate uses stage fusion.
B16, prompt execution, caches, admission, logits selection and precision unchanged.

compare_r64_ffn_fused builds64 independent actual512-token histories for each
control/candidate pass, compares all262144 BF16 vocabulary logits per live request,
and covers widths64,17,31 plus4, physical slot/page remapping, full-page transition,
position0 request reuse, and15-token prompt interleave. All bits must match,
all logits must be finite; no numeric tolerance. Stage timings are diagnostic,
not HTTP/SLO or official throughput evidence.
