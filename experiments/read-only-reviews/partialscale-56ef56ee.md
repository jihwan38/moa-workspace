# B16 F4 per960 output-partial scaling review

Frozen56ef56ee has no concrete source-level scale-axis mismatch. UP/GATE scale
HBM[L/240,H/960,L%240] is moved through actual DMA to the identical output row
and H960 part as its packed weight; cluster=L/7680, slice=L/120%64,H/960,
local=L%120. DOWN uses the analogous H row/L960 mapping. Each new per-part scale
VRF holds120 F32values=480bytes. The original output-row scales and token scales
remain after cross-part reduction. Seven unchanged quantizer/GeGLU helpers were
compared by whitespace-normalized body hashes in the companion JSON.

The proposed intended arithmetic is local F32 dot(E4activation,F4lookup), then
per960 F32scale, then the original4/16-way cross-slice sum, token/row scales,
BF16 dot boundary, original globals and GeGLU boundaries. It changes the weight
model; do not use original E4 bit parity as its intended oracle or silently
replace this ordering with elementwise reconstructed E4 weights. The new static
F4 quantizer/ties/zero contract and supplied-weight error are not yet validated.

Root's resource build reported E0599: inter-slice reduction is unavailable after
vector arithmetic in that typestate. A materialized F32 partial followed by the
original reduction is a supported bounded repair, but the complete DMA/liveness
cost must be included. This review makes no successful compile/runtime claim.

The epilogue preserves post-attention→pre-FFN norm→quant/UP/GATE/GeGLU→hidden
quant/DOWN→post-FFN state ordering, with the original retained HBM buffers.
No production source was edited during this review.
